"""CLI: efficiency benchmark (wall and process CPU time, optional WCSS / NMI)."""

from __future__ import annotations

import argparse
import json
import sys
import torch

from evaluation.backends.fast_kmeans import run_fast_kmeans
from evaluation.backends.minibatch_kmeans import run_minibatch_kmeans
from evaluation.backends.flash_kmeans import run_flash_kmeans
from evaluation.backends.faiss_backend import run_faiss_kmeans
from evaluation.backends.sklearn_backend import (
    run_sklearn_kmeans,
    run_sklearn_minibatch_kmeans,
)

from evaluation.data import gaussian_mixture, train_val_split
from evaluation.metrics import ari_score, nmi_score, wcss_and_assign_labels_torch
from evaluation.backends.methods import (
    FASTKMEANS,
    FAISSKMEANS,
    FLASHKMEANS,
    MINIBATCHKMEANS,
    MethodName,
    SKLEARNKMEANS,
    SKLEARNMINIBATCHKMEANS,
    VALID_METHODS,
)
from evaluation.nvitop_process_memory import (
    NvitopProcessGpuMemory,
    nvitop_process_gpu_memory_during,
    torch_device_index,
)

# Rows used to fit centroids; remainder is held out for --eval-quality (WCSS / NMI).
_TRAIN_FRACTION = 0.75


def _quality_on_split(
    x: torch.Tensor,
    y: torch.Tensor,
    centroids: torch.Tensor,
    *,
    eval_device: torch.device,
    eval_chunk_rows: int,
) -> tuple[float, float, float, float]:
    """WCSS, WCSS per point, NMI, ARI for ``x`` / ``y`` vs ``centroids``."""
    ev_kw: dict = {"chunk_rows": eval_chunk_rows}
    if x.device != eval_device:
        ev_kw["eval_device"] = eval_device
    wcss, pred = wcss_and_assign_labels_torch(x, centroids, **ev_kw)
    if pred.device != y.device:
        pred = pred.to(y.device)
    n_pts = int(x.shape[0])
    wpp = wcss / n_pts if n_pts > 0 else float("nan")
    return wcss, wpp, nmi_score(y, pred), ari_score(y, pred)


def _benchmark_row(
    method: str,
    out: dict,
    gpu_nvitop: NvitopProcessGpuMemory | None,
    gpu_memory_series: list[dict[str, float | int]] | None,
    gpu_memory_series_interval_s: float | None,
    *,
    n: int,
    n_train: int,
    train_fraction: float,
    k: int,
    d: int,
    niter: int,
    eval_quality: bool,
    eval_device: torch.device,
    eval_chunk_rows: int,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
) -> dict | None:
    w = out["wall_time_s"]
    row: dict = {
        "method": method,
        "n": n,
        "n_train": n_train,
        "train_fraction": train_fraction,
        "k": k,
        "d": d,
        "niter": niter,
        "wall_time_s": w,
        "cpu_time_s": float(out["cpu_time_s"]),
    }
    if "iters_done" in out:
        row["backend_iterations_done"] = out["iters_done"]
    if out.get("batch_size") is not None:
        row["batch_size"] = out["batch_size"]
    if out.get("reassignment_ratio") is not None:
        row["reassignment_ratio"] = out["reassignment_ratio"]
    if out.get("flash_large_n"):
        row["flash_large_n"] = True
    if out.get("timing_repeats") not in (None, 1) or out.get("timing_warmup_runs"):
        row["timing_warmup_runs"] = int(out.get("timing_warmup_runs", 0))
        row["timing_repeats"] = int(out.get("timing_repeats", 1))
    if gpu_nvitop is not None:
        g = gpu_nvitop
        row["gpu_process_baseline_memory_bytes_nvitop"] = g.baseline_bytes
        row["gpu_process_baseline_memory_mib_nvitop"] = round(
            g.baseline_bytes / (1024**2), 3
        )
        row["gpu_process_peak_memory_bytes_nvitop"] = g.peak_bytes
        row["gpu_process_peak_memory_mib_nvitop"] = round(
            g.peak_bytes / (1024**2), 3
        )
        row["gpu_process_peak_memory_delta_bytes_nvitop"] = g.delta_bytes
        row["gpu_process_peak_memory_delta_mib_nvitop"] = round(
            g.delta_bytes / (1024**2), 3
        )
    if gpu_memory_series is not None:
        row["gpu_memory_series_nvitop"] = gpu_memory_series
        if gpu_memory_series_interval_s is not None:
            row["gpu_memory_series_interval_s"] = gpu_memory_series_interval_s
    if eval_quality:
        if out.get("centroids") is None:
            return None
        c = out["centroids"]
        if not isinstance(c, torch.Tensor):
            c = torch.as_tensor(c, dtype=torch.float32)
        c_eval = c.detach()
        ev = eval_device
        n_val = int(x_val.shape[0])
        print(
            f"Eval-quality: WCSS + NMI on train (n={n_train}) and val (n={n_val}) "
            f"for {method} on {ev} (k={k}, d={d}, chunk_rows={eval_chunk_rows})...",
            flush=True,
        )
        wcss_tr, wpp_tr, nmi_tr, ari_tr = _quality_on_split(
            x_train, y_train, c_eval, eval_device=ev, eval_chunk_rows=eval_chunk_rows
        )
        wcss_va, wpp_va, nmi_va, ari_va = _quality_on_split(
            x_val, y_val, c_eval, eval_device=ev, eval_chunk_rows=eval_chunk_rows
        )
        row["quality"] = {
            "train": {
                "n": n_train,
                "wcss": wcss_tr,
                "wcss_per_point": wpp_tr,
                "nmi": nmi_tr,
                "ari": ari_tr,
            },
            "val": {
                "n": n_val,
                "wcss": wcss_va,
                "wcss_per_point": wpp_va,
                "nmi": nmi_va,
                "ari": ari_va,
            },
        }
    return row


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Wall and process CPU time for niter Lloyd iterations (backend-dependent). "
            "On CUDA, JSON may include nvitop per-process GPU memory (baseline, peak, delta) "
            "during the backend fit."
        ),
    )
    p.add_argument("--n", type=int, default=65_536, help="Number of points")
    p.add_argument("--k", type=int, default=1024, help="Number of clusters")
    p.add_argument("--d", type=int, default=128, help="Feature dimension")
    p.add_argument("--seed", type=int, default=42, help="Random seed (data + init)")
    p.add_argument("--niter", type=int, default=1, help="Number of Lloyd iterations")
    p.add_argument(
        "--method",
        choices=sorted(VALID_METHODS),
        default=FLASHKMEANS,
        help="Which backend to benchmark (flash skipped if ENABLE_FLASH_KMEANS is False)",
    )
    p.add_argument("--device", type=str, default="cuda:0", help="Device (e.g. cuda:0)")
    p.add_argument(
        "--batch-size",
        type=int,
        default=16_384,
        metavar="B",
        help="MiniBatchKMeans batch size (default: 16384; capped at n by each backend)",
    )
    p.add_argument(
        "--reassignment-ratio",
        type=float,
        default=0.01,
        metavar="R",
        help="reassignment_ratio for minibatchkmeans and sklearnminibatchkmeans (default: 0.01)",
    )
    p.add_argument(
        "--eval-quality",
        action="store_true",
        help="WCSS + NMI on train (75%%) and val (25%%) holdout",
    )
    p.add_argument(
        "--eval-chunk-rows",
        type=int,
        default=8192,
        metavar="R",
        help="Row chunk size for WCSS / label assignment (avoids OOM on GPU; default: 8192)",
    )
    p.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        metavar="PATH",
        help="Write results as a JSON array to this file (UTF-8)",
    )
    p.add_argument(
        "--gpu-memory-series",
        action="store_true",
        help=(
            "On CUDA, record per-process GPU memory over time (nvitop) during the backend "
            "fit for curve comparison (adds gpu_memory_series_nvitop to JSON)."
        ),
    )
    p.add_argument(
        "--gpu-memory-series-interval",
        type=float,
        default=0.05,
        metavar="SEC",
        help="Approximate spacing between series samples when --gpu-memory-series (default: 0.05).",
    )
    p.add_argument(
        "--no-flash-pin",
        action="store_true",
        help="Do not page-lock the flash-kmeans training matrix before the fit.",
    )
    p.add_argument(
        "--flash-warmup-runs",
        type=int,
        default=2,
        metavar="N",
        help="Uncounted flash-kmeans warmup runs before the clock (default: 2).",
    )
    p.add_argument(
        "--flash-timed-repeats",
        type=int,
        default=2,
        metavar="N",
        help="Timed flash-kmeans runs to average (default: 2).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    n = args.n
    k = args.k
    d = args.d
    seed = args.seed
    niter = args.niter
    method: MethodName = args.method
    device = args.device
    eval_quality = args.eval_quality
    eval_chunk_rows = args.eval_chunk_rows
    output_path = args.output
    batch_size = args.batch_size
    reassignment_ratio = args.reassignment_ratio
    gpu_memory_series = args.gpu_memory_series
    gpu_memory_series_interval_s = float(args.gpu_memory_series_interval)
    flash_pin = not args.no_flash_pin
    flash_warmup_runs = args.flash_warmup_runs
    flash_timed_repeats = args.flash_timed_repeats
    if flash_warmup_runs < 0 or flash_timed_repeats < 1:
        print(
            "evaluation.perf_test: --flash-warmup-runs must be >= 0 and "
            "--flash-timed-repeats must be >= 1.",
            file=sys.stderr,
        )
        sys.exit(1)

    if eval_chunk_rows < 1:
        print("evaluation.perf_test: --eval-chunk-rows must be >= 1.", file=sys.stderr)
        sys.exit(1)

    if batch_size < 1:
        print("evaluation.perf_test: --batch-size must be >= 1.", file=sys.stderr)
        sys.exit(1)

    if reassignment_ratio < 0:
        print("evaluation.perf_test: --reassignment-ratio must be >= 0.", file=sys.stderr)
        sys.exit(1)

    if gpu_memory_series and gpu_memory_series_interval_s <= 0:
        print(
            "evaluation.perf_test: --gpu-memory-series-interval must be > 0.",
            file=sys.stderr,
        )
        sys.exit(1)

    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        print(
            "evaluation.perf_test: --device requests CUDA but torch.cuda.is_available() is False.",
            file=sys.stderr,
        )
        sys.exit(1)

    print("Generating Gaussian mixture data...")
    x_cpu, y_true = gaussian_mixture(n, k, d, seed=seed)

    try:
        x_train, y_train, x_val, y_val = train_val_split(
            x_cpu, y_true, train_fraction=_TRAIN_FRACTION, seed=seed
        )
    except ValueError as e:
        print(f"evaluation.perf_test: {e}", file=sys.stderr)
        sys.exit(1)

    del x_cpu
    n_train = int(x_train.shape[0])
    if n_train < k:
        print(
            f"evaluation.perf_test: need n_train >= k for initialization; got n_train={n_train}, k={k}. "
            "Increase --n.",
            file=sys.stderr,
        )
        sys.exit(1)

    if eval_quality:
        if x_val is None or int(x_val.shape[0]) < 1:
            print("evaluation.perf_test: val split is empty; cannot compute quality.", file=sys.stderr)
            sys.exit(1)

    if (
        flash_pin
        and method == FLASHKMEANS
        and dev.type == "cuda"
        and x_train.device.type == "cpu"
        and not x_train.is_pinned()
    ):
        pinned_train = x_train.contiguous().pin_memory()
        if not eval_quality:
            x_val = None
            y_val = None
        x_train = pinned_train

    print("Running benchmarks...")

    def _run_backend() -> dict:
        if method == FLASHKMEANS:
            return run_flash_kmeans(
                x_train,
                k,
                niter,
                seed,
                dev,
                warmup_runs=flash_warmup_runs,
                timed_repeats=flash_timed_repeats,
            )
        if method == FASTKMEANS:
            return run_fast_kmeans(x_train, k, niter, seed, dev)
        if method == MINIBATCHKMEANS:
            return run_minibatch_kmeans(
                x_train,
                k,
                niter,
                seed,
                dev,
                batch_size=batch_size,
                reassignment_ratio=reassignment_ratio,
            )
        if method == SKLEARNKMEANS:
            return run_sklearn_kmeans(x_train, k, niter, seed, dev)
        if method == SKLEARNMINIBATCHKMEANS:
            return run_sklearn_minibatch_kmeans(
                x_train,
                k,
                niter,
                seed,
                dev,
                batch_size=batch_size,
                reassignment_ratio=reassignment_ratio,
            )
        if method == FAISSKMEANS:
            return run_faiss_kmeans(x_train, k, niter, seed, dev)
        print(f"evaluation.perf_test: unknown method {method!r}.", file=sys.stderr)
        sys.exit(1)

    nvitop_stats: NvitopProcessGpuMemory | None = None
    nvitop_series: list[dict[str, float | int]] | None = None
    series_interval_out: float | None = None
    if dev.type == "cuda":
        out, nvitop_stats, nvitop_series = nvitop_process_gpu_memory_during(
            torch_device_index(dev),
            _run_backend,
            record_series=gpu_memory_series,
            series_interval_s=gpu_memory_series_interval_s,
        )
        if gpu_memory_series:
            series_interval_out = gpu_memory_series_interval_s
    else:
        out = _run_backend()
        if gpu_memory_series:
            print(
                "evaluation.perf_test: --gpu-memory-series ignored on non-CUDA device.",
                file=sys.stderr,
            )

    row = _benchmark_row(
        method,
        out,
        nvitop_stats,
        nvitop_series,
        series_interval_out,
        n=n,
        n_train=n_train,
        train_fraction=_TRAIN_FRACTION,
        k=k,
        d=d,
        niter=niter,
        eval_quality=eval_quality,
        eval_device=dev,
        eval_chunk_rows=eval_chunk_rows,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
    )
    rows = [row] if row is not None else []

    if output_path is not None:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2)

    for r in rows:
        print(json.dumps(r, indent=2))


if __name__ == "__main__":
    main()
