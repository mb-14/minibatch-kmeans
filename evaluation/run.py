"""CLI: clustering quality evaluation on clustbench datasets (ARI / NMI vs references)."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from typing import Any

import numpy as np
import torch

from evaluation.backends.faiss_backend import run_faiss_kmeans
from evaluation.backends.fast_kmeans import run_fast_kmeans
from evaluation.backends.flash_kmeans import run_flash_kmeans
from evaluation.backends.methods import (
    FAISSKMEANS,
    FASTKMEANS,
    FLASHKMEANS,
    MINIBATCHKMEANS,
    SKLEARNKMEANS,
    SKLEARNMINIBATCHKMEANS,
    VALID_METHODS,
    MethodName,
)
from evaluation.backends.minibatch_kmeans import run_minibatch_kmeans
from evaluation.backends.sklearn_backend import (
    run_sklearn_kmeans,
    run_sklearn_minibatch_kmeans,
)
from evaluation.cli import fail, non_negative_float, positive_int, require_clustbench
from evaluation.datasets import LoadedBenchmark, load_clustbench_dataset_or_skip
from evaluation.heuristics import scaled_niter
from evaluation.metrics import assign_labels_torch, clustbench_best_ari_and_nmi
from evaluation.presets import BENCHMARK_TASKS

PROG = "evaluation.run"


def _parse_methods(
    raw: list[str] | None,
    *,
    device: torch.device,
) -> list[MethodName]:
    if not raw:
        candidates: list[MethodName] = [
            FASTKMEANS,
            MINIBATCHKMEANS,
            SKLEARNMINIBATCHKMEANS,
        ]
        if device.type == "cuda":
            candidates.insert(0, FLASHKMEANS)
        try:
            import faiss  # noqa: F401
        except ImportError:
            pass
        else:
            candidates.append(FAISSKMEANS)
        return candidates
    out: list[MethodName] = []
    for m in raw:
        if m == FLASHKMEANS and device.type != "cuda":
            print(
                f"{PROG}: skipping flashkmeans (requires CUDA device).",
                file=sys.stderr,
            )
            continue
        out.append(m)  # type: ignore[arg-type]
    if not out:
        print(f"{PROG}: no methods left to run.", file=sys.stderr)
        sys.exit(1)
    return out


def _fit_centroids(
    method: MethodName,
    x: torch.Tensor,
    k: int,
    niter: int,
    seed: int,
    device: torch.device,
    *,
    batch_size: int,
    reassignment_ratio: float | None = None,
    reassign_centers: bool = True,
) -> torch.Tensor:
    n, _d = x.shape
    mb_bs = int(min(n, batch_size))
    if method == FLASHKMEANS:
        out = run_flash_kmeans(x, k, niter, seed, device)
    elif method == FASTKMEANS:
        out = run_fast_kmeans(x, k, niter, seed, device)
    elif method == MINIBATCHKMEANS:
        out = run_minibatch_kmeans(
            x,
            k,
            niter,
            seed,
            device,
            batch_size=mb_bs,
            reassign_centers=reassign_centers,
        )
    elif method == SKLEARNKMEANS:
        out = run_sklearn_kmeans(x, k, niter, seed, device)
    elif method == SKLEARNMINIBATCHKMEANS:
        out = run_sklearn_minibatch_kmeans(
            x,
            k,
            niter,
            seed,
            device,
            batch_size=mb_bs,
            reassignment_ratio=reassignment_ratio,
        )
    elif method == FAISSKMEANS:
        out = run_faiss_kmeans(x, k, niter, seed, device)
    else:
        raise ValueError(method)
    c = out.get("centroids")
    if c is None:
        raise RuntimeError(f"{method} returned no centroids")
    return torch.as_tensor(c, dtype=torch.float32)


def _mean_stdev(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    m = statistics.mean(values)
    if len(values) == 1:
        return m, 0.0
    return m, statistics.stdev(values)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluate k-means backends on benchmark datasets (best ARI/NMI over references).",
    )
    p.add_argument(
        "--data-path",
        type=str,
        required=True,
        metavar="DIR",
        help="Clustering-benchmarks suite root (folders sipu/, mnist/, …). "
        "See https://clustering-benchmarks.gagolewski.com/weave/clustbench-usage.html",
    )
    p.add_argument(
        "--methods",
        nargs="*",
        default=None,
        choices=sorted(VALID_METHODS),
        metavar="M",
        help="Backends: flashkmeans fastkmeans minibatchkmeans sklearnkmeans "
        "sklearnminibatchkmeans faisskmeans (default: all applicable).",
    )
    p.add_argument("--device", type=str, default=None, help="torch device (default: cuda:0 or cpu)")
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base RNG seed; trial i uses seed+i (default: 42).",
    )
    p.add_argument(
        "--n-trials",
        type=positive_int,
        default=1,
        metavar="N",
        help="Number of independent runs (seeds seed..seed+N-1). Mean ± sample stdev on best ARI/NMI.",
    )
    p.add_argument(
        "--reassignment-ratio",
        type=non_negative_float,
        default=0.1,
        metavar="R",
        help="reassignment_ratio for sklearnminibatchkmeans (default: 0.1).",
    )
    p.add_argument(
        "--no-reassign-centers",
        dest="reassign_centers",
        action="store_false",
        help="Run minibatchkmeans with reassign_centers=False (default: on).",
    )
    p.add_argument(
        "--batch-size",
        type=positive_int,
        default=16_384,
        metavar="B",
        help="Minibatch size for minibatchkmeans and sklearnminibatchkmeans (default: 16384; capped at n per dataset).",
    )
    p.add_argument(
        "--eval-chunk-rows",
        type=positive_int,
        default=8192,
        metavar="R",
        help="Row chunk size for assignment to centroids.",
    )
    p.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Write JSON results to this path.",
    )
    return p


def _best_score(values: list[float]) -> float | None:
    """Max ignoring NaN scores; None if there are none."""
    return max((v for v in values if not math.isnan(v)), default=None)


def _scores_over_references(
    method: MethodName,
    loaded: LoadedBenchmark,
    seed: int,
    niter: int,
    dev: torch.device,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    """Fit once per reference partition (its own k) and score against that reference."""
    x_cpu = loaded.x
    n = x_cpu.shape[0]
    per_ref: list[dict[str, Any]] = []
    for ref_idx, ref in enumerate(loaded.labels):
        k = int(loaded.n_clusters[ref_idx])
        if k < 1 or n < k:
            print(f"  skip ref {ref_idx}: invalid k={k} or n={n}", file=sys.stderr)
            continue
        torch.manual_seed(seed)
        if dev.type == "cuda":
            torch.cuda.manual_seed_all(seed)

        try:
            centroids = _fit_centroids(
                method,
                x_cpu,
                k,
                niter,
                seed,
                dev,
                batch_size=args.batch_size,
                reassignment_ratio=args.reassignment_ratio,
                reassign_centers=args.reassign_centers,
            )
        except Exception as e:  # noqa: BLE001 - one failed backend run must not abort the batch
            print(f"  {method} seed {seed} ref {ref_idx} (k={k}) failed: {e}", file=sys.stderr)
            continue

        pred = assign_labels_torch(
            x_cpu,
            centroids,
            chunk_rows=args.eval_chunk_rows,
            eval_device=dev if x_cpu.device != dev else None,
        )
        y_np = pred.cpu().numpy().astype(np.int64)
        ari, nmi = clustbench_best_ari_and_nmi([ref], y_np, k)
        per_ref.append({"ref_index": ref_idx, "k": k, "ari": ari, "nmi": nmi})
    return per_ref


def _method_row(
    battery: str,
    dataset: str,
    method: MethodName,
    loaded: LoadedBenchmark,
    seeds: list[int],
    dev: torch.device,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Result row for one (dataset, method): best ARI/NMI per seed, then mean ± stdev."""
    n, d = loaded.x.shape
    niter = scaled_niter(n)
    by_seed: list[dict[str, Any]] = []
    per_ref: list[dict[str, Any]] = []
    for seed in seeds:
        per_ref = _scores_over_references(method, loaded, seed, niter, dev, args)
        by_seed.append(
            {
                "seed": seed,
                "best_ari": _best_score([r["ari"] for r in per_ref]),
                "best_nmi": _best_score([r["nmi"] for r in per_ref]),
            }
        )

    ari_vals = [float(x["best_ari"]) for x in by_seed if x["best_ari"] is not None]
    nmi_vals = [float(x["best_nmi"]) for x in by_seed if x["best_nmi"] is not None]
    ari_mean, ari_std = _mean_stdev(ari_vals)
    nmi_mean, nmi_std = _mean_stdev(nmi_vals)

    row: dict[str, Any] = {
        "battery": battery,
        "dataset": dataset,
        "n": n,
        "d": d,
        "method": method,
        "niter": niter,
        "device": str(dev),
        "n_trials": args.n_trials,
        "seeds": seeds,
        "best_ari": ari_mean if ari_vals else None,
        "best_nmi": nmi_mean if nmi_vals else None,
        "best_ari_std": ari_std if ari_vals else None,
        "best_nmi_std": nmi_std if nmi_vals else None,
        "by_seed": by_seed,
        # Per-reference detail is only meaningful for a single seed.
        "references": per_ref if len(seeds) == 1 else [],
    }
    if method in (MINIBATCHKMEANS, SKLEARNMINIBATCHKMEANS):
        row["batch_size"] = int(min(n, args.batch_size))
    return row


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    require_clustbench(PROG)

    dev = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    methods = _parse_methods(args.methods, device=dev)
    seeds: list[int] = [args.seed + i for i in range(args.n_trials)]

    rows: list[dict[str, Any]] = []
    n_loaded = 0
    for battery, dataset in BENCHMARK_TASKS:
        print(f"Loading {battery}/{dataset}...", flush=True)
        loaded = load_clustbench_dataset_or_skip(battery, dataset, path=args.data_path, prog=PROG)
        if loaded is None:
            continue
        n_loaded += 1
        rows.extend(
            _method_row(battery, dataset, method, loaded, seeds, dev, args) for method in methods
        )

    if n_loaded == 0:
        fail(
            PROG,
            f"none of the {len(BENCHMARK_TASKS)} datasets could be loaded from "
            f"--data-path {args.data_path!r}; check the path.",
        )

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2)
    elif rows:
        json.dump(rows, sys.stdout, indent=2)
        print(file=sys.stdout)


if __name__ == "__main__":
    main()
