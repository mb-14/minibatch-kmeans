"""CLI: clustering quality evaluation on clustbench datasets (ARI / NMI vs references)."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from typing import Any

import numpy as np
import torch

from evaluation.backends.fast_kmeans import run_fast_kmeans
from evaluation.backends.flash_kmeans import run_flash_kmeans
from evaluation.backends.minibatch_kmeans import run_minibatch_kmeans
from evaluation.backends.faiss_backend import run_faiss_kmeans
from evaluation.backends.sklearn_backend import (
    run_sklearn_kmeans,
    run_sklearn_minibatch_kmeans,
)
from evaluation.metrics import assign_labels_torch, clustbench_best_ari_and_nmi
from evaluation.datasets import load_clustbench_dataset
from evaluation.heuristics import scaled_niter
from evaluation.presets import BENCHMARK_TASKS
import clustbench
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
                "evaluation.run: skipping flashkmeans (requires CUDA device).",
                file=sys.stderr,
            )
            continue
        out.append(m)  # type: ignore[arg-type]
    if not out:
        print("evaluation.run: no methods left to run.", file=sys.stderr)
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
            reassignment_ratio=reassignment_ratio,
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


def main(argv: list[str] | None = None) -> None:
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
        type=int,
        default=1,
        metavar="N",
        help="Number of independent runs (seeds seed..seed+N-1). Mean ± sample stdev on best ARI/NMI.",
    )
    p.add_argument(
        "--reassignment-ratio",
        type=float,
        default=0.1,
        metavar="R",
        help="reassignment_ratio for minibatchkmeans and sklearnminibatchkmeans (default: 0.1).",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=16_384,
        metavar="B",
        help="Minibatch size for minibatchkmeans and sklearnminibatchkmeans (default: 16384; capped at n per dataset).",
    )
    p.add_argument(
        "--eval-chunk-rows",
        type=int,
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
    args = p.parse_args(argv)

    if args.eval_chunk_rows < 1:
        print("evaluation.run: --eval-chunk-rows must be >= 1.", file=sys.stderr)
        sys.exit(1)
    if args.reassignment_ratio is not None and args.reassignment_ratio < 0:
        print("evaluation.run: --reassignment-ratio must be >= 0.", file=sys.stderr)
        sys.exit(1)
    if args.batch_size < 1:
        print("evaluation.run: --batch-size must be >= 1.", file=sys.stderr)
        sys.exit(1)
    if args.n_trials < 1:
        print("evaluation.run: --n-trials must be >= 1.", file=sys.stderr)
        sys.exit(1)

    dev = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    methods = _parse_methods(args.methods, device=dev)
    seeds_list: list[int] = [args.seed + i for i in range(args.n_trials)]

    tasks = list(BENCHMARK_TASKS)

    rows: list[dict[str, Any]] = []

    for battery, dataset in tasks:
        print(f"Loading {battery}/{dataset}...", flush=True)
        try:
            loaded = load_clustbench_dataset(
                battery,
                dataset,
                path=args.data_path,
            )
        except Exception as e:
            print(f"evaluation.run: failed to load {battery}/{dataset}: {e}", file=sys.stderr)
            continue

        x_cpu = loaded.x
        n, d = x_cpu.shape
        ref_labels = loaded.labels
        n_clusters = loaded.n_clusters
        niter_ds = scaled_niter(n)

        for method in methods:
            by_seed: list[dict[str, Any]] = []
            references_single: list[dict[str, Any]] = []
            for seed in seeds_list:
                best_ari = float("-inf")
                best_nmi = float("-inf")
                per_ref: list[dict[str, Any]] = []
                for ref_idx in range(len(ref_labels)):
                    k = int(n_clusters[ref_idx])
                    if k < 1 or n < k:
                        print(
                            f"  skip ref {ref_idx}: invalid k={k} or n={n}",
                            file=sys.stderr,
                        )
                        continue
                    torch.manual_seed(seed)
                    if dev.type == "cuda":
                        torch.cuda.manual_seed_all(seed)

                    try:
                        centroids = _fit_centroids(
                            method,
                            x_cpu,
                            k,
                            niter_ds,
                            seed,
                            dev,
                            batch_size=args.batch_size,
                            reassignment_ratio=args.reassignment_ratio,
                        )
                    except Exception as e:
                        print(
                            f"  {method} seed {seed} ref {ref_idx} (k={k}) failed: {e}",
                            file=sys.stderr,
                        )
                        continue

                    pred = assign_labels_torch(
                        x_cpu,
                        centroids,
                        chunk_rows=args.eval_chunk_rows,
                        eval_device=dev if x_cpu.device != dev else None,
                    )
                    y_np = pred.cpu().numpy().astype(np.int64)

                    ari, nmi = clustbench_best_ari_and_nmi([ref_labels[ref_idx]], y_np, k)
                    best_ari = max(best_ari, ari)
                    best_nmi = max(best_nmi, nmi)
                    ref_row: dict[str, Any] = {
                        "ref_index": ref_idx,
                        "k": k,
                        "ari": ari,
                        "nmi": nmi,
                    }
                    per_ref.append(ref_row)

                by_seed.append(
                    {
                        "seed": seed,
                        "best_ari": (best_ari if per_ref and best_ari > float("-inf") else None),
                        "best_nmi": (best_nmi if per_ref and best_nmi > float("-inf") else None),
                    }
                )
                if len(seeds_list) == 1:
                    references_single = per_ref

            ari_vals = [x["best_ari"] for x in by_seed if x["best_ari"] is not None]
            nmi_vals = [x["best_nmi"] for x in by_seed if x["best_nmi"] is not None]
            ari_mean, ari_std = _mean_stdev([float(v) for v in ari_vals])
            nmi_mean, nmi_std = _mean_stdev([float(v) for v in nmi_vals])

            row: dict[str, Any] = {
                "battery": battery,
                "dataset": dataset,
                "n": n,
                "d": d,
                "method": method,
                "niter": niter_ds,
                "device": str(dev),
                "n_trials": args.n_trials,
                "seeds": seeds_list,
                "best_ari": ari_mean if ari_vals else None,
                "best_nmi": nmi_mean if nmi_vals else None,
                "best_ari_std": ari_std if ari_vals else None,
                "best_nmi_std": nmi_std if nmi_vals else None,
                "by_seed": by_seed,
                "references": references_single if len(seeds_list) == 1 else [],
            }
            if method in (MINIBATCHKMEANS, SKLEARNMINIBATCHKMEANS):
                row["batch_size"] = int(min(n, args.batch_size))
            rows.append(row)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2)
    elif rows:
        json.dump(rows, sys.stdout, indent=2)
        print(file=sys.stdout)


if __name__ == "__main__":
    main()
