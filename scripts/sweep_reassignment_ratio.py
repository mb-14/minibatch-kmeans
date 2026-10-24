#!/usr/bin/env python3
"""Sweep reassignment_ratio ρ for local MiniBatchKMeans vs fastkmeans baseline.

Simulated: Gaussian-mixture scenarios (well_separated, overlapping, imbalanced).
Real: the default tasks in ``evaluation.presets.BENCHMARK_TASKS`` (same reference logic as evaluation.run).

Writes JSON lines or a JSON array with ARI/NMI, Δ vs fastkmeans; minibatch batch size from
``--batch-size`` (capped at ``n`` per dataset).

Sim mode precomputes ``n_iter`` / ``batch_size`` once per ``(n, k, d)`` grid row (not once per
scenario). Synthetic data is built **once per** ``(scenario, n, k, d)`` (see ``--data-seed``),
then reused for all trial seeds and ρ values. Optional ``--sim-cache-dir`` stores those tensors
so repeated sweeps skip Gaussian generation.

Trials: use ``--seed`` and ``--n-trials`` (seeds ``seed``, ``seed+1``, …) like
``evaluation.run``; or set ``--seeds`` to a comma list to override. For **sim** mode,
those values seed **k-means only**; Gaussian data uses ``--data-seed`` (default: same as
``--seed``) so every trial sees the same ``(x, y_true)`` per grid cell.

Default simulated ``(n, k, d)`` grid spans up to **n=2e6**, **k=1024**, **d=4096** (a
non-exhaustive set of triples, not a full factorial). Override with ``--sim-grid-json``. The largest settings
need very large CPU/GPU memory for float32 data.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Literal

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch
from tqdm import tqdm

from evaluation.backends.fast_kmeans import run_fast_kmeans
from evaluation.backends.minibatch_kmeans import run_minibatch_kmeans
from evaluation.data import gaussian_mixture, gaussian_mixture_imbalanced
from evaluation.datasets import load_clustbench_dataset
from evaluation.heuristics import scaled_niter
from evaluation.metrics import (
    ari_score,
    assign_labels_torch,
    clustbench_best_ari_and_nmi,
    nmi_score,
)
from evaluation.presets import BENCHMARK_TASKS

ScenarioName = Literal["well_separated", "overlapping", "imbalanced"]

DEFAULT_RHO_GRID = [
    0.0,
    0.001,
    0.01,
    0.1,
    0.3,
]

# Sparse grid: cover n up to 2e6, k up to 1024, d up to 4096 (not a full factorial).
# Largest tuple (2e6, 1024, 4096) needs on the order of tens of GB RAM for float32 X alone.
DEFAULT_SIM_GRID: list[tuple[int, int, int]] = [
    (262_144, 512, 512),
    (1_048_576, 512, 1024),
    (2_000_000, 256, 128),
    (2_000_000, 512, 256),
    (100_000, 1024, 256),
    (512_000, 1024, 512),
    (2_000_000, 1024, 1024),
    (64_000, 256, 4096),
    (200_000, 512, 4096),
    (2_000_000, 1024, 4096),
]

QUICK_SIM_GRID: list[tuple[int, int, int]] = DEFAULT_SIM_GRID[:3]
QUICK_RHO = [0.0, 0.01, 0.1]


def _load_sim_grid_json(path: Path) -> list[tuple[int, int, int]]:
    """Load ``[[n,k,d], ...]`` from JSON; validates ``n >= k >= 1``, ``d >= 1``."""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        raise ValueError("sim grid JSON must be a list of [n, k, d] triples")
    out: list[tuple[int, int, int]] = []
    for i, row in enumerate(raw):
        if not isinstance(row, (list, tuple)) or len(row) != 3:
            raise ValueError(f"sim grid row {i}: expected [n, k, d], got {row!r}")
        n, k, d = (int(row[0]), int(row[1]), int(row[2]))
        if n < 1 or k < 1 or d < 1:
            raise ValueError(f"sim grid row {i}: n, k, d must be >= 1")
        if n < k:
            raise ValueError(f"sim grid row {i}: need n >= k, got n={n}, k={k}")
        out.append((n, k, d))
    if not out:
        raise ValueError("sim grid JSON is empty")
    return out


def _make_sim_data(
    scenario: ScenarioName,
    n: int,
    k: int,
    d: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if scenario == "well_separated":
        return gaussian_mixture(n, k, d, seed=seed, center_scale=10.0, noise_scale=5.0)
    if scenario == "overlapping":
        return gaussian_mixture(n, k, d, seed=seed, center_scale=8.0, noise_scale=14.0)
    if scenario == "imbalanced":
        return gaussian_mixture_imbalanced(n, k, d, seed=seed, center_scale=10.0, noise_scale=5.0)
    raise ValueError(scenario)


def _sim_grid_rows_with_niter_and_batch(
    grid: list[tuple[int, int, int]],
    batch_size: int,
) -> list[tuple[int, int, int, int, int]]:
    """One row per valid ``(n, k, d)`` with ``niter`` and effective ``batch_size`` (``min(n, batch_size)``)."""
    rows: list[tuple[int, int, int, int, int]] = []
    for n, k, d in grid:
        if n < k:
            continue
        niter = scaled_niter(n)
        bs = int(min(n, batch_size))
        rows.append((n, k, d, niter, bs))
    return rows


def _sim_cache_path(
    cache_dir: Path,
    scenario: str,
    n: int,
    k: int,
    d: int,
    data_seed: int,
) -> Path:
    """Cache file for synthetic data (keyed by data RNG, not k-means trial seed)."""
    return cache_dir / f"{scenario}_n{n}_k{k}_d{d}_dseed{data_seed}.pt"


def _load_or_make_sim_data(
    scenario: ScenarioName,
    n: int,
    k: int,
    d: int,
    data_seed: int,
    *,
    cache_dir: Path | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate Gaussian-mixture data, or load from ``cache_dir`` when present (skips generation)."""
    if cache_dir is not None:
        path = _sim_cache_path(cache_dir, scenario, n, k, d, data_seed)
        if path.is_file():
            try:
                payload = torch.load(path, map_location="cpu")
                x = torch.as_tensor(payload["x"], dtype=torch.float32)
                y_true = torch.as_tensor(payload["y_true"], dtype=torch.int64)
            except Exception:
                pass
            else:
                if x.shape == (n, d) and y_true.shape == (n,):
                    return x, y_true

    x, y_true = _make_sim_data(scenario, n, k, d, data_seed)
    x = x.cpu()
    y_true = y_true.cpu()
    if cache_dir is not None:
        path = _sim_cache_path(cache_dir, scenario, n, k, d, data_seed)
        cache_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"x": x, "y_true": y_true}, path)
    return x, y_true


def _metrics_from_centroids(
    x: torch.Tensor,
    y_true: torch.Tensor,
    centroids: torch.Tensor,
    *,
    device: torch.device,
    chunk_rows: int,
) -> tuple[float, float]:
    pred = assign_labels_torch(
        x,
        centroids,
        chunk_rows=chunk_rows,
        eval_device=device if x.device != device else None,
    )
    yt = y_true.to(device=pred.device)
    ari = ari_score(yt, pred)
    nmi = nmi_score(yt, pred)
    return float(ari), float(nmi)


def run_sim_sweep(
    *,
    scenarios: list[ScenarioName],
    grid: list[tuple[int, int, int]],
    rho_values: list[float],
    seeds: list[int],
    data_seed: int,
    device: torch.device,
    eval_chunk_rows: int,
    batch_size: int,
    progress: bool = True,
    sim_cache_dir: Path | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    grid_rows = _sim_grid_rows_with_niter_and_batch(grid, batch_size)
    grid_points = len(grid_rows)
    n_cfg = len(scenarios) * grid_points
    per_cfg = len(seeds) * len(rho_values)
    outer = tqdm(
        total=n_cfg,
        desc="sim sweep",
        unit="cfg",
        disable=not progress or n_cfg == 0,
        file=sys.stderr,
    )
    try:
        for scenario in scenarios:
            for n, k, d, niter, bs in grid_rows:
                outer.set_postfix_str(f"{scenario} n={n} k={k} d={d}", refresh=False)
                x, y_true = _load_or_make_sim_data(
                    scenario,
                    n,
                    k,
                    d,
                    data_seed,
                    cache_dir=sim_cache_dir,
                )

                inner = tqdm(
                    total=per_cfg,
                    desc="  seed×rho",
                    unit="row",
                    leave=False,
                    disable=not progress or per_cfg == 0,
                    file=sys.stderr,
                )
                try:
                    for seed in seeds:
                        torch.manual_seed(seed)
                        if device.type == "cuda":
                            torch.cuda.manual_seed_all(seed)

                        fast_out = run_fast_kmeans(x, k, niter, seed, device)
                        fast_c = fast_out["centroids"]
                        assert fast_c is not None
                        fa, fn = _metrics_from_centroids(
                            x, y_true, fast_c, device=device, chunk_rows=eval_chunk_rows
                        )

                        for rho in rho_values:
                            inner.set_postfix_str(f"seed={seed} rho={rho}", refresh=False)
                            mb_out = run_minibatch_kmeans(
                                x,
                                k,
                                niter,
                                seed,
                                device,
                                batch_size=bs,
                                reassignment_ratio=rho,
                            )
                            mb_c = mb_out["centroids"]
                            ma, mn = _metrics_from_centroids(
                                x, y_true, mb_c, device=device, chunk_rows=eval_chunk_rows
                            )
                            rows.append(
                                {
                                    "kind": "sim",
                                    "scenario": scenario,
                                    "n": n,
                                    "k": k,
                                    "d": d,
                                    "data_seed": data_seed,
                                    "seed": seed,
                                    "niter": niter,
                                    "reassignment_ratio": rho,
                                    "batch_size": bs,
                                    "device": str(device),
                                    "fastkmeans_ari": fa,
                                    "fastkmeans_nmi": fn,
                                    "minibatch_ari": ma,
                                    "minibatch_nmi": mn,
                                    "delta_ari": ma - fa,
                                    "delta_nmi": mn - fn,
                                    "minibatch_wall_s": mb_out["wall_time_s"],
                                    "fastkmeans_wall_s": fast_out["wall_time_s"],
                                }
                            )
                            inner.update(1)
                finally:
                    inner.close()
                outer.update(1)
    finally:
        outer.close()
    return rows


def run_real_sweep(
    *,
    tasks: list[tuple[str, str]],
    rho_values: list[float],
    seeds: list[int],
    device: torch.device,
    eval_chunk_rows: int,
    data_path: str,
    batch_size: int,
    progress: bool = True,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    outer = tqdm(
        tasks,
        desc="real sweep",
        unit="dataset",
        disable=not progress,
        file=sys.stderr,
    )
    try:
        for battery, dataset in outer:
            outer.set_postfix_str(f"{battery}/{dataset}", refresh=False)
            try:
                loaded = load_clustbench_dataset(
                    battery,
                    dataset,
                    path=data_path,
                )
            except Exception as e:
                print(
                    f"sweep_reassignment_ratio: skip {battery}/{dataset}: {e}",
                    file=sys.stderr,
                )
                continue

            x_cpu = loaded.x
            n, d = x_cpu.shape
            ref_labels = loaded.labels
            n_clusters = loaded.n_clusters
            niter = scaled_niter(n)

            valid_ref_indices = [
                ref_idx
                for ref_idx in range(len(ref_labels))
                if int(n_clusters[ref_idx]) >= 1 and n >= int(n_clusters[ref_idx])
            ]
            n_ref_cfg = len(valid_ref_indices)
            per_ref = len(seeds) * len(rho_values)
            mid = tqdm(
                valid_ref_indices,
                desc=f"  {battery}/{dataset}",
                unit="cfg",
                leave=False,
                disable=not progress or n_ref_cfg == 0,
                file=sys.stderr,
            )
            try:
                for ref_idx in mid:
                    k = int(n_clusters[ref_idx])
                    bs = int(min(n, batch_size))
                    mid.set_postfix_str(f"k={k} ref={ref_idx}", refresh=False)

                    inner = tqdm(
                        total=per_ref,
                        desc="    seed×rho",
                        unit="row",
                        leave=False,
                        disable=not progress or per_ref == 0,
                        file=sys.stderr,
                    )
                    try:
                        for seed in seeds:
                            torch.manual_seed(seed)
                            if device.type == "cuda":
                                torch.cuda.manual_seed_all(seed)

                            fast_out = run_fast_kmeans(x_cpu, k, niter, seed, device)
                            fast_c = fast_out["centroids"]
                            assert fast_c is not None
                            pred_f = assign_labels_torch(
                                x_cpu,
                                fast_c,
                                chunk_rows=eval_chunk_rows,
                                eval_device=device if x_cpu.device != device else None,
                            )
                            y_np_f = pred_f.cpu().numpy().astype(np.int64)
                            fa, fn = clustbench_best_ari_and_nmi([ref_labels[ref_idx]], y_np_f, k)

                            for rho in rho_values:
                                inner.set_postfix_str(f"seed={seed} rho={rho}", refresh=False)
                                mb_out = run_minibatch_kmeans(
                                    x_cpu,
                                    k,
                                    niter,
                                    seed,
                                    device,
                                    batch_size=bs,
                                    reassignment_ratio=rho,
                                )
                                mb_c = mb_out["centroids"]
                                pred_m = assign_labels_torch(
                                    x_cpu,
                                    mb_c,
                                    chunk_rows=eval_chunk_rows,
                                    eval_device=device if x_cpu.device != device else None,
                                )
                                y_np_m = pred_m.cpu().numpy().astype(np.int64)
                                ma, mn = clustbench_best_ari_and_nmi([ref_labels[ref_idx]], y_np_m, k)

                                rows.append(
                                    {
                                        "kind": "real",
                                        "battery": battery,
                                        "dataset": dataset,
                                        "ref_index": ref_idx,
                                        "n": n,
                                        "k": k,
                                        "d": d,
                                        "seed": seed,
                                        "niter": niter,
                                        "reassignment_ratio": rho,
                                        "batch_size": bs,
                                        "device": str(device),
                                        "fastkmeans_ari": fa,
                                        "fastkmeans_nmi": fn,
                                        "minibatch_ari": ma,
                                        "minibatch_nmi": mn,
                                        "delta_ari": ma - fa,
                                        "delta_nmi": mn - fn,
                                        "minibatch_wall_s": mb_out["wall_time_s"],
                                        "fastkmeans_wall_s": fast_out["wall_time_s"],
                                    }
                                )
                                inner.update(1)
                    finally:
                        inner.close()
            finally:
                mid.close()
    finally:
        outer.close()
    return rows


def _parse_rho_list(raw: str) -> list[float]:
    parts = [p.strip() for p in raw.replace(",", " ").split() if p.strip()]
    return [float(p) for p in parts]


def _sim_scenarios_and_grid(args: argparse.Namespace) -> tuple[list[ScenarioName], list[tuple[int, int, int]]]:
    if args.quick:
        return ["well_separated"], QUICK_SIM_GRID
    if args.sim_grid_json is not None:
        sim_grid = _load_sim_grid_json(args.sim_grid_json)
    else:
        sim_grid = list(DEFAULT_SIM_GRID)
    return ["well_separated", "overlapping", "imbalanced"], sim_grid


def _print_aligned_table(
    headers: list[str],
    rows: list[list[object]],
    *,
    right_align: list[bool],
) -> None:
    """Print a fixed-width table; ``right_align[i]`` selects rjust vs ljust for column ``i``."""
    if len(headers) != len(right_align):
        raise ValueError("headers and right_align length mismatch")
    str_rows = [[str(x) for x in row] for row in rows]
    ncols = len(headers)
    widths = [len(headers[i]) for i in range(ncols)]
    for row in str_rows:
        for i in range(ncols):
            widths[i] = max(widths[i], len(row[i]))
    sep = "  "

    def _cell(s: str, col: int) -> str:
        w = widths[col]
        return s.rjust(w) if right_align[col] else s.ljust(w)

    print(sep.join(_cell(headers[i], i) for i in range(ncols)), flush=True)
    for row in str_rows:
        print(sep.join(_cell(row[i], i) for i in range(ncols)), flush=True)


def _print_sim_heuristics(*, grid: list[tuple[int, int, int]], batch_size: int) -> None:
    """One row per distinct ``(n, k, d)`` (batch_size is ``min(n, batch_size)``)."""
    headers = ["kind", "n", "k", "d", "n_iter", "batch_size"]
    right_align = [False, True, True, True, True, True]
    seen: set[tuple[int, int, int]] = set()
    rows: list[list[object]] = []
    for n, k, d in grid:
        if n < k:
            continue
        key = (n, k, d)
        if key in seen:
            continue
        seen.add(key)
        ni = scaled_niter(n)
        bs = int(min(n, batch_size))
        rows.append(["sim", n, k, d, ni, bs])
    _print_aligned_table(headers, rows, right_align=right_align)


def _print_real_heuristics(
    *,
    tasks: list[tuple[str, str]],
    data_path: str,
    batch_size: int,
) -> None:
    headers = ["kind", "battery", "dataset", "ref_index", "n", "k", "d", "n_iter", "batch_size"]
    right_align = [False, False, False, True, True, True, True, True, True]
    rows: list[list[object]] = []
    for battery, dataset in tasks:
        try:
            loaded = load_clustbench_dataset(
                battery,
                dataset,
                path=data_path,
            )
        except Exception as e:
            print(
                f"sweep_reassignment_ratio: skip {battery}/{dataset}: {e}",
                file=sys.stderr,
            )
            continue
        x_cpu = loaded.x
        n, d = int(x_cpu.shape[0]), int(x_cpu.shape[1])
        n_clusters = loaded.n_clusters
        for ref_idx in range(len(loaded.labels)):
            k = int(n_clusters[ref_idx])
            if k < 1 or n < k:
                continue
            ni = scaled_niter(n)
            bs = int(min(n, batch_size))
            rows.append(["real", battery, dataset, ref_idx, n, k, d, ni, bs])
    _print_aligned_table(headers, rows, right_align=right_align)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--mode",
        choices=("sim", "real", "both"),
        default="sim",
        help="sim: Gaussian mixtures; real: clustbench; both: run sim then real",
    )
    p.add_argument("--device", type=str, default=None, help="torch device (default: cuda:0 or cpu)")
    p.add_argument(
        "--eval-chunk-rows",
        type=int,
        default=8192,
        metavar="R",
        help="Chunk rows for label assignment",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=16_384,
        metavar="B",
        help="Minibatch batch size for MiniBatchKMeans (default: 16384; capped at n per dataset)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base k-means RNG seed when using --n-trials (default: 42; trials use seed, seed+1, …). "
        "Sim Gaussian data uses --data-seed instead (default: same as this value).",
    )
    p.add_argument(
        "--data-seed",
        type=int,
        default=None,
        metavar="S",
        help="RNG seed for sim Gaussian mixtures only (default: --seed). Ignored outside sim mode.",
    )
    p.add_argument(
        "--n-trials",
        type=int,
        default=3,
        metavar="N",
        help="Number of independent runs with seeds seed, seed+1, …, seed+N-1 (default: 5). Ignored if --seeds is set.",
    )
    p.add_argument(
        "--seeds",
        type=str,
        default=None,
        metavar="LIST",
        help="Comma-separated seeds; if set, overrides --seed and --n-trials.",
    )
    p.add_argument(
        "--rho",
        type=str,
        default=None,
        help=f"Space/comma-separated ρ values (default: {DEFAULT_RHO_GRID})",
    )
    p.add_argument(
        "--sim-grid-json",
        type=Path,
        default=None,
        metavar="PATH",
        help="JSON list of [n,k,d] triples (n>=k); overrides built-in sim grid. Ignored with --quick.",
    )
    p.add_argument(
        "--sim-cache-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="Load/store simulated (x, y_true) per (scenario, n, k, d, --data-seed); skips "
        "Gaussian generation on cache hits. Sim mode only.",
    )
    p.add_argument(
        "--quick",
        action="store_true",
        help="Small grid, three ρ, two seeds (for smoke tests)",
    )
    p.add_argument(
        "--data-path",
        type=str,
        default=None,
        metavar="DIR",
        help="Local clustering-data-v1 root (required for --mode real or both)",
    )
    p.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Write JSON array to this path",
    )
    p.add_argument(
        "--json-lines",
        action="store_true",
        help="Print one JSON object per line to stdout (default: compact JSON array)",
    )
    p.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars on stderr (for logs / CI)",
    )
    p.add_argument(
        "--print-params",
        action="store_true",
        help="Print derived n_iter and batch_size for sim grid and/or real datasets, then exit (no fit/eval).",
    )
    args = p.parse_args(argv)

    if args.mode in ("real", "both") and args.data_path is None:
        print(
            "sweep_reassignment_ratio: --data-path is required for --mode real or both.",
            file=sys.stderr,
        )
        sys.exit(1)

    scenarios: list[ScenarioName]
    sim_grid: list[tuple[int, int, int]]
    try:
        scenarios, sim_grid = _sim_scenarios_and_grid(args)
    except (OSError, ValueError, json.JSONDecodeError) as e:
        print(f"sweep_reassignment_ratio: --sim-grid-json: {e}", file=sys.stderr)
        sys.exit(1)

    if args.print_params:
        if args.mode in ("sim", "both"):
            print("## sim", file=sys.stderr, flush=True)
            _print_sim_heuristics(grid=sim_grid, batch_size=args.batch_size)
        if args.mode in ("real", "both"):
            try:
                import clustbench  # noqa: F401
            except ImportError:
                print(
                    "sweep_reassignment_ratio: clustering-benchmarks required for real mode. "
                    'pip install -e ".[eval]"',
                    file=sys.stderr,
                )
                sys.exit(1)
            print("## real", file=sys.stderr, flush=True)
            _print_real_heuristics(
                tasks=list(BENCHMARK_TASKS),
                data_path=args.data_path,
                batch_size=args.batch_size,
            )
        return

    if args.eval_chunk_rows < 1:
        print("sweep_reassignment_ratio: --eval-chunk-rows must be >= 1.", file=sys.stderr)
        sys.exit(1)
    if args.n_trials < 1:
        print("sweep_reassignment_ratio: --n-trials must be >= 1.", file=sys.stderr)
        sys.exit(1)
    if args.batch_size < 1:
        print("sweep_reassignment_ratio: --batch-size must be >= 1.", file=sys.stderr)
        sys.exit(1)

    dev = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))

    rho_values = _parse_rho_list(args.rho) if args.rho else (QUICK_RHO if args.quick else DEFAULT_RHO_GRID)

    def _resolve_seed_list(*, quick: bool) -> list[int]:
        if args.seeds is not None:
            out = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
            if not out:
                print("sweep_reassignment_ratio: --seeds must list at least one integer.", file=sys.stderr)
                sys.exit(1)
            return out
        if quick:
            return [args.seed + i for i in range(2)]
        return [args.seed + i for i in range(args.n_trials)]

    seed_list = _resolve_seed_list(quick=args.quick)
    sim_data_seed = args.data_seed if args.data_seed is not None else args.seed

    all_rows: list[dict[str, Any]] = []

    show_progress = not args.no_progress

    if args.mode in ("sim", "both"):
        all_rows.extend(
            run_sim_sweep(
                scenarios=scenarios,
                grid=sim_grid,
                rho_values=rho_values,
                seeds=seed_list,
                data_seed=sim_data_seed,
                device=dev,
                eval_chunk_rows=args.eval_chunk_rows,
                batch_size=args.batch_size,
                progress=show_progress,
                sim_cache_dir=args.sim_cache_dir,
            )
        )

    if args.mode in ("real", "both"):
        try:
            import clustbench  # noqa: F401
        except ImportError:
            print(
                "sweep_reassignment_ratio: clustering-benchmarks required for real mode. "
                'pip install -e ".[eval]"',
                file=sys.stderr,
            )
            sys.exit(1)
        tasks = list(BENCHMARK_TASKS)
        all_rows.extend(
            run_real_sweep(
                tasks=tasks,
                rho_values=rho_values,
                seeds=seed_list,
                device=dev,
                eval_chunk_rows=args.eval_chunk_rows,
                data_path=args.data_path,
                batch_size=args.batch_size,
                progress=show_progress,
            )
        )

    if args.json_lines:
        for row in all_rows:
            print(json.dumps(row), flush=True)
    else:
        print(json.dumps(all_rows))

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(all_rows, f, indent=2)


if __name__ == "__main__":
    main()
