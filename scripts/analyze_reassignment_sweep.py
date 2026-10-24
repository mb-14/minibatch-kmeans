#!/usr/bin/env python3
"""Aggregate sweep JSON: summary stats, ρ curves, heatmaps (Δ vs fastkmeans)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError as e:
    print(f"analyze_reassignment_sweep: need matplotlib and numpy: {e}", file=sys.stderr)
    sys.exit(1)


def _mean_std(vals: list[float]) -> tuple[float, float]:
    if not vals:
        return float("nan"), float("nan")
    m = statistics.mean(vals)
    if len(vals) == 1:
        return m, 0.0
    return m, statistics.stdev(vals)


def _load_rows(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("expected JSON array")
    return data


def _group_key_sim(row: dict[str, Any]) -> tuple[str, str, int, int, int]:
    return (
        row.get("kind", "sim"),
        str(row["scenario"]),
        int(row["n"]),
        int(row["k"]),
        int(row["d"]),
    )


def _group_key_real(row: dict[str, Any]) -> tuple[str, str, str, int, int, int, int]:
    return (
        row.get("kind", "real"),
        str(row["battery"]),
        str(row["dataset"]),
        int(row["ref_index"]),
        int(row["n"]),
        int(row["k"]),
        int(row["d"]),
    )


def build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-cell stats: best ρ by mean minibatch ARI, comparison to ρ=0.01 and fastkmeans."""
    by_sim: dict[tuple[Any, ...], dict[float, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    by_real: dict[tuple[Any, ...], dict[float, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    fast_ari_sim: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    fast_ari_real: dict[tuple[Any, ...], list[float]] = defaultdict(list)

    for row in rows:
        rho = float(row["reassignment_ratio"])
        ma = float(row["minibatch_ari"])
        fa = float(row["fastkmeans_ari"])
        kind = row.get("kind", "sim")
        if kind == "sim":
            key = _group_key_sim(row)
            by_sim[key][rho].append(ma)
            fast_ari_sim[key].append(fa)
        else:
            key = _group_key_real(row)
            by_real[key][rho].append(ma)
            fast_ari_real[key].append(fa)

    def _cell_summary(
        key: tuple[Any, ...],
        by_rho: dict[float, list[float]],
        fast_vals: list[float],
    ) -> dict[str, Any]:
        rho_means = {r: _mean_std(v)[0] for r, v in by_rho.items()}
        if not rho_means:
            return {}
        best_rho = max(rho_means, key=lambda r: rho_means[r])
        ref_01 = rho_means.get(0.01)
        m_fast, s_fast = _mean_std(fast_vals)
        m_best = rho_means[best_rho]
        gap_best = m_best - m_fast
        gap_01 = (ref_01 - m_fast) if ref_01 is not None else None
        return {
            "key": list(key) if isinstance(key, tuple) else key,
            "best_rho": best_rho,
            "mean_ari_at_best_rho": m_best,
            "mean_fastkmeans_ari": m_fast,
            "stdev_fastkmeans_ari": s_fast,
            "gap_ari_minibatch_minus_fast_at_best_rho": gap_best,
            "mean_ari_at_rho_0_01": ref_01,
            "gap_ari_at_rho_0_01": gap_01,
            "rho_grid": sorted(rho_means.keys()),
        }

    sim_cells = [_cell_summary(k, dict(by_sim[k]), fast_ari_sim[k]) for k in by_sim]
    real_cells = [_cell_summary(k, dict(by_real[k]), fast_ari_real[k]) for k in by_real]
    sim_cells = [c for c in sim_cells if c]
    real_cells = [c for c in real_cells if c]

    return {
        "n_rows": len(rows),
        "sim_cells": sim_cells,
        "real_cells": real_cells,
    }


def plot_sim_curves(
    rows: list[dict[str, Any]],
    out_dir: Path,
) -> list[Path]:
    """One PNG per (scenario, n, k, d): ARI vs ρ with fastkmeans reference line."""
    out_dir.mkdir(parents=True, exist_ok=True)
    groups: dict[tuple[str, int, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("kind") != "sim":
            continue
        key = (str(row["scenario"]), int(row["n"]), int(row["k"]), int(row["d"]))
        groups[key].append(row)

    paths: list[Path] = []
    for key, g in sorted(groups.items()):
        scenario, n, k, d = key
        rhos = sorted({float(r["reassignment_ratio"]) for r in g})
        means: list[float] = []
        stds: list[float] = []
        fast_m: list[float] = []
        for rho in rhos:
            chunk = [r for r in g if float(r["reassignment_ratio"]) == rho]
            aris = [float(r["minibatch_ari"]) for r in chunk]
            m, s = _mean_std(aris)
            means.append(m)
            stds.append(s)
            fast_m.extend([float(r["fastkmeans_ari"]) for r in chunk])
        fm, _ = _mean_std(fast_m)

        fig, ax = plt.subplots(figsize=(7, 4))
        x = np.arange(len(rhos))
        ax.errorbar(x, means, yerr=stds, fmt="o-", capsize=3, label="minibatch ARI")
        ax.axhline(fm, color="C1", linestyle="--", label=f"fastkmeans ARI (mean={fm:.4f})")
        ax.set_xticks(x)
        ax.set_xticklabels([str(r) for r in rhos], rotation=45, ha="right")
        ax.set_xlabel("reassignment_ratio ρ")
        ax.set_ylabel("ARI")
        ax.set_title(f"sim:{scenario} n={n} k={k} d={d}")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fname = f"sim_{scenario}_n{n}_k{k}_d{d}.png".replace("/", "_")
        p = out_dir / fname
        fig.savefig(p, dpi=120)
        plt.close(fig)
        paths.append(p)
    return paths


def plot_real_curves(
    rows: list[dict[str, Any]],
    out_dir: Path,
) -> list[Path]:
    """One PNG per (battery, dataset, ref_index) with valid rows."""
    out_dir.mkdir(parents=True, exist_ok=True)
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("kind") != "real":
            continue
        key = (str(row["battery"]), str(row["dataset"]), int(row["ref_index"]))
        groups[key].append(row)

    paths: list[Path] = []
    for key, g in sorted(groups.items()):
        battery, dataset, ref_idx = key
        rhos = sorted({float(r["reassignment_ratio"]) for r in g})
        means: list[float] = []
        stds: list[float] = []
        fast_m: list[float] = []
        for rho in rhos:
            chunk = [r for r in g if float(r["reassignment_ratio"]) == rho]
            aris = [float(r["minibatch_ari"]) for r in chunk]
            m, s = _mean_std(aris)
            means.append(m)
            stds.append(s)
            fast_m.extend([float(r["fastkmeans_ari"]) for r in chunk])
        fm, _ = _mean_std(fast_m)

        fig, ax = plt.subplots(figsize=(7, 4))
        x = np.arange(len(rhos))
        ax.errorbar(x, means, yerr=stds, fmt="o-", capsize=3, label="minibatch ARI")
        ax.axhline(fm, color="C1", linestyle="--", label=f"fastkmeans ARI (mean={fm:.4f})")
        ax.set_xticks(x)
        ax.set_xticklabels([str(r) for r in rhos], rotation=45, ha="right")
        ax.set_xlabel("reassignment_ratio ρ")
        ax.set_ylabel("ARI (clustbench best)")
        ax.set_title(f"{battery}/{dataset} ref={ref_idx}")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        safe = f"real_{battery}_{dataset}_ref{ref_idx}.png".replace("/", "_")
        p = out_dir / safe
        fig.savefig(p, dpi=120)
        plt.close(fig)
        paths.append(p)
    return paths


def plot_delta_heatmap_sim(
    rows: list[dict[str, Any]],
    out_path: Path,
) -> Path | None:
    """2D heatmap: rows = scenario, cols = ρ; value = mean ΔARI vs fastkmeans (aggregated over seeds and n,k,d)."""
    sim_rows = [r for r in rows if r.get("kind") == "sim"]
    if not sim_rows:
        return None
    scenarios = sorted({str(r["scenario"]) for r in sim_rows})
    rhos = sorted({float(r["reassignment_ratio"]) for r in sim_rows})
    mat = np.full((len(scenarios), len(rhos)), np.nan)
    for i, sc in enumerate(scenarios):
        for j, rho in enumerate(rhos):
            deltas = [
                float(r["delta_ari"])
                for r in sim_rows
                if str(r["scenario"]) == sc and float(r["reassignment_ratio"]) == rho
            ]
            if deltas:
                mat[i, j] = statistics.mean(deltas)
    fig, ax = plt.subplots(figsize=(max(8, len(rhos)), max(4, len(scenarios))))
    im = ax.imshow(mat, aspect="auto", cmap="RdBu", vmin=-0.2, vmax=0.2)
    ax.set_xticks(np.arange(len(rhos)))
    ax.set_xticklabels([str(r) for r in rhos], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(scenarios)))
    ax.set_yticklabels(scenarios)
    ax.set_xlabel("ρ")
    ax.set_ylabel("scenario")
    ax.set_title("Mean ΔARI (minibatch − fastkmeans), sim (all n,k,d, seeds)")
    fig.colorbar(im, ax=ax, label="ΔARI")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", "-i", type=Path, required=True, help="JSON array from sweep")
    p.add_argument(
        "--figures-dir",
        type=Path,
        default=None,
        help="Write PNG figures here (default: reports/figures/reassignment_sweep)",
    )
    p.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Write summary JSON (default: stdout only)",
    )
    args = p.parse_args(argv)

    rows = _load_rows(args.input)
    summary = build_summary(rows)
    summary_json = json.dumps(summary, indent=2)
    print(summary_json, flush=True)

    fig_dir = args.figures_dir or Path("reports/figures/reassignment_sweep")
    sim_pngs = plot_sim_curves(rows, fig_dir / "sim")
    real_pngs = plot_real_curves(rows, fig_dir / "real")
    hm = plot_delta_heatmap_sim(rows, fig_dir / "heatmap_sim_delta_ari.png")

    print(
        f"Wrote {len(sim_pngs)} sim curve(s), {len(real_pngs)} real curve(s), "
        f"heatmap={hm is not None} under {fig_dir}",
        file=sys.stderr,
        flush=True,
    )

    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.summary_json, "w", encoding="utf-8") as f:
            f.write(summary_json)


if __name__ == "__main__":
    main()
