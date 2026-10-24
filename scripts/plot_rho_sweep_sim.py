#!/usr/bin/env python3
"""Visualize minibatch − fastkmeans (ΔARI, ΔNMI) from rho_sweep_sim.json.

Aggregates mean ± seed stdev, then plots vs reassignment_ratio ρ for each
sim scenario, across (n, k) configs."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError as e:
    print(f"plot_rho_sweep_sim: need matplotlib and numpy: {e}", file=sys.stderr)
    sys.exit(1)

SCENARIOS = ("imbalanced", "overlapping", "well_separated")


def _fmt_n(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M".replace(".0M", "M")
    if n >= 1000:
        return f"{n // 1000}k"
    return str(n)


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


def aggregate(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """One dict per (scenario, n, k, rho) with mean/std over seeds."""
    groups: dict[tuple[str, int, int, float], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        if r.get("kind") != "sim":
            continue
        key = (
            str(r["scenario"]),
            int(r["n"]),
            int(r["k"]),
            float(r["reassignment_ratio"]),
        )
        groups[key].append(r)

    out: list[dict[str, Any]] = []
    for key in sorted(groups.keys()):
        gs = groups[key]
        da = [float(x["delta_ari"]) for x in gs]
        dn = [float(x["delta_nmi"]) for x in gs]
        ma, sa = _mean_std(da)
        mn, sn = _mean_std(dn)
        out.append(
            {
                "scenario": key[0],
                "n": key[1],
                "k": key[2],
                "rho": key[3],
                "mean_delta_ari": ma,
                "std_delta_ari": sa,
                "mean_delta_nmi": mn,
                "std_delta_nmi": sn,
                "n_seeds": len(gs),
            }
        )
    return out


def _configs_for_scenario(agg: list[dict[str, Any]], scenario: str) -> list[tuple[int, int]]:
    cfgs = {
        (int(a["n"]), int(a["k"]))
        for a in agg
        if a["scenario"] == scenario
    }
    return sorted(cfgs, key=lambda t: (t[1], t[0]))


def _k_linestyle(k: int) -> str:
    if k <= 256:
        return "-"
    if k <= 512:
        return "--"
    return "-."


def plot_delta_lines(
    agg: list[dict[str, Any]],
    out_path: Path,
) -> None:
    """2 rows (ΔARI, ΔNMI) × 3 cols (scenarios): lines per (n,k), ρ on x."""
    rhos = sorted({float(a["rho"]) for a in agg})
    x = np.arange(len(rhos))
    ns_global = sorted({int(a["n"]) for a in agg})
    try:
        tab10 = plt.colormaps["tab10"]
    except AttributeError:
        tab10 = plt.cm.get_cmap("tab10")
    n_colors = {n: tab10(i % 10) for i, n in enumerate(ns_global)}

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True)
    for col, scen in enumerate(SCENARIOS):
        cfgs = _configs_for_scenario(agg, scen)
        ax_ari = axes[0, col]
        ax_nmi = axes[1, col]
        for n, k in cfgs:
            ys_ari: list[float] = []
            ys_nmi: list[float] = []
            e_ari: list[float] = []
            e_nmi: list[float] = []
            for rho in rhos:
                row = next(
                    (
                        a
                        for a in agg
                        if a["scenario"] == scen
                        and int(a["n"]) == n
                        and int(a["k"]) == k
                        and float(a["rho"]) == rho
                    ),
                    None,
                )
                if row is None:
                    ys_ari.append(float("nan"))
                    ys_nmi.append(float("nan"))
                    e_ari.append(float("nan"))
                    e_nmi.append(float("nan"))
                else:
                    ys_ari.append(row["mean_delta_ari"])
                    ys_nmi.append(row["mean_delta_nmi"])
                    e_ari.append(row["std_delta_ari"])
                    e_nmi.append(row["std_delta_nmi"])
            label = f"n={_fmt_n(n)} k={k}"
            color = n_colors[n]
            ls = _k_linestyle(k)
            ax_ari.plot(
                x,
                ys_ari,
                marker="o",
                linestyle=ls,
                color=color,
                linewidth=1.4,
                markersize=4,
                label=label,
                alpha=0.9,
            )
            ax_ari.errorbar(
                x,
                ys_ari,
                yerr=e_ari,
                fmt="none",
                ecolor=color,
                elinewidth=1.0,
                capsize=2,
                alpha=0.85,
            )
            ax_nmi.plot(
                x,
                ys_nmi,
                marker="o",
                linestyle=ls,
                color=color,
                linewidth=1.4,
                markersize=4,
                label=label,
                alpha=0.9,
            )
            ax_nmi.errorbar(
                x,
                ys_nmi,
                yerr=e_nmi,
                fmt="none",
                ecolor=color,
                elinewidth=1.0,
                capsize=2,
                alpha=0.85,
            )
        ax_ari.axhline(0.0, color="k", linewidth=0.7, alpha=0.35)
        ax_nmi.axhline(0.0, color="k", linewidth=0.7, alpha=0.35)
        ax_ari.set_title(scen.replace("_", " "))
        ax_ari.set_ylabel("ΔARI (minibatch − fastkmeans)")
        ax_nmi.set_ylabel("ΔNMI (minibatch − fastkmeans)")
        ax_nmi.set_xlabel("ρ (reassignment_ratio)")
        ax_ari.grid(True, alpha=0.3)
        ax_nmi.grid(True, alpha=0.3)
        if col == 2:
            ax_ari.legend(
                bbox_to_anchor=(1.02, 1),
                loc="upper left",
                fontsize=7,
                title="(n, k)",
                title_fontsize=8,
            )
            ax_nmi.legend(
                bbox_to_anchor=(1.02, 1),
                loc="upper left",
                fontsize=7,
                title="(n, k)",
                title_fontsize=8,
            )
        ax_nmi.set_xticks(x)
        ax_nmi.set_xticklabels([str(r) for r in rhos], rotation=30, ha="right")

    fig.suptitle(
        "Sim sweep: metric gap vs ρ (mean ± stdev over seeds)",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_heatmaps(
    agg: list[dict[str, Any]],
    out_path: Path,
    metric: str,
) -> None:
    """3×3 grid: rows = scenario, cols = k; y = n, x = ρ, color = Δ metric."""
    assert metric in ("ari", "nmi")
    field = "mean_delta_ari" if metric == "ari" else "mean_delta_nmi"
    rhos = sorted({float(a["rho"]) for a in agg})
    ks = [256, 512, 1024]

    fig, axes = plt.subplots(3, 3, figsize=(12, 10), sharex=True, layout="constrained")
    vmin = min(a[field] for a in agg)
    vmax = max(a[field] for a in agg)
    # symmetric diverging scale around 0 when both signs
    if vmin < 0 < vmax:
        m = max(abs(vmin), abs(vmax))
        vmin, vmax = -m, m

    for i, scen in enumerate(SCENARIOS):
        for j, k in enumerate(ks):
            ax = axes[i, j]
            rows_n = sorted(
                {
                    int(a["n"])
                    for a in agg
                    if a["scenario"] == scen and int(a["k"]) == k
                },
                reverse=True,
            )
            if not rows_n:
                ax.set_visible(False)
                continue
            mat = np.full((len(rows_n), len(rhos)), np.nan)
            for ri, n in enumerate(rows_n):
                for ci, rho in enumerate(rhos):
                    row = next(
                        (
                            a
                            for a in agg
                            if a["scenario"] == scen
                            and int(a["n"]) == n
                            and int(a["k"]) == k
                            and float(a["rho"]) == rho
                        ),
                        None,
                    )
                    if row is not None:
                        mat[ri, ci] = row[field]
            im = ax.imshow(
                mat,
                aspect="auto",
                cmap="RdBu_r",
                vmin=vmin,
                vmax=vmax,
                origin="upper",
            )
            ax.set_xticks(range(len(rhos)))
            ax.set_xticklabels([str(r) for r in rhos], rotation=35, ha="right", fontsize=8)
            ax.set_yticks(range(len(rows_n)))
            ax.set_yticklabels([_fmt_n(n) for n in rows_n], fontsize=8)
            if i == 0:
                ax.set_title(f"k = {k}")
            if j == 0:
                ax.set_ylabel(f"{scen.replace('_', ' ')}\nn", fontsize=9)
            if i == 2:
                ax.set_xlabel("ρ")

    mappable = plt.cm.ScalarMappable(
        cmap="RdBu_r",
        norm=plt.Normalize(vmin=vmin, vmax=vmax),
    )
    mappable.set_array([])
    cbar = fig.colorbar(
        mappable,
        ax=axes.ravel().tolist(),
        shrink=0.55,
        label=f"mean Δ{'ARI' if metric == 'ari' else 'NMI'}",
    )
    fig.suptitle(
        f"Δ{'ARI' if metric == 'ari' else 'NMI'} heatmap (minibatch − fastkmeans)",
        fontsize=12,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input",
        type=Path,
        default=_ROOT / "reports" / "rho_sweep_sim.json",
        help="Sweep JSON (sim rows)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=_ROOT / "figures" / "rho_sweep_sim",
        help="Directory for PNG outputs",
    )
    args = p.parse_args()

    rows = _load_rows(args.input)
    agg = aggregate(rows)
    if not agg:
        print("No aggregated rows; check JSON.", file=sys.stderr)
        sys.exit(1)

    out = args.output_dir
    plot_delta_lines(agg, out / "delta_vs_rho_lines.png")
    plot_heatmaps(agg, out / "heatmap_delta_ari.png", "ari")
    plot_heatmaps(agg, out / "heatmap_delta_nmi.png", "nmi")
    print(f"Wrote:\n  {out / 'delta_vs_rho_lines.png'}\n  {out / 'heatmap_delta_ari.png'}\n  {out / 'heatmap_delta_nmi.png'}")


if __name__ == "__main__":
    main()
