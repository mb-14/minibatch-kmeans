#!/usr/bin/env python3
"""Visualize minibatch − fastkmeans (ΔARI, ΔNMI) from rho_sweep_real.json.

Aggregates mean ± stdev over seeds per (battery, dataset, ref_index, ρ)."""

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
    print(f"plot_rho_sweep_real: need matplotlib and numpy: {e}", file=sys.stderr)
    sys.exit(1)


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


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One dict per (battery, dataset, ref_index, rho) with mean/std over seeds."""
    groups: dict[tuple[str, str, int, float], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        if r.get("kind") != "real":
            continue
        key = (
            str(r["battery"]),
            str(r["dataset"]),
            int(r["ref_index"]),
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
        r0 = gs[0]
        out.append(
            {
                "battery": key[0],
                "dataset": key[1],
                "ref_index": key[2],
                "rho": key[3],
                "n": int(r0["n"]),
                "k": int(r0["k"]),
                "d": int(r0["d"]),
                "mean_delta_ari": ma,
                "std_delta_ari": sa,
                "mean_delta_nmi": mn,
                "std_delta_nmi": sn,
                "n_seeds": len(gs),
            }
        )
    return out


def _panel_order(agg: list[dict[str, Any]]) -> list[tuple[str, str]]:
    keys = {(a["battery"], a["dataset"]) for a in agg}
    return sorted(keys, key=lambda t: (t[0], t[1]))


def _refs_for_panel(
    agg: list[dict[str, Any]], battery: str, dataset: str
) -> list[int]:
    return sorted(
        {int(a["ref_index"]) for a in agg if a["battery"] == battery and a["dataset"] == dataset}
    )


def plot_delta_lines(agg: list[dict[str, Any]], out_path: Path) -> None:
    """2 rows (ΔARI, ΔNMI) × len(panels) cols: one line per ref_index if multiple."""
    panels = _panel_order(agg)
    rhos = sorted({float(a["rho"]) for a in agg})
    x = np.arange(len(rhos))
    try:
        tab10 = plt.colormaps["tab10"]
    except AttributeError:
        tab10 = plt.cm.get_cmap("tab10")

    ncols = len(panels)
    fig, axes = plt.subplots(2, ncols, figsize=(3.2 * ncols, 6.5), sharex=True, squeeze=False)
    for col, (bat, ds) in enumerate(panels):
        ax_ari = axes[0, col]
        ax_nmi = axes[1, col]
        refs = _refs_for_panel(agg, bat, ds)
        for ri, ref in enumerate(refs):
            color = tab10(ri % 10)
            r0 = next(
                a
                for a in agg
                if a["battery"] == bat and a["dataset"] == ds and int(a["ref_index"]) == ref
            )
            n, k = int(r0["n"]), int(r0["k"])
            ys_ari: list[float] = []
            ys_nmi: list[float] = []
            e_ari: list[float] = []
            e_nmi: list[float] = []
            for rho in rhos:
                row = next(
                    (
                        a
                        for a in agg
                        if a["battery"] == bat
                        and a["dataset"] == ds
                        and int(a["ref_index"]) == ref
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
            label = f"ref {ref}" if len(refs) > 1 else None
            ax_ari.plot(
                x,
                ys_ari,
                marker="o",
                linestyle="-",
                color=color,
                linewidth=1.5,
                markersize=5,
                label=label,
                alpha=0.95,
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
                linestyle="-",
                color=color,
                linewidth=1.5,
                markersize=5,
                label=label,
                alpha=0.95,
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
        title = f"{bat} · {ds}"
        ax_ari.set_title(title, fontsize=10)
        ax_ari.text(
            0.02,
            0.98,
            f"n={_fmt_n(n)}, k={k}",
            transform=ax_ari.transAxes,
            va="top",
            fontsize=8,
            color="0.35",
        )
        ax_ari.set_ylabel("ΔARI" if col == 0 else "")
        ax_nmi.set_ylabel("ΔNMI" if col == 0 else "")
        ax_nmi.set_xlabel("ρ")
        ax_ari.grid(True, alpha=0.3)
        ax_nmi.grid(True, alpha=0.3)
        if len(refs) > 1:
            ax_ari.legend(loc="best", fontsize=7)
            ax_nmi.legend(loc="best", fontsize=7)
        ax_nmi.set_xticks(x)
        ax_nmi.set_xticklabels([str(r) for r in rhos], rotation=30, ha="right", fontsize=8)

    fig.suptitle(
        "Real-data sweep: minibatch − fastkmeans vs ρ (mean ± stdev over seeds)",
        fontsize=12,
        y=1.01,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_heatmap(
    agg: list[dict[str, Any]],
    out_path: Path,
    metric: str,
) -> None:
    """Rows = datasets (battery · dataset), cols = ρ."""
    assert metric in ("ari", "nmi")
    field = "mean_delta_ari" if metric == "ari" else "mean_delta_nmi"
    panels = _panel_order(agg)
    rhos = sorted({float(a["rho"]) for a in agg})
    nrows = len(panels)
    ncols = len(rhos)
    mat = np.full((nrows, ncols), np.nan)
    ylabels: list[str] = []
    for i, (bat, ds) in enumerate(panels):
        ylabels.append(f"{bat} · {ds}")
        ref = _refs_for_panel(agg, bat, ds)[0]
        for j, rho in enumerate(rhos):
            row = next(
                (
                    a
                    for a in agg
                    if a["battery"] == bat
                    and a["dataset"] == ds
                    and int(a["ref_index"]) == ref
                    and float(a["rho"]) == rho
                ),
                None,
            )
            if row is not None:
                mat[i, j] = row[field]

    vmin = float(np.nanmin(mat))
    vmax = float(np.nanmax(mat))
    if vmin < 0 < vmax:
        m = max(abs(vmin), abs(vmax))
        vmin, vmax = -m, m

    fig, ax = plt.subplots(figsize=(max(8, 1.2 * ncols), max(4, 0.55 * nrows)))
    im = ax.imshow(
        mat,
        aspect="auto",
        cmap="RdBu_r",
        vmin=vmin,
        vmax=vmax,
        origin="upper",
    )
    ax.set_xticks(range(ncols))
    ax.set_xticklabels([str(r) for r in rhos], rotation=35, ha="right")
    ax.set_yticks(range(nrows))
    ax.set_yticklabels(ylabels, fontsize=9)
    ax.set_xlabel("ρ (reassignment_ratio)")
    ax.set_ylabel("dataset")
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, label=f"mean Δ{'ARI' if metric == 'ari' else 'NMI'}")
    fig.suptitle(
        f"Δ{'ARI' if metric == 'ari' else 'NMI'} (minibatch − fastkmeans)",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input",
        type=Path,
        default=_ROOT / "reports" / "rho_sweep_real.json",
        help="Sweep JSON (real rows)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=_ROOT / "figures" / "rho_sweep_real",
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
    plot_heatmap(agg, out / "heatmap_delta_ari.png", "ari")
    plot_heatmap(agg, out / "heatmap_delta_nmi.png", "nmi")
    print(
        f"Wrote:\n  {out / 'delta_vs_rho_lines.png'}\n"
        f"  {out / 'heatmap_delta_ari.png'}\n"
        f"  {out / 'heatmap_delta_nmi.png'}"
    )


if __name__ == "__main__":
    main()
