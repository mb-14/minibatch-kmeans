#!/usr/bin/env python3
"""Build aggregate figures and a markdown report from a ρ sweep JSON (e.g. rho_sweep_all.json)."""

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
    print(f"generate_rho_sweep_report: need matplotlib and numpy: {e}", file=sys.stderr)
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


def plot_pooled_delta_vs_rho(
    rows: list[dict[str, Any]],
    out_path: Path,
) -> Path:
    """Mean ΔARI and ΔNMI vs ρ (one point per ρ, across all rows)."""
    rhos = sorted({float(r["reassignment_ratio"]) for r in rows})
    d_ari: list[float] = []
    d_nmi: list[float] = []
    d_ari_std: list[float] = []
    d_nmi_std: list[float] = []
    for rho in rhos:
        chunk = [r for r in rows if float(r["reassignment_ratio"]) == rho]
        a = [float(r["delta_ari"]) for r in chunk]
        n = [float(r["delta_nmi"]) for r in chunk]
        ma, sa = _mean_std(a)
        mn, sn = _mean_std(n)
        d_ari.append(ma)
        d_nmi.append(mn)
        d_ari_std.append(sa)
        d_nmi_std.append(sn)

    x = np.arange(len(rhos))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.errorbar(x, d_ari, yerr=d_ari_std, fmt="o-", capsize=3, label="mean ΔARI (± stdev across configs)")
    ax.errorbar(x, d_nmi, yerr=d_nmi_std, fmt="s-", capsize=3, label="mean ΔNMI")
    ax.axhline(0.0, color="k", linewidth=0.8, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([str(r) for r in rhos], rotation=45, ha="right")
    ax.set_xlabel("reassignment_ratio ρ")
    ax.set_ylabel("minibatch − fastkmeans")
    ax.set_title("Pooled quality gap vs ρ (all configs)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_pooled_ari_vs_rho(
    rows: list[dict[str, Any]],
    out_path: Path,
) -> Path:
    """Mean minibatch ARI vs ρ with fastkmeans reference (horizontal)."""
    rhos = sorted({float(r["reassignment_ratio"]) for r in rows})
    mb: list[float] = []
    fast_vals = [float(r["fastkmeans_ari"]) for r in rows]
    fm, _ = _mean_std(fast_vals)
    for rho in rhos:
        chunk = [r for r in rows if float(r["reassignment_ratio"]) == rho]
        a = [float(r["minibatch_ari"]) for r in chunk]
        mb.append(_mean_std(a)[0])

    x = np.arange(len(rhos))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x, mb, "o-", label="mean minibatch ARI")
    ax.axhline(fm, color="C1", linestyle="--", label=f"mean fastkmeans ARI ({fm:.4f})")
    ax.set_xticks(x)
    ax.set_xticklabels([str(r) for r in rhos], rotation=45, ha="right")
    ax.set_xlabel("reassignment_ratio ρ")
    ax.set_ylabel("ARI")
    ax.set_title("Pooled ARI vs ρ")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_heatmap_battery_rho(
    rows: list[dict[str, Any]],
    out_path: Path,
) -> Path | None:
    real = [r for r in rows if r.get("kind") == "real"]
    if not real:
        return None
    batteries = sorted({str(r["battery"]) for r in real})
    rhos = sorted({float(r["reassignment_ratio"]) for r in real})
    mat = np.full((len(batteries), len(rhos)), np.nan)
    for i, bat in enumerate(batteries):
        for j, rho in enumerate(rhos):
            deltas = [
                float(r["delta_ari"])
                for r in real
                if str(r["battery"]) == bat and float(r["reassignment_ratio"]) == rho
            ]
            if deltas:
                mat[i, j] = statistics.mean(deltas)
    fig, ax = plt.subplots(figsize=(max(8, len(rhos)), max(4, len(batteries) * 0.6)))
    im = ax.imshow(mat, aspect="auto", cmap="RdBu", vmin=-0.08, vmax=0.08)
    ax.set_xticks(np.arange(len(rhos)))
    ax.set_xticklabels([str(r) for r in rhos], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(batteries)))
    ax.set_yticklabels(batteries)
    ax.set_xlabel("ρ")
    ax.set_ylabel("battery")
    ax.set_title("Mean ΔARI (minibatch − fastkmeans), real")
    fig.colorbar(im, ax=ax, label="ΔARI")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_heatmap_dataset_rho(
    rows: list[dict[str, Any]],
    out_path: Path,
) -> Path | None:
    real = [r for r in rows if r.get("kind") == "real"]
    if not real:
        return None
    datasets = sorted({str(r["dataset"]) for r in real})
    rhos = sorted({float(r["reassignment_ratio"]) for r in real})
    mat = np.full((len(datasets), len(rhos)), np.nan)
    for i, ds in enumerate(datasets):
        for j, rho in enumerate(rhos):
            deltas = [
                float(r["delta_ari"])
                for r in real
                if str(r["dataset"]) == ds and float(r["reassignment_ratio"]) == rho
            ]
            if deltas:
                mat[i, j] = statistics.mean(deltas)
    fig_h = max(6, len(datasets) * 0.35)
    fig, ax = plt.subplots(figsize=(max(9, len(rhos)), fig_h))
    im = ax.imshow(mat, aspect="auto", cmap="RdBu", vmin=-0.2, vmax=0.2)
    ax.set_xticks(np.arange(len(rhos)))
    ax.set_xticklabels([str(r) for r in rhos], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(datasets)))
    ax.set_yticklabels(datasets, fontsize=8)
    ax.set_xlabel("ρ")
    ax.set_ylabel("dataset")
    ax.set_title("Mean ΔARI (minibatch − fastkmeans) by dataset")
    fig.colorbar(im, ax=ax, label="ΔARI")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_speed_ratio_vs_rho(
    rows: list[dict[str, Any]],
    out_path: Path,
) -> Path:
    rhos = sorted({float(r["reassignment_ratio"]) for r in rows})
    means: list[float] = []
    stds: list[float] = []
    for rho in rhos:
        chunk = [r for r in rows if float(r["reassignment_ratio"]) == rho]
        ratios = [
            float(r["minibatch_wall_s"]) / float(r["fastkmeans_wall_s"])
            for r in chunk
            if float(r.get("fastkmeans_wall_s") or 0) > 0
        ]
        m, s = _mean_std(ratios)
        means.append(m)
        stds.append(s)

    x = np.arange(len(rhos))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.errorbar(x, means, yerr=stds, fmt="o-", capsize=3, color="C2")
    ax.axhline(1.0, color="k", linewidth=0.8, alpha=0.5, label="parity")
    ax.set_xticks(x)
    ax.set_xticklabels([str(r) for r in rhos], rotation=45, ha="right")
    ax.set_xlabel("reassignment_ratio ρ")
    ax.set_ylabel("minibatch wall time / fastkmeans wall time")
    ax.set_title("Pooled speed ratio vs ρ")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def _pct_positive_delta(rows: list[dict[str, Any]], rho: float) -> float:
    chunk = [r for r in rows if float(r["reassignment_ratio"]) == rho]
    if not chunk:
        return float("nan")
    pos = sum(1 for r in chunk if float(r["delta_ari"]) > 0)
    return 100.0 * pos / len(chunk)


def write_markdown(
    out_md: Path,
    input_json: Path,
    figures: dict[str, str],
    n_rows: int,
    rhos: list[float],
) -> None:
    """figures: name -> relative path from repo root."""
    lines = [
        "# ρ sweep report (real benchmarks)",
        "",
        f"Generated from [`{input_json.relative_to(_ROOT) if input_json.is_relative_to(_ROOT) else input_json}`]({input_json.relative_to(_ROOT) if input_json.is_relative_to(_ROOT) else input_json}).",
        "",
        f"- **Rows in JSON:** {n_rows}",
        f"- **ρ values:** {', '.join(str(r) for r in rhos)}",
        "",
        "Mini-batch **k**-means is compared to **fastkmeans** on the same data, **k**, **seed**, and **niter**. **ΔARI** / **ΔNMI** are minibatch − fastkmeans (clustbench reference labels).",
        "",
        "## Aggregate figures",
        "",
    ]
    order = [
        ("pooled_delta_vs_rho", "Pooled **ΔARI** and **ΔNMI** vs ρ (mean across all dataset/ref configurations)."),
        ("pooled_ari_vs_rho", "Pooled mean **minibatch ARI** vs ρ with mean **fastkmeans** ARI as a horizontal reference."),
        ("heatmap_battery_rho", "**Battery × ρ:** mean ΔARI (clustbench families: sipu, fcps, wut, …)."),
        ("heatmap_dataset_rho", "**Dataset × ρ:** mean ΔARI (some datasets appear in multiple batteries)."),
        ("speed_ratio_vs_rho", "**Wall-time ratio** minibatch / fastkmeans vs ρ (same hardware; minibatch is often slower here)."),
    ]
    for key, caption in order:
        path = figures.get(key)
        if not path:
            continue
        lines.append(f"### {caption.split('(')[0].strip()}")
        lines.append("")
        lines.append(f"![{key}]({path})")
        lines.append("")
        lines.append(caption)
        lines.append("")

    lines.extend(
        [
            "## Quick read",
            "",
            "- **Larger ρ** often **reduces** the pooled gap to fastkmeans on **NMI** and moves **ΔARI** toward zero or slightly positive; **ρ = 0** is the harshest pooled average for minibatch in this file.",
            "- **Heatmaps** show where minibatch is **better** (red) or **worse** (blue) than fastkmeans; scale is **ΔARI**.",
            "- **Speed:** ratios **> 1** mean minibatch took **longer** wall time than fastkmeans in this benchmark setup.",
            "",
            "## Regenerate",
            "",
            "```bash",
            f"conda run -n kmeans-bench python scripts/generate_rho_sweep_report.py \\",
            f"  --input {input_json.relative_to(_ROOT) if input_json.is_relative_to(_ROOT) else input_json} \\",
            f"  --out-dir reports/figures/rho_sweep_all \\",
            f"  --report reports/rho_sweep_all_report.md",
            "```",
            "",
            "Optional per-cell ARI curves (one PNG per dataset ref):",
            "",
            "```bash",
            "conda run -n kmeans-bench python scripts/analyze_reassignment_sweep.py \\",
            f"  -i {input_json.relative_to(_ROOT) if input_json.is_relative_to(_ROOT) else input_json} \\",
            "  --figures-dir reports/figures/rho_sweep_all/per_cell",
            "```",
            "",
        ]
    )

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input",
        "-i",
        type=Path,
        default=_ROOT / "reports" / "rho_sweep_all.json",
        help="Sweep JSON array",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=_ROOT / "reports" / "figures" / "rho_sweep_all",
        help="Directory for PNG figures",
    )
    p.add_argument(
        "--report",
        type=Path,
        default=_ROOT / "reports" / "rho_sweep_all_report.md",
        help="Output markdown path",
    )
    args = p.parse_args(argv)

    rows = _load_rows(args.input)
    rhos = sorted({float(r["reassignment_ratio"]) for r in rows})

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {
        "pooled_delta_vs_rho": out_dir / "pooled_delta_vs_rho.png",
        "pooled_ari_vs_rho": out_dir / "pooled_ari_vs_rho.png",
        "heatmap_battery_rho": out_dir / "heatmap_delta_ari_battery_rho.png",
        "heatmap_dataset_rho": out_dir / "heatmap_delta_ari_dataset_rho.png",
        "speed_ratio_vs_rho": out_dir / "speed_ratio_vs_rho.png",
    }

    plot_pooled_delta_vs_rho(rows, paths["pooled_delta_vs_rho"])
    plot_pooled_ari_vs_rho(rows, paths["pooled_ari_vs_rho"])
    hb = plot_heatmap_battery_rho(rows, paths["heatmap_battery_rho"])
    plot_heatmap_dataset_rho(rows, paths["heatmap_dataset_rho"])
    plot_speed_ratio_vs_rho(rows, paths["speed_ratio_vs_rho"])

    fig_md: dict[str, str] = {}
    for key, p in paths.items():
        if key == "heatmap_battery_rho" and hb is None:
            continue
        try:
            fig_md[key] = str(p.resolve().relative_to(_ROOT))
        except ValueError:
            fig_md[key] = str(p)

    write_markdown(
        args.report.resolve(),
        args.input.resolve(),
        fig_md,
        len(rows),
        rhos,
    )

    print(f"Wrote figures under {out_dir}", file=sys.stderr)
    print(f"Wrote report {args.report}", file=sys.stderr)


if __name__ == "__main__":
    main()
