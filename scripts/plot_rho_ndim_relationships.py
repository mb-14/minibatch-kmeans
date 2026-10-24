#!/usr/bin/env python3
"""Plots and stats: how (n, k, d) interact with reassignment_ratio ρ in sweep JSON.

Expects rows with optional multiple seeds per (battery, dataset, ref, ρ); aggregates
mean ± stdev over seeds before plotting."""

from __future__ import annotations

import argparse
import json
import math
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
    print(f"plot_rho_ndim_relationships: need matplotlib and numpy: {e}", file=sys.stderr)
    sys.exit(1)


def _mean_std(vals: list[float]) -> tuple[float, float]:
    if not vals:
        return float("nan"), float("nan")
    m = statistics.mean(vals)
    if len(vals) == 1:
        return m, 0.0
    return m, statistics.stdev(vals)


def _pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2 or len(xs) != len(ys):
        return float("nan")
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    deny = math.sqrt(sum((y - my) ** 2 for y in ys))
    if denx == 0 or deny == 0:
        return float("nan")
    return num / (denx * deny)


def _load_rows(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("expected JSON array")
    return data


def aggregate_by_config_rho(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per (battery, dataset, ref, ρ) with mean over seeds."""
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        key = (
            str(r["battery"]),
            str(r["dataset"]),
            int(r["ref_index"]),
            float(r["reassignment_ratio"]),
        )
        groups[key].append(r)

    out: list[dict[str, Any]] = []
    for key, gs in sorted(groups.items()):
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
                "mean_delta_ari": _mean_std([float(x["delta_ari"]) for x in gs])[0],
                "std_delta_ari": _mean_std([float(x["delta_ari"]) for x in gs])[1],
                "mean_delta_nmi": _mean_std([float(x["delta_nmi"]) for x in gs])[0],
                "mean_mb_ari": _mean_std([float(x["minibatch_ari"]) for x in gs])[0],
            }
        )
    return out


def plot_delta_vs_rho_by_d(
    agg: list[dict[str, Any]],
    out_path: Path,
) -> None:
    """One line per distinct d (mean ΔARI over configs sharing that d)."""
    ds = sorted({int(a["d"]) for a in agg})
    rhos = sorted({float(a["rho"]) for a in agg})
    fig, ax = plt.subplots(figsize=(9, 5))
    for d in ds:
        means: list[float] = []
        for rho in rhos:
            chunk = [a for a in agg if int(a["d"]) == d and float(a["rho"]) == rho]
            if chunk:
                means.append(statistics.mean(a["mean_delta_ari"] for a in chunk))
            else:
                means.append(float("nan"))
        ax.plot(
            np.arange(len(rhos)),
            means,
            "o-",
            label=f"d={d}",
        )
    ax.axhline(0.0, color="k", linewidth=0.6, alpha=0.4)
    ax.set_xticks(np.arange(len(rhos)))
    ax.set_xticklabels([str(r) for r in rhos], rotation=45, ha="right")
    ax.set_xlabel("ρ")
    ax.set_ylabel("mean ΔARI (over configs, then mean over seeds)")
    ax.set_title("ΔARI vs ρ by embedding dimension d")
    ax.legend(loc="best", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _bin_log_n(n: int, edges: list[float]) -> int:
    ln = math.log10(max(n, 1))
    for i in range(len(edges) - 1):
        if edges[i] <= ln < edges[i + 1]:
            return i
    return len(edges) - 2


def plot_delta_vs_rho_by_log_n_bins(
    agg: list[dict[str, Any]],
    out_path: Path,
) -> None:
    """Quartile bins of log10(n)."""
    ns = sorted({int(a["n"]) for a in agg})
    log_ns = [math.log10(max(n, 1)) for n in ns]
    q1, q2, q3 = np.quantile(log_ns, [0.25, 0.5, 0.75])
    edges = [min(log_ns) - 1e-6, q1, q2, q3, max(log_ns) + 1e-6]
    labels = [
        f"log₁₀n∈[{edges[0]:.2f},{q1:.2f})",
        f"[{q1:.2f},{q2:.2f})",
        f"[{q2:.2f},{q3:.2f})",
        f"[{q3:.2f},{edges[-1]:.2f}]",
    ]

    rhos = sorted({float(a["rho"]) for a in agg})
    fig, ax = plt.subplots(figsize=(9, 5))

    for bi in range(4):
        means: list[float] = []
        for rho in rhos:
            chunk = [
                a
                for a in agg
                if float(a["rho"]) == rho and _bin_log_n(int(a["n"]), edges) == bi
            ]
            if chunk:
                means.append(statistics.mean(a["mean_delta_ari"] for a in chunk))
            else:
                means.append(float("nan"))
        ax.plot(np.arange(len(rhos)), means, "o-", label=labels[bi])

    ax.axhline(0.0, color="k", linewidth=0.6, alpha=0.4)
    ax.set_xticks(np.arange(len(rhos)))
    ax.set_xticklabels([str(r) for r in rhos], rotation=45, ha="right")
    ax.set_xlabel("ρ")
    ax.set_ylabel("mean ΔARI")
    ax.set_title("ΔARI vs ρ by quartile of log₁₀(n)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_delta_vs_rho_by_k_bins(
    agg: list[dict[str, Any]],
    out_path: Path,
) -> None:
    """Tertiles of k across unique configs (same k repeated across ρ)."""
    one_per_cfg = {}
    for a in agg:
        key = (a["battery"], a["dataset"], int(a["ref_index"]))
        if key not in one_per_cfg:
            one_per_cfg[key] = int(a["k"])
    cfg_ks = list(one_per_cfg.values())
    t1, t2 = np.quantile(cfg_ks, [1 / 3, 2 / 3])

    def bin_k(k: int) -> int:
        if k <= t1:
            return 0
        if k <= t2:
            return 1
        return 2

    rhos = sorted({float(a["rho"]) for a in agg})
    fig, ax = plt.subplots(figsize=(9, 5))
    labels = [
        f"k tertile low (k ≤ {t1:.1f})",
        f"k tertile mid ({t1:.1f} < k ≤ {t2:.1f})",
        f"k tertile high (k > {t2:.1f})",
    ]
    for bi in range(3):
        means: list[float] = []
        for rho in rhos:
            chunk = [
                a for a in agg if float(a["rho"]) == rho and bin_k(int(a["k"])) == bi
            ]
            if chunk:
                means.append(statistics.mean(a["mean_delta_ari"] for a in chunk))
            else:
                means.append(float("nan"))
        ax.plot(np.arange(len(rhos)), means, "o-", label=labels[bi])

    ax.axhline(0.0, color="k", linewidth=0.6, alpha=0.4)
    ax.set_xticks(np.arange(len(rhos)))
    ax.set_xticklabels([str(r) for r in rhos], rotation=45, ha="right")
    ax.set_xlabel("ρ")
    ax.set_ylabel("mean ΔARI")
    ax.set_title("ΔARI vs ρ by k tertile (approximate bins)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_improvement_rho01_minus_rho0_vs_log_n(
    agg: list[dict[str, Any]],
    out_path: Path,
) -> None:
    """Per config: ΔARI(ρ=0.1) − ΔARI(ρ=0); scatter vs log10(n)."""
    by_cfg: dict[tuple[str, str, int], dict[float, dict[str, Any]]] = defaultdict(dict)
    for a in agg:
        key = (a["battery"], a["dataset"], int(a["ref_index"]))
        by_cfg[key][float(a["rho"])] = a

    xs: list[float] = []
    ys: list[float] = []
    cs: list[int] = []
    for key, m in by_cfg.items():
        if 0.0 not in m or 0.1 not in m:
            continue
        a0, a1 = m[0.0], m[0.1]
        xs.append(math.log10(max(int(a0["n"]), 1)))
        ys.append(float(a1["mean_delta_ari"]) - float(a0["mean_delta_ari"]))
        cs.append(int(a0["k"]))

    fig, ax = plt.subplots(figsize=(8, 5))
    sc = ax.scatter(xs, ys, c=cs, cmap="viridis", s=55, alpha=0.85, edgecolors="k", linewidths=0.3)
    ax.axhline(0.0, color="k", linewidth=0.6, alpha=0.4)
    ax.set_xlabel("log₁₀(n)")
    ax.set_ylabel("ΔARI at ρ=0.1 minus ΔARI at ρ=0")
    ax.set_title("Gain from high ρ vs sample size n (color = k)")
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("k")
    ax.grid(True, alpha=0.3)
    r_im = _pearson(xs, ys)
    ax.text(
        0.02,
        0.98,
        f"Pearson(log₁₀n, gain) = {r_im:.3f}",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_scatter_n_k_color_delta_rho(
    agg: list[dict[str, Any]],
    rho: float,
    out_path: Path,
) -> None:
    """log10(n) vs k; color = mean ΔARI at fixed ρ; point size ∝ log10(d)."""
    chunk = [a for a in agg if abs(float(a["rho"]) - rho) < 1e-12]
    xs = [math.log10(max(int(a["n"]), 1)) for a in chunk]
    ys = [int(a["k"]) for a in chunk]
    col = [float(a["mean_delta_ari"]) for a in chunk]
    sizes = [40 + 25 * math.log10(max(int(a["d"]), 2)) for a in chunk]

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    sc = ax.scatter(xs, ys, c=col, cmap="RdBu", vmin=-0.12, vmax=0.12, s=sizes, alpha=0.9, edgecolors="k", linewidths=0.35)
    ax.set_xlabel("log₁₀(n)")
    ax.set_ylabel("k")
    ax.set_title(f"ΔARI vs (n, k) at ρ = {rho} (size ∝ log₁₀d)")
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("mean ΔARI")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_best_rho_scatter(
    agg: list[dict[str, Any]],
    out_path: Path,
) -> None:
    """Argmax over ρ of mean minibatch ARI (after seed mean)."""
    by_cfg: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for a in agg:
        key = (a["battery"], a["dataset"], int(a["ref_index"]))
        by_cfg[key].append(a)

    xs: list[float] = []
    ys: list[float] = []
    best_rhos: list[float] = []
    for key, rows_c in by_cfg.items():
        by_rho: dict[float, list[dict[str, Any]]] = defaultdict(list)
        for a in rows_c:
            by_rho[float(a["rho"])].append(a)
        rho_means = {
            rho: statistics.mean(x["mean_mb_ari"] for x in lst)
            for rho, lst in by_rho.items()
        }
        if not rho_means:
            continue
        br = max(rho_means, key=lambda r: rho_means[r])
        r0 = rows_c[0]
        xs.append(math.log10(max(int(r0["n"]), 1)))
        ys.append(int(r0["k"]))
        best_rhos.append(br)

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    # discrete rho levels -> numeric index for colormap
    rhos_sorted = sorted({float(a["rho"]) for a in agg})

    def _rho_index(r: float) -> int:
        for i, x in enumerate(rhos_sorted):
            if abs(x - r) < 1e-9:
                return i
        return 0

    idx = [_rho_index(r) for r in best_rhos]
    sc = ax.scatter(xs, ys, c=idx, cmap="plasma", vmin=0, vmax=len(rhos_sorted) - 1, s=70, alpha=0.9, edgecolors="k", linewidths=0.35)
    ax.set_xlabel("log₁₀(n)")
    ax.set_ylabel("k")
    ax.set_title("ρ with highest mean minibatch ARI (per dataset ref)")
    cbar = fig.colorbar(sc, ax=ax, ticks=range(len(rhos_sorted)))
    cbar.ax.set_yticklabels([str(r) for r in rhos_sorted])
    cbar.set_label("best ρ")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def compute_insights_text(
    agg: list[dict[str, Any]],
    rows: list[dict[str, Any]],
) -> str:
    """Pearson correlations on 28 config means at ρ=0.1."""
    a01 = [a for a in agg if abs(float(a["rho"]) - 0.1) < 1e-12]
    ln = [math.log10(max(int(a["n"]), 1)) for a in a01]
    lk = [float(a["k"]) for a in a01]
    ld = [math.log10(max(int(a["d"]), 1)) for a in a01]
    y = [float(a["mean_delta_ari"]) for a in a01]
    lines = [
        "### Correlations (n=28 dataset refs, seed-mean ΔARI at ρ=0.1)",
        "",
        f"- Pearson(mean ΔARI, log₁₀n) = {_pearson(ln, y):.3f}",
        f"- Pearson(mean ΔARI, k) = {_pearson(lk, y):.3f}",
        f"- Pearson(mean ΔARI, log₁₀d) = {_pearson(ld, y):.3f}",
        "",
    ]
    # improvement
    by_cfg: dict[tuple[str, str, int], dict[float, dict[str, Any]]] = defaultdict(dict)
    for a in agg:
        key = (a["battery"], a["dataset"], int(a["ref_index"]))
        by_cfg[key][float(a["rho"])] = a
    gains: list[float] = []
    g_ln: list[float] = []
    for m in by_cfg.values():
        if 0.0 not in m or 0.1 not in m:
            continue
        gains.append(float(m[0.1]["mean_delta_ari"]) - float(m[0.0]["mean_delta_ari"]))
        g_ln.append(math.log10(max(int(m[0.0]["n"]), 1)))
    lines.extend(
        [
            "### Gain from ρ = 0.1 vs ρ = 0",
            "",
            f"- Mean gain (ΔARI(0.1) − ΔARI(0)) over configs: **{statistics.mean(gains):.4f}**",
            f"- Pearson(gain, log₁₀n) = **{_pearson(g_ln, gains):.3f}** (larger **n** tends to associate with larger **improvement** when increasing ρ to 0.1)",
            "",
        ]
    )
    lines.extend(
        [
            "### How to read the figures",
            "",
            "- **By d / log n / k:** curves show **mean ΔARI** vs ρ within each bin (equal weight per **dataset ref**, not per point).",
            "- **Gain vs log n:** each point is one clustbench reference; **positive** y means minibatch catches up more to fastkmeans at high ρ.",
            "- **Best ρ scatter:** which ρ maximizes **mean minibatch ARI** (after averaging seeds); useful when quality matters more than a fixed default.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input",
        "-i",
        type=Path,
        default=_ROOT / "reports" / "rho_sweep_all.json",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=_ROOT / "reports" / "figures" / "rho_sweep_all" / "ndim",
    )
    p.add_argument(
        "--insights-md",
        type=Path,
        default=_ROOT / "reports" / "rho_ndim_insights.md",
        help="Append insights section (creates/overwrites with figures + text)",
    )
    args = p.parse_args(argv)

    rows = _load_rows(args.input)
    agg = aggregate_by_config_rho(rows)
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    plot_delta_vs_rho_by_d(agg, out / "delta_ari_vs_rho_by_d.png")
    plot_delta_vs_rho_by_log_n_bins(agg, out / "delta_ari_vs_rho_by_log10n_quartile.png")
    plot_delta_vs_rho_by_k_bins(agg, out / "delta_ari_vs_rho_by_k_bins.png")
    plot_improvement_rho01_minus_rho0_vs_log_n(agg, out / "gain_delta_ari_vs_log10n.png")
    plot_scatter_n_k_color_delta_rho(agg, 0.1, out / "scatter_log10n_k_delta_ari_rho0.1.png")
    plot_best_rho_scatter(agg, out / "scatter_log10n_k_best_rho.png")

    insight_body = compute_insights_text(agg, rows)
    rel = str(out.relative_to(_ROOT))
    md = "\n".join(
        [
            "# ρ vs (n, k, d): plots and notes",
            "",
            f"Source: [`{args.input.relative_to(_ROOT) if args.input.is_relative_to(_ROOT) else args.input}`]({args.input.relative_to(_ROOT) if args.input.is_relative_to(_ROOT) else args.input}).",
            "",
            insight_body,
            "## Figures",
            "",
            f"![ΔARI vs ρ by d]({rel}/delta_ari_vs_rho_by_d.png)",
            "",
            f"![ΔARI vs ρ by log n quartile]({rel}/delta_ari_vs_rho_by_log10n_quartile.png)",
            "",
            f"![ΔARI vs ρ by k bins]({rel}/delta_ari_vs_rho_by_k_bins.png)",
            "",
            f"![Gain vs log n]({rel}/gain_delta_ari_vs_log10n.png)",
            "",
            f"![scatter n k delta ρ=0.1]({rel}/scatter_log10n_k_delta_ari_rho0.1.png)",
            "",
            f"![best ρ]({rel}/scatter_log10n_k_best_rho.png)",
            "",
            "## Regenerate",
            "",
            "```bash",
            "conda run -n kmeans-bench python scripts/plot_rho_ndim_relationships.py \\",
            f"  -i {args.input.relative_to(_ROOT) if args.input.is_relative_to(_ROOT) else args.input} \\",
            f"  --out-dir {rel} \\",
            f"  --insights-md reports/rho_ndim_insights.md",
            "```",
            "",
        ]
    )
    args.insights_md.parent.mkdir(parents=True, exist_ok=True)
    args.insights_md.write_text(md, encoding="utf-8")

    print(f"Wrote figures under {out}", file=sys.stderr)
    print(f"Wrote {args.insights_md}", file=sys.stderr)


if __name__ == "__main__":
    main()
