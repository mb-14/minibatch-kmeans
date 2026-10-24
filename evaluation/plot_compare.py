"""Bar and heatmap figures comparing clustering backends from eval JSON rows."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

METHOD_ORDER: tuple[str, ...] = (
    "minibatchkmeans",
    "fastkmeans",
    "sklearnminibatchkmeans",
    "sklearnkmeans",
    "faisskmeans",
    "cuvskmeans",
    "superkmeans",
)

METHOD_LABELS: dict[str, str] = {
    "minibatchkmeans": "MiniBatch (torch)",
    "fastkmeans": "fast_kmeans",
    "sklearnminibatchkmeans": "MiniBatch (sklearn)",
    "sklearnkmeans": "KMeans (sklearn)",
    "faisskmeans": "Faiss Clustering",
    "cuvskmeans": "cuVS K-Means",
    "superkmeans": "SuperKMeans",
}

BATTERY_ORDER: tuple[str, ...] = ("sipu", "mnist")


def _load_rows(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array in {path}")
    return data


def _pivot_nested(rows: list[Mapping[str, Any]]) -> dict[str, dict[str, dict[str, Mapping[str, Any]]]]:
    """battery -> dataset -> method -> row."""
    out: dict[str, dict[str, dict[str, Mapping[str, Any]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for r in rows:
        out[str(r["battery"])][str(r["dataset"])][str(r["method"])] = r
    return {b: {d: dict(mm) for d, mm in dm.items()} for b, dm in out.items()}


def _ranks_high_best(scores: Sequence[float]) -> list[int]:
    """Competition ranks: 1 = best (highest score). Ties share the same rank; next rank skips."""
    s = np.asarray(scores, dtype=float)
    n = len(s)
    if n == 0:
        return []
    order = np.argsort(-s)
    ranks = [0] * n
    rank = 1
    i = 0
    while i < n:
        v = float(s[order[i]])
        j = i + 1
        while j < n and np.isclose(float(s[order[j]]), v, rtol=0.0, atol=1e-15):
            j += 1
        span = j - i
        for k in range(i, j):
            ranks[order[k]] = rank
        rank += span
        i = j
    return ranks


def _method_colors() -> dict[str, Any]:
    return {m: plt.cm.tab10(i / 9.0) for i, m in enumerate(METHOD_ORDER)}


def _plot_battery_grouped(
    ax: Any,
    battery: str,
    nested: Mapping[str, Mapping[str, Mapping[str, Any]]],
    metric: str,
    std_key: str,
) -> None:
    by_ds = nested.get(battery, {})
    if not by_ds:
        ax.set_visible(False)
        return
    datasets = sorted(by_ds.keys())
    methods_present: set[str] = set()
    for d in datasets:
        methods_present.update(by_ds[d].keys())
    methods = [m for m in METHOD_ORDER if m in methods_present]
    if not methods:
        ax.set_visible(False)
        return

    n_ds = len(datasets)
    n_m = len(methods)
    x = np.arange(n_ds, dtype=float)
    width = 0.8 / max(n_m, 1)
    colors = plt.cm.tab10(np.linspace(0, 0.9, max(n_m, 1)))

    for j, method in enumerate(methods):
        ys: list[float] = []
        es: list[float] = []
        for d in datasets:
            row = by_ds[d].get(method)
            if row is None:
                ys.append(float("nan"))
                es.append(0.0)
            else:
                ys.append(float(row[metric]))
                es.append(float(row.get(std_key) or 0.0))
        offset = (j - (n_m - 1) / 2.0) * width
        ax.bar(
            x + offset,
            ys,
            width,
            yerr=es,
            capsize=2 if any(e > 0 for e in es) else 0,
            label=METHOD_LABELS.get(method, method),
            color=colors[j],
        )

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=25, ha="right")
    ylabel = "ARI" if metric == "best_ari" else "NMI"
    ax.set_ylabel(ylabel)
    ax.set_title(battery)
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", alpha=0.3)


def write_compare_figures_from_json(json_path: Path, out_dir: Path) -> Path:
    """Load eval JSON rows and write comparison PNGs into ``out_dir``."""
    rows = _load_rows(json_path)
    nested = _pivot_nested(rows)
    out_dir.mkdir(parents=True, exist_ok=True)

    batteries = [b for b in BATTERY_ORDER if b in nested]
    if not batteries:
        batteries = sorted(nested.keys())

    for metric, std_key in (("best_ari", "best_ari_std"), ("best_nmi", "best_nmi_std")):
        fig, axes = plt.subplots(
            len(batteries),
            1,
            figsize=(9.5, 3.2 * len(batteries)),
            squeeze=False,
        )
        for ax_row, bat in zip(axes[:, 0], batteries):
            _plot_battery_grouped(ax_row, bat, nested, metric, std_key)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center", ncol=len(handles), bbox_to_anchor=(0.5, 1.02))
        title = "ARI" if metric == "best_ari" else "NMI"
        fig.suptitle(f"{title} by dataset (mean ± std over seeds)", y=1.06)
        fig.tight_layout()
        out = out_dir / f"eval_compare_{metric.replace('best_', '')}.png"
        fig.savefig(out, dpi=160, bbox_inches="tight")
        plt.close(fig)

    _write_heatmaps(nested, out_dir)
    _write_dot_rank_plots(nested, out_dir)

    return out_dir


def _write_heatmaps(
    nested: Mapping[str, Mapping[str, Mapping[str, Any]]],
    out_dir: Path,
) -> None:
    tasks: list[tuple[str, str]] = []
    for b in BATTERY_ORDER:
        if b not in nested:
            continue
        for d in sorted(nested[b].keys()):
            tasks.append((b, d))
    if not tasks:
        for b in sorted(nested.keys()):
            for d in sorted(nested[b].keys()):
                tasks.append((b, d))

    methods = [m for m in METHOD_ORDER if any(m in nested[b][d] for b, d in tasks)]
    if not methods:
        return

    for metric, _std_key in (("best_ari", "best_ari_std"), ("best_nmi", "best_nmi_std")):
        mat = np.full((len(methods), len(tasks)), np.nan)
        for j, (b, d) in enumerate(tasks):
            for i, m in enumerate(methods):
                row = nested.get(b, {}).get(d, {}).get(m)
                if row is not None:
                    mat[i, j] = float(row[metric])
        fig, ax = plt.subplots(figsize=(max(8, len(tasks) * 1.1), 4))
        im = ax.imshow(mat, aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis")
        ax.set_xticks(np.arange(len(tasks)))
        ax.set_xticklabels([f"{b}/{d}" for b, d in tasks], rotation=45, ha="right")
        ax.set_yticks(np.arange(len(methods)))
        ax.set_yticklabels([METHOD_LABELS.get(m, m) for m in methods])
        ht = "ARI" if metric == "best_ari" else "NMI"
        ax.set_title(f"{ht} (heatmap, mean over seeds)")
        fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
        fig.tight_layout()
        out = out_dir / f"eval_compare_heatmap_{metric.replace('best_', '')}.png"
        fig.savefig(out, dpi=160, bbox_inches="tight")
        plt.close(fig)


def _write_dot_rank_plots(
    nested: Mapping[str, Mapping[str, Mapping[str, Any]]],
    out_dir: Path,
) -> None:
    """Per-dataset scatter: x = rank (1 = best), y = metric; color = method."""
    batteries = [b for b in BATTERY_ORDER if b in nested]
    if not batteries:
        batteries = sorted(nested.keys())
    max_cols = max((len(nested[b]) for b in batteries), default=0)
    if max_cols == 0:
        return

    colors = _method_colors()

    for metric in ("best_ari", "best_nmi"):
        fig, axes = plt.subplots(
            len(batteries),
            max_cols,
            figsize=(3.4 * max_cols, 2.8 * len(batteries)),
            squeeze=False,
        )
        for ri, bat in enumerate(batteries):
            ds_list = sorted(nested[bat].keys())
            for ci in range(max_cols):
                ax = axes[ri, ci]
                if ci >= len(ds_list):
                    ax.set_visible(False)
                    continue
                d = ds_list[ci]
                by_m = nested[bat][d]
                methods = [m for m in METHOD_ORDER if m in by_m]
                scores = [float(by_m[m][metric]) for m in methods]
                ranks = _ranks_high_best(scores)
                for m, rk, sc in zip(methods, ranks, scores):
                    ax.scatter(
                        rk,
                        sc,
                        s=100,
                        c=[colors[m]],
                        zorder=3,
                        edgecolors="white",
                        linewidths=0.6,
                    )
                ax.set_xticks(list(range(1, len(methods) + 1)))
                ax.set_xlim(0.5, len(methods) + 0.5)
                ax.set_ylim(0.0, 1.0)
                ax.set_xlabel("Rank (1 = best)")
                ylab = "ARI" if metric == "best_ari" else "NMI"
                ax.set_ylabel(ylab)
                ax.set_title(f"{bat}/{d}")
                ax.grid(axis="both", alpha=0.3)
        legend_methods = [
            m
            for m in METHOD_ORDER
            if any(m in nested[b][d] for b in batteries for d in nested[b])
        ]
        leg_handles = [
            Line2D(
                [0],
                [0],
                linestyle="none",
                marker="o",
                markersize=9,
                markerfacecolor=colors[m],
                markeredgecolor="white",
                markeredgewidth=0.6,
                label=METHOD_LABELS.get(m, m),
            )
            for m in legend_methods
        ]
        fig.legend(
            handles=leg_handles,
            loc="upper center",
            ncol=min(len(leg_handles), 4),
            bbox_to_anchor=(0.5, 1.01),
        )
        title = "ARI" if metric == "best_ari" else "NMI"
        fig.suptitle(f"{title} dot + rank (mean score; rank within dataset)", y=1.06)
        fig.tight_layout()
        out = out_dir / f"eval_compare_dot_rank_{metric.replace('best_', '')}.png"
        fig.savefig(out, dpi=160, bbox_inches="tight")
        plt.close(fig)


def write_compare_figures(reports_dir: Path) -> Path:
    """Load ``reports_dir/eval.json`` and write figures to ``reports_dir/figures``."""
    path = reports_dir / "eval.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing eval JSON: {path}")
    out = reports_dir / "figures"
    return write_compare_figures_from_json(path, out)
