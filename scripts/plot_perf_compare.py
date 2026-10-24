"""Bar charts from combined ``evaluation.perf_test`` JSON (one row per method)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np

METHOD_ORDER: tuple[str, ...] = ("faisskmeans", "fastkmeans", "minibatchkmeans")
METHOD_LABELS: dict[str, str] = {
    "faisskmeans": "Faiss",
    "fastkmeans": "fast_kmeans",
    "minibatchkmeans": "MiniBatch (torch)",
}


def _load_rows(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array in {path}")
    return data


def _by_method(rows: list[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    return {str(r["method"]): r for r in rows}


def _process_gpu_peak_mib(row: Mapping[str, Any]) -> tuple[float, bool]:
    """(MiB, True) for whole-process peak; (MiB, False) if legacy JSON had only Δ."""
    if "gpu_process_peak_memory_mib_nvitop" in row:
        return float(row["gpu_process_peak_memory_mib_nvitop"]), True
    v = row.get("gpu_process_peak_memory_delta_mib_nvitop")
    return (float(v) if v is not None else float("nan")), False


def plot_perf_compare(rows: list[Mapping[str, Any]], out_path: Path, *, dpi: int = 160) -> None:
    by_m = _by_method(rows)
    methods = [m for m in METHOD_ORDER if m in by_m]
    if not methods:
        raise ValueError("No recognized methods in JSON (expected faisskmeans / fastkmeans / minibatchkmeans).")
    labels = [METHOD_LABELS.get(m, m) for m in methods]
    x = np.arange(len(methods))
    colors = plt.cm.Set2(np.linspace(0, 1, len(methods)))

    wall = [float(by_m[m]["wall_time_s"]) for m in methods]
    mem_pairs = [_process_gpu_peak_mib(by_m[m]) for m in methods]
    mem = [p[0] for p in mem_pairs]
    mem_is_peak = all(p[1] for p in mem_pairs)
    has_mem = any(np.isfinite(m) for m in mem)
    nmi = [float(by_m[m]["quality"]["val"]["nmi"]) for m in methods]
    ari = [float(by_m[m]["quality"]["val"]["ari"]) for m in methods]

    r0 = rows[0]
    subtitle = (
        f"n={r0.get('n')}, k={r0.get('k')}, d={r0.get('d')}, "
        f"niter={r0.get('niter')}, val fraction ≈ {1.0 - float(r0.get('train_fraction', 0.75)):.2f}"
    )

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8))
    ax_w, ax_m, ax_nmi, ax_ari = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    ax_w.bar(x, wall, color=colors, edgecolor="0.3", linewidth=0.6)
    ax_w.set_xticks(x)
    ax_w.set_xticklabels(labels, rotation=15, ha="right")
    ax_w.set_ylabel("Wall time (s)")
    ax_w.set_title("Wall time (lower is better)")
    ax_w.grid(axis="y", alpha=0.35)

    if has_mem:
        mem_plot = [m if np.isfinite(m) else 0.0 for m in mem]
        ax_m.bar(x, mem_plot, color=colors, edgecolor="0.3", linewidth=0.6)
        for i, m in enumerate(mem):
            if not np.isfinite(m):
                ax_m.text(
                    float(x[i]),
                    ax_m.get_ylim()[1] * 0.02,
                    "n/a",
                    ha="center",
                    fontsize=9,
                    rotation=90,
                    va="bottom",
                )
    else:
        ax_m.bar(x, [0] * len(methods), color=colors)
        ax_m.text(0.5, 0.5, "No GPU nvitop memory fields", ha="center", va="center", transform=ax_m.transAxes)
    ax_m.set_xticks(x)
    ax_m.set_xticklabels(labels, rotation=15, ha="right")
    if mem_is_peak:
        ax_m.set_ylabel("Process GPU memory at peak (MiB)")
        ax_m.set_title("Whole-process GPU memory peak (nvitop; lower is better)")
    else:
        ax_m.set_ylabel("Δ GPU memory during fit (MiB)")
        ax_m.set_title("Legacy JSON: Δ only (re-run perf_test for peak footprint)")
    ax_m.grid(axis="y", alpha=0.35)

    ax_nmi.bar(x, nmi, color=colors, edgecolor="0.3", linewidth=0.6)
    ax_nmi.set_xticks(x)
    ax_nmi.set_xticklabels(labels, rotation=15, ha="right")
    ax_nmi.set_ylabel("NMI")
    ax_nmi.set_ylim(0.0, 1.0)
    ax_nmi.set_title("NMI on validation split (higher is better)")
    ax_nmi.grid(axis="y", alpha=0.35)

    ax_ari.bar(x, ari, color=colors, edgecolor="0.3", linewidth=0.6)
    ax_ari.set_xticks(x)
    ax_ari.set_xticklabels(labels, rotation=15, ha="right")
    ax_ari.set_ylabel("ARI")
    ax_ari.set_ylim(-0.05, 1.0)
    ax_ari.set_title("ARI on validation split (higher is better)")
    ax_ari.grid(axis="y", alpha=0.35)

    fig.suptitle("perf_test comparison\n" + subtitle, fontsize=12, y=1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Plot perf_test JSON (combined rows).")
    p.add_argument("json_path", type=Path, help="Path to JSON array from merged perf_test runs")
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("figures/perf_compare.png"),
        help="Output PNG path",
    )
    p.add_argument("--dpi", type=int, default=160)
    args = p.parse_args()
    rows = _load_rows(args.json_path)
    plot_perf_compare(rows, args.output, dpi=args.dpi)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
