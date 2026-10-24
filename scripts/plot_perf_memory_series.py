"""Overlay per-process GPU memory (nvitop) time series from perf_test JSON rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np

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


def plot_memory_series(
    rows: Sequence[Mapping[str, Any]],
    out_path: Path,
    *,
    dpi: int = 160,
    y_unit: str = "mib",
) -> None:
    """``y_unit`` is ``mib`` (default) or ``bytes``."""
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.cm.Set2(np.linspace(0, 1, max(len(rows), 1)))

    for i, row in enumerate(rows):
        method = str(row["method"])
        series = row.get("gpu_memory_series_nvitop")
        if not series:
            raise ValueError(
                f"Row for {method} has no gpu_memory_series_nvitop; "
                "re-run perf_test with --gpu-memory-series."
            )
        t = np.array([float(p["t_s"]) for p in series], dtype=float)
        b = np.array([int(p["process_gpu_memory_bytes"]) for p in series], dtype=float)
        if y_unit == "mib":
            y = b / (1024**2)
            ylab = "Process GPU memory (MiB)"
        else:
            y = b
            ylab = "Process GPU memory (bytes)"
        label = METHOD_LABELS.get(method, method)
        ax.plot(t, y, label=label, color=colors[i], linewidth=1.8, marker="o", markersize=2)

    ax.set_xlabel("Time from start of backend fit (s)")
    ax.set_ylabel(ylab)
    ax.set_title("Per-process GPU memory over time (nvitop)")
    ax.grid(alpha=0.35)
    ax.legend(loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Plot gpu_memory_series_nvitop curves from perf_test JSON (one or more rows)."
    )
    p.add_argument(
        "json_path",
        type=Path,
        help="JSON array: one file with multiple methods, or merge rows manually",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("figures/perf_gpu_memory_series.png"),
        help="Output PNG path",
    )
    p.add_argument("--dpi", type=int, default=160)
    p.add_argument(
        "--bytes",
        action="store_true",
        help="Y axis in bytes instead of MiB",
    )
    args = p.parse_args()
    rows = _load_rows(args.json_path)
    plot_memory_series(rows, args.output, dpi=args.dpi, y_unit="bytes" if args.bytes else "mib")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
