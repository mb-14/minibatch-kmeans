"""CLI: write PNG comparison figures from eval JSON (default: ``results_eval.json``)."""

from __future__ import annotations

import argparse
from pathlib import Path

from evaluation.plot_compare import write_compare_figures_from_json


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description="Plot ARI/NMI comparison bars and heatmaps from eval JSON.")
    p.add_argument(
        "--input",
        "-i",
        type=Path,
        default=root / "results_eval.json",
        help="Path to eval JSON array (default: project results_eval.json)",
    )
    p.add_argument(
        "--out",
        "-o",
        type=Path,
        default=root / "figures" / "eval_compare",
        help="Output directory for PNGs (default: figures/eval_compare)",
    )
    args = p.parse_args()
    fig_dir = write_compare_figures_from_json(args.input, args.out)
    print(f"Wrote {fig_dir}/*.png")


if __name__ == "__main__":
    main()
