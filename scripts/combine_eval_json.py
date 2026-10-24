"""Merge multiple evaluation.run JSON outputs into one array for plotting.

Rows are keyed by (battery, dataset, method); later inputs override earlier ones
for the same key.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _row_key(row: dict) -> tuple[str, str, str]:
    return (str(row["battery"]), str(row["dataset"]), str(row["method"]))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description="Merge eval JSON arrays (evaluation.run output).",
    )
    p.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        metavar="PATH",
        help="Eval JSON files (first wins unless overridden by later files)",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Write merged JSON array here",
    )
    args = p.parse_args(argv)

    merged: dict[tuple[str, str, str], dict] = {}
    for path in args.inputs:
        if not path.is_file():
            print(f"combine_eval_json: missing file {path}", file=sys.stderr)
            sys.exit(1)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            print(f"combine_eval_json: expected JSON array in {path}", file=sys.stderr)
            sys.exit(1)
        for row in data:
            if not isinstance(row, dict):
                continue
            merged[_row_key(row)] = row

    out_list = sorted(merged.values(), key=_row_key)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out_list, f, indent=2)
    print(f"Wrote {len(out_list)} rows to {args.output}")


if __name__ == "__main__":
    main()
