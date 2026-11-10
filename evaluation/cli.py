"""Shared helpers for the evaluation command-line tools."""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Callable
from typing import NoReturn, TypeVar

N = TypeVar("N", int, float)


def fail(prog: str, msg: str) -> NoReturn:
    """Print ``prog: msg`` to stderr and exit with status 1."""
    print(f"{prog}: {msg}", file=sys.stderr)
    sys.exit(1)


def require_clustbench(prog: str) -> None:
    """Exit early if clustering-benchmarks is missing, rather than skipping every dataset."""
    try:
        import clustbench  # noqa: F401 - availability check only
    except ImportError:
        fail(
            prog,
            'clustering-benchmarks is required for clustbench datasets. pip install -e ".[eval]"',
        )


def _bounded(convert: Callable[[str], N], ok: Callable[[N], bool], what: str) -> Callable[[str], N]:
    """argparse ``type=`` that converts with ``convert`` and rejects values failing ``ok``."""

    def parse(raw: str) -> N:
        try:
            value = convert(raw)
        except ValueError:
            raise argparse.ArgumentTypeError(f"invalid value: {raw!r}") from None
        if not (math.isfinite(value) and ok(value)):
            raise argparse.ArgumentTypeError(f"must be a finite number {what}, got {raw}")
        return value

    return parse


positive_int = _bounded(int, lambda v: v >= 1, ">= 1")
non_negative_int = _bounded(int, lambda v: v >= 0, ">= 0")
positive_float = _bounded(float, lambda v: v > 0, "> 0")
non_negative_float = _bounded(float, lambda v: v >= 0, ">= 0")
