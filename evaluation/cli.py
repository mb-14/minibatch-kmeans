"""Shared helpers for the evaluation command-line tools."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from typing import NoReturn


def fail(prog: str, msg: str) -> NoReturn:
    """Print ``prog: msg`` to stderr and exit with status 1."""
    print(f"{prog}: {msg}", file=sys.stderr)
    sys.exit(1)


def _bounded(
    convert: Callable[[str], float], ok: Callable[[float], bool], what: str
) -> Callable[[str], float]:
    def parse(raw: str) -> float:
        try:
            value = convert(raw)
        except ValueError:
            raise argparse.ArgumentTypeError(f"invalid value: {raw!r}") from None
        if not ok(value):
            raise argparse.ArgumentTypeError(f"must be {what}, got {raw}")
        return value

    parse.__name__ = convert.__name__  # argparse names the type in some messages
    return parse


positive_int = _bounded(int, lambda v: v >= 1, ">= 1")
non_negative_int = _bounded(int, lambda v: v >= 0, ">= 0")
positive_float = _bounded(float, lambda v: v > 0, "> 0")
non_negative_float = _bounded(float, lambda v: v >= 0, ">= 0")
