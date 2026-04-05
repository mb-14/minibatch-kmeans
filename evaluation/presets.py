"""Default clustbench evaluation tasks: ``(battery, dataset)`` pairs."""

from __future__ import annotations

# Benchmark Suite v1.1.0 — see https://clustering-benchmarks.gagolewski.com/weave/suite-v1.html

BENCHMARK_TASKS: list[tuple[str, str]] = [
    ("sipu", "worms_64"),
    ("sipu", "birch1"),
    ("sipu", "birch2"),
    ("mnist", "digits"),
    ("mnist", "fashion"),
]
