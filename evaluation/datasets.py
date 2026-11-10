"""Load clustering benchmark datasets via ``clustering-benchmarks`` (clustbench)."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class LoadedBenchmark:
    """Torch-ready view of a clustbench dataset."""

    battery: str
    dataset: str
    x: torch.Tensor  # float32 CPU dense ``(n, d)``
    labels: list[np.ndarray]  # reference partitions (numpy, int)
    n_clusters: np.ndarray  # k per reference


def missing_clustbench_data(path: str, tasks: list[tuple[str, str]]) -> bool:
    """True if ``path`` holds none of the ``<battery>/<dataset>.data.gz`` files for ``tasks``.

    A cheap up-front check for a wrong ``--data-path``, before any long-running work.
    """
    root = Path(path).expanduser()
    return not any((root / battery / f"{dataset}.data.gz").is_file() for battery, dataset in tasks)


def load_clustbench_dataset_or_skip(
    battery: str,
    dataset: str,
    *,
    path: str,
    prog: str,
) -> LoadedBenchmark | None:
    """``load_clustbench_dataset``, but print a warning and return None for a missing or bad file.

    Only file and parse errors are caught, so bugs still raise.
    """
    try:
        return load_clustbench_dataset(battery, dataset, path=path)
    except (OSError, EOFError, ValueError) as e:
        print(f"{prog}: skipping {battery}/{dataset}: {e}", file=sys.stderr)
        return None


def load_clustbench_dataset(
    battery: str,
    dataset: str,
    *,
    path: str,
) -> LoadedBenchmark:
    """
    Load benchmark data the same way as upstream ``clustbench.load_dataset``.

    ``path`` is the directory that contains battery folders (``sipu/``, ``fcps/``, …)—
    equivalent to ``data_path`` in
    https://clustering-benchmarks.gagolewski.com/weave/clustbench-usage.html
    """
    import clustbench

    b: Any = clustbench.load_dataset(battery, dataset, path=path)
    data = np.asarray(b.data, dtype=np.float64)
    x = torch.from_numpy(data).to(dtype=torch.float32)
    labels = [np.asarray(lab, dtype=np.int64) for lab in b.labels]
    n_clusters = np.asarray(b.n_clusters, dtype=np.int64)
    return LoadedBenchmark(
        battery=battery,
        dataset=dataset,
        x=x,
        labels=labels,
        n_clusters=n_clusters,
    )
