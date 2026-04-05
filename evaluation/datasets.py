"""Load clustering benchmark datasets via ``clustering-benchmarks`` (clustbench)."""

from __future__ import annotations

from dataclasses import dataclass
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
