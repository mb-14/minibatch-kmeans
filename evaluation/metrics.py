"""Clustering metrics for benchmark evaluation (WCSS, ARI/NMI, clustbench reference scoring)."""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from minibatch_kmeans._utils import _as_device

# Rows of X processed per chunk in WCSS / assignment (avoids O(n*k) and full float64 X).
_DEFAULT_CHUNK_ROWS = 8192


def _row_chunk(x: torch.Tensor, start: int, end: int) -> torch.Tensor:
    return x[start:end]


def wcss_torch(
    x: torch.Tensor,
    centroids: torch.Tensor,
    *,
    chunk_rows: int = _DEFAULT_CHUNK_ROWS,
    eval_device: torch.device | str | None = None,
) -> float:
    """
    Within-cluster sum of squares (full data, current centroids).

    Assigns each row to nearest centroid in Euclidean distance, then sums squared distances.
    Accumulates in float64 for stability. Processes ``x`` in row chunks so peak memory stays
    O(chunk_rows * max(d, k)) instead of O(n * k).

    If ``eval_device`` is set and differs from ``x.device``, ``centroids`` are placed on
    ``eval_device`` once and each chunk ``x[start:end]`` is moved there before distance work
    (so ``x`` can remain on CPU while evaluation runs on GPU without materializing all of ``x``).
    """
    if chunk_rows < 1:
        raise ValueError("chunk_rows must be >= 1")
    if eval_device is None:
        dev = x.device
    else:
        dev = _as_device(eval_device)

    n = x.shape[0]

    if eval_device is None:
        staged = False
        c = centroids
    else:
        staged = x.device != dev
        c = centroids.to(device=dev, dtype=torch.float32)
    c64 = c.double()
    total = torch.zeros((), dtype=torch.float64, device=dev)
    for start in range(0, n, chunk_rows):
        end = min(start + chunk_rows, n)
        if staged:
            x64 = _row_chunk(x, start, end).to(dev).double()
        else:
            x64 = _row_chunk(x, start, end).double()
        dists = torch.cdist(x64, c64, p=2.0) ** 2
        total = total + dists.min(dim=1).values.sum()
    return float(total.item())


def assign_labels_torch(
    x: torch.Tensor,
    centroids: torch.Tensor,
    *,
    chunk_rows: int = _DEFAULT_CHUNK_ROWS,
    eval_device: torch.device | str | None = None,
) -> torch.Tensor:
    """Nearest-centroid labels (same distance definition as ``wcss_torch``), chunked over rows."""
    if chunk_rows < 1:
        raise ValueError("chunk_rows must be >= 1")
    if eval_device is None:
        dev = x.device
    else:
        dev = _as_device(eval_device)

    n = x.shape[0]

    if eval_device is None:
        staged = False
        c = centroids
    else:
        staged = x.device != dev
        c = centroids.to(device=dev, dtype=torch.float32)
    c64 = c.double()
    parts = []
    for start in range(0, n, chunk_rows):
        end = min(start + chunk_rows, n)
        if staged:
            x64 = _row_chunk(x, start, end).to(dev).double()
        else:
            x64 = _row_chunk(x, start, end).double()
        dists = torch.cdist(x64, c64, p=2.0) ** 2
        parts.append(dists.argmin(dim=1).to(torch.int64))
    return torch.cat(parts, dim=0)


def wcss_and_assign_labels_torch(
    x: torch.Tensor,
    centroids: torch.Tensor,
    *,
    chunk_rows: int = _DEFAULT_CHUNK_ROWS,
    eval_device: torch.device | str | None = None,
) -> tuple[float, torch.Tensor]:
    """
    Same as calling ``wcss_torch`` then ``assign_labels_torch``, but one pass over ``x``
    (one ``cdist`` per chunk instead of two).
    """
    if chunk_rows < 1:
        raise ValueError("chunk_rows must be >= 1")
    if eval_device is None:
        dev = x.device
    else:
        dev = _as_device(eval_device)

    n = x.shape[0]

    if eval_device is None:
        staged = False
        c = centroids
    else:
        staged = x.device != dev
        c = centroids.to(device=dev, dtype=torch.float32)
    c64 = c.double()
    total = torch.zeros((), dtype=torch.float64, device=dev)
    parts = []
    for start in range(0, n, chunk_rows):
        end = min(start + chunk_rows, n)
        if staged:
            x64 = _row_chunk(x, start, end).to(dev).double()
        else:
            x64 = _row_chunk(x, start, end).double()
        dists = torch.cdist(x64, c64, p=2.0) ** 2
        total = total + dists.min(dim=1).values.sum()
        parts.append(dists.argmin(dim=1).to(torch.int64))
    pred = torch.cat(parts, dim=0)
    return float(total.item()), pred


def _labels_to_numpy_int64(y: torch.Tensor) -> np.ndarray:
    return y.detach().cpu().numpy().astype(np.int64, copy=False).ravel()


def ari_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """Adjusted Rand index (``sklearn.metrics.adjusted_rand_score``)."""
    if y_true.numel() != y_pred.numel():
        raise ValueError("y_true and y_pred must have the same length.")
    return float(
        adjusted_rand_score(
            _labels_to_numpy_int64(y_true),
            _labels_to_numpy_int64(y_pred),
        )
    )


def nmi_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """Normalized mutual information (``average_method='arithmetic'``)."""
    return float(
        normalized_mutual_info_score(
            _labels_to_numpy_int64(y_true),
            _labels_to_numpy_int64(y_pred),
            average_method="arithmetic",
        )
    )


def _labels_for_clustbench_get_score(y_pred: np.ndarray, k: int) -> np.ndarray:
    """
    ``clustbench.get_score`` requires ``results[k]`` with values in ``1..k`` (see clustbench/score.py).
    Our k-means backends use nearest-centroid indices ``0..k-1``.
    """
    y = np.asarray(y_pred, dtype=np.int64)
    lo, hi = int(np.min(y)), int(np.max(y))
    if lo >= 1 and hi <= k:
        return y
    if lo >= 0 and hi <= k - 1:
        return y + 1
    raise ValueError(
        f"Cannot map partition to clustbench 1..k labels: min={lo}, max={hi}, k={k}"
    )


def _metric_ari_numpy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(adjusted_rand_score(y_true, y_pred))


def _metric_nmi_numpy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(
        normalized_mutual_info_score(y_true, y_pred, average_method="arithmetic")
    )


def clustbench_best_score(
    reference_labels: list[np.ndarray],
    y_pred: np.ndarray,
    k: int,
    *,
    metric: Callable[[np.ndarray, np.ndarray], float] | None = None,
) -> float:
    """
    Best similarity to any reference partition, using clustbench's noise handling.

    ``y_pred`` must be a length-``n`` integer partition with ``k`` clusters.
    ``results`` is ``{k: y_pred}`` as required by ``clustbench.get_score``.
    """
    import clustbench

    if metric is None:
        metric = _metric_ari_numpy
    y_pred_cb = _labels_for_clustbench_get_score(y_pred, k)
    return float(
        clustbench.get_score(
            reference_labels,
            {int(k): y_pred_cb},
            metric=metric,
            compute_max=True,
        )
    )


def clustbench_best_ari_and_nmi(
    reference_labels: list[np.ndarray],
    y_pred: np.ndarray,
    k: int,
) -> tuple[float, float]:
    """ARI and NMI each maximized over references (same ``y_pred``; two separate ``get_score`` calls)."""
    ari = clustbench_best_score(
        reference_labels, y_pred, k, metric=_metric_ari_numpy
    )
    nmi = clustbench_best_score(
        reference_labels, y_pred, k, metric=_metric_nmi_numpy
    )
    return ari, nmi
