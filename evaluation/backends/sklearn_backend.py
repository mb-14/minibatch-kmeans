"""scikit-learn: KMeans and MiniBatchKMeans (CPU NumPy fit; centroids as torch.Tensor)."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from sklearn.cluster import KMeans, MiniBatchKMeans

from evaluation.backends.timing import run_timed_nvtx
from minibatch_kmeans._utils import nvtx_range


def _x_to_float32_numpy(x: torch.Tensor) -> np.ndarray:
    t = torch.as_tensor(x, dtype=torch.float32).detach()
    if t.layout is not torch.strided:
        raise ValueError("sklearn backends require dense strided input")
    t = t.cpu().contiguous()
    return t.numpy()


def run_sklearn_kmeans(
    x: torch.Tensor,
    k: int,
    niter: int,
    seed: int,
    device: torch.device,
    batch_size: int | None = None,
) -> dict[str, Any]:
    _ = batch_size
    with nvtx_range("sklearn_kmeans:as_numpy", device):
        x_np = _x_to_float32_numpy(x)
    n, _d = x_np.shape

    def _fit() -> KMeans:
        km_inner = KMeans(
            n_clusters=k,
            max_iter=max(1, niter),
            n_init=1,
            random_state=seed,
            init="random",
            verbose=False,
        )
        km_inner.fit(x_np)
        return km_inner

    km, elapsed, cpu_s = run_timed_nvtx(device, "sklearn_kmeans:fit", _fit)

    centers = torch.tensor(
        np.asarray(km.cluster_centers_, dtype=np.float32),
        dtype=torch.float32,
        device=device,
    )

    return {
        "centroids": centers,
        "wall_time_s": elapsed,
        "cpu_time_s": cpu_s,
        "iters_done": int(km.n_iter_),
    }


def run_sklearn_minibatch_kmeans(
    x: torch.Tensor,
    k: int,
    niter: int,
    seed: int,
    device: torch.device,
    batch_size: int | None = None,
    *,
    reassignment_ratio: float | None = None,
) -> dict[str, Any]:
    with nvtx_range("sklearn_minibatch_kmeans:prepare", device):
        x_fit = _x_to_float32_numpy(x)
    n, _d = x_fit.shape

    bs = batch_size if batch_size is not None else min(1024, max(1, n))
    bs = int(min(bs, n))
    rr = 0.01 if reassignment_ratio is None else float(reassignment_ratio)

    def _fit() -> MiniBatchKMeans:
        km_inner = MiniBatchKMeans(
            n_clusters=k,
            max_iter=max(1, niter),
            batch_size=bs,
            n_init=1,
            random_state=seed,
            init="random",
            verbose=False,
            reassignment_ratio=rr,
        )
        km_inner.fit(x_fit)
        return km_inner

    km, elapsed, cpu_s = run_timed_nvtx(
        device, "sklearn_minibatch_kmeans:fit", _fit
    )

    centers = torch.tensor(
        np.asarray(km.cluster_centers_, dtype=np.float32),
        dtype=torch.float32,
        device=device,
    )

    return {
        "centroids": centers,
        "wall_time_s": elapsed,
        "cpu_time_s": cpu_s,
        "timing_epochs": max(1, niter),
        "batch_size": bs,
        "reassignment_ratio": rr,
        "iters_done": int(km.n_iter_),
    }
