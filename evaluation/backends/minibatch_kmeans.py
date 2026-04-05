"""Local PyTorch MiniBatchKMeans (``minibatch-kmeans`` package)."""

from __future__ import annotations

from typing import Any

import torch

from evaluation.backends.timing import run_timed_nvtx
from minibatch_kmeans._utils import nvtx_range
from minibatch_kmeans import MiniBatchKMeans


def run_minibatch_kmeans(
    x: torch.Tensor,
    k: int,
    niter: int,
    seed: int,
    device: torch.device,
    batch_size: int | None = None,
    *,
    reassignment_ratio: float | None = None,
) -> dict[str, Any]:
    with nvtx_range("minibatch_kmeans:backend:as_tensor", device):
        x = torch.as_tensor(x, dtype=torch.float32)
        if x.layout is not torch.strided:
            raise ValueError("minibatch_kmeans backend requires dense strided input")
    n, _d = x.shape
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)

    bs = batch_size if batch_size is not None else min(1024, max(1, n))

    rr = 0.01 if reassignment_ratio is None else float(reassignment_ratio)

    with nvtx_range("minibatch_kmeans:backend:construct", device):
        km = MiniBatchKMeans(
            n_clusters=k,
            verbose=False,
            random_state=rng,
            reassignment_ratio=rr,
            device=device,
            dtype=torch.float32,
        )

    def _fit() -> torch.Tensor:
        return km.fit(
            x,
            batch_size=bs,
            max_iter=max(1, niter),
            tol=0.0,
            max_no_improvement=None,
        )

    centers, elapsed, cpu_s = run_timed_nvtx(
        device, "minibatch_kmeans:fit", _fit
    )

    return {
        "centroids": centers.detach(),
        "wall_time_s": elapsed,
        "cpu_time_s": cpu_s,
        "timing_epochs": max(1, niter),
        "batch_size": int(min(bs, n)),
        "reassignment_ratio": rr,
    }
