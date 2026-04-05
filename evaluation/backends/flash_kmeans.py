"""flash-kmeans: batch_kmeans_Euclid (Triton)."""

from __future__ import annotations

from typing import Any

import torch

from evaluation.backends.timing import run_timed_nvtx
from flash_kmeans.kmeans_triton_impl import batch_kmeans_Euclid


def run_flash_kmeans(
    x: torch.Tensor,
    k: int,
    niter: int,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    """
    Run Euclidean Lloyd for ``niter`` iterations.

    ``tol=0.0`` avoids early exit so all ``niter`` steps run unless shift is exactly zero.
    Initialization matches fastkmeans: ``torch.manual_seed(seed)`` then ``randperm(n)[:k]`` rows.

    On CUDA, the Triton implementation requires ``k >= 16`` (same ``tl.dot`` constraint as upstream);
    for smaller ``k`` we delegate to :func:`run_fast_kmeans` (PyTorch path on GPU).
    """
    x = torch.as_tensor(x, dtype=torch.float32)
    n, d = x.shape

    if device.type == "cuda" and k < 16:
        from evaluation.backends.fast_kmeans import run_fast_kmeans

        out = run_fast_kmeans(x, k, niter, seed, device)
        out["flash_fallback_to_fastkmeans"] = True
        return out

    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    t = x.contiguous().to(device=device, dtype=torch.float32)
    x_b = t.unsqueeze(0)

    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    idx = torch.randperm(n, device=t.device)[:k]
    init_centroids = t[idx].unsqueeze(0)

    def _run_flash() -> tuple[torch.Tensor, torch.Tensor, int]:
        return batch_kmeans_Euclid(
            x_b,
            k,
            max_iters=niter,
            tol=0.0,
            init_centroids=init_centroids,
            verbose=False,
        )

    (cluster_ids, centroids, iters_done), elapsed, cpu_s = run_timed_nvtx(
        device, "flash_kmeans:batch_kmeans_Euclid", _run_flash
    )

    return {
        "centroids": centroids.squeeze(0).detach(),
        "labels": cluster_ids.squeeze(0).detach(),
        "wall_time_s": elapsed,
        "cpu_time_s": cpu_s,
        "iters_done": int(iters_done),
    }
