"""faiss: :class:`faiss.Clustering` with :class:`faiss.IndexFlatL2` or GPU flat L2 index.

Dense float32 training vectors only. Matches the setup in Meta's MNIST k-means bench
(``max_points_per_centroid`` so the full set is used, ``GpuIndexFlatL2`` when GPU Faiss is available).

See https://github.com/facebookresearch/faiss/blob/main/benchs/kmeans_mnist.py
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from evaluation.backends.timing import run_timed_nvtx
from minibatch_kmeans._utils import nvtx_range


def _x_to_float32_numpy_dense(x: torch.Tensor) -> np.ndarray:
    if x.layout is not torch.strided:
        raise ValueError("faiss k-means requires a dense strided float tensor")
    t = torch.as_tensor(x, dtype=torch.float32).detach().cpu().contiguous()
    return t.numpy()


def _make_index_flat_l2(faiss: Any, d: int, device: torch.device) -> Any:
    """CPU :class:`IndexFlatL2`, or :class:`GpuIndexFlatL2` if Faiss GPU bindings exist."""
    if (
        device.type == "cuda"
        and hasattr(faiss, "StandardGpuResources")
        and faiss.get_num_gpus() > 0
    ):
        gpu_id = 0 if device.index is None else int(device.index)
        if gpu_id >= faiss.get_num_gpus():
            gpu_id = 0
        res = faiss.StandardGpuResources()
        cfg = faiss.GpuIndexFlatConfig()
        cfg.useFloat16 = False
        cfg.device = gpu_id
        return faiss.GpuIndexFlatL2(res, d, cfg)
    return faiss.IndexFlatL2(d)


def run_faiss_kmeans(
    x: torch.Tensor,
    k: int,
    niter: int,
    seed: int,
    device: torch.device,
    batch_size: int | None = None,
) -> dict[str, Any]:
    """
    Run Faiss Lloyd-style k-means for ``niter`` iterations.

    Uses the same index style as Meta's benchmark: flat L2 on CPU, or
    ``GpuIndexFlatL2`` on the CUDA device index when Faiss was built with GPU support.
    Centroids are returned as ``torch.float32`` on ``device``.
    """
    _ = batch_size
    try:
        import faiss
    except ImportError as e:
        raise ImportError(
            "faiss backend requires the faiss package "
            "(e.g. `pip install faiss-cpu` or conda `faiss-gpu`)."
        ) from e

    with nvtx_range("faiss_kmeans:as_numpy", device):
        x_np = _x_to_float32_numpy_dense(x)
    _n, d = x_np.shape
    d_i, k_i = int(d), int(k)

    def _fit() -> tuple[np.ndarray, int]:
        clus = faiss.Clustering(d_i, k_i)
        clus.niter = max(1, int(niter))
        clus.verbose = False
        clus.max_points_per_centroid = 10_000_000
        clus.min_points_per_centroid = 1
        clus.seed = int(seed)
        index = _make_index_flat_l2(faiss, d_i, device)
        clus.train(x_np, index)
        cent_flat = faiss.vector_float_to_array(clus.centroids)
        centers = np.asarray(cent_flat, dtype=np.float32).reshape(k_i, d_i)
        iters_done = int(clus.iteration_stats.size())
        return centers, iters_done

    (centers_np, iters_done), elapsed, cpu_s = run_timed_nvtx(
        device, "faiss_kmeans:train", _fit
    )

    centers = torch.tensor(centers_np, dtype=torch.float32, device=device)

    return {
        "centroids": centers,
        "wall_time_s": elapsed,
        "cpu_time_s": cpu_s,
        "iters_done": iters_done,
    }
