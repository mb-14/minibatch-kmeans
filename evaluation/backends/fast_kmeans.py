"""fastkmeans: FastKMeans (PyTorch chunked path; Triton disabled in this harness).

Upstream ``fastkmeans`` types ``train`` / ``predict`` as taking ``np.ndarray`` (Faiss-style).
Implementation converts with ``torch.from_numpy`` and runs ``_kmeans_torch_double_chunked``
on tensors; NumPy is only that API boundary and ``centroids.numpy()`` after the run.
``use_triton=False`` avoids upstream Triton edge cases (asserts for some ``k`` / shapes).
See https://github.com/AnswerDotAI/fastkmeans/blob/main/fastkmeans/kmeans.py
"""

from __future__ import annotations

import math
from typing import Any

import torch

from evaluation.backends.timing import run_timed_nvtx
from fastkmeans import FastKMeans


class FastKMeansTorch:
    """
    Same constructor and behavior as :class:`fastkmeans.FastKMeans`, but ``train`` /
    ``fit`` / ``predict`` / ``fit_predict`` use ``torch.Tensor``. Upstream still receives
    ``ndarray`` at its public entry points only.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._km = FastKMeans(*args, **kwargs)

    @property
    def centroids(self) -> torch.Tensor | None:
        c = self._km.centroids
        if c is None:
            return None
        return torch.from_numpy(c)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._km, name)

    @staticmethod
    def _to_train_array(x: torch.Tensor):
        """Float32 contiguous CPU array for upstream ``train`` (Faiss-style signature)."""
        t = torch.as_tensor(x, dtype=torch.float32).detach().cpu().contiguous()
        return t.numpy()

    def train(self, data: torch.Tensor) -> torch.Tensor:
        self._km.train(self._to_train_array(data))
        assert self._km.centroids is not None
        return torch.from_numpy(self._km.centroids).to(dtype=torch.float32)

    def fit(self, data: torch.Tensor) -> FastKMeansTorch:
        self.train(data)
        return self

    def predict(self, data: torch.Tensor) -> torch.Tensor:
        labels_np = self._km.predict(self._to_train_array(data))
        return torch.from_numpy(labels_np).to(dtype=torch.int64)

    def fit_predict(self, data: torch.Tensor) -> torch.Tensor:
        self.fit(data)
        return self.predict(data)


def run_fast_kmeans(
    x: torch.Tensor,
    k: int,
    niter: int,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    """
    Run ``FastKMeans`` for ``niter`` Lloyd iterations.

    Triton is **disabled** (``use_triton=False``): the chunked PyTorch path runs on GPU/CPU and
    avoids device-side asserts from the upstream Triton kernels for some ``k`` / problem sizes.

    ``tol=-inf`` disables early stopping on centroid shift (same idea as fastkmeans speedbench).
    ``max_points_per_centroid=None`` uses the full dataset (no subsampling).
    Init uses the same seed as flash (``randperm(n)[:k]`` rows after ``manual_seed(seed)``).
    """
    x = torch.as_tensor(x, dtype=torch.float32)
    n, d = x.shape

    km = FastKMeansTorch(
        d=d,
        k=k,
        niter=niter,
        tol=-math.inf,
        gpu=device.type == "cuda",
        seed=seed,
        max_points_per_centroid=None,
        device=device,
        dtype=torch.float32,
        use_triton=False,
        verbose=False,
    )

    _, elapsed, cpu_s = run_timed_nvtx(
        device, "fast_kmeans:train", lambda: km.train(x)
    )

    centroids = km.centroids
    assert centroids is not None
    centroids = centroids.to(device=device, dtype=torch.float32)

    return {
        "centroids": centroids,
        "labels": None,
        "wall_time_s": elapsed,
        "cpu_time_s": cpu_s,
        "iters_done": niter,
    }
