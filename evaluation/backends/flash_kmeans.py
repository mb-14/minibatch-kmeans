"""flash-kmeans: batched Triton Lloyd, or chunked CPU→GPU Lloyd for large n."""

from __future__ import annotations

from typing import Any

import torch

from evaluation.backends.timing import run_timed_nvtx, sync_cuda_if_needed
from flash_kmeans.kmeans_large import kmeans_largeN
from flash_kmeans.kmeans_triton_impl import batch_kmeans_Euclid

# Same cutoff as ``FlashKMeans.chunk_size_data_cpu``: CPU matrices with more
# rows than this are trained by ``kmeans_largeN`` (chunked copies), not by
# uploading the full matrix first.
_LARGE_N_CPU_ROWS = 1_048_576
_WARMUP_ITERS = 2


def _pin_cpu(x: torch.Tensor) -> torch.Tensor:
    """Page-lock a CPU matrix so chunked copies can overlap compute."""
    pinned = x.contiguous()
    if pinned.device.type == "cpu" and not pinned.is_pinned():
        return pinned.pin_memory()
    return pinned


def _warmup(device: torch.device, fn, runs: int) -> None:
    for _ in range(runs):
        fn()
    sync_cuda_if_needed(device)


def _average_runs(
    device: torch.device,
    name: str,
    fn,
    repeats: int,
) -> tuple[Any, float, float]:
    result = None
    elapsed_sum = 0.0
    cpu_sum = 0.0
    for _ in range(repeats):
        result, elapsed, cpu_s = run_timed_nvtx(device, name, fn)
        elapsed_sum += elapsed
        cpu_sum += cpu_s
    if result is None:
        raise RuntimeError("flash-kmeans timing repeat count must be >= 1")
    return result, elapsed_sum / repeats, cpu_sum / repeats


def run_flash_kmeans(
    x: torch.Tensor,
    k: int,
    niter: int,
    seed: int,
    device: torch.device,
    *,
    warmup_runs: int = 0,
    timed_repeats: int = 1,
) -> dict[str, Any]:
    """
    Run Euclidean Lloyd for ``niter`` iterations.

    ``tol=0.0`` avoids early exit so all ``niter`` steps run unless shift is exactly zero.
    Initialization matches fastkmeans: ``torch.manual_seed(seed)`` then ``randperm(n)[:k]`` rows.

    When ``x`` is on CPU and has more than 1,048,576 rows, the timed region is
    :func:`flash_kmeans.kmeans_large.kmeans_largeN`, which copies each chunk to the
    GPU inside the call. Smaller inputs copy the full matrix to ``device`` inside
    the timed region, then run
    :func:`flash_kmeans.kmeans_triton_impl.batch_kmeans_Euclid`.

    CPU inputs are pinned before timing. ``warmup_runs`` calls of
    ``min(2, niter)`` iterations run before the clock. ``timed_repeats``
    full runs are averaged.
    """
    if warmup_runs < 0 or timed_repeats < 1:
        raise ValueError("warmup_runs must be >= 0 and timed_repeats must be >= 1")
    x = torch.as_tensor(x, dtype=torch.float32)
    n, _d = x.shape
    use_large_n = (
        device.type == "cuda"
        and x.device.type == "cpu"
        and n > _LARGE_N_CPU_ROWS
    )
    if use_large_n:
        return _run_flash_large_n(
            x, k, niter, seed, device, warmup_runs=warmup_runs, timed_repeats=timed_repeats
        )

    x_cpu = _pin_cpu(x)

    def _run_flash(iters: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        t = x_cpu.to(device=device, dtype=torch.float32)
        idx = torch.randperm(n, device=t.device)[:k]
        init_centroids = t[idx].unsqueeze(0)
        return batch_kmeans_Euclid(
            t.unsqueeze(0),
            k,
            max_iters=iters,
            tol=0.0,
            init_centroids=init_centroids,
            verbose=False,
        )

    if warmup_runs:
        _warmup(device, lambda: _run_flash(min(_WARMUP_ITERS, niter)), warmup_runs)
    (cluster_ids, centroids, iters_done), elapsed, cpu_s = _average_runs(
        device,
        "flash_kmeans:batch_kmeans_Euclid",
        lambda: _run_flash(niter),
        timed_repeats,
    )

    return {
        "centroids": centroids.squeeze(0).detach(),
        "labels": cluster_ids.squeeze(0).detach(),
        "wall_time_s": elapsed,
        "cpu_time_s": cpu_s,
        "iters_done": int(iters_done),
        "timing_warmup_runs": warmup_runs,
        "timing_repeats": timed_repeats,
    }


def _run_flash_large_n(
    x: torch.Tensor,
    k: int,
    niter: int,
    seed: int,
    device: torch.device,
    *,
    warmup_runs: int,
    timed_repeats: int,
) -> dict[str, Any]:
    """Time ``kmeans_largeN``, including its chunked CPU→GPU copies."""
    n = int(x.shape[0])
    x_cpu = _pin_cpu(x)

    def _run(iters: int) -> tuple[torch.Tensor, torch.Tensor]:
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        init_centroids = x_cpu[torch.randperm(n)[:k]].contiguous()
        return kmeans_largeN(
            x_cpu,
            k,
            max_iters=iters,
            tol=0.0,
            verbose=False,
            init_centroids=init_centroids,
            device=device,
            dtype=torch.float32,
        )

    if warmup_runs:
        _warmup(device, lambda: _run(min(_WARMUP_ITERS, niter)), warmup_runs)
    (labels, centroids), elapsed, cpu_s = _average_runs(
        device,
        "flash_kmeans:kmeans_largeN",
        lambda: _run(niter),
        timed_repeats,
    )
    return {
        "centroids": centroids.detach(),
        "labels": labels.detach(),
        "wall_time_s": elapsed,
        "cpu_time_s": cpu_s,
        "iters_done": int(niter),
        "flash_large_n": True,
        "timing_warmup_runs": warmup_runs,
        "timing_repeats": timed_repeats,
    }
