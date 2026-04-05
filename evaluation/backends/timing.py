"""Shared CUDA sync + wall/CPU timing around NVTX ranges for backend benchmarks."""

from __future__ import annotations

import time
from typing import Callable, TypeVar

import torch

from minibatch_kmeans._utils import nvtx_range

T = TypeVar("T")


def sync_cuda_if_needed(device: torch.device | str) -> None:
    dev = device if isinstance(device, torch.device) else torch.device(device)
    if dev.type == "cuda":
        torch.cuda.synchronize()


def run_timed_nvtx(
    device: torch.device | str,
    nvtx_name: str,
    fn: Callable[[], T],
) -> tuple[T, float, float]:
    """Sync, time ``fn`` inside ``nvtx_name``, sync again, return value and (wall_s, cpu_s)."""
    sync_cuda_if_needed(device)
    t0 = time.perf_counter()
    c0 = time.process_time()
    with nvtx_range(nvtx_name, device):
        result = fn()
    sync_cuda_if_needed(device)
    elapsed = time.perf_counter() - t0
    cpu_s = time.process_time() - c0
    return result, elapsed, cpu_s
