"""Utilities used by the minibatch_kmeans implementation."""

from __future__ import annotations

import secrets
from contextlib import contextmanager
from typing import Iterator

import torch


def default_cpu_generator() -> torch.Generator:
    g = torch.Generator(device="cpu")
    g.manual_seed(secrets.randbits(64))
    return g


def _as_device(device: torch.device | str) -> torch.device:
    return device if isinstance(device, torch.device) else torch.device(device)


@contextmanager
def nvtx_range(name: str, device: torch.device | str) -> Iterator[None]:
    """CUDA NVTX range for Nsight Systems timelines (no-op on CPU)."""
    dev = _as_device(device)
    if dev.type == "cuda":
        torch.cuda.nvtx.range_push(name)
        try:
            yield
        finally:
            torch.cuda.nvtx.range_pop()
    else:
        yield
