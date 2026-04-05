"""Per-process GPU memory via [nvitop](https://github.com/XuehaiPan/nvitop) while a callable runs.

Reports **this Python process** GPU memory (bytes), not whole-device usage.
"""

from __future__ import annotations

import os
import threading
from typing import Callable, TypeVar

T = TypeVar("T")

_POLL_INTERVAL_S = 0.01


def _process_gpu_bytes(device_index: int, pid: int) -> int | None:
    try:
        from nvitop import Device
    except ImportError:
        return None
    try:
        d = Device(device_index)
        gp = d.processes().get(pid)
        if gp is None:
            return None
        return int(gp.gpu_memory())
    except Exception:
        return None


def nvitop_process_gpu_memory_peak_delta_bytes_during(
    device_index: int,
    fn: Callable[[], T],
) -> tuple[T, int | None]:
    """
    Run ``fn`` while polling nvitop for this process's reported GPU memory.

    Returns ``(result, delta_bytes)`` where ``delta_bytes`` is
    ``max(observed) - baseline`` with baseline taken immediately before the poll loop
    (``None`` if ``nvitop`` is not installed).
    """
    try:
        from nvitop import Device  # noqa: F401
    except ImportError:
        return fn(), None

    pid = os.getpid()
    baseline = _process_gpu_bytes(device_index, pid)
    if baseline is None:
        baseline = 0

    peak_holder: list[int] = [baseline]

    stop = threading.Event()

    def _poll() -> None:
        while not stop.wait(_POLL_INTERVAL_S):
            u = _process_gpu_bytes(device_index, pid)
            if u is not None and u > peak_holder[0]:
                peak_holder[0] = u

    poller = threading.Thread(target=_poll, daemon=True)
    poller.start()
    try:
        result = fn()
    finally:
        stop.set()
        poller.join(timeout=30.0)

    final = _process_gpu_bytes(device_index, pid)
    peak = peak_holder[0]
    if final is not None:
        peak = max(peak, final)
    delta = max(0, peak - baseline)
    return result, delta


def torch_device_index(device: object) -> int:
    """CUDA ordinal for device index 0, 1, … (``cuda:0`` -> ``0``)."""
    idx = getattr(device, "index", None)
    return 0 if idx is None else int(idx)
