"""Per-process GPU memory via [nvitop](https://github.com/XuehaiPan/nvitop) while a callable runs.

All values refer to **this Python process** (PID) on the chosen CUDA device — not total GPU
card usage from other processes. ``peak_bytes`` is the maximum reported process footprint
during the timed section (whole-process VRAM on that device at peak).
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, TypeVar

T = TypeVar("T")

_POLL_INTERVAL_S = 0.01


@dataclass(frozen=True)
class NvitopProcessGpuMemory:
    """GPU memory attributed to the process by nvitop (one device)."""

    baseline_bytes: int
    """Process GPU memory immediately before ``fn`` runs."""
    peak_bytes: int
    """Maximum process GPU memory observed while ``fn`` runs (and right after)."""
    delta_bytes: int
    """``peak_bytes - baseline_bytes`` (non-negative)."""


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


def nvitop_process_gpu_memory_during(
    device_index: int,
    fn: Callable[[], T],
    *,
    record_series: bool = False,
    series_interval_s: float = 0.05,
) -> tuple[T, NvitopProcessGpuMemory | None, list[dict[str, Any]] | None]:
    """
    Run ``fn`` while polling nvitop for this process's reported GPU memory on ``device_index``.

    Returns ``(result, stats, series)`` where ``stats`` is ``None`` if nvitop is unavailable,
    and ``series`` is ``None`` unless ``record_series`` is True (then a list of
    ``{"t_s", "process_gpu_memory_bytes"}`` samples for comparing usage over time).

    ``t_s`` is seconds from the start of ``fn``; the first row uses ``t_s=0`` with the
    pre-``fn`` baseline read.
    """
    try:
        from nvitop import Device  # noqa: F401
    except ImportError:
        return fn(), None, None

    if record_series and series_interval_s <= 0:
        raise ValueError("series_interval_s must be > 0 when record_series is True")

    pid = os.getpid()
    baseline = _process_gpu_bytes(device_index, pid)
    if baseline is None:
        baseline = 0

    peak_holder: list[int] = [baseline]
    series: list[dict[str, Any]] | None = [] if record_series else None
    t_start: list[float | None] = [None]
    last_series_bucket: list[int] = [-1]

    stop = threading.Event()

    def _poll() -> None:
        while not stop.wait(_POLL_INTERVAL_S):
            u = _process_gpu_bytes(device_index, pid)
            if u is not None and u > peak_holder[0]:
                peak_holder[0] = u
            ts = t_start[0]
            if series is None or ts is None:
                continue
            elapsed = time.perf_counter() - ts
            b = int(elapsed // series_interval_s)
            if b > last_series_bucket[0]:
                last_series_bucket[0] = b
                ub = u if u is not None else baseline
                series.append(
                    {
                        "t_s": round(b * series_interval_s, 4),
                        "process_gpu_memory_bytes": ub,
                    }
                )

    poller = threading.Thread(target=_poll, daemon=True)
    poller.start()
    try:
        if record_series and series is not None:
            series.append(
                {"t_s": 0.0, "process_gpu_memory_bytes": baseline}
            )
            last_series_bucket[0] = 0
        t_start[0] = time.perf_counter()
        result = fn()
    finally:
        stop.set()
        poller.join(timeout=30.0)

    final = _process_gpu_bytes(device_index, pid)
    peak = peak_holder[0]
    if final is not None:
        peak = max(peak, final)
    delta = max(0, peak - baseline)
    stats = NvitopProcessGpuMemory(
        baseline_bytes=baseline, peak_bytes=peak, delta_bytes=delta
    )
    if record_series and series is not None:
        t_end = time.perf_counter() - (t_start[0] or time.perf_counter())
        fv = final if final is not None else peak
        series.append(
            {
                "t_s": round(t_end, 4),
                "process_gpu_memory_bytes": fv if fv is not None else baseline,
            }
        )
    return result, stats, series if record_series else None


def nvitop_process_gpu_memory_peak_delta_bytes_during(
    device_index: int,
    fn: Callable[[], T],
) -> tuple[T, int | None]:
    """
    Same as :func:`nvitop_process_gpu_memory_during` but returns only ``delta_bytes`` for
    backward compatibility.
    """
    result, stats, _s = nvitop_process_gpu_memory_during(device_index, fn)
    if stats is None:
        return result, None
    return result, stats.delta_bytes


def torch_device_index(device: object) -> int:
    """CUDA ordinal for device index 0, 1, … (``cuda:0`` -> ``0``)."""
    idx = getattr(device, "index", None)
    return 0 if idx is None else int(idx)
