"""Fused distance + argmin assignment on CUDA (Triton).

The kernel streams centroid tiles through ``tl.dot`` and keeps a running
min/argmin per row, so the ``(n, k)`` distance matrix is never stored.
"""

from __future__ import annotations

import torch

from ._utils import row_sq_norms

_IMPORT_ERROR: ImportError | None = None
try:
    import triton
    import triton.language as tl
except ImportError as exc:
    triton = None
    tl = None
    _IMPORT_ERROR = exc

AVAILABLE = triton is not None
_NO_TRITON = "triton cannot be imported"

if AVAILABLE:
    # Unspecialized sizes: one compiled variant for every batch size.
    @triton.jit(do_not_specialize=["n_rows", "n_cent"])
    def _assign_kernel(
        x_ptr,
        c_ptr,
        c_sq_ptr,
        out_min_ptr,
        out_idx_ptr,
        n_rows,
        n_cent,
        d,
        stride_xm,
        stride_cn,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        # int64: n * d can exceed 2**31.
        rows64 = pid_m.to(tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M)
        row_ok = rows64 < n_rows
        offs_k = tl.arange(0, BLOCK_K)

        best = tl.full((BLOCK_M,), float("inf"), tl.float32)
        best_idx = tl.zeros((BLOCK_M,), tl.int32)

        for n0 in range(0, n_cent, BLOCK_N):
            cols = n0 + tl.arange(0, BLOCK_N)
            col_ok = cols < n_cent
            cols64 = cols.to(tl.int64)
            acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
            for k0 in range(0, d, BLOCK_K):
                ks = k0 + offs_k
                k_ok = ks < d
                xt = tl.load(
                    x_ptr + rows64[:, None] * stride_xm + ks[None, :],
                    mask=row_ok[:, None] & k_ok[None, :],
                    other=0.0,
                )
                ct = tl.load(
                    c_ptr + cols64[None, :] * stride_cn + ks[:, None],
                    mask=col_ok[None, :] & k_ok[:, None],
                    other=0.0,
                )
                # Full float32 products; Triton's default is TF32.
                acc = tl.dot(xt, ct, acc, input_precision="ieee")
            c_sq = tl.load(c_sq_ptr + cols, mask=col_ok, other=0.0)
            dist = c_sq[None, :] - 2.0 * acc
            dist = tl.where(col_ok[None, :], dist, float("inf"))
            # NaN wins, as in torch.argmin: held as -inf, restored before the store.
            dist = tl.where(dist != dist, float("-inf"), dist)  # noqa: PLR0124 - NaN test
            tile_min, tile_arg = tl.min(
                dist, axis=1, return_indices=True, return_indices_tie_break_left=True
            )
            # Strict "<": ties keep the lower index.
            better = tile_min < best
            best_idx = tl.where(better, n0 + tile_arg, best_idx)
            best = tl.where(better, tile_min, best)

        best = tl.where(best == float("-inf"), float("nan"), best)
        tl.store(out_min_ptr + rows64, best, mask=row_ok)
        tl.store(out_idx_ptr + rows64, best_idx.to(tl.int64), mask=row_ok)


# Fixed, not autotuned: the tile shape sets the summation order, so autotuning
# would make results depend on timing.
_BLOCK_M, _BLOCK_N, _BLOCK_K = 64, 128, 16
_NUM_WARPS, _NUM_STAGES = 4, 4
# Triton vectorizes the row loads only when the feature count, the row stride
# and the data pointer are divisible by _ALIGN (Triton's fixed divisibility
# check); other inputs are copied into fresh rows zero-padded to a multiple of
# _PAD_TO, which leaves every dot product unchanged. Padding to a multiple of
# the feature tile keeps the tile boundaries, so the summation order, and the
# result bits, match the unpadded run (_PAD_TO must stay a multiple of _ALIGN).
# X is copied in chunks of at most _PAD_BYTES (256 MB), or of one row tile
# when a tile is larger than that, so the copy stays bounded however many rows
# X has; smaller chunks were slower at 1000 features. Chunks are whole row
# tiles, which keeps every chunk's output pointers aligned (an unaligned one
# compiles a second kernel variant). With fewer than _PAD_MIN_K centers the
# kernel is short and the copy costs more than the faster loads save (up to
# 1.8x slower at k=64), so X is only copied when its columns are not
# contiguous. At 1024 centers padding breaks even at 200-1000 features and wins
# below; small widths already win from about 400 centers.
_ALIGN, _PAD_TO, _PAD_BYTES, _PAD_MIN_K = 16, _BLOCK_K, 1 << 28, 1024
assert _PAD_TO % _ALIGN == 0


def unsupported_reason(X: torch.Tensor, C: torch.Tensor) -> str | None:
    """Why :func:`nearest_centroid` cannot handle these inputs, or ``None`` if it can."""
    if X.dim() != 2 or C.dim() != 2 or X.shape[1] != C.shape[1]:
        return "X and the centers must be 2D with the same number of features"
    if X.dtype != C.dtype:
        return f"X is {X.dtype} but the centers are {C.dtype}"
    if C.device != X.device:
        return f"X is on {X.device} but the centers are on {C.device}"
    if C.shape[0] == 0:
        return "there are no centers"
    if not AVAILABLE:
        return _NO_TRITON + (f" ({_IMPORT_ERROR!r})" if _IMPORT_ERROR else "")
    if X.device.type != "cuda":
        return f"{X.device} is not a CUDA device"
    if X.dtype != torch.float32:
        return f"the kernel needs float32, got {X.dtype}"
    return None


def nearest_centroid(X: torch.Tensor, C: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Nearest row of ``C`` for each row of ``X`` (float32 on CUDA).

    Returns
    -------

    **partial** (`torch.Tensor`, shape ``(n,)``, float32):
        ``min_j ||c_j||^2 - 2 x.c_j``; add ``||x||^2`` for squared distances.

    **labels** (`torch.Tensor`, shape ``(n,)``, int64):
        Index of the minimizing centroid (lowest index on exact ties).
    """
    n, d = X.shape
    if n == 0:
        return X.new_empty(0), torch.empty(0, dtype=torch.int64, device=X.device)
    # The norms come from the unpadded, contiguous centers: zeros, or another
    # memory layout, could change their rounding and with it the labels.
    C = C.contiguous()
    c_sq = row_sq_norms(C)
    partial = torch.empty(n, device=X.device, dtype=torch.float32)
    labels = torch.empty(n, device=X.device, dtype=torch.int64)
    pad = -d % _PAD_TO
    if pad == 0 and _aligned(X):  # X is used as it is
        if not _aligned(C):
            C = _padded(C, 0)
        _launch(X, C, c_sq, partial, labels)
        return partial, labels
    if X.stride(1) == 1 and C.shape[0] < _PAD_MIN_K:  # too few centers to pay for a copy
        _launch(X, C, c_sq, partial, labels)
        return partial, labels
    if pad or not _aligned(C):
        C = _padded(C, pad)
    row_bytes = max(1, (d + pad) * X.element_size())  # d may be 0
    rows = max(1, _PAD_BYTES // row_bytes // _BLOCK_M) * _BLOCK_M
    for begin in range(0, n, rows):
        end = min(begin + rows, n)
        _launch(_padded(X[begin:end], pad), C, c_sq, partial[begin:end], labels[begin:end])
    return partial, labels


def _aligned(T: torch.Tensor) -> bool:
    return T.stride(1) == 1 and T.stride(0) % _ALIGN == 0 and T.data_ptr() % _ALIGN == 0


def _padded(T: torch.Tensor, pad: int) -> torch.Tensor:
    """A fresh contiguous copy of ``T`` with ``pad`` zero columns appended."""
    d = T.shape[1]
    out = T.new_empty(T.shape[0], d + pad)
    out[:, d:].zero_()  # only the pad columns: the rest is overwritten
    out[:, :d] = T
    return out


def _launch(
    X: torch.Tensor,
    C: torch.Tensor,
    c_sq: torch.Tensor,
    partial: torch.Tensor,
    labels: torch.Tensor,
) -> None:
    """Run the kernel on ``X`` and ``C`` with unit column stride, writing ``partial`` and ``labels``."""
    n, d = X.shape
    # Launch on X's device and its current stream, which the prefetcher waits on.
    with torch.cuda.device(X.device):
        _assign_kernel[(triton.cdiv(n, _BLOCK_M),)](
            X,
            C,
            c_sq,
            partial,
            labels,
            n,
            C.shape[0],
            d,
            X.stride(0),
            C.stride(0),
            BLOCK_M=_BLOCK_M,
            BLOCK_N=_BLOCK_N,
            BLOCK_K=_BLOCK_K,
            num_warps=_NUM_WARPS,
            num_stages=_NUM_STAGES,
        )
