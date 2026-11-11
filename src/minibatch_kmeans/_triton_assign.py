"""Fused nearest-centroid assignment on CUDA (Triton).

One kernel computes, for each row ``x`` of ``X``, the minimum over centroids of
``||c||^2 - 2 x.c`` and its index, streaming centroid tiles through ``tl.dot``
(half-precision inputs, float32 accumulation) and keeping a running min/argmin
per row in registers. Only ``(min, label)`` per row is written, so the
``(n, k)`` distance matrix is never materialized.

The caller adds ``||x||^2`` afterwards when it needs inertia.

``triton`` is optional: :data:`AVAILABLE` is false when it cannot be imported,
and callers must then use the PyTorch path.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised only without triton installed
    triton = None
    tl = None

AVAILABLE = triton is not None

_TL_DTYPES = {}

if AVAILABLE:
    _TL_DTYPES = {torch.bfloat16: tl.bfloat16, torch.float16: tl.float16}

    @triton.jit
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
        cent_per_split,
        N_SPLIT: tl.constexpr,
        DOT_DTYPE: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_s = tl.program_id(1)
        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        row_ok = rows < n_rows
        rows64 = rows.to(tl.int64)
        offs_k = tl.arange(0, BLOCK_K)

        best = tl.full((BLOCK_M,), float("inf"), tl.float32)
        best_idx = tl.zeros((BLOCK_M,), tl.int32)

        cent_begin = pid_s * cent_per_split
        cent_end = tl.minimum(cent_begin + cent_per_split, n_cent)
        for n0 in range(cent_begin, cent_end, BLOCK_N):
            cols = n0 + tl.arange(0, BLOCK_N)
            col_ok = cols < cent_end
            cols64 = cols.to(tl.int64)
            acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
            for k0 in range(0, d, BLOCK_K):
                ks = k0 + offs_k
                k_ok = ks < d
                xt = tl.load(
                    x_ptr + rows64[:, None] * stride_xm + ks[None, :],
                    mask=row_ok[:, None] & k_ok[None, :],
                    other=0.0,
                ).to(DOT_DTYPE)
                ct = tl.load(
                    c_ptr + cols64[None, :] * stride_cn + ks[:, None],
                    mask=col_ok[None, :] & k_ok[:, None],
                    other=0.0,
                ).to(DOT_DTYPE)
                acc = tl.dot(xt, ct, acc)
            c_sq = tl.load(c_sq_ptr + cols, mask=col_ok, other=0.0)
            dist = c_sq[None, :] - 2.0 * acc
            dist = tl.where(col_ok[None, :], dist, float("inf"))
            tile_min, tile_arg = tl.min(
                dist, axis=1, return_indices=True, return_indices_tie_break_left=True
            )
            # Strict "<": on a tie the earlier tile (lower index) wins.
            better = tile_min < best
            best_idx = tl.where(better, n0 + tile_arg, best_idx)
            best = tl.where(better, tile_min, best)

        out = rows64 * N_SPLIT + pid_s
        tl.store(out_min_ptr + out, best, mask=row_ok)
        tl.store(out_idx_ptr + out, best_idx, mask=row_ok)


# One fixed tile config, no autotuning: autotuning costs seconds per new shape
# in every fresh process, and configs with different BLOCK_K sum the dot
# products in different orders, so the chosen config would change results.
_BLOCK_M, _BLOCK_N, _BLOCK_K = 128, 128, 32
_NUM_WARPS, _NUM_STAGES = 8, 3


def supported(X: torch.Tensor, C: torch.Tensor, assign_dtype: torch.dtype | None) -> bool:
    """Whether :func:`nearest_centroid` handles these inputs."""
    return (
        AVAILABLE
        and assign_dtype in _TL_DTYPES
        and X.is_cuda
        and C.device == X.device
        and X.dtype == torch.float32
        and C.dtype == torch.float32
        and X.dim() == 2
        and C.dim() == 2
        and X.shape[1] == C.shape[1]
        and X.shape[0] > 0
        and C.shape[0] > 0
    )


def nearest_centroid(
    X: torch.Tensor, C: torch.Tensor, assign_dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """Nearest row of ``C`` for each row of ``X``.

    Both are rounded to ``assign_dtype`` (bfloat16 or float16) for the cross
    term, accumulated in float32. ``||c||^2`` is computed in float32.

    Returns
    -------

    **partial** (`torch.Tensor`, shape ``(n,)``, float32):
        ``min_j ||c_j||^2 - 2 x.c_j``; add ``||x||^2`` for squared distances.

    **labels** (`torch.Tensor`, shape ``(n,)``, int64):
        Index of the minimizing centroid (lowest index on exact ties).
    """
    if X.stride(1) != 1:
        X = X.contiguous()
    C = C.contiguous()
    n, d = X.shape
    k = C.shape[0]
    c_sq = torch.linalg.vector_norm(C, ord=2, dim=1).square()

    # Few row blocks (small batches) leave SMs idle at large k: split the
    # centroids across a second grid axis and reduce the partial results.
    sms = torch.cuda.get_device_properties(X.device).multi_processor_count
    row_blocks = triton.cdiv(n, _BLOCK_M)
    n_split = 1
    while row_blocks * n_split < 2 * sms and k // (n_split * 2) >= 512:
        n_split *= 2
    cent_per_split = triton.cdiv(k, n_split)

    out_min = torch.empty((n, n_split), device=X.device, dtype=torch.float32)
    out_idx = torch.empty((n, n_split), device=X.device, dtype=torch.int32)

    grid = (row_blocks, n_split)
    _assign_kernel[grid](
        X,
        C,
        c_sq,
        out_min,
        out_idx,
        n,
        k,
        d,
        X.stride(0),
        C.stride(0),
        cent_per_split,
        N_SPLIT=n_split,
        DOT_DTYPE=_TL_DTYPES[assign_dtype],
        BLOCK_M=_BLOCK_M,
        BLOCK_N=_BLOCK_N,
        BLOCK_K=_BLOCK_K,
        num_warps=_NUM_WARPS,
        num_stages=_NUM_STAGES,
    )
    if n_split == 1:
        return out_min.squeeze(1), out_idx.squeeze(1).long()
    # Splits cover increasing centroid ranges; argmin returns the first
    # minimum, so exact ties still go to the lowest index.
    partial, which = out_min.min(dim=1)
    labels = out_idx.gather(1, which.unsqueeze(1)).squeeze(1).long()
    return partial, labels
