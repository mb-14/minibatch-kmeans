"""Fused distance + argmin assignment on CUDA (Triton).

The kernel streams centroid tiles through ``tl.dot`` and keeps a running
min/argmin per row, so the ``(n, k)`` distance matrix is never stored.
:func:`nearest_centroid` uses full float32 products; :func:`nearest_centroid_bf16`
rounds both operands to bfloat16 (float32 accumulation), which runs on tensor
cores.
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
        mu_ptr,
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
        BF16: tl.constexpr,
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
                if BF16:
                    # C arrives shifted and rounded; shift and round X here, tile by tile.
                    mu = tl.load(mu_ptr + ks, mask=k_ok, other=0.0)
                    xt = (xt - mu[None, :]).to(tl.bfloat16)
                    acc = tl.dot(xt, ct, acc)
                else:
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
# bfloat16 tiles, from closed PR #11 (tuned on a GB10).
_BF16_CONFIG = {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 32, "num_warps": 8, "num_stages": 3}


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
    if X.stride(1) != 1:
        X = X.contiguous()
    C = C.contiguous()
    n, d = X.shape
    k = C.shape[0]
    if n == 0:
        return X.new_empty(0), torch.empty(0, dtype=torch.int64, device=X.device)
    c_sq = row_sq_norms(C)
    partial = torch.empty(n, device=X.device, dtype=torch.float32)
    labels = torch.empty(n, device=X.device, dtype=torch.int64)

    # Launch on X's device and its current stream, which the prefetcher waits on.
    with torch.cuda.device(X.device):
        _assign_kernel[(triton.cdiv(n, _BLOCK_M),)](
            X,
            c_sq,  # unused (BF16=False)
            C,
            c_sq,
            partial,
            labels,
            n,
            k,
            d,
            X.stride(0),
            C.stride(0),
            BLOCK_M=_BLOCK_M,
            BLOCK_N=_BLOCK_N,
            BLOCK_K=_BLOCK_K,
            BF16=False,
            num_warps=_NUM_WARPS,
            num_stages=_NUM_STAGES,
        )
    return partial, labels


def _bf16_shift(C: torch.Tensor) -> torch.Tensor:
    """Offset subtracted from ``X`` and ``C`` before rounding them to bfloat16.

    Distances do not change under a common shift, but rounding error grows
    with magnitude. Centering on the centers' mean keeps the error relative to
    the centers' spread, not to the data's distance from the origin. The mean
    is snapped to a power-of-two grid no finer than that spread, so integer
    data stays integer. NaN centers are ignored. Where the grid is too fine
    for the mean (no spread, as with one center), the mean itself is used.
    (From closed PR #13.)
    """
    mu = torch.nan_to_num(C.nanmean(0))
    spread = torch.nan_to_num(C - mu).abs().amax()
    step = torch.exp2(torch.floor(torch.log2(spread.clamp(min=torch.finfo(C.dtype).tiny))))
    snapped = torch.round(mu / step) * step
    return torch.where(torch.isfinite(snapped), snapped, mu)


def nearest_centroid_bf16(X: torch.Tensor, C: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Nearest row of ``C`` for each row of ``X``, from bfloat16 products.

    ``X`` and ``C`` (float32, CUDA) are shifted by :func:`_bf16_shift` and
    rounded to bfloat16; products accumulate in float32. Near-ties may resolve
    differently from :func:`nearest_centroid`. Same return values as
    :func:`nearest_centroid`; ``partial`` comes from the rounded products.
    """
    if X.stride(1) != 1:
        X = X.contiguous()
    n, d = X.shape
    k = C.shape[0]
    if n == 0:
        return X.new_empty(0), torch.empty(0, dtype=torch.int64, device=X.device)
    mu = _bf16_shift(C).contiguous()
    Ch = (C - mu).to(torch.bfloat16).contiguous()
    c_sq = row_sq_norms(Ch.float())  # norms of the rounded centers, as in the products
    partial = torch.empty(n, device=X.device, dtype=torch.float32)
    labels = torch.empty(n, device=X.device, dtype=torch.int64)
    cfg = _BF16_CONFIG
    with torch.cuda.device(X.device):
        _assign_kernel[(triton.cdiv(n, cfg["BLOCK_M"]),)](
            X,
            mu,
            Ch,
            c_sq,
            partial,
            labels,
            n,
            k,
            d,
            X.stride(0),
            Ch.stride(0),
            BF16=True,
            **cfg,
        )
    # The kernel's minimum is over shifted points: ||c-mu||^2 - 2 (x-mu).(c-mu).
    # Adding ||mu||^2 - 2 x.mu makes it ||c||^2 - 2 x.c, as nearest_centroid
    # returns, without a shifted copy of X.
    return partial.add_(mu.dot(mu) - 2 * (X @ mu)), labels
