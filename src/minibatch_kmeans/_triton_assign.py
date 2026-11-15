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
_FP32_CONFIG = {
    "BLOCK_M": _BLOCK_M,
    "BLOCK_N": _BLOCK_N,
    "BLOCK_K": _BLOCK_K,
    "num_warps": _NUM_WARPS,
    "num_stages": _NUM_STAGES,
}
# bfloat16 tiles, from closed PR #11 (tuned on a GB10). The compiled kernel
# uses 65.8 KB of shared memory (Triton 3.6), within the 99 KB per block that
# every GPU with bfloat16 tensor cores (compute capability 8.0+) allows.
_BF16_CONFIG = {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 32, "num_warps": 8, "num_stages": 3}
# Triton vectorizes the row loads only when the feature count, the row stride
# and the data pointer are divisible by _ALIGN (Triton's fixed divisibility
# check); other inputs are copied into fresh rows zero-padded to a multiple of
# the feature tile (_PAD_TO, _BF16_PAD_TO), which leaves every dot product
# unchanged. Padding to a multiple of the feature tile keeps the tile
# boundaries, so the summation order, and the result bits, match the unpadded
# run (the pad widths must stay multiples of _ALIGN).
# X is copied in chunks of at most _PAD_BYTES (256 MB), or of one row tile
# when a tile is larger than that, so the copy stays bounded however many rows
# X has; smaller chunks were slower at 1000 features. Chunks are whole row
# tiles, which keeps every chunk's output pointers aligned (an unaligned one
# compiles a second kernel variant). With fewer than _PAD_MIN_K centers the
# float32 kernel is short and the copy costs more than the faster loads save
# (up to 1.8x slower at k=64), so X is only copied when its columns are not
# contiguous. At 1024 centers padding breaks even at 200-1000 features and wins
# below; small widths already win from about 400 centers. The bfloat16 kernel
# is bound by its loads, so padding pays from fewer centers: it loses at 256
# and wins at 512 (1.06-1.27x at 5-1000 features), up to 2.3x at 16384.
_ALIGN, _PAD_TO, _PAD_BYTES, _PAD_MIN_K = 16, _BLOCK_K, 1 << 28, 1024
_BF16_PAD_TO, _BF16_PAD_MIN_K = _BF16_CONFIG["BLOCK_K"], 512
assert _PAD_TO % _ALIGN == 0 and _BF16_PAD_TO % _ALIGN == 0


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
    # The norms come from the unpadded, contiguous centers: zeros, or another
    # memory layout, could change their rounding and with it the labels.
    C = C.contiguous()
    return _assign(X, C, row_sq_norms(C), None, _FP32_CONFIG, _PAD_TO, _PAD_MIN_K)


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
    if mu.numel() == 0:  # no features
        return mu
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
    if X.shape[0] == 0:
        return X.new_empty(0), torch.empty(0, dtype=torch.int64, device=X.device)
    # The shift (a mean over C) and the X @ mu correction below round
    # differently on other memory layouts, so both start from contiguous rows.
    C = C.contiguous()
    if X.stride(1) != 1:
        X = X.contiguous()
    mu = _bf16_shift(C)
    Ch = (C - mu).to(torch.bfloat16).contiguous()
    c_sq = row_sq_norms(Ch.float())  # norms of the rounded centers, as in the products
    partial, labels = _assign(X, Ch, c_sq, mu, _BF16_CONFIG, _BF16_PAD_TO, _BF16_PAD_MIN_K)
    # The kernel's minimum is over shifted points: ||c-mu||^2 - 2 (x-mu).(c-mu).
    # Adding ||mu||^2 - 2 x.mu makes it ||c||^2 - 2 x.c, as nearest_centroid
    # returns, without a shifted copy of X.
    return partial.add_(mu.dot(mu) - 2 * (X @ mu)), labels


def _assign(
    X: torch.Tensor,
    C: torch.Tensor,
    c_sq: torch.Tensor,
    mu: torch.Tensor | None,
    cfg: dict,
    pad_to: int,
    min_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Launch the kernel on ``X`` and the contiguous, kernel-ready centers ``C``,
    zero-padding the features as the comment on ``_ALIGN`` describes.

    ``mu`` is the bfloat16 shift (padded with zeros, so pad columns stay 0
    after the shift), or ``None`` for the float32 kernel.
    """
    n, d = X.shape
    if n == 0:
        return X.new_empty(0), torch.empty(0, dtype=torch.int64, device=X.device)
    partial = torch.empty(n, device=X.device, dtype=torch.float32)
    labels = torch.empty(n, device=X.device, dtype=torch.int64)
    pad = -d % pad_to
    if pad == 0 and _aligned(X):  # X is used as it is
        if not _aligned(C):
            C = _padded(C, 0)
        _launch(X, C, c_sq, mu, partial, labels, cfg)
        return partial, labels
    if X.stride(1) == 1 and C.shape[0] < min_k:  # too few centers to pay for a copy
        _launch(X, C, c_sq, mu, partial, labels, cfg)
        return partial, labels
    if pad or not _aligned(C):
        C = _padded(C, pad)
    if mu is not None and pad:
        mu = _padded(mu[None], pad)[0]
    row_bytes = max(1, (d + pad) * X.element_size())  # d may be 0
    block_m = cfg["BLOCK_M"]
    rows = max(1, _PAD_BYTES // row_bytes // block_m) * block_m
    for begin in range(0, n, rows):
        end = min(begin + rows, n)
        chunk = _padded(X[begin:end], pad)
        _launch(chunk, C, c_sq, mu, partial[begin:end], labels[begin:end], cfg)
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
    mu: torch.Tensor | None,
    partial: torch.Tensor,
    labels: torch.Tensor,
    cfg: dict,
) -> None:
    """Run the kernel on ``X`` and ``C`` with unit column stride, writing ``partial`` and ``labels``."""
    n, d = X.shape
    # Launch on X's device and its current stream, which the prefetcher waits on.
    with torch.cuda.device(X.device):
        _assign_kernel[(triton.cdiv(n, cfg["BLOCK_M"]),)](
            X,
            c_sq if mu is None else mu,  # mu is unused when BF16=False
            C,
            c_sq,
            partial,
            labels,
            n,
            C.shape[0],
            d,
            X.stride(0),
            C.stride(0),
            BF16=mu is not None,
            **cfg,
        )
