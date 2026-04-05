"""Dataset-scaled iteration counts for evaluation runs."""

from __future__ import annotations

# --- Iteration count (inverse to sample size) ----------------------------

# At ``n == REFERENCE_N``, unclamped ``raw == BASE_NITER``; result is then clamped to [NITER_MIN, NITER_MAX].
# ``REFERENCE_N`` is the median ``n`` over :data:`evaluation.presets.BENCHMARK_TASKS` (real suite).
REFERENCE_N = 100_000
NITER_MIN = 10
NITER_MAX = 50
BASE_NITER = (NITER_MIN + NITER_MAX) // 2


def scaled_niter(n: int) -> int:
    """
    Lloyd / minibatch iterations for ``n`` points: ``BASE_NITER * REFERENCE_N / n``, rounded and
    clamped to ``[NITER_MIN, NITER_MAX]``. Larger ``n`` → fewer iterations.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    raw = BASE_NITER * REFERENCE_N / n
    return int(max(NITER_MIN, min(NITER_MAX, round(raw))))
