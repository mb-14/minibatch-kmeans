"""evaluation.heuristics: scaled_niter."""

from __future__ import annotations

import unittest

from evaluation.heuristics import (
    BASE_NITER,
    NITER_MAX,
    NITER_MIN,
    REFERENCE_N,
    scaled_niter,
)


class TestScaledNiter(unittest.TestCase):
    def test_reference_n_matches_base(self) -> None:
        self.assertEqual(scaled_niter(REFERENCE_N), BASE_NITER)

    def test_larger_n_fewer_iters(self) -> None:
        self.assertGreater(scaled_niter(10_000), scaled_niter(100_000))

    def test_clamped(self) -> None:
        self.assertLessEqual(scaled_niter(100), NITER_MAX)
        self.assertGreaterEqual(scaled_niter(10_000_000), NITER_MIN)

    def test_base_is_midpoint(self) -> None:
        self.assertEqual(BASE_NITER, (NITER_MIN + NITER_MAX) // 2)

    def test_invalid_n(self) -> None:
        with self.assertRaises(ValueError):
            scaled_niter(0)


if __name__ == "__main__":
    unittest.main()
