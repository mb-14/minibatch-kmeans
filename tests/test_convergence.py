"""Stopping rules of MiniBatchKMeans.fit: centers tolerance and inertia EWA stall."""

from __future__ import annotations

import contextlib
import io
import unittest

import torch

from minibatch_kmeans import MiniBatchKMeans
from minibatch_kmeans._core import _ConvergenceState

N_SAMPLES = 1000
BATCH_SIZE = 100
N_STEPS = 50


class TestConvergenceState(unittest.TestCase):
    def test_first_update_sets_ewa(self) -> None:
        state = _ConvergenceState()
        state.update_ewa(4.0, batch_size=BATCH_SIZE, n_samples=N_SAMPLES)
        self.assertEqual(state.ewa, 4.0)

    def test_update_blends_with_alpha(self) -> None:
        state = _ConvergenceState()
        state.update_ewa(4.0, batch_size=BATCH_SIZE, n_samples=N_SAMPLES)
        state.update_ewa(2.0, batch_size=BATCH_SIZE, n_samples=N_SAMPLES)
        alpha = BATCH_SIZE * 2.0 / (N_SAMPLES + 1)
        self.assertAlmostEqual(state.ewa, 4.0 * (1 - alpha) + 2.0 * alpha)

    def test_alpha_is_capped_at_one(self) -> None:
        state = _ConvergenceState()
        state.update_ewa(4.0, batch_size=500, n_samples=10)
        state.update_ewa(2.0, batch_size=500, n_samples=10)
        self.assertEqual(state.ewa, 2.0)

    def test_stalled_counts_non_improvements(self) -> None:
        state = _ConvergenceState()
        state.ewa = 5.0
        self.assertFalse(state.stalled(2))  # first value is the new minimum
        self.assertEqual(state.ewa_min, 5.0)
        state.ewa = 6.0
        self.assertFalse(state.stalled(2))
        state.ewa = 7.0
        self.assertTrue(state.stalled(2))
        self.assertEqual(state.no_improvement, 2)

    def test_improvement_resets_counter(self) -> None:
        state = _ConvergenceState()
        state.ewa = 5.0
        state.stalled(2)
        state.ewa = 6.0
        state.stalled(2)
        state.ewa = 4.0
        self.assertFalse(state.stalled(2))
        self.assertEqual(state.no_improvement, 0)
        self.assertEqual(state.ewa_min, 4.0)

    def test_none_never_stalls(self) -> None:
        state = _ConvergenceState()
        state.ewa = 5.0
        state.stalled(None)
        for ewa in (6.0, 7.0, 8.0, 9.0):
            state.ewa = ewa
            self.assertFalse(state.stalled(None))


class TestMiniBatchConvergence(unittest.TestCase):
    """Drive ``_mini_batch_convergence`` step by step with hand-picked inputs."""

    def setUp(self) -> None:
        self.km = MiniBatchKMeans(n_clusters=2, verbose=0)
        self.km._cluster_centers = torch.zeros(2, 3)
        self.state = _ConvergenceState()

    def _check(
        self,
        step: int,
        *,
        batch_inertia: float | None,
        shift: float = 1.0,
        tol: float = 0.0,
        max_no_improvement: int | None = None,
    ) -> bool:
        centers_new = self.km._cluster_centers + shift
        return self.km._mini_batch_convergence(
            step,
            N_STEPS,
            N_SAMPLES,
            BATCH_SIZE,
            centers_new,
            batch_inertia,
            tol,
            max_no_improvement,
            self.state,
        )

    def test_first_step_never_converges_or_touches_ewa(self) -> None:
        self.assertFalse(self._check(0, batch_inertia=500.0, shift=0.0, tol=1.0))
        self.assertIsNone(self.state.ewa)

    def test_ewa_uses_mean_batch_inertia(self) -> None:
        self._check(0, batch_inertia=500.0)
        self._check(1, batch_inertia=500.0)
        self.assertEqual(self.state.ewa, 500.0 / BATCH_SIZE)

    def test_small_center_change_converges(self) -> None:
        self._check(0, batch_inertia=None)
        self.assertFalse(self._check(1, batch_inertia=None, shift=1.0, tol=1e-3))
        self.assertTrue(self._check(2, batch_inertia=None, shift=0.0, tol=1e-3))

    def test_zero_tol_disables_center_check(self) -> None:
        self._check(0, batch_inertia=None)
        self.assertFalse(self._check(1, batch_inertia=None, shift=0.0, tol=0.0))

    def test_inertia_stall_converges(self) -> None:
        inertias = [500.0, 500.0, 600.0, 700.0]  # step 0 skipped, then a minimum, then 2 worse
        results = [
            self._check(i, batch_inertia=v, max_no_improvement=2) for i, v in enumerate(inertias)
        ]
        self.assertEqual(results, [False, False, False, True])

    def test_center_check_wins_over_stall_count(self) -> None:
        """Tolerance is checked before the stall counter, which then isn't advanced."""
        self._check(0, batch_inertia=500.0, max_no_improvement=1)
        self._check(1, batch_inertia=500.0, max_no_improvement=1)
        converged = self._check(2, batch_inertia=600.0, shift=0.0, tol=1e-3, max_no_improvement=1)
        self.assertTrue(converged)
        self.assertEqual(self.state.no_improvement, 0)

    def test_no_inertia_never_stalls(self) -> None:
        for i in range(10):
            self.assertFalse(self._check(i, batch_inertia=None, max_no_improvement=1))


class TestFitStoppingPaths(unittest.TestCase):
    """End-to-end: each stopping rule ends ``fit`` early and says why."""

    def _fit(self, **kw: object) -> list[str]:
        torch.manual_seed(0)
        x = torch.randn(5000, 8)
        km = MiniBatchKMeans(n_clusters=4, verbose=1, random_state=torch.Generator().manual_seed(1))
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            km.fit(x, batch_size=500, max_iter=20, **kw)
        return out.getvalue().splitlines()

    def test_small_center_change(self) -> None:
        lines = self._fit(tol=5e-2, max_no_improvement=None)
        self.assertTrue(lines[-1].startswith("Converged (small centers change)"))
        self.assertLess(len(lines) - 1, 200)

    def test_lack_of_improvement(self) -> None:
        lines = self._fit(tol=0.0, max_no_improvement=3)
        self.assertTrue(lines[-1].startswith("Converged (lack of improvement in inertia)"))
        self.assertLess(len(lines) - 1, 200)

    def test_runs_all_steps_when_both_disabled(self) -> None:
        lines = self._fit(tol=0.0, max_no_improvement=None)
        self.assertFalse(any(line.startswith("Converged") for line in lines))
        self.assertEqual(len(lines), 200)  # max_iter * n_samples // batch_size


if __name__ == "__main__":
    unittest.main()
