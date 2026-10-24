"""Smoke tests for scikit-learn backend runners."""

from __future__ import annotations

import unittest

import torch

from evaluation.backends.sklearn_backend import (
    run_sklearn_kmeans,
    run_sklearn_minibatch_kmeans,
)


class TestSklearnBackends(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.n = 64
        self.k = 4
        self.d = 8
        self.niter = 5
        self.seed = 42
        self.device = torch.device("cpu")
        self.x = torch.randn(self.n, self.d, dtype=torch.float32)

    def test_sklearn_kmeans_shape_and_finite(self) -> None:
        out = run_sklearn_kmeans(
            self.x, self.k, self.niter, self.seed, self.device
        )
        c = out["centroids"]
        self.assertEqual(c.shape, (self.k, self.d))
        self.assertTrue(torch.isfinite(c).all())
        self.assertGreater(out["wall_time_s"], 0.0)
        self.assertGreaterEqual(out["cpu_time_s"], 0.0)
        self.assertIn("iters_done", out)

    def test_sklearn_minibatch_shape_and_finite(self) -> None:
        out = run_sklearn_minibatch_kmeans(
            self.x, self.k, self.niter, self.seed, self.device, batch_size=32
        )
        c = out["centroids"]
        self.assertEqual(c.shape, (self.k, self.d))
        self.assertTrue(torch.isfinite(c).all())
        self.assertGreater(out["wall_time_s"], 0.0)
        self.assertGreaterEqual(out["cpu_time_s"], 0.0)
        self.assertEqual(out["batch_size"], 32)
        self.assertIn("iters_done", out)


if __name__ == "__main__":
    unittest.main()
