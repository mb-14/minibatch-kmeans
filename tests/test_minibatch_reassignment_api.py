"""API: run_minibatch_kmeans accepts reassignment_ratio and returns it."""

from __future__ import annotations

import unittest

import torch

from evaluation.backends.minibatch_kmeans import run_minibatch_kmeans


class TestMinibatchReassignmentApi(unittest.TestCase):
    def setUp(self) -> None:
        self.n = 128
        self.k = 4
        self.d = 8
        self.niter = 20
        self.seed = 7
        self.device = torch.device("cpu")
        torch.manual_seed(0)
        self.x = torch.randn(self.n, self.d, dtype=torch.float32)

    def test_default_reassignment_ratio_recorded(self) -> None:
        out = run_minibatch_kmeans(
            self.x,
            self.k,
            self.niter,
            self.seed,
            self.device,
            batch_size=32,
        )
        self.assertEqual(out["reassignment_ratio"], 0.01)
        self.assertEqual(out["centroids"].shape, (self.k, self.d))

    def test_zero_reassignment_ratio(self) -> None:
        out = run_minibatch_kmeans(
            self.x,
            self.k,
            self.niter,
            self.seed,
            self.device,
            batch_size=32,
            reassignment_ratio=0.0,
        )
        self.assertEqual(out["reassignment_ratio"], 0.0)
        self.assertTrue(torch.isfinite(out["centroids"]).all())


if __name__ == "__main__":
    unittest.main()
