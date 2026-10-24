"""Smoke tests for Faiss k-means backend (optional faiss-cpu / faiss-gpu)."""

from __future__ import annotations

import unittest

import torch

try:
    import faiss  # noqa: F401

    _HAVE_FAISS = True
except ImportError:
    _HAVE_FAISS = False

from evaluation.backends.faiss_backend import run_faiss_kmeans


@unittest.skipUnless(_HAVE_FAISS, "faiss not installed")
class TestFaissBackend(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.n = 200
        self.k = 4
        self.d = 8
        self.niter = 5
        self.seed = 42
        self.device = torch.device("cpu")
        self.x = torch.randn(self.n, self.d, dtype=torch.float32)

    def test_faiss_kmeans_shape_and_finite(self) -> None:
        out = run_faiss_kmeans(
            self.x, self.k, self.niter, self.seed, self.device
        )
        c = out["centroids"]
        self.assertEqual(c.shape, (self.k, self.d))
        self.assertTrue(torch.isfinite(c).all())
        self.assertGreater(out["wall_time_s"], 0.0)
        self.assertGreaterEqual(out["cpu_time_s"], 0.0)
        self.assertIn("iters_done", out)
        self.assertGreaterEqual(out["iters_done"], 1)


if __name__ == "__main__":
    unittest.main()
