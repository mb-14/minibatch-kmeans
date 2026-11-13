"""API: run_minibatch_kmeans runs with center reassignment on (default) and off."""

from __future__ import annotations

import unittest
from unittest import mock

import torch

from evaluation.backends import minibatch_kmeans as backend
from minibatch_kmeans import MiniBatchKMeans


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

    def _run(self, **kwargs: bool) -> tuple[torch.Tensor, bool]:
        """Centroids, and the ``reassign_centers`` value the model was built with."""
        with mock.patch.object(backend, "MiniBatchKMeans", wraps=MiniBatchKMeans) as model:
            out = backend.run_minibatch_kmeans(
                self.x, self.k, self.niter, self.seed, self.device, batch_size=32, **kwargs
            )
        self.assertNotIn("reassignment_ratio", out)
        return out["centroids"], model.call_args.kwargs["reassign_centers"]

    def test_reassignment_is_on_by_default(self) -> None:
        centers, switch = self._run()
        self.assertIs(switch, True)
        self.assertEqual(centers.shape, (self.k, self.d))

    def test_reassignment_can_be_turned_off(self) -> None:
        centers, switch = self._run(reassign_centers=False)
        self.assertIs(switch, False)
        self.assertTrue(torch.isfinite(centers).all())


if __name__ == "__main__":
    unittest.main()
