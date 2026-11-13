"""API: run_minibatch_kmeans runs with center reassignment on (default) and off."""

from __future__ import annotations

import unittest
from unittest import mock

import torch

from evaluation import perf_test, run
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
        self.assertIs(out["reassign_centers"], model.call_args.kwargs["reassign_centers"])
        return out["centroids"], model.call_args.kwargs["reassign_centers"]

    def test_reassignment_is_on_by_default(self) -> None:
        centers, switch = self._run()
        self.assertIs(switch, True)
        self.assertEqual(centers.shape, (self.k, self.d))

    def test_reassignment_can_be_turned_off(self) -> None:
        centers, switch = self._run(reassign_centers=False)
        self.assertIs(switch, False)
        self.assertTrue(torch.isfinite(centers).all())

    def test_command_line_flag_reaches_the_backend(self) -> None:
        # --no-reassign-centers is the only way to get the no-reassignment baseline.
        for argv, expected in (([], True), (["--no-reassign-centers"], False)):
            args = run._build_parser().parse_args(["--data-path", "unused", *argv])
            self.assertIs(args.reassign_centers, expected)
            self.assertIs(perf_test.parse_args(argv).reassign_centers, expected)
            with mock.patch.object(
                run, "run_minibatch_kmeans", wraps=run.run_minibatch_kmeans
            ) as b:
                run._fit_centroids(
                    run.MINIBATCHKMEANS,
                    self.x,
                    self.k,
                    2,
                    self.seed,
                    self.device,
                    batch_size=32,
                    reassign_centers=args.reassign_centers,
                )
            self.assertIs(b.call_args.kwargs["reassign_centers"], expected)
            with mock.patch.object(
                perf_test, "run_minibatch_kmeans", wraps=perf_test.run_minibatch_kmeans
            ) as b:
                perf_test._run_backend(
                    perf_test.MINIBATCHKMEANS,
                    self.x,
                    self.k,
                    2,
                    self.seed,
                    self.device,
                    batch_size=32,
                    reassignment_ratio=0.01,
                    reassign_centers=expected,
                    flash_warmup_runs=0,
                    flash_timed_repeats=1,
                )
            self.assertIs(b.call_args.kwargs["reassign_centers"], expected)


if __name__ == "__main__":
    unittest.main()
