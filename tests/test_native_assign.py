"""PyTorch assignment path: ||x||^2 is added to the row minimum, not to every distance."""

import math
import unittest

import torch

from minibatch_kmeans import MiniBatchKMeans

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _model(C):
    km = MiniBatchKMeans(C.shape[0], device=C.device, dtype=C.dtype)
    km._cluster_centers = C
    return km


class TestNativeAssign(unittest.TestCase):
    def test_matches_float64_distances(self):
        for dev in DEVICES:
            with self.subTest(dev):
                g = torch.Generator(device=dev).manual_seed(0)
                X = torch.randn(3000, 24, device=dev, generator=g) + 3.0
                C = torch.randn(200, 24, device=dev, generator=g) + 3.0
                labels, inertia = _model(C)._labels_inertia(X)
                true = torch.cdist(X.double(), C.double()).square()
                best = true.min(1).values
                chosen = true.gather(1, labels[:, None]).squeeze(1)
                # A label may differ from float64 only on a near-tie.
                tol = 1e-5 * (best + X.double().square().sum(1)) + 1e-4
                self.assertTrue(bool(((chosen - best) <= tol).all()))
                self.assertAlmostEqual(inertia / best.sum().item(), 1.0, delta=1e-5)

    def test_exact_ties_take_lowest_index(self):
        # Small integers: every product and sum is exact, so the tie is exact.
        X = torch.tensor([[1.0, 2.0], [0.0, 0.0], [5.0, 5.0]])
        C = torch.tensor([[9.0, 9.0], [1.0, 2.0], [1.0, 2.0], [0.0, 0.0]])
        for dev in DEVICES:
            with self.subTest(dev):
                labels, inertia = _model(C.to(dev))._labels_inertia(X.to(dev))
                self.assertEqual(labels.tolist(), [1, 3, 1])  # 1 and 2 tie on rows 0 and 2
                self.assertEqual(inertia, 0.0 + 0.0 + 25.0)

    def test_nan_center_wins_like_argmin(self):
        for dev in DEVICES:
            with self.subTest(dev):
                X = torch.randn(50, 4, device=dev)
                C = torch.randn(6, 4, device=dev)
                C[4] = float("nan")
                labels, inertia = _model(C)._labels_inertia(X)
                self.assertTrue(bool((labels == 4).all()))
                self.assertTrue(math.isnan(inertia))

    def test_without_inertia(self):
        X, C = torch.randn(40, 3), torch.randn(5, 3)
        labels, inertia = _model(C)._labels_inertia(X, return_inertia=False)
        self.assertIsNone(inertia)
        self.assertEqual(labels.tolist(), _model(C)._labels_inertia(X)[0].tolist())


if __name__ == "__main__":
    unittest.main()
