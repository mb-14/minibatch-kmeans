"""Tests for benchmark synthetic data helpers."""

from __future__ import annotations

import unittest

import torch

from evaluation.data import gaussian_mixture, gaussian_mixture_imbalanced, train_val_split


class TestTrainValSplit(unittest.TestCase):
    def test_full_data_returns_none_val(self) -> None:
        x = torch.randn(20, 3)
        y = torch.arange(20)
        xt, yt, xv, yv = train_val_split(x, y, train_fraction=1.0, seed=0)
        self.assertIs(xt, x)
        self.assertIs(yt, y)
        self.assertIsNone(xv)
        self.assertIsNone(yv)

    def test_holdout_shapes_and_disjoint(self) -> None:
        x, y = gaussian_mixture(1000, 8, 4, seed=42)
        xt, yt, xv, yv = train_val_split(x, y, train_fraction=0.8, seed=42)
        self.assertEqual(xt.shape, (800, 4))
        self.assertEqual(yt.shape, (800,))
        self.assertEqual(xv.shape, (200, 4))
        self.assertEqual(yv.shape, (200,))

    def test_deterministic_per_seed(self) -> None:
        x, y = gaussian_mixture(500, 10, 2, seed=7)
        a = train_val_split(x, y, train_fraction=0.7, seed=123)
        b = train_val_split(x, y, train_fraction=0.7, seed=123)
        for t, u in zip(a, b, strict=True):
            self.assertTrue(torch.equal(t, u))

    def test_invalid_fraction(self) -> None:
        x = torch.randn(10, 2)
        y = torch.zeros(10, dtype=torch.int64)
        with self.assertRaises(ValueError):
            train_val_split(x, y, train_fraction=0.0, seed=0)
        with self.assertRaises(ValueError):
            train_val_split(x, y, train_fraction=1.1, seed=0)

    def test_too_small_for_two_non_empty_parts(self) -> None:
        x = torch.randn(5, 2)
        y = torch.zeros(5, dtype=torch.int64)
        with self.assertRaises(ValueError):
            train_val_split(x, y, train_fraction=0.01, seed=0)


class TestGaussianMixtureImbalanced(unittest.TestCase):
    def test_shape_and_labels(self) -> None:
        x, y = gaussian_mixture_imbalanced(500, 8, 4, seed=1)
        self.assertEqual(x.shape, (500, 4))
        self.assertEqual(y.shape, (500,))
        self.assertTrue((y >= 0).all() and (y < 8).all())

    def test_deterministic(self) -> None:
        a = gaussian_mixture_imbalanced(200, 5, 3, seed=99)
        b = gaussian_mixture_imbalanced(200, 5, 3, seed=99)
        self.assertTrue(torch.equal(a[0], b[0]))
        self.assertTrue(torch.equal(a[1], b[1]))


if __name__ == "__main__":
    unittest.main()
