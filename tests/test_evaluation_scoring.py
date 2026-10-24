"""Tests for clustbench scoring label convention."""

import unittest

import numpy as np

from evaluation.metrics import _labels_for_clustbench_get_score


class TestClustbenchLabels(unittest.TestCase):
    def test_zero_based_to_one_based(self):
        y = np.array([0, 1, 14], dtype=np.int64)
        out = _labels_for_clustbench_get_score(y, 15)
        self.assertTrue(np.array_equal(out, np.array([1, 2, 15])))

    def test_one_based_unchanged(self):
        y = np.array([1, 2, 15], dtype=np.int64)
        out = _labels_for_clustbench_get_score(y, 15)
        self.assertTrue(np.array_equal(out, y))


if __name__ == "__main__":
    unittest.main()
