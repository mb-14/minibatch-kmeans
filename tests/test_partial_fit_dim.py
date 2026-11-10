"""partial_fit rejects a column count that differs from the fitted centers with ValueError."""

import unittest

import torch

from minibatch_kmeans import MiniBatchKMeans


class TestPartialFitDimCheck(unittest.TestCase):
    def test_mismatched_columns_raise_value_error(self) -> None:
        km = MiniBatchKMeans(n_clusters=3)
        km.partial_fit(torch.randn(100, 4))
        centers, counts = km.cluster_centers.clone(), km._counts.clone()
        with self.assertRaisesRegex(ValueError, "dim 5"):
            km.partial_fit(torch.randn(100, 5))
        # The rejected batch leaves the model untouched.
        torch.testing.assert_close(km.cluster_centers, centers)
        torch.testing.assert_close(km._counts, counts)

    def test_fit_may_change_dimension(self) -> None:
        km = MiniBatchKMeans(n_clusters=3)
        km.fit(torch.randn(100, 4), batch_size=32, max_iter=1, max_no_improvement=None)
        km.fit(torch.randn(100, 5), batch_size=32, max_iter=1, max_no_improvement=None)
        self.assertEqual(km.cluster_centers.shape, (3, 5))


if __name__ == "__main__":
    unittest.main()
