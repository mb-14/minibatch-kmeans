"""Half-precision models keep per-cluster counts in float32 so updates stay finite and exact."""

import unittest

import torch

from minibatch_kmeans import MiniBatchKMeans

_DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestHalfPrecisionCounts(unittest.TestCase):
    def _fit(self, dtype: torch.dtype) -> MiniBatchKMeans:
        g = torch.Generator().manual_seed(0)
        x = torch.randn(20000, 4, generator=g).to(dtype)
        # No reassignment, so counts only ever grow by whole batches.
        km = MiniBatchKMeans(n_clusters=4, dtype=dtype, device=_DEV, reassignment_ratio=0.0)
        # ~390 steps x 1024 rows over 4 clusters: counts pass 65504 (float16 max).
        km.fit(x, batch_size=1024, max_iter=20, max_no_improvement=None)
        return km

    def test_float16_stays_finite(self) -> None:
        km = self._fit(torch.float16)
        self.assertEqual(km.cluster_centers.dtype, torch.float16)
        self.assertTrue(torch.isfinite(km.cluster_centers).all())
        self.assertTrue(torch.isfinite(km._counts).all())
        self.assertEqual(km._counts.dtype, torch.float32)

    def test_bfloat16_counts_keep_growing(self) -> None:
        km = self._fit(torch.bfloat16)
        # With no reassignment, counts sum to the rows processed, independent of
        # labels, so they must match a float32 run exactly. bf16 counts round off.
        expected = self._fit(torch.float32)._counts.sum().item()
        self.assertGreater(expected, 4 * 65536)
        self.assertEqual(km._counts.sum().item(), expected)


if __name__ == "__main__":
    unittest.main()
