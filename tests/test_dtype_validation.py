"""MiniBatchKMeans accepts only float32/float64; half precision breaks the update rule."""

import unittest

import torch

from minibatch_kmeans import MiniBatchKMeans


class TestDtypeValidation(unittest.TestCase):
    def test_half_precision_rejected(self) -> None:
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype), self.assertRaisesRegex(ValueError, "float32 or"):
                MiniBatchKMeans(n_clusters=2, dtype=dtype)

    def test_non_float_rejected(self) -> None:
        with self.assertRaises(ValueError):
            MiniBatchKMeans(n_clusters=2, dtype=torch.int64)

    def test_float32_and_float64_accepted(self) -> None:
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                km = MiniBatchKMeans(n_clusters=2, dtype=dtype)
                km.fit(torch.randn(20, 3, dtype=dtype), batch_size=8, max_iter=1)
                self.assertEqual(km.cluster_centers.dtype, dtype)


if __name__ == "__main__":
    unittest.main()
