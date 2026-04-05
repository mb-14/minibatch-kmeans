"""Save/load roundtrip for MiniBatchKMeans."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from minibatch_kmeans import MiniBatchKMeans


class TestMiniBatchKMeansIO(unittest.TestCase):
    def test_roundtrip_fit_predict_match(self) -> None:
        torch.manual_seed(0)
        x = torch.randn(80, 5, dtype=torch.float32)
        rng = torch.Generator(device="cpu")
        rng.manual_seed(42)
        km = MiniBatchKMeans(
            n_clusters=4,
            verbose=False,
            reassignment_ratio=0.02,
            random_state=rng,
            dtype=torch.float32,
        )
        km.fit(x, batch_size=32, max_iter=2, max_no_improvement=None)
        pred_before = km.predict(x)

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "m.pt"
            km.save(path)
            loaded = MiniBatchKMeans.load(path)

        self.assertEqual(loaded.n_clusters, km.n_clusters)
        self.assertEqual(loaded.verbose, km.verbose)
        self.assertEqual(loaded.reassignment_ratio, km.reassignment_ratio)
        self.assertEqual(loaded.dtype, km.dtype)
        self.assertTrue(torch.allclose(loaded.cluster_centers, km.cluster_centers))
        self.assertTrue(torch.allclose(loaded._counts, km._counts))
        self.assertEqual(loaded._n_since_last_reassign, km._n_since_last_reassign)
        self.assertTrue(torch.equal(loaded.predict(x), pred_before))

    def test_roundtrip_partial_fit(self) -> None:
        torch.manual_seed(1)
        rng = torch.Generator(device="cpu")
        rng.manual_seed(7)
        km = MiniBatchKMeans(
            n_clusters=3,
            random_state=rng,
            dtype=torch.float32,
        )
        for chunk in torch.randn(120, 4, dtype=torch.float32).split(40):
            km.partial_fit(chunk)

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "p.pt"
            km.save(path)
            loaded = MiniBatchKMeans.load(path)

        x = torch.randn(30, 4, dtype=torch.float32)
        self.assertTrue(torch.equal(km.predict(x), loaded.predict(x)))

    def test_unfitted_roundtrip_raises_on_use(self) -> None:
        km = MiniBatchKMeans(n_clusters=2, dtype=torch.float32)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "u.pt"
            km.save(path)
            loaded = MiniBatchKMeans.load(path)

        self.assertEqual(loaded.n_clusters, 2)
        with self.assertRaises(ValueError):
            _ = loaded.cluster_centers
        with self.assertRaises(ValueError):
            loaded.predict(torch.randn(10, 3, dtype=torch.float32))

    def test_unknown_format_version(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "bad.pt"
            torch.save({"format_version": 999}, path)
            with self.assertRaises(ValueError) as ctx:
                MiniBatchKMeans.load(path)
            self.assertIn("format_version", str(ctx.exception).lower())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_load_to_cuda(self) -> None:
        torch.manual_seed(2)
        x = torch.randn(50, 3, dtype=torch.float32)
        km = MiniBatchKMeans(n_clusters=3, dtype=torch.float32)
        km.fit(x, batch_size=16, max_iter=1, max_no_improvement=None)

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "c.pt"
            km.save(path)
            loaded = MiniBatchKMeans.load(path, device=torch.device("cuda:0"))

        self.assertEqual(loaded.device.type, "cuda")
        self.assertEqual(loaded.cluster_centers.device.type, "cuda")
        self.assertEqual(loaded._counts.device.type, "cuda")
        x_cuda = x.to("cuda:0")
        pred = loaded.predict(x_cuda)
        self.assertEqual(pred.device.type, "cuda")


if __name__ == "__main__":
    unittest.main()
