"""Save/load roundtrip for MiniBatchKMeans."""

from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
import warnings
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
            reassign_centers=False,
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
        self.assertIs(loaded.reassign_centers, False)
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

        self.assertIs(loaded.reassign_centers, True)
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

    def test_checkpoint_stores_the_switch_as_version_2(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path, _ = self._save_fitted(td)
            payload = torch.load(path, weights_only=False)
        self.assertEqual(payload["format_version"], 2)
        self.assertIs(payload["reassign_centers"], True)
        self.assertNotIn("reassignment_ratio", payload)

    def test_checkpoints_with_a_reassignment_ratio_load(self) -> None:
        # Version 1 (released) stored a number; 0 meant off. Unreleased
        # version 2 files stored "auto". None of them has reassign_centers.
        # The number was stored as given, so it may be a tensor or numpy scalar.
        cases = [(1, 0, False), (1, 0.0, False), (1, 0.05, True), (2, "auto", True)]
        cases += [(1, torch.tensor(0.0), False), (1, torch.tensor(0.01), True)]
        for version, ratio, expected in cases:
            with self.subTest(version=version, ratio=ratio), tempfile.TemporaryDirectory() as td:
                path, x = self._save_fitted(td)
                payload = torch.load(path, weights_only=False)
                del payload["reassign_centers"]
                payload.update(format_version=version, reassignment_ratio=ratio)
                torch.save(payload, path)
                with warnings.catch_warnings():
                    warnings.simplefilter("error")  # loading must not warn
                    loaded = MiniBatchKMeans.load(path)
                self.assertIs(loaded.reassign_centers, expected)
                self.assertEqual(loaded.predict(x).shape, (50,))

    def _save_fitted(self, td: str, **legacy_fields: object) -> tuple[Path, torch.Tensor]:
        """Save a small fitted CPU model; optionally add fields older versions wrote."""
        x = torch.randn(50, 3)
        km = MiniBatchKMeans(n_clusters=3, verbose=1)
        with contextlib.redirect_stdout(io.StringIO()):
            km.fit(x, batch_size=16, max_iter=1, max_no_improvement=None)
        path = Path(td) / "m.pt"
        km.save(path)
        if legacy_fields:
            payload = torch.load(path, weights_only=False)
            payload.update(legacy_fields)
            torch.save(payload, path)
        return path, x

    def test_machine_settings_are_not_saved(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path, _ = self._save_fitted(td)
            payload = torch.load(path, weights_only=False)
            loaded = MiniBatchKMeans.load(path)
            noisy = MiniBatchKMeans.load(path, verbose=1)
        self.assertNotIn("verbose", payload)
        self.assertNotIn("device_str", payload)
        self.assertFalse(loaded.verbose)  # the saved model had verbose=1
        self.assertTrue(noisy.verbose)

    def test_load_ignores_legacy_machine_fields(self) -> None:
        # Runs without CUDA: an old checkpoint saved on CUDA with verbose=1.
        with tempfile.TemporaryDirectory() as td:
            path, x = self._save_fitted(td, device_str="cuda:0", verbose=True)
            loaded = MiniBatchKMeans.load(path)
        self.assertFalse(loaded.verbose)
        self.assertEqual(loaded.device.type, "cpu")
        self.assertEqual(loaded.cluster_centers.device.type, "cpu")
        self.assertEqual(loaded._counts.device.type, "cpu")
        self.assertEqual(loaded.predict(x).shape, (50,))

    def test_load_accepts_dict_and_callable_map_location(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path, x = self._save_fitted(td)
            for map_location in ({"cuda:0": "cpu"}, lambda storage, _loc: storage):
                with self.subTest(map_location=type(map_location).__name__):
                    loaded = MiniBatchKMeans.load(path, map_location=map_location)
                    self.assertEqual(loaded.device.type, "cpu")
                    self.assertEqual(loaded.predict(x).shape, (50,))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_load_cuda_checkpoint_on_cpu_by_default(self) -> None:
        x = torch.randn(50, 3)
        km = MiniBatchKMeans(n_clusters=3, device=torch.device("cuda:0"))
        km.fit(x, batch_size=16, max_iter=1, max_no_improvement=None)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "m.pt"
            km.save(path)
            loaded = MiniBatchKMeans.load(path)
        self.assertEqual(loaded.device.type, "cpu")
        self.assertEqual(loaded.cluster_centers.device.type, "cpu")

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
