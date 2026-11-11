import unittest
import unittest.mock

import torch

import minibatch_kmeans._core as core
from minibatch_kmeans import MiniBatchKMeans


def _blobs(n=400, d=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    centers = torch.tensor([[0.0] * d, [10.0] * d, [-10.0] * d, [10.0, -10.0, 0.0][:d]])
    labels = torch.arange(n) % 4
    return centers[labels] + 0.1 * torch.randn(n, d, generator=g)


def _fit(X, assign_dtype, device="cpu"):
    km = MiniBatchKMeans(
        4,
        reassignment_ratio=0.0,
        assign_dtype=assign_dtype,
        device=torch.device(device),
        random_state=torch.Generator().manual_seed(0),
    )
    km.fit(X, batch_size=64, max_iter=5, max_no_improvement=None)
    return km


def _inertia(X, km):
    return torch.cdist(X, km.cluster_centers).min(1).values.square().mean().item()


class TestAssignDtype(unittest.TestCase):
    def test_distances_use_rounded_inputs(self):
        g = torch.Generator().manual_seed(0)
        X, C = torch.randn(50, 8, generator=g), torch.randn(6, 8, generator=g)
        for dt in core._ASSIGN_DTYPES:
            got = core._pairwise_squared_distances(X, C, dt)
            mu = core._assign_shift(C)
            Xr, Cr = (X - mu).to(dt).float(), (C - mu).to(dt).float()
            torch.testing.assert_close(got, torch.cdist(Xr, Cr).square(), rtol=1e-4, atol=1e-4)

    def test_offset_data_assigns_like_full_precision(self):
        # Far from the origin, rounding the raw values would swamp the
        # spacing between centers; the shift keeps the error relative to it.
        g = torch.Generator().manual_seed(0)
        for offset in (300.0, 1e5):
            C = offset + 10 * torch.randn(64, 32, generator=g)
            X = C[torch.randint(0, 64, (2000,), generator=g)] + torch.randn(2000, 32, generator=g)
            want = torch.cdist(X.double(), C.double()).argmin(1)
            for dt in core._ASSIGN_DTYPES:
                with self.subTest(offset=offset, dtype=dt):
                    got = core._pairwise_squared_distances(X, C, dt)
                    self.assertGreaterEqual(got.min().item(), -1e-3 * got.max().item())
                    self.assertGreater((got.argmin(1) == want).float().mean().item(), 0.999)

    def test_integer_data_is_exact(self):
        # uint8-style data and integer centers: the shift stays integral, so
        # bf16 represents every value and the distances are exact.
        g = torch.Generator().manual_seed(0)
        X = torch.randint(0, 256, (300, 64), generator=g).float()
        C = torch.randint(0, 256, (20, 64), generator=g).float()
        got = core._pairwise_squared_distances(X, C, torch.bfloat16)
        torch.testing.assert_close(got, torch.cdist(X.double(), C.double()).square().float())

    def test_nan_center_does_not_poison_others(self):
        g = torch.Generator().manual_seed(0)
        X, C = torch.randn(50, 8, generator=g), torch.randn(6, 8, generator=g)
        C[2] = float("nan")
        got = core._pairwise_squared_distances(X, C, torch.bfloat16)
        self.assertTrue(bool(got[:, 2].isnan().all()))
        self.assertTrue(bool(got[:, [0, 1, 3, 4, 5]].isfinite().all()))

    def test_inertia_is_exact(self):
        g = torch.Generator().manual_seed(0)
        X = 300 + torch.randn(500, 16, generator=g)
        km = MiniBatchKMeans(8, assign_dtype=torch.bfloat16)
        km._cluster_centers = X[:8].clone()
        labels, inertia = km._labels_inertia(X)
        want = (X - km._cluster_centers[labels]).square().sum().item()
        self.assertAlmostEqual(inertia, want, delta=1e-6 * want)

    def test_load_checkpoint_without_assign_dtype(self):
        import os
        import tempfile

        km = MiniBatchKMeans(4)
        km.fit(_blobs(), batch_size=64, max_iter=1)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.pt")
            km.save(path)
            payload = torch.load(path, weights_only=False)
            del payload["assign_dtype_str"]  # as written before assign_dtype existed
            torch.save(payload, path)
            back = MiniBatchKMeans.load(path)
        self.assertIsNone(back.assign_dtype)
        torch.testing.assert_close(back.cluster_centers, km.cluster_centers)

    def test_none_is_full_precision(self):
        g = torch.Generator().manual_seed(0)
        X, C = torch.randn(50, 8, generator=g), torch.randn(6, 8, generator=g)
        torch.testing.assert_close(
            core._pairwise_squared_distances(X, C, None),
            torch.cdist(X, C).square(),
            rtol=1e-4,
            atol=1e-4,
        )

    def test_bad_assign_dtype_raises(self):
        with self.assertRaises(ValueError):
            MiniBatchKMeans(4, assign_dtype=torch.int8)
        with self.assertRaises(ValueError):
            MiniBatchKMeans(4, assign_dtype=torch.float32)

    def test_bf16_fit_keeps_fp32_state(self):
        X = _blobs()
        km = _fit(X, torch.bfloat16)
        self.assertEqual(km.cluster_centers.dtype, torch.float32)
        self.assertEqual(km._counts.dtype, torch.float32)
        # Same seed in full precision reaches the same solution.
        want = _inertia(X, _fit(X, None))
        self.assertAlmostEqual(_inertia(X, km), want, delta=1e-3 * want)

    def test_save_load_keeps_assign_dtype(self):
        import os
        import tempfile

        km = MiniBatchKMeans(4, assign_dtype=torch.bfloat16)
        km.fit(_blobs(), batch_size=64, max_iter=1)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.pt")
            km.save(path)
            back = MiniBatchKMeans.load(path)
        self.assertEqual(back.assign_dtype, torch.bfloat16)
        self.assertIsNone(MiniBatchKMeans(4).assign_dtype)


@unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
class TestAssignDtypeCuda(unittest.TestCase):
    def test_fused_matches_rounded_reference(self):
        g = torch.Generator().manual_seed(0)
        X, C = torch.randn(300, 64, generator=g), torch.randn(40, 64, generator=g)
        for dt in core._ASSIGN_DTYPES:
            got = core._pairwise_squared_distances(X.cuda(), C.cuda(), dt).cpu()
            want = core._pairwise_squared_distances(X, C, dt)
            torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-3)
            self.assertEqual(got.dtype, torch.float32)

    def test_torch_without_addmm_out_dtype(self):
        # Older torch has no out_dtype on addmm: fall back to float32 products.
        g = torch.Generator().manual_seed(0)
        X, C = torch.randn(300, 64, generator=g).cuda(), torch.randn(40, 64, generator=g).cuda()
        want = core._pairwise_squared_distances(X, C, torch.bfloat16)
        real_addmm = torch.addmm

        def old_addmm(*args, **kwargs):
            if "out_dtype" in kwargs:
                raise TypeError("addmm() got an unexpected keyword argument 'out_dtype'")
            return real_addmm(*args, **kwargs)

        with (
            unittest.mock.patch.object(torch, "addmm", old_addmm),
            unittest.mock.patch.dict(core._ADDMM_OUT_DTYPE, {"supported": True}),
        ):
            got = core._pairwise_squared_distances(X, C, torch.bfloat16)
            self.assertFalse(core._ADDMM_OUT_DTYPE["supported"])
        torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-3)

    def test_bf16_fit_on_gpu(self):
        X = _blobs().cuda()
        km = _fit(X, torch.bfloat16, device="cuda")
        want = _inertia(X, _fit(X, None, "cuda"))
        self.assertAlmostEqual(_inertia(X, km), want, delta=1e-3 * want)


if __name__ == "__main__":
    unittest.main()
