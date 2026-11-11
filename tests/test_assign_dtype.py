import unittest

import torch

import minibatch_kmeans._core as core
from minibatch_kmeans import MiniBatchKMeans


def _model(k=4, device="cpu", rr=0.0, seed=0):
    return MiniBatchKMeans(
        n_clusters=k,
        reassignment_ratio=rr,
        device=torch.device(device),
        random_state=torch.Generator().manual_seed(seed),
    )


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
            Xr, Cr = X.to(dt).float(), C.to(dt).float()
            want = X.square().sum(1, keepdim=True) + C.square().sum(1) - 2 * Xr @ Cr.T
            torch.testing.assert_close(got, want)

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

    def test_bf16_fit_on_gpu(self):
        X = _blobs().cuda()
        km = _fit(X, torch.bfloat16, device="cuda")
        want = _inertia(X, _fit(X, None, "cuda"))
        self.assertAlmostEqual(_inertia(X, km), want, delta=1e-3 * want)


if __name__ == "__main__":
    unittest.main()
