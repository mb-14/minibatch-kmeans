"""Fused Triton assignment kernel (CUDA, half-precision ``assign_dtype``)."""

import unittest
from unittest import mock

import torch

from minibatch_kmeans import MiniBatchKMeans
from minibatch_kmeans import _core as core
from minibatch_kmeans import _triton_assign as ta

# n, k, d: tails on every axis, k below one tile, d below the 16-wide dot
# minimum, and small-n / large-k shapes that split the centroids.
SHAPES = [
    (1, 1, 1),
    (7, 3, 5),
    (100, 17, 15),
    (1000, 257, 33),
    (2049, 1000, 100),
    (300, 5000, 64),
    (513, 70, 300),
]


def _check_against_reference(tc, X, C, dt):
    ref = core._pairwise_squared_distances(X, C, dt)
    ref_labels = ref.argmin(1)
    partial, labels = ta.nearest_centroid(X, C, dt)
    tc.assertEqual(labels.dtype, torch.int64)
    tc.assertEqual(labels.shape, (X.shape[0],))
    # A label may differ from the reference only on a near-tie: the reference
    # distance to the chosen center must be (almost) the reference minimum.
    best = ref.gather(1, ref_labels[:, None]).squeeze(1)
    chosen = ref.gather(1, labels[:, None]).squeeze(1)
    x_sq = X.square().sum(1)
    tol = 1e-5 * (best.abs() + x_sq) + 1e-4
    tc.assertTrue(bool(((chosen - best).abs() <= tol).all()))
    torch.testing.assert_close((partial + x_sq).sum(), best.sum(), rtol=1e-5, atol=1e-3)


@unittest.skipUnless(torch.cuda.is_available() and ta.AVAILABLE, "needs CUDA and triton")
class TestTritonKernel(unittest.TestCase):
    def test_matches_pytorch_path(self):
        g = torch.Generator(device="cuda").manual_seed(0)
        for n, k, d in SHAPES:
            X = torch.randn(n, d, device="cuda", generator=g)
            C = torch.randn(k, d, device="cuda", generator=g)
            for dt in core._ASSIGN_DTYPES:
                with self.subTest(n=n, k=k, d=d, dtype=dt):
                    _check_against_reference(self, X, C, dt)

    def test_exact_ties_take_lowest_index(self):
        g = torch.Generator(device="cuda").manual_seed(0)
        base = torch.randn(600, 32, device="cuda", generator=g)
        C = torch.cat([base, base, base])  # spans several tiles and splits
        X = C[torch.randint(0, len(C), (999,), device="cuda", generator=g)]
        _, labels = ta.nearest_centroid(X, C, torch.bfloat16)
        self.assertTrue(bool((labels < len(base)).all()))

    def test_non_contiguous_rows(self):
        g = torch.Generator(device="cuda").manual_seed(0)
        X = torch.randn(64, 200, device="cuda", generator=g)[:, :40:2]
        C = torch.randn(9, 20, device="cuda", generator=g)
        _check_against_reference(self, X, C, torch.bfloat16)

    def test_labels_inertia_uses_kernel(self):
        g = torch.Generator(device="cuda").manual_seed(0)
        X = torch.randn(500, 16, device="cuda", generator=g)
        km = MiniBatchKMeans(30, assign_dtype=torch.bfloat16, device=torch.device("cuda"))
        km._cluster_centers = torch.randn(30, 16, device="cuda", generator=g)
        with mock.patch.object(ta, "nearest_centroid", wraps=ta.nearest_centroid) as spy:
            labels, inertia = km._labels_inertia(X)
        spy.assert_called_once()
        ref = core._pairwise_squared_distances(X, km._cluster_centers, torch.bfloat16)
        best = ref.min(1).values
        chosen = ref.gather(1, labels[:, None]).squeeze(1)
        self.assertTrue(bool(((chosen - best).abs() <= 1e-5 * best.abs() + 1e-4).all()))
        self.assertAlmostEqual(inertia, best.sum().item(), delta=1e-4 * inertia)

    def test_fit_from_cpu_data_matches_pytorch_path(self):
        # CPU data goes through the pinned-memory prefetcher; 1000 rows in
        # batches of 96 leave a short last batch in each pass.
        g = torch.Generator().manual_seed(0)
        X = torch.randn(1000, 24, generator=g) + 4 * torch.randint(0, 6, (1000, 1), generator=g)

        def fit():
            km = MiniBatchKMeans(
                40,
                reassignment_ratio=0.0,
                assign_dtype=torch.bfloat16,
                device=torch.device("cuda"),
                random_state=torch.Generator().manual_seed(0),
            )
            km.fit(
                X,
                batch_size=96,
                max_iter=3,
                sampling="permutation",
                max_no_improvement=None,
                init_centroids=X[:40].cuda(),
            )
            return km.cluster_centers

        with mock.patch.object(ta, "nearest_centroid", wraps=ta.nearest_centroid) as spy:
            got = fit()
        self.assertTrue(spy.called)
        with mock.patch.object(ta, "supported", return_value=False):
            want = fit()
        torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-4)


class TestFallback(unittest.TestCase):
    def _uses_kernel(self, X, C, assign_dtype):
        km = MiniBatchKMeans(len(C), assign_dtype=assign_dtype, device=X.device, dtype=X.dtype)
        km._cluster_centers = C
        with mock.patch.object(ta, "nearest_centroid", wraps=ta.nearest_centroid) as spy:
            labels, inertia = km._labels_inertia(X)
        want = core._pairwise_squared_distances(X, C, assign_dtype)
        if not spy.called:
            torch.testing.assert_close(labels, want.argmin(1))
            self.assertAlmostEqual(inertia, want.min(1).values.sum().item(), places=3)
        return spy.called

    def test_cpu_uses_pytorch(self):
        X, C = torch.randn(40, 8), torch.randn(5, 8)
        self.assertFalse(self._uses_kernel(X, C, torch.bfloat16))

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_cuda_without_half_assign_uses_pytorch(self):
        X, C = torch.randn(40, 8, device="cuda"), torch.randn(5, 8, device="cuda")
        self.assertFalse(self._uses_kernel(X, C, None))

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_float64_uses_pytorch(self):
        X = torch.randn(40, 8, device="cuda", dtype=torch.float64)
        C = torch.randn(5, 8, device="cuda", dtype=torch.float64)
        self.assertFalse(self._uses_kernel(X, C, torch.bfloat16))

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_missing_triton_uses_pytorch(self):
        X, C = torch.randn(40, 8, device="cuda"), torch.randn(5, 8, device="cuda")
        with mock.patch.object(ta, "AVAILABLE", False):
            self.assertFalse(self._uses_kernel(X, C, torch.bfloat16))


if __name__ == "__main__":
    unittest.main()
