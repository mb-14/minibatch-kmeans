"""Fused Triton assignment kernel (CUDA, float32)."""

import contextlib
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import torch

from minibatch_kmeans import MiniBatchKMeans
from minibatch_kmeans import _core as core
from minibatch_kmeans import _triton_assign as ta

# TRITON_INTERPRET=1 runs the kernel on the CPU (CI has no GPU): the logic
# tests then use CPU tensors, and tests that need a real GPU skip.
INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"
KERNEL_DEVICE = "cpu" if INTERPRET else "cuda"
CUDA_KERNEL = ta.AVAILABLE and not INTERPRET and torch.cuda.is_available()
CAN_RUN_KERNEL = CUDA_KERNEL or (ta.AVAILABLE and INTERPRET)
NEEDS_CUDA_KERNEL = "needs triton and a GPU (not the interpreter)"


def _why(reason):
    """The reason "triton" gives: ``reason``, unless Triton is not installed."""
    return reason if ta.AVAILABLE else "cannot be imported"


class KernelTestCase(unittest.TestCase):
    """Runs with TF32 off, whatever ran before, and restores the setting.

    The comparisons with the PyTorch path use tolerances for full float32
    products.
    """

    def setUp(self):
        tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        self.addCleanup(setattr, torch.backends.cuda.matmul, "allow_tf32", tf32)


# n, k, d: tails on every axis, k below one tile, d below one 16-wide step,
# many centroid tiles, and wide rows.
SHAPES = [
    (1, 1, 1),
    (7, 3, 5),
    (100, 17, 15),
    (1000, 257, 33),
    (2049, 1000, 100),
    (300, 5000, 64),
    (513, 70, 512),
    (200, 300, 1027),  # wide rows, not a multiple of the feature tile
]


def _check_against_reference(tc, X, C):
    ref = core._pairwise_squared_distances(X, C)
    ref_labels = ref.argmin(1)
    partial, labels = ta.nearest_centroid(X, C)
    tc.assertEqual(labels.dtype, torch.int64)
    tc.assertEqual(labels.shape, (X.shape[0],))
    # A label may differ from the reference only on a near-tie (summation
    # order): the reference distance to the chosen center must be (almost)
    # the reference minimum.
    best = ref.gather(1, ref_labels[:, None]).squeeze(1)
    chosen = ref.gather(1, labels[:, None]).squeeze(1)
    x_sq = X.square().sum(1)
    tol = 1e-5 * (best.abs() + x_sq) + 1e-4
    tc.assertTrue(bool(((chosen - best).abs() <= tol).all()))
    # Both paths expand ||x - c||^2 in float32, whose cancellation error
    # scales with ||x||^2, not with the distance.
    atol = 1e-6 * x_sq.sum().item() + 1e-3
    torch.testing.assert_close((partial + x_sq).sum(), best.sum(), rtol=1e-5, atol=atol)
    return (labels == ref_labels).float().mean().item()


@unittest.skipUnless(CAN_RUN_KERNEL, "needs triton, and CUDA or TRITON_INTERPRET=1")
class TestKernelLogic(KernelTestCase):
    def setUp(self):
        super().setUp()
        if INTERPRET:
            # nearest_centroid is CUDA-only; stand in for its CUDA call.
            patch = mock.patch.object(torch.cuda, "device", return_value=contextlib.nullcontext())
            patch.start()
            self.addCleanup(patch.stop)

    def test_matches_pytorch_path(self):
        g = torch.Generator(device=KERNEL_DEVICE).manual_seed(0)
        for n, k, d in SHAPES:
            X = torch.randn(n, d, device=KERNEL_DEVICE, generator=g)
            C = torch.randn(k, d, device=KERNEL_DEVICE, generator=g)
            with self.subTest(n=n, k=k, d=d):
                _check_against_reference(self, X, C)

    def test_offset_data_is_full_precision(self):
        # Strict float32 (no TF32): data far from the origin still assigns
        # like the PyTorch path.
        g = torch.Generator(device=KERNEL_DEVICE).manual_seed(0)
        C = 300 + 10 * torch.randn(3000, 48, device=KERNEL_DEVICE, generator=g)
        X = C[torch.randint(0, 3000, (700,), device=KERNEL_DEVICE, generator=g)]
        X = X + torch.randn(700, 48, device=KERNEL_DEVICE, generator=g)
        self.assertGreater(_check_against_reference(self, X, C), 0.999)

    def test_exact_ties_take_lowest_index(self):
        g = torch.Generator(device=KERNEL_DEVICE).manual_seed(0)
        # Small integers: every product and sum is exact in float32, so the
        # duplicates tie exactly in any summation order (the interpreter's
        # BLAS-backed dot is not bitwise position-invariant on random floats).
        base = torch.randint(-4, 5, (600, 32), device=KERNEL_DEVICE, generator=g).float()
        C = torch.cat([base, base, base])  # spans several tiles
        X = C[torch.randint(0, len(C), (999,), device=KERNEL_DEVICE, generator=g)]
        _, labels = ta.nearest_centroid(X, C)
        self.assertTrue(bool((labels < len(base)).all()))

    def test_nan_center_wins_like_argmin(self):
        g = torch.Generator(device=KERNEL_DEVICE).manual_seed(0)
        X = torch.randn(300, 16, device=KERNEL_DEVICE, generator=g)
        for k, bad in [(40, 7), (5000, 3000)]:  # one tile; many tiles
            C = torch.randn(k, 16, device=KERNEL_DEVICE, generator=g)
            C[bad] = float("nan")
            partial, labels = ta.nearest_centroid(X, C)
            torch.testing.assert_close(labels, core._pairwise_squared_distances(X, C).argmin(1))
            self.assertTrue(bool(partial.isnan().all()))

    def test_non_contiguous_rows(self):
        g = torch.Generator(device=KERNEL_DEVICE).manual_seed(0)
        X = torch.randn(64, 200, device=KERNEL_DEVICE, generator=g)[:, :40:2]
        C = torch.randn(9, 20, device=KERNEL_DEVICE, generator=g)
        _check_against_reference(self, X, C)


@unittest.skipUnless(CUDA_KERNEL, NEEDS_CUDA_KERNEL)
class TestKernelCuda(KernelTestCase):
    def test_one_compile_for_all_batch_sizes(self):
        # Sizes are not specialized, so streaming batches of varying size
        # (partial_fit, predict) do not trigger hidden recompiles.
        g = torch.Generator(device="cuda").manual_seed(0)
        C = torch.randn(4096, 64, device="cuda", generator=g)

        def variants():
            kernel = ta._assign_kernel
            if hasattr(kernel, "device_caches"):  # Triton >= 3.3
                return sum(len(v[0]) for v in kernel.device_caches.values())
            return sum(len(per_device) for per_device in kernel.cache.values())

        ta.nearest_centroid(torch.randn(8, 64, device="cuda", generator=g), C)
        before = variants()
        for n in (7, 100, 5000, 16001, 16384):
            ta.nearest_centroid(torch.randn(n, 64, device="cuda", generator=g), C)
        self.assertEqual(variants(), before)

    def test_ignores_allow_tf32(self):
        # Always full float32 products, whatever PyTorch's matmul setting is.
        g = torch.Generator(device="cuda").manual_seed(0)
        X = torch.randn(2000, 256, device="cuda", generator=g)
        C = torch.randn(500, 256, device="cuda", generator=g)
        ieee, _ = ta.nearest_centroid(X, C)
        torch.backends.cuda.matmul.allow_tf32 = True
        also_ieee, _ = ta.nearest_centroid(X, C)
        self.assertTrue(torch.equal(ieee, also_ieee))

    def test_labels_inertia_uses_kernel(self):
        g = torch.Generator(device="cuda").manual_seed(0)
        X = torch.randn(500, 16, device="cuda", generator=g)
        km = MiniBatchKMeans(30, device=torch.device("cuda"), assignment_backend="triton")
        km._cluster_centers = torch.randn(30, 16, device="cuda", generator=g)
        with mock.patch.object(ta, "nearest_centroid", wraps=ta.nearest_centroid) as spy:
            labels, inertia = km._labels_inertia(X)
        spy.assert_called_once()
        ref = core._pairwise_squared_distances(X, km._cluster_centers)
        best = ref.min(1).values
        chosen = ref.gather(1, labels[:, None]).squeeze(1)
        self.assertTrue(bool(((chosen - best).abs() <= 1e-5 * best.abs() + 1e-4).all()))
        self.assertAlmostEqual(inertia, best.sum().item(), delta=1e-5 * inertia)

    def test_fit_from_cpu_data_matches_pytorch_path(self):
        # CPU data goes through the pinned-memory prefetcher. Both fits use
        # the same seed, so they start from the same centers.
        g = torch.Generator().manual_seed(0)
        X = torch.randn(1000, 24, generator=g) + 4 * torch.randint(0, 6, (1000, 1), generator=g)

        def fit(backend):
            km = MiniBatchKMeans(
                40,
                reassign_centers=False,
                device=torch.device("cuda"),
                random_state=torch.Generator().manual_seed(0),
                assignment_backend=backend,
            )
            with mock.patch.object(ta, "nearest_centroid", wraps=ta.nearest_centroid) as spy:
                km.fit(X, batch_size=96, max_iter=3, max_no_improvement=None)
            self.assertEqual(spy.called, backend == "triton")
            return km.cluster_centers

        def inertia(centers):
            return torch.cdist(X, centers.cpu()).min(1).values.square().mean().item()

        got = fit("triton")
        want = fit("native")
        # Summation order differs, so a near-tie can flip a label and nudge a
        # center; compare the fit quality.
        self.assertAlmostEqual(inertia(got), inertia(want), delta=1e-3 * inertia(want))


class TestAssignmentBackend(KernelTestCase):
    def _model(self, X, C, backend):
        km = MiniBatchKMeans(len(C), device=X.device, dtype=X.dtype, assignment_backend=backend)
        km._cluster_centers = C
        return km

    def test_bad_value_raises(self):
        for bad in ("cute", "auto", ""):
            with self.subTest(bad), self.assertRaises(ValueError):
                MiniBatchKMeans(4, assignment_backend=bad)
        km = MiniBatchKMeans(4)
        with self.assertRaises(ValueError):
            km.assignment_backend = "Triton"  # also when set later
        self.assertEqual(km.assignment_backend, "native")

    def test_object_pickled_before_the_option_existed(self):
        X = torch.randn(64, 4)
        km = MiniBatchKMeans(4)
        km.fit(X, batch_size=32, max_iter=1)
        del km._assignment_backend  # such an object has no such attribute after unpickling
        self.assertEqual(km.assignment_backend, "native")
        self.assertEqual(km.predict(X).shape, (64,))

    def test_default_is_native(self):
        self.assertEqual(MiniBatchKMeans(4).assignment_backend, "native")

    def test_default_never_imports_the_kernel_module(self):
        script = """
import sys, torch
from minibatch_kmeans import MiniBatchKMeans
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
km = MiniBatchKMeans(64, device=device)
X = torch.randn(1 << 16, 8, device=device)
km.fit(X, batch_size=1 << 16, max_iter=1)
km.predict(X)
assert "minibatch_kmeans._triton_assign" not in sys.modules
"""
        result = subprocess.run(
            [sys.executable, "-c", script], check=False, capture_output=True, text=True, timeout=600
        )
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])

    def test_triton_raises_where_it_cannot_run(self):
        cases = [("not a CUDA device", torch.randn(40, 8), torch.randn(5, 8))]
        if torch.cuda.is_available():
            cases.append(
                (
                    "float32",
                    torch.randn(40, 8, device="cuda", dtype=torch.float64),
                    torch.randn(5, 8, device="cuda", dtype=torch.float64),
                )
            )
        for why, X, C in cases:
            with self.subTest(why), self.assertRaisesRegex(RuntimeError, _why(why)):
                self._model(X, C, "triton")._labels_inertia(X)

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_triton_raises_without_triton(self):
        X, C = torch.randn(40, 8, device="cuda"), torch.randn(5, 8, device="cuda")
        with (
            mock.patch.object(ta, "AVAILABLE", False),
            self.assertRaisesRegex(RuntimeError, "triton cannot be imported"),
        ):
            self._model(X, C, "triton")._labels_inertia(X)

    def test_dtype_message_names_both(self):
        X, C = torch.randn(40, 8), torch.randn(5, 8, dtype=torch.float64)
        self.assertIn("centers are torch.float64", ta.unsupported_reason(X, C))

    def test_triton_fit_and_partial_fit_raise_where_it_cannot_run(self):
        X = torch.randn(64, 4)
        for train in ("fit", "partial_fit"):
            km = MiniBatchKMeans(4, assignment_backend="triton")
            with self.subTest(train), self.assertRaisesRegex(RuntimeError, _why("not a CUDA")):
                getattr(km, train)(X)

    @unittest.skipUnless(CUDA_KERNEL, NEEDS_CUDA_KERNEL)
    def test_triton_predict_on_empty_input(self):
        g = torch.Generator(device="cuda").manual_seed(0)
        km = MiniBatchKMeans(5, device=torch.device("cuda"), assignment_backend="triton")
        km._cluster_centers = torch.randn(5, 8, device="cuda", generator=g)
        labels = km.predict(torch.empty(0, 8, device="cuda"))
        self.assertEqual((labels.shape, labels.dtype), ((0,), torch.int64))

    def test_load_rejects_empty_backend_override(self):
        km = MiniBatchKMeans(4)
        km.fit(torch.randn(64, 4), batch_size=32, max_iter=1)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.pt")
            km.save(path)
            with self.assertRaises(ValueError):
                MiniBatchKMeans.load(path, assignment_backend="")

    def test_backend_is_not_saved_and_load_chooses_it(self):
        X = torch.randn(64, 4)
        km = MiniBatchKMeans(4)
        km.fit(X, batch_size=32, max_iter=1)
        km.assignment_backend = "triton"
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.pt")
            km.save(path)
            self.assertNotIn("assignment_backend", torch.load(path, weights_only=False))
            self.assertEqual(MiniBatchKMeans.load(path).assignment_backend, "native")
            back = MiniBatchKMeans.load(path, assignment_backend="triton")
            self.assertEqual(back.assignment_backend, "triton")
            torch.testing.assert_close(back.cluster_centers, km.cluster_centers)


if __name__ == "__main__":
    unittest.main()
