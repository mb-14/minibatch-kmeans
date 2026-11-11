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


class TestSampler(unittest.TestCase):
    def _batches(self, n, bs, max_iter, sampling):
        s = core._BatchIndexSampler(n, bs, max_iter, sampling, torch.Generator().manual_seed(1))
        batches = list(s)
        self.assertEqual(len(batches), s.n_steps)
        return batches

    def test_permutation_pass_uses_every_row_once(self):
        batches = self._batches(10, 4, 1, "permutation")
        self.assertEqual([len(b) for b in batches], [4, 4, 2])
        self.assertEqual(sorted(torch.cat(batches).tolist()), list(range(10)))

    def test_permutation_fractional_pass(self):
        batches = self._batches(10, 4, 1.5, "permutation")
        self.assertEqual([len(b) for b in batches], [4, 4, 2, 4, 1])
        tail = torch.cat(batches[3:])
        self.assertEqual(len(set(tail.tolist())), 5)  # partial pass: distinct rows

    def test_permutation_passes_are_reshuffled(self):
        batches = self._batches(1000, 1000, 2, "permutation")
        self.assertFalse(torch.equal(batches[0], batches[1]))

    def test_replacement_step_count(self):
        self.assertEqual(len(self._batches(100, 10, 1.5, "replacement")), 15)
        self.assertEqual(len(self._batches(100, 10, 0.01, "replacement")), 1)

    def test_unknown_mode_raises(self):
        with self.assertRaises(ValueError):
            core._BatchIndexSampler(10, 4, 1, "bogus", torch.Generator())


class TestFitOptions(unittest.TestCase):
    def test_init_centroids_full_batch_step_is_lloyd(self):
        X = _blobs()
        init = X[:4].clone()
        km = _model()
        km.fit(
            X,
            batch_size=len(X),
            max_iter=1,
            max_no_improvement=None,
            init_centroids=init,
            sampling="permutation",
        )
        # One full-batch step from zero counts gives each center the mean of its rows.
        d = torch.cdist(X, init)
        labels = d.argmin(1)
        want = torch.stack([X[labels == j].mean(0) for j in range(4)])
        torch.testing.assert_close(km.cluster_centers, want)

    def test_init_centroids_is_copied(self):
        X = _blobs()
        init = X[:4].clone()
        before = init.clone()
        _model().fit(X, batch_size=32, max_iter=1, init_centroids=init)
        torch.testing.assert_close(init, before)

    def test_init_centroids_bad_shape_raises(self):
        X = _blobs()
        with self.assertRaises(ValueError):
            _model().fit(X, init_centroids=X[:3])
        with self.assertRaises(ValueError):
            _model().fit(X, init_centroids=X[:4, :2])

    def test_init_centroids_bad_dtype_raises(self):
        X = _blobs()
        with self.assertRaises(ValueError):
            _model().fit(X, init_centroids=X[:4].double())

    def test_fractional_max_iter(self):
        X = _blobs()
        steps = []
        km = _model()
        orig = km._mini_batch_step

        def counting(Xb, **kw):
            steps.append(Xb.shape[0])
            return orig(Xb, **kw)

        km._mini_batch_step = counting
        km.fit(X, batch_size=100, max_iter=1.5, max_no_improvement=None)
        self.assertEqual(len(steps), 6)
        steps.clear()
        km.fit(X, batch_size=100, max_iter=1.5, max_no_improvement=None, sampling="permutation")
        self.assertEqual(steps, [100, 100, 100, 100, 100, 100])

    def test_max_iter_must_be_positive(self):
        with self.assertRaises(ValueError):
            _model().fit(_blobs(), max_iter=0)

    def test_bad_sampling_raises(self):
        with self.assertRaises(ValueError):
            _model().fit(_blobs(), sampling="bogus")

    def test_permutation_is_reproducible(self):
        X = _blobs()
        a = _model(seed=5).fit(X, batch_size=64, max_iter=2, sampling="permutation").clone()
        b = _model(seed=5).fit(X, batch_size=64, max_iter=2, sampling="permutation").clone()
        torch.testing.assert_close(a, b)

    def test_permutation_finds_blobs(self):
        X = _blobs()
        km = _model(rr=0.01)
        # Start from one row of each blob: a random start can merge blobs in
        # either sampling mode, which is not what this test checks.
        km.fit(
            X,
            batch_size=64,
            max_iter=5,
            sampling="permutation",
            max_no_improvement=None,
            init_centroids=X[:4].clone(),
        )
        inertia = torch.cdist(X, km.cluster_centers).min(1).values.square().mean()
        self.assertLess(inertia.item(), 1.0)


@unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
class TestFitOptionsCuda(unittest.TestCase):
    def test_indices_drawn_on_gpu(self):
        X = _blobs().cuda()
        seen = []
        orig = core._BatchIndexSampler.__iter__

        def spy(self):
            for idx in orig(self):
                seen.append(idx.device.type)
                yield idx

        core._BatchIndexSampler.__iter__ = spy
        try:
            _model(device="cuda").fit(X, batch_size=64, max_iter=1, sampling="permutation")
        finally:
            core._BatchIndexSampler.__iter__ = orig
        self.assertTrue(seen)
        self.assertEqual(set(seen), {"cuda"})

    def test_permutation_cpu_and_gpu_data_match(self):
        X = _blobs(n=4000)
        kw = {
            "batch_size": 300,
            "max_iter": 2.5,
            "sampling": "permutation",
            "max_no_improvement": None,
        }
        a = _model(device="cuda", rr=0.01).fit(X, **kw)  # staged through the prefetcher
        b = _model(device="cuda", rr=0.01).fit(X.cuda(), **kw)
        torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-4)

    def test_init_centroids_on_gpu(self):
        X = _blobs().cuda()
        init = X[:4].clone()
        km = _model(device="cuda")
        km.fit(
            X,
            batch_size=len(X),
            max_iter=1,
            max_no_improvement=None,
            init_centroids=init,
            sampling="permutation",
        )
        labels = torch.cdist(X, init).argmin(1)
        want = torch.stack([X[labels == j].mean(0) for j in range(4)])
        torch.testing.assert_close(km.cluster_centers, want)


if __name__ == "__main__":
    unittest.main()
