import unittest

import torch

from minibatch_kmeans import MiniBatchKMeans


@unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
class TestPrefetch(unittest.TestCase):
    def _fit(self, X, **kw):
        km = MiniBatchKMeans(
            n_clusters=16,
            device=torch.device("cuda"),
            random_state=torch.Generator().manual_seed(3),
        )
        km.fit(X, batch_size=512, max_iter=5, **kw)
        return km

    def test_cpu_data_matches_gpu_data(self):
        torch.manual_seed(0)
        X = torch.randn(20_000, 32)
        a = self._fit(X, max_no_improvement=None)
        b = self._fit(X.cuda(), max_no_improvement=None)
        torch.testing.assert_close(a.cluster_centers, b.cluster_centers, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(a._counts, b._counts)

    def test_same_seed_is_reproducible(self):
        torch.manual_seed(0)
        X = torch.randn(20_000, 32)
        a = self._fit(X)
        b = self._fit(X)
        torch.testing.assert_close(a.cluster_centers, b.cluster_centers)

    def test_early_stop_shuts_down_thread(self):
        import threading

        torch.manual_seed(0)
        X = torch.randn(20_000, 32)
        self._fit(X, tol=1e9)  # converges on the first check
        names = [t.name for t in threading.enumerate()]
        self.assertNotIn("minibatch-kmeans-prefetch", names)


if __name__ == "__main__":
    unittest.main()
