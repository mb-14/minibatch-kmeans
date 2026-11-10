import threading
import time
import unittest
from unittest import mock

import torch

import minibatch_kmeans._core as core
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

    def _thread_names(self):
        return [t.name for t in threading.enumerate()]

    def test_early_stop_shuts_down_thread_promptly(self):
        torch.manual_seed(0)
        X = torch.randn(20_000, 32)
        steps = []
        orig_next = core._BatchPrefetcher.next
        orig_close = core._BatchPrefetcher.close
        close_times = []

        def counting_next(self):
            steps.append(1)
            return orig_next(self)

        def timed_close(self):
            # Give the worker time to finish its in-flight batch and park
            # waiting for a free slot, which is the case close() must handle.
            time.sleep(0.02)
            t0 = time.perf_counter()
            orig_close(self)
            close_times.append(time.perf_counter() - t0)

        with mock.patch.object(core._BatchPrefetcher, "next", counting_next), \
                mock.patch.object(core._BatchPrefetcher, "close", timed_close):
            # The convergence check skips step 1, so a huge tol stops at step 2.
            self._fit(X, tol=1e9)

        n_steps = (5 * 20_000) // 512
        self.assertEqual(len(steps), 2, f"expected an early stop, ran {len(steps)} of {n_steps} steps")
        self.assertNotIn("minibatch-kmeans-prefetch", self._thread_names())
        # The worker is parked waiting for a free slot; close() must wake it,
        # not wait for a polling timeout.
        self.assertLess(close_times[0], 0.05)

    def test_worker_error_is_raised_in_fit(self):
        torch.manual_seed(0)
        X = torch.randn(20_000, 32)

        def broken_index_select(*args, **kwargs):
            raise RuntimeError("gather failed")

        with mock.patch.object(core.torch, "index_select", broken_index_select):
            with self.assertRaisesRegex(RuntimeError, "gather failed"):
                self._fit(X, max_no_improvement=None)
        self.assertNotIn("minibatch-kmeans-prefetch", self._thread_names())


if __name__ == "__main__":
    unittest.main()
