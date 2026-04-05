"""Unit tests for WCSS / NMI (no GPU required)."""

import unittest

import torch

from evaluation.metrics import (
    ari_score,
    assign_labels_torch,
    nmi_score,
    wcss_and_assign_labels_torch,
    wcss_torch,
)


class TestWcss(unittest.TestCase):
    def test_wcss_zero(self):
        x = torch.tensor([[0.0, 0.0], [2.0, 0.0]], dtype=torch.float32)
        c = torch.tensor([[0.0, 0.0], [2.0, 0.0]], dtype=torch.float32)
        self.assertEqual(wcss_torch(x, c), 0.0)

    def test_wcss_two_points_one_center(self):
        c2 = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
        x2 = torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
        self.assertAlmostEqual(wcss_torch(x2, c2), 2.0, places=5)

    def test_wcss_chunking_matches(self):
        torch.manual_seed(0)
        x = torch.randn(200, 64, dtype=torch.float32)
        c = torch.randn(16, 64, dtype=torch.float32)
        ref = wcss_torch(x, c, chunk_rows=200)
        for ch in (1, 3, 17, 8192):
            self.assertAlmostEqual(wcss_torch(x, c, chunk_rows=ch), ref, places=5)

    def test_assign_labels_matches_chunking(self):
        torch.manual_seed(1)
        x = torch.randn(100, 32, dtype=torch.float32)
        c = torch.randn(8, 32, dtype=torch.float32)
        ref = assign_labels_torch(x, c, chunk_rows=100)
        for ch in (1, 7, 50):
            self.assertTrue(torch.equal(assign_labels_torch(x, c, chunk_rows=ch), ref))

    def test_wcss_and_assign_matches_separate(self):
        torch.manual_seed(4)
        x = torch.randn(120, 40, dtype=torch.float32)
        c = torch.randn(10, 40, dtype=torch.float32)
        for ch in (1, 13, 120):
            w_sep = wcss_torch(x, c, chunk_rows=ch)
            a_sep = assign_labels_torch(x, c, chunk_rows=ch)
            w_both, a_both = wcss_and_assign_labels_torch(x, c, chunk_rows=ch)
            self.assertAlmostEqual(w_both, w_sep, places=5)
            self.assertTrue(torch.equal(a_both, a_sep))
        if torch.cuda.is_available():
            d = torch.device("cuda:0")
            for ch in (7, 64):
                w_sep = wcss_torch(x, c, chunk_rows=ch, eval_device=d)
                a_sep = assign_labels_torch(x, c, chunk_rows=ch, eval_device=d)
                w_both, a_both = wcss_and_assign_labels_torch(
                    x, c, chunk_rows=ch, eval_device=d
                )
                self.assertAlmostEqual(w_both, w_sep, places=4)
                self.assertTrue(torch.equal(a_both.cpu(), a_sep.cpu()))

    def test_eval_device_cpu_explicit_matches(self):
        torch.manual_seed(2)
        x = torch.randn(50, 16, dtype=torch.float32)
        c = torch.randn(4, 16, dtype=torch.float32)
        ref_w = wcss_torch(x, c, chunk_rows=50)
        ref_a = assign_labels_torch(x, c, chunk_rows=50)
        for ch in (3, 11, 50):
            self.assertAlmostEqual(
                wcss_torch(x, c, chunk_rows=ch, eval_device="cpu"),
                ref_w,
                places=5,
            )
            self.assertTrue(
                torch.equal(
                    assign_labels_torch(x, c, chunk_rows=ch, eval_device="cpu"),
                    ref_a,
                )
            )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_eval_device_cuda_matches_cpu_reference(self):
        torch.manual_seed(3)
        x = torch.randn(80, 24, dtype=torch.float32)
        c = torch.randn(6, 24, dtype=torch.float32)
        ref_w = wcss_torch(x, c, chunk_rows=80)
        ref_a = assign_labels_torch(x, c, chunk_rows=80)
        d = torch.device("cuda:0")
        for ch in (5, 17, 80):
            self.assertAlmostEqual(
                wcss_torch(x, c, chunk_rows=ch, eval_device=d),
                ref_w,
                places=4,
            )
            self.assertTrue(
                torch.equal(
                    assign_labels_torch(x, c, chunk_rows=ch, eval_device=d).cpu(),
                    ref_a,
                )
            )


class TestAri(unittest.TestCase):
    def test_ari_perfect(self):
        y = torch.tensor([0, 0, 1, 1], dtype=torch.int64)
        self.assertAlmostEqual(ari_score(y, y), 1.0, places=7)

    def test_ari_permutation(self):
        a = torch.tensor([0, 0, 1, 1], dtype=torch.int64)
        b = torch.tensor([1, 1, 0, 0], dtype=torch.int64)
        self.assertAlmostEqual(ari_score(a, b), 1.0, places=7)

    def test_ari_random_partition(self):
        a = torch.tensor([0, 0, 0, 0], dtype=torch.int64)
        b = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
        self.assertLess(ari_score(a, b), 1.0)

    def test_ari_matches_sklearn_if_available(self):
        try:
            from sklearn.metrics import adjusted_rand_score
        except ImportError:
            self.skipTest("sklearn not installed")
        torch.manual_seed(0)
        y_true = torch.randint(0, 4, (200,), dtype=torch.int64)
        y_pred = torch.randint(0, 4, (200,), dtype=torch.int64)
        ref = adjusted_rand_score(y_true.numpy(), y_pred.numpy())
        self.assertAlmostEqual(ari_score(y_true, y_pred), ref, places=10)


class TestNmi(unittest.TestCase):
    def test_nmi_perfect(self):
        y = torch.tensor([0, 0, 1, 1], dtype=torch.int64)
        self.assertAlmostEqual(nmi_score(y, y), 1.0, places=7)

    def test_nmi_permutation(self):
        a = torch.tensor([0, 0, 1, 1], dtype=torch.int64)
        b = torch.tensor([1, 1, 0, 0], dtype=torch.int64)
        self.assertAlmostEqual(nmi_score(a, b), 1.0, places=7)

    def test_nmi_incomplete(self):
        a = torch.tensor([0, 0, 0, 0], dtype=torch.int64)
        b = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
        self.assertAlmostEqual(nmi_score(a, b), 0.0, places=7)


if __name__ == "__main__":
    unittest.main()
