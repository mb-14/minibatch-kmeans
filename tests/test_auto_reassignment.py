"""reassignment_ratio="auto": the threshold is a fraction of the mean cluster weight."""

import unittest

import torch

from minibatch_kmeans import MiniBatchKMeans


def _model(counts: list[float], ratio, centers=None) -> tuple[MiniBatchKMeans, torch.Tensor]:
    """A fitted model with the given counts whose next step runs a periodic check.

    The default centers are far apart (10 units between neighbors).
    """
    k = len(counts)
    km = MiniBatchKMeans(k, reassignment_ratio=ratio, random_state=torch.Generator().manual_seed(0))
    if centers is None:
        centers = [[10.0 * j, 0.0] for j in range(k)]
    km._cluster_centers = torch.tensor(centers)
    km._counts = torch.tensor(counts)
    km._n_since_last_reassign = 10**9
    return km, 1000.0 + torch.arange(2.0 * 4 * k).reshape(4 * k, 2)


def _reassigned(km: MiniBatchKMeans, batch: torch.Tensor) -> list[int]:
    before = km._cluster_centers.clone()
    out = km._maybe_reassign_centers(batch, km._cluster_centers)
    return (out != before).any(dim=1).nonzero().flatten().tolist()


class TestAutoReassignment(unittest.TestCase):
    def test_default_is_auto(self) -> None:
        self.assertEqual(MiniBatchKMeans(3).reassignment_ratio, "auto")

    def test_invalid_values_raise(self) -> None:
        for bad in ("mean", "Auto", -0.1):
            with self.assertRaisesRegex(ValueError, "reassignment_ratio"):
                MiniBatchKMeans(3, reassignment_ratio=bad)

    def test_one_large_cluster_does_not_raise_the_threshold(self) -> None:
        # The 300s are small next to the largest cluster, but in use.
        counts = [10_000.0, 300.0, 300.0, 300.0]
        self.assertEqual(_reassigned(*_model(counts, "auto")), [])
        self.assertEqual(_reassigned(*_model(counts, 0.05)), [1, 2, 3])

    def test_cluster_far_below_the_mean_count_is_reassigned(self) -> None:
        counts = [100.0, 120.0, 1.0, 90.0]
        km, batch = _model(counts, "auto")
        self.assertEqual(_reassigned(km, batch), [2])
        self.assertEqual(km._counts.tolist(), [100.0, 120.0, 90.0, 90.0])

    def test_one_of_two_centers_sharing_a_spot_is_reassigned(self) -> None:
        # Centers 1 and 2 are 0.1 apart: removing either adds almost no error.
        centers = [[0.0, 0.0], [10.0, 0.0], [10.1, 0.0], [20.0, 0.0], [30.0, 0.0]]
        counts = [100.0, 60.0, 50.0, 100.0, 100.0]
        self.assertEqual(_reassigned(*_model(counts, "auto", centers)), [1])
        self.assertEqual(_reassigned(*_model(counts, 0.01, centers)), [])

    def test_small_cluster_far_from_the_others_is_kept(self) -> None:
        centers = [[0.0, 0.0], [10.0, 0.0], [20.0, 0.0], [500.0, 0.0]]
        self.assertEqual(_reassigned(*_model([100.0, 100.0, 100.0, 5.0], "auto", centers)), [])

    def test_decision_does_not_depend_on_rows_seen(self) -> None:
        centers = [[0.0, 0.0], [10.0, 0.0], [10.1, 0.0], [20.0, 0.0], [30.0, 0.0]]
        for scale in (1.0, 1000.0):
            counts = [c * scale for c in (100.0, 60.0, 50.0, 100.0, 1.0)]
            self.assertEqual(_reassigned(*_model(counts, "auto", centers)), [1, 4])

    def test_check_forced_by_an_empty_cluster_uses_counts_only(self) -> None:
        centers = [[0.0, 0.0], [10.0, 0.0], [10.1, 0.0], [20.0, 0.0], [30.0, 0.0]]
        km, batch = _model([100.0, 60.0, 50.0, 100.0, 0.0], "auto", centers)
        km._n_since_last_reassign = 0
        self.assertEqual(_reassigned(km, batch), [4])

    def test_zero_disables(self) -> None:
        self.assertEqual(_reassigned(*_model([100.0, 0.0, 100.0], 0)), [])

    def test_partial_fit_revives_a_dead_center(self) -> None:
        torch.manual_seed(0)
        km = MiniBatchKMeans(3, random_state=torch.Generator().manual_seed(1))
        km.partial_fit(torch.randn(64, 2))
        km._cluster_centers[0] = 1e6  # no row will ever be nearest to this center
        km._counts[0] = 0.0
        for _ in range(3):
            km.partial_fit(torch.randn(64, 2))
        self.assertLess(km.cluster_centers[0].abs().max().item(), 10.0)
        self.assertGreater(km._counts[0].item(), 0.0)


if __name__ == "__main__":
    unittest.main()
