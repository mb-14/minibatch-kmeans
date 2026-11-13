"""Center reassignment: which clusters are marked, and where their centers go."""

import pickle
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from minibatch_kmeans import MiniBatchKMeans, _core


def _model(
    counts: list[float], centers=None, reassign_centers: bool = True
) -> tuple[MiniBatchKMeans, torch.Tensor]:
    """A fitted model with the given counts whose next step runs a periodic check.

    The default centers are far apart (10 units between neighbors). The batch
    has four rows per center, each one unit away from it.
    """
    k = len(counts)
    km = MiniBatchKMeans(
        k, reassign_centers=reassign_centers, random_state=torch.Generator().manual_seed(0)
    )
    if centers is None:
        centers = [[10.0 * j, 0.0] for j in range(k)]
    km._cluster_centers = torch.tensor(centers)
    km._counts = torch.tensor(counts)
    km._n_since_last_reassign = 10**9
    offsets = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    return km, (km._cluster_centers[:, None, :] + offsets).reshape(4 * k, 2)


def _step(km: MiniBatchKMeans, batch: torch.Tensor) -> torch.Tensor:
    labels, _ = km._labels_inertia(batch, return_inertia=False)
    return km._maybe_reassign_centers(batch, km._cluster_centers, labels)


def _reassigned(km: MiniBatchKMeans, batch: torch.Tensor) -> list[int]:
    """Clusters whose count changed: the marked ones and their donors."""
    before = km._counts.clone()
    _step(km, batch)
    return (km._counts != before).nonzero().flatten().tolist()


def _marked(km: MiniBatchKMeans, batch: torch.Tensor) -> list[int]:
    """Clusters that a periodic check marks."""
    labels, _ = km._labels_inertia(batch, return_inertia=False)
    row_errors = (batch - km._cluster_centers[labels]).square().sum(dim=1)
    return km._under_used(km._cluster_centers, True, row_errors).nonzero().flatten().tolist()


class TestAutoReassignment(unittest.TestCase):
    def test_on_by_default(self) -> None:
        self.assertIs(MiniBatchKMeans(3).reassign_centers, True)

    def test_only_a_bool_is_accepted(self) -> None:
        for bad in ("auto", 0.01, 0, 1, None, torch.tensor(True)):
            with self.assertRaisesRegex(ValueError, "reassign_centers"):
                MiniBatchKMeans(3, reassign_centers=bad)
        km = MiniBatchKMeans(3)
        with self.assertRaisesRegex(ValueError, "reassign_centers"):
            km.reassign_centers = 0  # also when set later
        self.assertIs(km.reassign_centers, True)

    def test_numpy_boolean_is_accepted(self) -> None:
        for value in (np.bool_(False), np.bool_(True), np.any(np.zeros(3))):
            km = MiniBatchKMeans(3, reassign_centers=value)
            self.assertIs(km.reassign_centers, bool(value))

    def test_object_pickled_before_the_switch_existed(self) -> None:
        # Such an object has a reassignment_ratio attribute and no switch; 0 meant off.
        for ratio, expected in ((0.01, True), (0, False), ("auto", True)):
            km = MiniBatchKMeans(3, random_state=torch.Generator().manual_seed(0))
            km.partial_fit(torch.randn(64, 2))
            del km._reassign_centers
            km.reassignment_ratio = ratio
            old = pickle.loads(pickle.dumps(km))
            self.assertIs(old.reassign_centers, expected)
            old.partial_fit(torch.randn(64, 2))
            with tempfile.TemporaryDirectory() as td:
                old.save(Path(td) / "m.pt")
                self.assertIs(MiniBatchKMeans.load(Path(td) / "m.pt").reassign_centers, expected)

    def test_old_reassignment_ratio_argument_warns_and_only_zero_has_an_effect(self) -> None:
        # 0 used to disable reassignment and still does; other values are ignored.
        for old, expected in ((0, False), (0.0, False), (0.01, True), ("auto", True)):
            with self.assertWarnsRegex(FutureWarning, "reassignment_ratio"):
                km = MiniBatchKMeans(3, reassignment_ratio=old)
            self.assertIs(km.reassign_centers, expected)
            self.assertNotIn("reassignment_ratio", vars(km))
        with self.assertWarns(FutureWarning):
            km = MiniBatchKMeans(3, reassign_centers=False, reassignment_ratio=0.01)
        self.assertIs(km.reassign_centers, False)

    def test_one_large_cluster_does_not_raise_the_threshold(self) -> None:
        # The 300s are small next to the largest cluster, but in use.
        counts = [10_000.0, 300.0, 300.0, 300.0]
        self.assertEqual(_marked(*_model(counts)), [])

    def test_cluster_far_below_the_mean_count_is_reassigned(self) -> None:
        counts = [100.0, 120.0, 1.0, 90.0]
        self.assertEqual(_marked(*_model(counts)), [2])

    def test_marked_center_takes_half_of_a_donor_cluster(self) -> None:
        km, batch = _model([100.0, 120.0, 1.0, 90.0])
        before = km._cluster_centers.clone()
        out = _step(km, batch)
        moved = (out != before).any(dim=1).nonzero().flatten().tolist()
        self.assertEqual(len(moved), 2)
        self.assertIn(2, moved)
        donor = moved[0] if moved[1] == 2 else moved[1]
        # Donor and new center sit on opposite sides of the donor's old center,
        # a tenth of the way to one of its rows (the rows are one unit away).
        torch.testing.assert_close((out[2] + out[donor]) / 2, before[donor])
        self.assertAlmostEqual((out[2] - before[donor]).norm().item(), 0.1, places=5)
        expected = [100.0, 120.0, 1.0, 90.0]
        expected[donor] = expected[2] = expected[donor] / 2
        self.assertEqual(km._counts.tolist(), expected)

    def test_one_of_two_centers_sharing_a_spot_is_reassigned(self) -> None:
        # Centers 1 and 2 are 0.1 apart: removing either adds almost no error.
        centers = [[0.0, 0.0], [10.0, 0.0], [10.1, 0.0], [20.0, 0.0], [30.0, 0.0]]
        counts = [100.0, 60.0, 50.0, 100.0, 100.0]
        self.assertEqual(_marked(*_model(counts, centers)), [1])

    def test_redundant_centers_are_found_when_they_are_the_majority(self) -> None:
        # Four of five centers share one spot, so the median removal cost is
        # itself tiny; the comparison with the batch error still marks one.
        centers = [[0.0, 0.0], [0.01, 0.0], [0.03, 0.0], [0.06, 0.0], [100.0, 0.0]]
        self.assertEqual(_marked(*_model([100.0] * 5, centers)), [0])

    def test_small_cluster_far_from_the_others_is_kept(self) -> None:
        centers = [[0.0, 0.0], [10.0, 0.0], [20.0, 0.0], [500.0, 0.0]]
        self.assertEqual(_marked(*_model([100.0, 100.0, 100.0, 5.0], centers)), [])

    def test_decision_does_not_depend_on_rows_seen(self) -> None:
        centers = [[0.0, 0.0], [10.0, 0.0], [10.1, 0.0], [20.0, 0.0], [30.0, 0.0]]
        for scale in (1.0, 1000.0):
            counts = [c * scale for c in (100.0, 60.0, 50.0, 100.0, 1.0)]
            self.assertEqual(_marked(*_model(counts, centers)), [1, 4])

    def test_check_forced_by_an_empty_cluster_uses_counts_only(self) -> None:
        centers = [[0.0, 0.0], [10.0, 0.0], [10.1, 0.0], [20.0, 0.0], [30.0, 0.0]]
        km, batch = _model([100.0, 60.0, 50.0, 100.0, 0.0], centers)
        km._n_since_last_reassign = 0
        changed = _reassigned(km, batch)  # center 4 and its donor
        self.assertEqual(len(changed), 2)
        self.assertIn(4, changed)
        self.assertGreater(km._counts[4].item(), 0.0)

    def test_same_seed_gives_the_same_centers(self) -> None:
        torch.manual_seed(0)
        X = torch.randn(4000, 4) + 6 * torch.randint(0, 3, (4000, 1))
        fits = []
        for _ in range(2):
            km = MiniBatchKMeans(64, random_state=torch.Generator().manual_seed(3))
            km.fit(X, batch_size=256, max_iter=5, tol=0.0, max_no_improvement=None)
            fits.append(km.cluster_centers)
        torch.testing.assert_close(fits[0], fits[1], rtol=0, atol=0)

    def test_removal_costs_do_not_depend_on_the_block_size(self) -> None:
        torch.manual_seed(0)
        centers = torch.randn(300, 5, dtype=torch.float64)
        counts = torch.rand(300, dtype=torch.float64) * 100
        dist = torch.cdist(centers, centers).fill_diagonal_(float("inf"))
        expected = dist.argmin(dim=1)
        for budget in (1 << 30, 300 * 8 * 7, 1):  # all rows, 7 rows, 1 row per block
            with mock.patch.object(_core, "_REMOVAL_COST_BYTES", budget):
                cost, nearest = _core._removal_costs(centers, counts)
            self.assertEqual(nearest.tolist(), expected.tolist(), budget)
            paired = counts * counts[expected] / (counts + counts[expected])
            torch.testing.assert_close(cost, paired * dist.min(dim=1).values ** 2)

    def test_false_disables(self) -> None:
        self.assertEqual(_reassigned(*_model([100.0, 0.0, 100.0], reassign_centers=False)), [])
        self.assertEqual(len(_reassigned(*_model([100.0, 0.0, 100.0]))), 2)

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
