"""``--sim-grid-json`` loader for sweep_reassignment_ratio."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.sweep_reassignment_ratio import (
    _load_sim_grid_json,
    _sim_grid_rows_with_niter_and_batch,
)
from scripts import sweep_reassignment_ratio as m


class TestSweepSimGridJson(unittest.TestCase):
    def test_load_valid(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "g.json"
            p.write_text(json.dumps([[100, 10, 8], [64, 8, 2]]), encoding="utf-8")
            g = _load_sim_grid_json(p)
            self.assertEqual(g, [(100, 10, 8), (64, 8, 2)])

    def test_rejects_n_lt_k(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "g.json"
            p.write_text(json.dumps([[5, 10, 2]]), encoding="utf-8")
            with self.assertRaises(ValueError):
                _load_sim_grid_json(p)

    def test_default_grid_has_max_bounds(self) -> None:
        ns = [t[0] for t in m.DEFAULT_SIM_GRID]
        ks = [t[1] for t in m.DEFAULT_SIM_GRID]
        ds = [t[2] for t in m.DEFAULT_SIM_GRID]
        self.assertGreaterEqual(max(ns), 2_000_000)
        self.assertGreaterEqual(max(ks), 1024)
        self.assertGreaterEqual(max(ds), 4096)

    def test_sim_grid_rows_skips_n_lt_k(self) -> None:
        rows = _sim_grid_rows_with_niter_and_batch([(100, 10, 8), (5, 10, 2)], 16384)
        self.assertEqual(len(rows), 1)
        n, k, d, niter, bs = rows[0]
        self.assertEqual((n, k, d), (100, 10, 8))
        self.assertGreater(niter, 0)
        self.assertEqual(bs, 100)


if __name__ == "__main__":
    unittest.main()
