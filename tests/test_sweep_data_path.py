"""sweep_reassignment_ratio must fail on a wrong --data-path instead of producing empty results."""

from __future__ import annotations

import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from evaluation.datasets import missing_clustbench_data
from scripts import sweep_reassignment_ratio as sweep


class _ExitTestCase(unittest.TestCase):
    def exit_code_and_stderr(self, fn, *args, **kwargs) -> tuple[int | str | None, str]:
        """Call ``fn`` with clustbench importable; return its exit code and stderr."""
        err = io.StringIO()
        with (
            mock.patch.dict(sys.modules, {"clustbench": mock.MagicMock()}),
            contextlib.redirect_stderr(err),
            contextlib.redirect_stdout(io.StringIO()),
            self.assertRaises(SystemExit) as cm,
        ):
            fn(*args, **kwargs)
        return cm.exception.code, err.getvalue()


class TestMissingClustbenchData(unittest.TestCase):
    def test_detects_missing_and_present(self) -> None:
        tasks = [("sipu", "a1"), ("mnist", "digits")]
        with tempfile.TemporaryDirectory() as d:
            self.assertTrue(missing_clustbench_data(d, tasks))
            (Path(d) / "mnist").mkdir()
            (Path(d) / "mnist" / "digits.data.gz").touch()
            self.assertFalse(missing_clustbench_data(d, tasks))


class TestSweepWrongDataPath(_ExitTestCase):
    def test_mode_both_fails_before_sim_sweep(self) -> None:
        """A wrong path must fail up front, not after hours of sim sweep."""
        with mock.patch.object(sweep, "run_sim_sweep") as sim:
            code, err = self.exit_code_and_stderr(
                sweep.main, ["--mode", "both", "--quick", "--data-path", "/wrong"]
            )
        self.assertEqual(code, 1)
        self.assertIn("no clustbench datasets could be loaded from --data-path '/wrong'", err)
        sim.assert_not_called()

    def test_real_sweep_fails_when_nothing_loads(self) -> None:
        with mock.patch.object(sweep, "load_clustbench_dataset_or_skip", return_value=None):
            code, _ = self.exit_code_and_stderr(
                sweep.run_real_sweep,
                tasks=[("sipu", "a1")],
                rho_values=[0.0],
                seeds=[0],
                device=torch.device("cpu"),
                eval_chunk_rows=8192,
                data_path="/corrupt",
                batch_size=1024,
                progress=False,
            )
        self.assertEqual(code, 1)

    def test_print_params_fails_when_nothing_loads(self) -> None:
        with mock.patch.object(sweep, "load_clustbench_dataset_or_skip", return_value=None):
            code, _ = self.exit_code_and_stderr(
                sweep._print_real_heuristics,
                tasks=[("sipu", "a1")],
                data_path="/corrupt",
                batch_size=1024,
            )
        self.assertEqual(code, 1)


if __name__ == "__main__":
    unittest.main()
