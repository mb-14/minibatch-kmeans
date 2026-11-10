"""Shared CLI helpers: argparse validators and the clustbench availability check."""

from __future__ import annotations

import argparse
import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from evaluation import run
from evaluation.cli import non_negative_float, non_negative_int, positive_float, positive_int


class TestValidators(unittest.TestCase):
    def test_accepts_in_range(self) -> None:
        self.assertEqual(positive_int("3"), 3)
        self.assertIsInstance(positive_int("3"), int)
        self.assertEqual(non_negative_int("0"), 0)
        self.assertEqual(positive_float("0.5"), 0.5)
        self.assertEqual(non_negative_float("0"), 0.0)

    def test_rejects_out_of_range(self) -> None:
        for parse, raw in [
            (positive_int, "0"),
            (non_negative_int, "-1"),
            (positive_float, "0"),
            (non_negative_float, "-0.1"),
        ]:
            with self.subTest(parse=parse, raw=raw), self.assertRaises(argparse.ArgumentTypeError):
                parse(raw)

    def test_rejects_non_finite(self) -> None:
        for raw in ("inf", "-inf", "nan"):
            with self.subTest(raw=raw), self.assertRaises(argparse.ArgumentTypeError):
                positive_float(raw)

    def test_rejects_garbage(self) -> None:
        with self.assertRaises(argparse.ArgumentTypeError):
            positive_int("1.5")

    def test_argparse_exits_with_usage_error(self) -> None:
        p = argparse.ArgumentParser(prog="tool")
        p.add_argument("--batch-size", type=positive_int)
        err = io.StringIO()
        with contextlib.redirect_stderr(err), self.assertRaises(SystemExit) as cm:
            p.parse_args(["--batch-size", "0"])
        self.assertEqual(cm.exception.code, 2)
        self.assertIn("argument --batch-size: must be a finite number >= 1, got 0", err.getvalue())


class TestRequireClustbench(unittest.TestCase):
    def test_run_fails_fast_without_clustbench(self) -> None:
        """Without the package, evaluation.run must exit 1, not write an empty result."""
        err = io.StringIO()
        with (
            mock.patch.dict(sys.modules, {"clustbench": None}),  # makes `import clustbench` fail
            contextlib.redirect_stderr(err),
            self.assertRaises(SystemExit) as cm,
        ):
            run.main(["--data-path", "unused", "--device", "cpu"])
        self.assertEqual(cm.exception.code, 1)
        self.assertIn("clustering-benchmarks is required", err.getvalue())


class TestRunNoDatasets(unittest.TestCase):
    def _main(self, argv: list[str]) -> tuple[int | str | None, str]:
        err = io.StringIO()
        with (
            mock.patch.dict(sys.modules, {"clustbench": mock.MagicMock()}),
            contextlib.redirect_stderr(err),
            contextlib.redirect_stdout(io.StringIO()),
            self.assertRaises(SystemExit) as cm,
        ):
            run.main(argv)
        return cm.exception.code, err.getvalue()

    def test_fails_when_no_dataset_loads(self) -> None:
        """A wrong --data-path must be an error, not an empty result with exit 0."""
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "out.json"
            with mock.patch.object(
                run, "load_clustbench_dataset_or_skip", return_value=None
            ) as load:
                code, err = self._main(["--data-path", "/wrong", "--device", "cpu", "-o", str(out)])
            self.assertEqual(load.call_count, len(run.BENCHMARK_TASKS))
            self.assertEqual(code, 1)
            self.assertIn("none of the", err)
            self.assertIn("'/wrong'", err)
            self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
