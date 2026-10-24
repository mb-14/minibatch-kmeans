"""evaluation.presets: default benchmark task list."""

from __future__ import annotations

import unittest

from evaluation.presets import BENCHMARK_TASKS


class TestBenchmarkTasks(unittest.TestCase):
    def test_default_tasks(self) -> None:
        self.assertEqual(
            BENCHMARK_TASKS,
            [
                ("sipu", "worms_64"),
                ("sipu", "birch1"),
                ("sipu", "birch2"),
                ("mnist", "digits"),
                ("mnist", "fashion"),
            ],
        )


if __name__ == "__main__":
    unittest.main()
