"""Tests for nvitop process memory helper (no GPU required for index helper)."""

from __future__ import annotations

import unittest

import torch

from evaluation.nvitop_process_memory import NvitopProcessGpuMemory, torch_device_index


class TestTorchDeviceIndex(unittest.TestCase):
    def test_cuda_defaults_to_zero(self) -> None:
        self.assertEqual(torch_device_index(torch.device("cuda")), 0)

    def test_cuda_explicit_index(self) -> None:
        self.assertEqual(torch_device_index(torch.device("cuda:3")), 3)


class TestNvitopProcessGpuMemoryDataclass(unittest.TestCase):
    def test_delta_non_negative(self) -> None:
        s = NvitopProcessGpuMemory(
            baseline_bytes=100, peak_bytes=250, delta_bytes=150
        )
        self.assertEqual(s.peak_bytes - s.baseline_bytes, s.delta_bytes)


if __name__ == "__main__":
    unittest.main()
