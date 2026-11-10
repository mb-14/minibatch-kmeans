"""load() places tensors on CPU by default, regardless of the device the model was saved from."""

import tempfile
import unittest
from pathlib import Path

import torch

from minibatch_kmeans import MiniBatchKMeans


class TestLoadDefaultsToCpu(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_cuda_checkpoint_loads_on_cpu_by_default(self) -> None:
        x = torch.randn(50, 3)
        km = MiniBatchKMeans(n_clusters=3, device=torch.device("cuda:0"))
        km.fit(x, batch_size=16, max_iter=1, max_no_improvement=None)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "m.pt"
            km.save(path)
            loaded = MiniBatchKMeans.load(path)
        self.assertEqual(loaded.device.type, "cpu")
        self.assertEqual(loaded.cluster_centers.device.type, "cpu")
        self.assertEqual(loaded._counts.device.type, "cpu")
        self.assertEqual(loaded.predict(x).shape, (50,))


if __name__ == "__main__":
    unittest.main()
