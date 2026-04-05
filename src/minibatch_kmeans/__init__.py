"""Mini-batch k-means clustering for PyTorch.

This package implements **mini-batch k-means** following [Sculley (WWW 2010)](https://doi.org/10.1145/1772690.1772862) (*Web-Scale K-Means Clustering*). [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MiniBatchKMeans.html)'s `MiniBatchKMeans` is based on the same algorithm but is **CPU-only**; this library is for PyTorch pipelines that want native tensors and optional CUDA acceleration.

### Quick start

```python
import torch
from minibatch_kmeans import MiniBatchKMeans

X = torch.randn(500, 4, dtype=torch.float32)
km = MiniBatchKMeans(n_clusters=8, dtype=torch.float32)
km.fit(X)
labels = km.predict(X)
```

### Incremental training

Use **partial_fit** for streaming or online updates:

```python
km = MiniBatchKMeans(n_clusters=8, dtype=torch.float32)
for batch in X.split(100):
    km.partial_fit(batch)
labels = km.predict(X)
```

### Save and load

Persist a fitted model (hyperparameters, CPU copies of learned tensors, online-learning state, and the CPU ``random_state``) with **`MiniBatchKMeans.save`**. Load with **`MiniBatchKMeans.load`**. By default tensors load on CPU; pass ``device=...`` to place the model on a GPU. Only load checkpoints you trust (same caveat as PyTorch’s ``torch.load`` without ``weights_only``).

```python
km.fit(X)
km.save("model.pt")

km2 = MiniBatchKMeans.load("model.pt")
labels = km2.predict(X)

# Optional: load weights onto CUDA
if torch.cuda.is_available():
    km_cuda = MiniBatchKMeans.load("model.pt", device=torch.device("cuda"))
```
"""

from importlib import metadata

from ._core import MiniBatchKMeans

__all__ = ["MiniBatchKMeans"]

try:
    __version__ = metadata.version("minibatch-kmeans")
except metadata.PackageNotFoundError:
    __version__ = "0.1.0"
