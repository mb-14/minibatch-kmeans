# minibatch-kmeans

This package implements **mini-batch k-means** following [Sculley (WWW 2010)](https://doi.org/10.1145/1772690.1772862) (*Web-Scale K-Means Clustering*). [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MiniBatchKMeans.html)'s `MiniBatchKMeans` is based on the same algorithm but is **CPU-only**; this library is for PyTorch pipelines that want native tensors and **GPU acceleration**.

## Requirements

- Python **3.10+** (see `pyproject.toml`)
- **PyTorch** 2.x (`torch>=2.0`; use [pytorch.org](https://pytorch.org/get-started/locally/) for the wheel that matches your platform / CUDA).

## Installation

```bash
pip install minibatch-kmeans
```

## Quick start

```python
import torch
from minibatch_kmeans import MiniBatchKMeans

X = torch.randn(5000, 4, dtype=torch.float32)
km = MiniBatchKMeans(n_clusters=8, dtype=torch.float32)
km.fit(X, batch_size=256, max_iter=20)
labels = km.predict(X)
```

## Incremental training

```python
km = MiniBatchKMeans(n_clusters=8, dtype=torch.float32)
for batch in X.split(256):
    km.partial_fit(batch)
labels = km.predict(X)
```

Example notebook: [notebooks/partial_fit_demo.ipynb](notebooks/partial_fit_demo.ipynb)

## Documentation

API reference: **[mb-14.github.io/minibatch-kmeans](https://mb-14.github.io/minibatch-kmeans/)**

## Benchmarks & evaluation

See **[evaluation/README.md](evaluation/README.md)** for benchmarks and evaluation.

## Development

```bash
pip install -e ".[dev]"
python -m unittest discover -s tests -v
```

Generate local API docs (HTML in `pdoc_html/`):

```bash
bash scripts/gen_api_docs.sh
```

## Related projects

- [AnswerDotAI/fastkmeans](https://github.com/AnswerDotAI/fastkmeans)
- [svg-project/flash-kmeans](https://github.com/svg-project/flash-kmeans)
- [Faiss wiki](https://github.com/facebookresearch/faiss/wiki)
- [rapidsai/cuml](https://github.com/rapidsai/cuml)

