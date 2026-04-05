# Benchmarks & evaluation

Commands below assume the **repository root** as the current working directory (so paths like `./clustering-data-v1` resolve correctly).

## Setup

Install evaluation dependencies:

```bash
pip install -e ".[eval]"
conda install -c conda-forge faiss-gpu
```

**One-time data setup** (clustering benchmarks):

- Run [`scripts/setup_eval.sh`](../scripts/setup_eval.sh): it checks out [clustering-data-v1](https://github.com/gagolews/clustering-data-v1) at tag **v1.1.0** next to the repo.

## Throughput

Synthetic timing / throughput (not the clustbench quality suite):

```bash
python -m evaluation.perf_test --device cuda:0 --n 2000000 --k 512 --d 4096 --niter 1 --method minibatchkmeans --batch_size 16384 -o results.json
```

## Quality evaluation (clustering benchmarks)

Quality runs follow [clustering-benchmarks](https://clustering-benchmarks.gagolewski.com/) ([Gagolewski, SoftwareX 2022](https://clustering-benchmarks.gagolewski.com/)): **external validity** against published reference partitions. Metrics match **scikit-learn** conventions (see `evaluation.metrics`).

### Metrics: ARI and NMI

Both compare predicted cluster labels to a reference partition (higher is better). **ARI** (Adjusted Rand Index) measures agreement on **point pairs** (same vs different cluster), adjusted so random agreement is ~0; it is **label-invariant** and can dip slightly below 0. **NMI** (Normalized Mutual Information) measures **shared information** between partitions, in **[0, 1]**, using the arithmetic-mean entropy normalizer. They emphasize different views (pairwise structure vs information); reporting both is usual.

### Datasets (`BENCHMARK_TASKS`)

The default task list is in [`evaluation/presets.py`](presets.py) (suite v1.1.0). Each entry is a `(battery, dataset)` pair:

| Pair | Brief description |
|------|-------------------|
| **sipu** / `worms_64` | 2D “worms” shape outlines (vectorized coordinates). |
| **sipu** / `birch1` | Synthetic Gaussian clusters (BIRCH-style; one geometry). |
| **sipu** / `birch2` | Same family as `birch1`, different cluster layout / difficulty. |
| **mnist** / `digits` | MNIST handwritten digits (flattened features). |
| **mnist** / `fashion` | Fashion-MNIST clothing images (flattened features). |

### Example: quality CLI

```bash
python -m evaluation.run \
  --data-path ./clustering-data-v1 \
  --device cuda:0 \
  --methods minibatchkmeans fastkmeans faisskmeans sklearnminibatchkmeans sklearnkmeans \
  --reassignment-ratio 0.1 \
  --n_trials 3 \
  -o results_eval.json
```

Key flags: `--batch-size` (default 16384) and `--reassignment-ratio` apply to mini-batch methods;
