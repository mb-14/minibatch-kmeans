"""Synthetic clustering datasets for benchmarking."""

from __future__ import annotations

import math

import torch


def train_val_split(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    train_fraction: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """
    Random row permutation, then first ``floor(train_fraction * n)`` rows for train.

    When ``train_fraction >= 1.0``, returns ``(x, y, None, None)`` — fit and optional
    quality metrics use the full matrix (no holdout).

    Raises
    ------
    ValueError
        If ``train_fraction`` is not in ``(0, 1]``, or if the implied train/val sizes are invalid.
    """
    if train_fraction > 1.0 or train_fraction <= 0.0:
        raise ValueError("train_fraction must be in (0, 1].")
    n = x.shape[0]
    if y.shape[0] != n:
        raise ValueError("x and y must have the same number of rows.")
    if train_fraction >= 1.0:
        return x, y, None, None

    g = torch.Generator()
    g.manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    n_train = int(math.floor(train_fraction * n))
    n_val = n - n_train
    if n_train < 1 or n_val < 1:
        raise ValueError(
            f"train_fraction={train_fraction} with n={n} yields n_train={n_train}, "
            f"n_val={n_val}; need both >= 1."
        )
    p = perm.to(device=x.device)
    x_perm = x[p]
    y_perm = y[p]
    x_train = x_perm[:n_train]
    y_train = y_perm[:n_train]
    x_val = x_perm[n_train:]
    y_val = y_perm[n_train:]
    return x_train, y_train, x_val, y_val


def gaussian_mixture(
    n_samples: int,
    n_clusters: int,
    n_features: int,
    seed: int,
    center_scale: float = 10.0,
    noise_scale: float = 5.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Structured clusters: points drawn as cluster_center + Gaussian noise.

    Returns
    -------
    x : torch.Tensor, shape (n_samples, n_features), float32
    y : torch.Tensor, shape (n_samples,), int64 — cluster index per point
    """
    g = torch.Generator()
    g.manual_seed(seed)
    centers = torch.randn(n_clusters, n_features, generator=g, dtype=torch.float32) * center_scale
    cluster_indices = torch.randint(
        0, n_clusters, (n_samples,), generator=g, dtype=torch.int64
    )
    x = torch.empty((n_samples, n_features), dtype=torch.float32)
    batch = 100_000
    for i in range(0, n_samples, batch):
        end = min(i + batch, n_samples)
        cidx = cluster_indices[i:end]
        noise = torch.randn(end - i, n_features, generator=g, dtype=torch.float32) * noise_scale
        x[i:end] = centers[cidx] + noise
    return x, cluster_indices


def gaussian_mixture_imbalanced(
    n_samples: int,
    n_clusters: int,
    n_features: int,
    seed: int,
    *,
    center_scale: float = 10.0,
    noise_scale: float = 5.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Same generative structure as ``gaussian_mixture``, but cluster assignment is
    sampled from a **skewed** categorical (softmax of ``linspace(0, 2, k)``) so
    high-index clusters have heavier mass than uniform sampling.
    """
    g = torch.Generator()
    g.manual_seed(seed)
    centers = torch.randn(n_clusters, n_features, generator=g, dtype=torch.float32) * center_scale
    logits = torch.linspace(0.0, 2.0, n_clusters, dtype=torch.float32)
    probs = torch.softmax(logits, dim=0)
    cluster_indices = torch.multinomial(
        probs, n_samples, replacement=True, generator=g
    )
    x = torch.empty((n_samples, n_features), dtype=torch.float32)
    batch = 100_000
    for i in range(0, n_samples, batch):
        end = min(i + batch, n_samples)
        cidx = cluster_indices[i:end]
        noise = torch.randn(end - i, n_features, generator=g, dtype=torch.float32) * noise_scale
        x[i:end] = centers[cidx] + noise
    return x, cluster_indices


def to_torch(x: torch.Tensor, device: torch.device) -> torch.Tensor:
    return x.to(device=device, dtype=torch.float32)
