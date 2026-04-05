from pathlib import Path
from typing import Optional

import torch

from ._utils import default_cpu_generator, nvtx_range

# Reassignment: threshold is _REASSIGN_K * n_clusters samples since last pass (empty clusters force sooner).
_REASSIGN_K = 10
_MAX_REASSIGN_FRAC = 0.5
_DENOM_EPS = 1e-10
_CHECKPOINT_FORMAT_VERSION = 1


def _dtype_to_str(dt: torch.dtype) -> str:
    return str(dt)


def _dtype_from_str(s: str) -> torch.dtype:
    name = s.split(".")[-1] if "." in s else s
    return getattr(torch, name)


class _ConvergenceState:
    __slots__ = ("ewa", "ewa_min", "no_improvement")

    def __init__(self) -> None:
        self.ewa: float | None = None
        self.ewa_min: float | None = None
        self.no_improvement = 0


def _pairwise_squared_distances(X: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
    x_sq = (X ** 2).sum(dim=1, keepdim=True)
    c_sq = (C ** 2).sum(dim=1)  # (n_clusters,)
    dist_sq = x_sq - 2 * X @ C.T + c_sq
    return dist_sq


class MiniBatchKMeans:
    """
    Mini-batch K-Means clustering (PyTorch).

    Accepts dense ``torch.Tensor`` inputs for :meth:`fit`, :meth:`predict`, and
    :meth:`partial_fit`.

    If ``random_state`` is ``None``, a CPU ``torch.Generator`` is created with a
    random seed (non-reproducible across runs).

    References
    ----------
    David Sculley. Web-Scale K-Means Clustering. In *Proceedings of the 19th
    International World Wide Web Conference (WWW)*, 2010.
    https://doi.org/10.1145/1772690.1772862
    """

    def __init__(
        self,
        n_clusters: int = 8,
        *,
        verbose: int = 0,
        random_state: Optional[torch.Generator] = None,
        reassignment_ratio: float = 0.01,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ):
        """
        Parameters
        ----------

        **n_clusters** (`int`, default ``8``):
            Number of clusters ``k``; the model learns ``k`` centroids.

        **verbose** (`int`, default ``0``):
            If nonzero, ``fit`` prints per-step batch inertia and convergence
            messages.

        **random_state** (`torch.Generator` or ``None``):
            CPU ``torch.Generator`` for sampling minibatches and initialization.
            If ``None``, a new generator is created with a random seed (runs are
            not reproducible).

        **reassignment_ratio** (`float`, default ``0.01``):
            During training, clusters whose weight is below this fraction of the
            largest cluster weight may be re-seeded from random data points; must
            be ``>= 0`` (``0`` disables reassignment logic).

        **device** (`torch.device`, default ``torch.device("cpu")``):
            Device storing ``cluster_centers`` and used for distance work.

        **dtype** (`torch.dtype`, default ``torch.float32``):
            Floating dtype of centroids; all input tensors must use the same
            dtype.
        """
        self.n_clusters = n_clusters
        self.verbose = bool(verbose)
        if random_state is None:
            self.random_state = default_cpu_generator()
        elif random_state.device.type != "cpu":
            raise ValueError("random_state must be a torch.Generator on CPU or None.")
        else:
            self.random_state = random_state
        if reassignment_ratio < 0:
            raise ValueError("reassignment_ratio should be >= 0.")
        self.reassignment_ratio = reassignment_ratio
        self.device = device
        self.dtype = dtype
        self._cluster_centers: Optional[torch.Tensor] = None

    @property
    def cluster_centers(self) -> torch.Tensor:
        """Learned cluster centroids, shape ``(n_clusters, n_features)``.

        Populated by the first call to ``fit`` or ``partial_fit`` that
        initializes centers.

        Returns
        -------

        **torch.Tensor**:
            Mutable view of the stored centers on ``device``.

        Raises
        ------

        **ValueError**:
            If no training step has run yet (centers not initialized).
        """
        if self._cluster_centers is None:
            raise ValueError("Model has not been fitted yet.")
        return self._cluster_centers

    def _labels_inertia(self, X: torch.Tensor, return_inertia: bool = True) -> tuple[torch.Tensor, float | None]:
        """E-step: assign labels and optionally compute inertia.

        Returns
        -------

        **labels** (`torch.Tensor`, shape ``(n_samples,)``, ``int64``):
            Nearest-centroid index per row.

        **inertia** (`float` or ``None``):
            Sum of squared distances to assigned centroids; ``None`` if
            ``return_inertia`` is false.
        """
        with nvtx_range("minibatch_kmeans:_labels_inertia", X.device):
            dist_sq = _pairwise_squared_distances(X, self._cluster_centers)
            labels = dist_sq.argmin(dim=1)

            if not return_inertia:
                return labels, None

            # Reuse E-step distances: inertia = sum_i dist_sq[i, labels[i]]
            inertia = dist_sq.gather(1, labels.unsqueeze(1)).squeeze(1).sum().item()
            return labels, inertia

    def _check_input(self, X: torch.Tensor, predict: bool = False) -> None:
        if X.dim() != 2:
            raise ValueError("Input tensor must be 2D.")
        if X.layout is not torch.strided:
            raise ValueError("Input must be a dense strided torch.Tensor.")
        if not predict and X.shape[0] < self.n_clusters:
            raise ValueError(
                f"n_samples={X.shape[0]} should be >= n_clusters={self.n_clusters}."
            )
        if X.dtype != self.dtype:
            raise ValueError(f"Input tensor dtype {X.dtype} must match model dtype {self.dtype}.")
        if predict and self._cluster_centers is None:
            raise ValueError("Model has not been fitted yet.")
        if predict and X.shape[1] != self._cluster_centers.shape[1]:
            raise ValueError(f"Input tensor dim {X.shape[1]} must match cluster centers dim {self._cluster_centers.shape[1]}.")

    def _init_centroids(self, X: torch.Tensor) -> torch.Tensor:
        """Pick initial centroids from ``X`` (``randperm``); reset ``_counts`` and reassign counter."""
        with nvtx_range("_init_centroids", self.device):
            device = self.device
            rng = self.random_state
            n_samples = X.shape[0]
            perm = torch.randperm(n_samples, device="cpu", generator=rng)
            seeds = perm[: self.n_clusters]
            self._cluster_centers = X[seeds].to(device).clone()
            self._counts = torch.zeros(self.n_clusters, dtype=X.dtype, device=device)
            self._n_since_last_reassign = 0

    def _mini_batch_convergence(
        self,
        step: int,
        n_steps: int,
        n_samples: int,
        batch_size: int,
        centers_new: torch.Tensor,
        batch_inertia: Optional[float],
        tol: float,
        max_no_improvement: Optional[int],
        inertia_convergence: _ConvergenceState,
    ) -> bool:
        with nvtx_range("_mini_batch_convergence", centers_new.device):
            if tol > 0.0:
                centers_squared_diff = ((centers_new - self._cluster_centers) ** 2).sum().item()
            else:
                centers_squared_diff = 0.0

            step = step + 1

            if step == 1:
                if batch_inertia is not None:
                    batch_inertia = batch_inertia / batch_size
                    if self.verbose:
                        print(
                            f"Minibatch step {step}/{n_steps}: mean batch inertia: {batch_inertia}"
                        )
                return False

            if batch_inertia is not None:
                batch_inertia = batch_inertia / batch_size
                if inertia_convergence.ewa is None:
                    inertia_convergence.ewa = batch_inertia
                else:
                    alpha = batch_size * 2.0 / (n_samples + 1)
                    alpha = min(alpha, 1.0)
                    inertia_convergence.ewa = (
                        inertia_convergence.ewa * (1 - alpha) + batch_inertia * alpha
                    )

                if self.verbose:
                    print(
                        f"Minibatch step {step}/{n_steps}: mean batch inertia: "
                        f"{batch_inertia}, ewa inertia: {inertia_convergence.ewa}"
                    )

            if tol > 0.0 and centers_squared_diff <= tol:
                if self.verbose:
                    print(f"Converged (small centers change) at step {step}/{n_steps}")
                return True

            if batch_inertia is not None:
                if (
                    inertia_convergence.ewa_min is None
                    or inertia_convergence.ewa < inertia_convergence.ewa_min
                ):
                    inertia_convergence.no_improvement = 0
                    inertia_convergence.ewa_min = inertia_convergence.ewa
                else:
                    inertia_convergence.no_improvement += 1

                if (
                    max_no_improvement is not None
                    and inertia_convergence.no_improvement >= max_no_improvement
                ):
                    if self.verbose:
                        print(
                            f"Converged (lack of improvement in inertia) at step {step}/{n_steps}"
                        )
                    return True

            return False

    def _random_reassign(self, batch_size: int) -> bool:
        self._n_since_last_reassign += batch_size
        reassign_thresh = _REASSIGN_K * self.n_clusters
        if (self._counts == 0).any().item() or self._n_since_last_reassign >= reassign_thresh:
            self._n_since_last_reassign = 0
            return True
        return False

    def _maybe_reassign_centers(
        self, X: torch.Tensor, centers: torch.Tensor
    ) -> torch.Tensor:
        """Replace under-used cluster centers with random points from this minibatch.

        When `_random_reassign` allows it (empty clusters or periodic refresh) and
        ``reassignment_ratio > 0``, clusters with weight below
        ``reassignment_ratio * max(weight)`` are candidates. Those center rows are
        replaced with random rows from ``X`` in a **new** tensor; the input
        ``centers`` tensor is not modified. ``self._counts`` is adjusted for
        reassigned clusters so the next updates stay balanced.

        Parameters
        ----------
        X : torch.Tensor
            Minibatch, shape ``(batch_size, n_features)``.
        centers : torch.Tensor
            Proposed centers after `_minibatch_update` (read-only here).

        Returns
        -------
        torch.Tensor
            Centers after any reassignment; same object as input if none.
        """
        random_reassign = self._random_reassign(X.shape[0])
        if not random_reassign or self.reassignment_ratio <= 0:
            return centers

        to_reassign = self._counts < self.reassignment_ratio * self._counts.max()
        n_to_reassign = to_reassign.sum().item()

        batch_size = X.shape[0]
        cap = int(_MAX_REASSIGN_FRAC * batch_size)
        if n_to_reassign > _MAX_REASSIGN_FRAC * batch_size:
            # Do not replace more than half the centers in one step; keep the
            # marked clusters with the largest weights and only reassign the rest.
            masked_indices = torch.where(to_reassign)[0]
            masked_weights = self._counts[to_reassign]
            n_keep = max(0, n_to_reassign - cap)
            _, top_in_masked = torch.topk(masked_weights, min(n_keep, len(masked_indices)))
            indices_dont_reassign = masked_indices[top_in_masked]
            to_reassign[indices_dont_reassign] = False
            n_to_reassign = to_reassign.sum().item()

        if n_to_reassign == 0:
            return centers

        # Distinct random minibatch rows as new center locations.
        perm = torch.randperm(
            X.shape[0], device="cpu", generator=self.random_state
        ).to(X.device)
        new_center_indices = perm[:n_to_reassign]
        if self.verbose:
            print(
                f"[MiniBatchKMeans] Reassigning {n_to_reassign} cluster centers."
            )
        out = centers.clone()
        out[to_reassign] = X[new_center_indices]

        # Match reassigned centers to a conservative weight (min among
        # clusters we kept, or 1 if everything was reassigned).
        non_empty = self._counts[~to_reassign]
        if non_empty.numel() > 0:
            self._counts[to_reassign] = non_empty.min()
        else:
            self._counts[to_reassign] = 1.0

        return out

    def _minibatch_update(
        self,
        X: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Incremental center update for MiniBatchKMeans.

        Updates ``self._counts`` in-place. Returns the new centers tensor.
        """
        with nvtx_range("_minibatch_update", X.device):
            _, n_features = X.shape
            n_clusters = self._cluster_centers.shape[0]

            batch_weighted = torch.zeros(
                (n_clusters, n_features), device=X.device, dtype=X.dtype
            )
            batch_weighted.index_add_(0, labels, X)
            batch_weights = torch.bincount(labels, minlength=n_clusters).to(X.dtype)

            mask = batch_weights > 0
            denom = (self._counts + batch_weights).unsqueeze(1).clamp(min=_DENOM_EPS)
            numerator = self._cluster_centers * self._counts.unsqueeze(1) + batch_weighted
            centers_new = torch.where(
                mask.unsqueeze(1), numerator / denom, self._cluster_centers
            )
            self._counts.copy_(torch.where(mask, self._counts + batch_weights, self._counts))

            return centers_new

    def _mini_batch_step(
        self,
        X: torch.Tensor,
        *,
        return_inertia: bool = True,
    ) -> tuple[torch.Tensor, float | None]:
        """One minibatch step. Returns (updated centers, batch inertia); inertia is None if not computed."""
        with nvtx_range("_mini_batch_step", X.device):
            labels, inertia = self._labels_inertia(X, return_inertia=return_inertia)

            updated = self._minibatch_update(X, labels)
            updated = self._maybe_reassign_centers(X, updated)

            return updated, inertia

    def fit(
        self,
        X: torch.Tensor,
        *,
        batch_size: int = 1024,
        max_iter: int = 100,
        tol: float = 0.0,
        max_no_improvement: Optional[int] = 10,
    ) -> torch.Tensor:
        """Train centroids by repeated random minibatches from ``X``.

        Each step draws ``batch_size`` rows with replacement, runs one
        assignment-and-update step, then checks convergence. When enabled,
        tracks batch inertia for logging and/or inertia-based early stopping.

        Parameters
        ----------

        **X** (`torch.Tensor`, shape ``(n_samples, n_features)``):
            Training data; 2-D, same ``dtype`` as the model. Must satisfy
            ``n_samples >= n_clusters``.

        **batch_size** (`int`, default ``1024``):
            Minibatch size (rows sampled per step). Capped at ``n_samples``.

        **max_iter** (`int`, default ``100``):
            Approximate number of full passes over the dataset; the total number
            of minibatch steps is ``(max_iter * n_samples) // batch_size`` (integer
            division).

        **tol** (`float`, default ``0.0``):
            If ``> 0``, stop when the sum of squared changes to all centers is
            at most ``tol`` after a step. ``0`` disables this criterion.

        **max_no_improvement** (`int` or ``None``, default ``10``):
            If not ``None``, stop when the exponentially weighted average of
            batch inertia fails to improve for this many consecutive minibatch steps.
            ``None`` disables inertia-based early stopping (internal inertia
            tracking is then inactive unless ``verbose`` requests batch inertia
            for logging).

        Returns
        -------

        **torch.Tensor** (shape ``(n_clusters, n_features)``):
            Fitted cluster centers (same tensor as ``cluster_centers``).

        Notes
        -----

        If ``device`` is CUDA, ``X`` may remain on CPU; each minibatch is moved to
        the device for computation so large datasets need not fit in GPU memory.
        """
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1.")
        if max_iter < 1:
            raise ValueError("max_iter must be >= 1.")
        self._check_input(X)
        need_batch_inertia = self.verbose or (max_no_improvement is not None)
        device = self.device
        rng = self.random_state
        n_samples = X.shape[0]
        batch_size = min(batch_size, n_samples)
        inertia_convergence = _ConvergenceState()

        self._init_centroids(X)

        n_steps = (max_iter * n_samples) // batch_size

        for i in range(n_steps):
            mb_idx = torch.randint(
                0, n_samples, (batch_size,), device="cpu", generator=rng
            )
            X_batch = X[mb_idx].to(device)

            centers_new, batch_inertia = self._mini_batch_step(
                X_batch,
                return_inertia=need_batch_inertia,
            )

            converged = self._mini_batch_convergence(
                i,
                n_steps,
                n_samples,
                batch_size,
                centers_new,
                batch_inertia,
                tol,
                max_no_improvement,
                inertia_convergence,
            )

            self._cluster_centers = centers_new
            if converged:
                break

        return self._cluster_centers

    def partial_fit(self, X: torch.Tensor) -> torch.Tensor:
        """Apply one mini-batch k-means update using all rows of ``X``.

        Use this for streaming or online learning. The first call initializes
        centers by taking ``n_clusters`` distinct rows from ``X`` (same rule as
        ``fit``), then applies one merge step with the full ``X`` as the batch.

        Parameters
        ----------

        **X** (`torch.Tensor`, shape ``(n_samples, n_features)``):
            One minibatch; ``n_samples`` must be ``>= n_clusters``. Every row
            participates in the update (batch size equals ``n_samples``).

        Returns
        -------

        **torch.Tensor** (shape ``(n_clusters, n_features)``):
            Updated cluster centers (same tensor as ``cluster_centers``).

        Raises
        ------

        **ValueError**:
            If ``X`` is not 2-D, dtypes differ, or the number of columns does not
            match fitted centers.
        """
        self._check_input(X)
        device = self.device

        if self._cluster_centers is None:
            self._init_centroids(X)

        X_batch = X.to(device)
        centers_new, _ = self._mini_batch_step(
            X_batch,
            return_inertia=False,
        )

        self._cluster_centers = centers_new

        return centers_new

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Nearest-centroid assignment (E-step) for each row of ``X``.

        Parameters
        ----------

        **X** (`torch.Tensor`, shape ``(n_samples, n_features)``):
            Observations to label; must match ``dtype``. Feature dimension must
            match ``cluster_centers.shape[1]``. Tensor is moved to ``device`` for
            distances.

        Returns
        -------

        **torch.Tensor** (shape ``(n_samples,)``, dtype ``int64``):
            Integer cluster index in ``0 .. n_clusters - 1`` per row.

        Raises
        ------

        **ValueError**:
            If ``X`` is not 2-D, dtypes differ, or the number of columns does not
            match fitted centers.
        """
        self._check_input(X, predict=True)
        X = X.to(device=self.device)
        return self._labels_inertia(X, return_inertia=False)[0]

    def fit_predict(
        self,
        X: torch.Tensor,
        *,
        batch_size: int = 1024,
        max_iter: int = 100,
        tol: float = 0.0,
        max_no_improvement: Optional[int] = 10,
    ) -> torch.Tensor:
        """Train on ``X``, then return cluster labels for every row of ``X``.

        Equivalent to ``self.fit(X, ...); return self.predict(X)`` with the
        same keyword arguments forwarded to ``fit``.

        Parameters
        ----------

        **X** (`torch.Tensor`, shape ``(n_samples, n_features)``):
            Training data; same constraints as ``fit`` / ``predict``.

        **batch_size** (`int`, default ``1024``):
            Minibatch size for ``fit``.

        **max_iter** (`int`, default ``100``):
            Approximate dataset passes for ``fit``.

        **tol** (`float`, default ``0.0``):
            Center-change tolerance for ``fit``.

        **max_no_improvement** (`int` or ``None``, default ``10``):
            Inertia early-stop threshold for ``fit``; ``None`` disables.

        Returns
        -------

        **torch.Tensor** (shape ``(n_samples,)``, dtype ``int64``):
            Cluster assignment for each training sample.

        Raises
        ------

        **ValueError**:
            If inputs or hyperparameters fail validation in ``fit`` or
            ``predict``.
        """
        self.fit(
            X,
            batch_size=batch_size,
            max_iter=max_iter,
            tol=tol,
            max_no_improvement=max_no_improvement,
        )
        return self.predict(X)

    def save(self, path: str | Path) -> None:
        """Serialize the estimator to ``path`` via :func:`torch.save`.

        Saves hyperparameters, CPU copies of learned tensors (if fitted), and
        the CPU ``random_state``. Checkpoints are intended for **trusted** use
        only (same caveat as :func:`torch.load` with ``weights_only=False``).

        Parameters
        ----------

        **path** (`str` or ``Path``):
            Output file path.
        """
        path = Path(path)
        payload: dict = {
            "format_version": _CHECKPOINT_FORMAT_VERSION,
            "n_clusters": self.n_clusters,
            "verbose": self.verbose,
            "reassignment_ratio": self.reassignment_ratio,
            "dtype_str": _dtype_to_str(self.dtype),
            "device_str": str(self.device),
            "fitted": self._cluster_centers is not None,
            "random_state": self.random_state.get_state(),
        }
        if self._cluster_centers is not None:
            payload["cluster_centers"] = self._cluster_centers.detach().cpu().clone()
            payload["counts"] = self._counts.detach().cpu().clone()
            payload["n_since_last_reassign"] = int(self._n_since_last_reassign)
        torch.save(payload, path)

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        map_location: str | torch.device | None = None,
        device: torch.device | None = None,
    ) -> "MiniBatchKMeans":
        """Load an estimator previously written with :meth:`save`.

        Loads tensors to ``map_location`` (default CPU), then optionally moves
        them to ``device`` and sets :attr:`device` accordingly.

        Parameters
        ----------

        **path** (`str` or ``Path``):
            File produced by :meth:`save`.

        **map_location** (`str`, :class:`torch.device`, optional):
            Forwarded to :func:`torch.load` for tensor placement; default ``"cpu"``.

        **device** (:class:`torch.device`, optional):
            If set, move ``cluster_centers`` and ``counts`` to this device and use
            it as the model device.

        Returns
        -------

        **MiniBatchKMeans**:
            Restored instance.

        Raises
        ------

        **ValueError**:
            If the checkpoint format is unknown or inconsistent.
        """
        path = Path(path)
        payload = torch.load(path, map_location=map_location or "cpu", weights_only=False)
        version = payload.get("format_version")
        if version != _CHECKPOINT_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported checkpoint format_version={version!r}; "
                f"expected {_CHECKPOINT_FORMAT_VERSION}."
            )
        target_device = device if device is not None else torch.device(payload["device_str"])
        dtype = _dtype_from_str(payload["dtype_str"])
        rng = torch.Generator(device="cpu")
        rng.set_state(payload["random_state"])
        inst = cls(
            n_clusters=int(payload["n_clusters"]),
            verbose=bool(payload["verbose"]),
            reassignment_ratio=float(payload["reassignment_ratio"]),
            random_state=rng,
            device=target_device,
            dtype=dtype,
        )
        if payload["fitted"]:
            centers = payload["cluster_centers"].to(device=target_device, dtype=dtype)
            counts = payload["counts"].to(device=target_device, dtype=dtype)
            if centers.shape[0] != inst.n_clusters:
                raise ValueError(
                    f"Checkpoint n_clusters mismatch: expected {inst.n_clusters}, "
                    f"got {centers.shape[0]} centers."
                )
            inst._cluster_centers = centers
            inst._counts = counts
            inst._n_since_last_reassign = int(payload["n_since_last_reassign"])
        return inst
