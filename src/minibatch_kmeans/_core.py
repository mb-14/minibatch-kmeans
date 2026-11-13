import queue
import threading
import warnings
from pathlib import Path

import torch

from ._utils import default_cpu_generator, nvtx_range, row_sq_norms

# Reassignment: threshold is _REASSIGN_K * n_clusters samples since last pass (empty clusters force sooner).
_REASSIGN_K = 10
_MAX_REASSIGN_FRAC = 0.5
# Reassignment (reassign_centers=True): a cluster is under-used when its count is below
# _AUTO_COUNT_FRAC of the mean count, or when removing it would add little
# error (see _removal_costs): less than _AUTO_COST_FRAC of the median removal
# cost, or less than _AUTO_GAIN_FRAC of the error an average cluster holds.
# The median, not the mean: one far-away cluster has a huge cost and would
# raise a mean threshold above the cost of healthy clusters. The median alone
# fails when most centers are redundant (a stream that has shown few of its
# groups so far), which is what the second reference covers.
_AUTO_COUNT_FRAC = 0.03
_AUTO_COST_FRAC = 0.05
_AUTO_GAIN_FRAC = 0.01
# A reassigned center takes over half of a donor cluster: the donor and the
# new center move this fraction of the way from the donor's mean toward one of
# the donor's rows, in opposite directions.
_AUTO_SPLIT_STEP = 0.1
# Rows per block of the pairwise search in _removal_costs, and the most bytes a
# block may take (the row count shrinks above k = 65536 in float32).
_REMOVAL_COST_CHUNK = 4096
_REMOVAL_COST_BYTES = 1 << 30
_DENOM_EPS = 1e-10
# Version 2 stores reassign_centers. Version 1 stored reassignment_ratio, which
# load() maps to it.
_CHECKPOINT_FORMAT_VERSION = 2
_READABLE_CHECKPOINT_VERSIONS = (1, 2)


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

    def update_ewa(self, mean_inertia: float, batch_size: int, n_samples: int) -> None:
        """Fold ``mean_inertia`` into the exponentially weighted average."""
        if self.ewa is None:
            self.ewa = mean_inertia
        else:
            alpha = min(batch_size * 2.0 / (n_samples + 1), 1.0)
            self.ewa = self.ewa * (1 - alpha) + mean_inertia * alpha

    def record_improvement(self) -> None:
        """Compare the current EWA with the best so far and update the no-improvement count."""
        if self.ewa is None:
            raise RuntimeError("record_improvement() called before update_ewa()")
        if self.ewa_min is None or self.ewa < self.ewa_min:
            self.no_improvement = 0
            self.ewa_min = self.ewa
        else:
            self.no_improvement += 1

    def is_stalled(self, max_no_improvement: int | None) -> bool:
        """True once the EWA has failed to improve ``max_no_improvement`` times in a row."""
        return max_no_improvement is not None and self.no_improvement >= max_no_improvement


class _BatchPrefetcher:
    """Gathers and uploads mini-batches on a background thread.

    ``X`` stays on the CPU. A worker thread draws batch indices, gathers the
    rows into one of two pinned buffers, and copies that buffer to the GPU on
    a side stream. While the GPU trains on batch ``i``, the worker prepares
    batch ``i + 1``.

    Slot life cycle (``slot = i % 2``):

    1. Worker waits until the slot is free, then waits for that slot's
       previous upload, so the pinned buffer is no longer being read.
    2. Worker gathers rows into ``pinned[slot]`` and, on the copy stream,
       waits for the training step that last used ``gpu[slot]``, then
       uploads and records ``copied[slot]``.
    3. :meth:`next` makes the compute stream wait for ``copied[slot]``.
    4. :meth:`release` records ``used[slot]`` on the compute stream and
       hands the slot back to the worker.
    """

    _N_SLOTS = 2

    def __init__(
        self,
        X: torch.Tensor,
        batch_size: int,
        n_steps: int,
        index_rng: torch.Generator,
        device: torch.device,
    ) -> None:
        self._X = X
        self._batch_size = batch_size
        self._n_steps = n_steps
        self._rng = index_rng
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        self._device = device
        n_features = int(X.shape[1])
        shape = (batch_size, n_features)
        self._pinned = [
            torch.empty(shape, dtype=X.dtype, pin_memory=True) for _ in range(self._N_SLOTS)
        ]
        self._gpu = [torch.empty(shape, dtype=X.dtype, device=device) for _ in range(self._N_SLOTS)]
        self._copied = [torch.cuda.Event() for _ in range(self._N_SLOTS)]
        self._used = [torch.cuda.Event() for _ in range(self._N_SLOTS)]
        self._used_recorded = [False] * self._N_SLOTS
        self._free = [threading.Semaphore(1) for _ in range(self._N_SLOTS)]
        self._ready: queue.Queue = queue.Queue()
        self._copy_stream = torch.cuda.Stream(device=device)
        self._stop = threading.Event()
        self._step = 0
        self._thread = threading.Thread(
            target=self._worker, name="minibatch-kmeans-prefetch", daemon=True
        )
        self._thread.start()

    def _acquire(self, slot: int) -> bool:
        """Wait until ``slot`` is free. Returns False if :meth:`close` was called."""
        self._free[slot].acquire()
        return not self._stop.is_set()

    def _worker(self) -> None:
        try:
            torch.cuda.set_device(self._device)
            n_samples = self._X.shape[0]
            for i in range(self._n_steps):
                slot = i % self._N_SLOTS
                if not self._acquire(slot):
                    return
                # The previous upload from this pinned buffer must finish
                # before the buffer is overwritten.
                self._copied[slot].synchronize()
                idx = torch.randint(
                    0, n_samples, (self._batch_size,), device="cpu", generator=self._rng
                )
                torch.index_select(self._X, 0, idx, out=self._pinned[slot])
                with torch.cuda.stream(self._copy_stream):
                    if self._used_recorded[slot]:
                        self._copy_stream.wait_event(self._used[slot])
                    self._gpu[slot].copy_(self._pinned[slot], non_blocking=True)
                    self._copied[slot].record(self._copy_stream)
                self._ready.put(slot)
        except BaseException as exc:  # noqa: BLE001 - re-raised in the training thread
            self._ready.put(exc)

    def next(self) -> torch.Tensor:
        """Return the next batch on the GPU, ready for the current stream."""
        item = self._ready.get()
        if isinstance(item, BaseException):
            raise item
        slot = item
        torch.cuda.current_stream(self._device).wait_event(self._copied[slot])
        return self._gpu[slot]

    def release(self) -> None:
        """Mark the batch from the last :meth:`next` as no longer needed."""
        slot = self._step % self._N_SLOTS
        self._used[slot].record(torch.cuda.current_stream(self._device))
        self._used_recorded[slot] = True
        self._step += 1
        self._free[slot].release()

    def close(self) -> None:
        self._stop.set()
        # Wake the worker if it is waiting for a free slot, so it sees the
        # stop flag at once instead of waiting for a step that never comes.
        for sem in self._free:
            sem.release()
        self._thread.join()
        torch.cuda.current_stream(self._device).wait_stream(self._copy_stream)


def _pairwise_squared_distances(X: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
    """Squared Euclidean distances from each row of ``X`` to each row of ``C``.

    The row norms use one reduction each, and the cross term is a single
    matrix product. That avoids a squared copy of ``X`` and a doubled copy of ``X``.
    """
    x_sq = row_sq_norms(X, keepdim=True)
    c_sq = row_sq_norms(C)
    return torch.addmm(c_sq, X, C.T, beta=1, alpha=-2).add_(x_sq)


_ASSIGNMENT_BACKENDS = ("native", "triton")
_DEFAULT_ASSIGNMENT_BACKEND = "native"


def _removal_costs(
    centers: torch.Tensor, counts: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Error added by removing each center, and the index of its nearest other center.

    Removing center ``j`` hands its rows to the nearest other center ``l``.
    Merging the two clusters adds
    ``counts[j] * counts[l] / (counts[j] + counts[l]) * |c_j - c_l|^2`` to the
    sum of squared distances (Ward's merge cost). It is small for an empty
    cluster and for two centers that sit close together.
    """
    k = centers.shape[0]
    sq_norms = centers.square().sum(dim=1)
    nearest = torch.empty(k, dtype=torch.int64, device=centers.device)
    # Each chunk holds a (rows, k) distance block; fewer rows at large k keep it
    # within the byte budget.
    chunk = max(1, min(_REMOVAL_COST_CHUNK, _REMOVAL_COST_BYTES // (k * centers.element_size())))
    for start in range(0, k, chunk):
        end = min(start + chunk, k)
        d = torch.addmm(sq_norms, centers[start:end], centers.T, beta=1, alpha=-2)
        d.add_(sq_norms[start:end, None])
        d.diagonal(start).fill_(float("inf"))
        nearest[start:end] = d.argmin(dim=1)
        del d  # or the next block is allocated while this one is still held
    # The expansion above loses the distance between two nearly equal centers
    # to rounding, so the distance to the chosen center is recomputed directly.
    dist_sq = (centers - centers[nearest]).square().sum(dim=1)
    paired = counts * counts[nearest] / (counts + counts[nearest]).clamp(min=_DENOM_EPS)
    return paired * dist_sq, nearest


def _sample_without_replacement(
    weights: torch.Tensor, n: int, generator: torch.Generator
) -> torch.Tensor:
    """``n`` distinct indices, drawn with probability proportional to ``weights``.

    At least ``n`` weights must be positive. Uses the Gumbel top-k trick, so the
    random numbers come from the CPU generator and the rest stays on the device.
    """
    u = torch.rand(weights.shape[0], generator=generator).to(weights.device)
    u.clamp_(1e-12, 1 - 1e-7)
    keys = torch.log(weights) - torch.log(-torch.log(u))  # log(0) = -inf: never drawn
    return torch.topk(keys, n).indices


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
        random_state: torch.Generator | None = None,
        reassign_centers: bool = True,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
        assignment_backend: str = _DEFAULT_ASSIGNMENT_BACKEND,
        reassignment_ratio: float | str | None = None,
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
            CPU ``torch.Generator`` for initialization and center reassignment.
            :meth:`fit` also takes one draw from it to seed a separate generator
            that samples the minibatches, so batch sampling stays reproducible
            when batches are prepared on a background thread. If ``None``, a new
            generator is created with a random seed (runs are not reproducible).

        **reassign_centers** (`bool`, default ``True``):
            Move the centers of under-used clusters during training. A cluster
            is under-used when its weight is far below the mean cluster weight,
            or when removing it would add little error because another center
            is close by (two centers sharing one group of points). The tests
            use the centers, their weights and the current minibatch, so they
            need no row count or training budget. The second test compares all
            pairs of centers at each check. An under-used center is moved next
            to a cluster with a large error in the current minibatch, and the
            two share that cluster. ``False`` never moves a center this way.

        **device** (`torch.device`, default ``torch.device("cpu")``):
            Device storing ``cluster_centers`` and used for distance work.

        **dtype** (`torch.dtype`, default ``torch.float32``):
            Floating dtype of centroids; all input tensors must use the same
            dtype. Must be ``torch.float32`` or ``torch.float64``: in half
            precision the running counts overflow and small center updates
            round away, so convert half-precision data with ``X.float()``.

        **assignment_backend** (`str`, default ``"native"``):
            How rows are assigned to centers in ``fit``, ``partial_fit`` and
            ``predict``. ``"native"``: PyTorch, which builds the
            ``(batch, n_clusters)`` distance matrix. ``"triton"``: a fused
            kernel that never stores that matrix. It needs Triton 3.0 or
            newer, and raises ``RuntimeError`` unless the model is on CUDA
            with ``torch.float32``. Results agree up to floating-point
            summation order.

        **reassignment_ratio** (deprecated):
            Ignored. Earlier versions took a number here (scikit-learn's
            threshold) or ``"auto"``. Passing any value warns and has no
            effect, including ``0``, which used to disable reassignment: use
            ``reassign_centers=False`` for that.
        """
        self.n_clusters = n_clusters
        self.verbose = bool(verbose)
        if random_state is None:
            self.random_state = default_cpu_generator()
        elif random_state.device.type != "cpu":
            raise ValueError("random_state must be a torch.Generator on CPU or None.")
        else:
            self.random_state = random_state
        if not isinstance(reassign_centers, bool):
            raise ValueError(f"reassign_centers must be True or False, got {reassign_centers!r}.")
        if reassignment_ratio is not None:
            warnings.warn(
                "reassignment_ratio is ignored and will be removed; use reassign_centers "
                "(False replaces reassignment_ratio=0).",
                FutureWarning,
                stacklevel=2,
            )
        self.reassign_centers = reassign_centers
        if dtype not in (torch.float32, torch.float64):
            raise ValueError(f"dtype must be torch.float32 or torch.float64, got {dtype}.")
        self.device = device
        self.dtype = dtype
        self.assignment_backend = assignment_backend
        self._cluster_centers: torch.Tensor | None = None

    @property
    def assignment_backend(self) -> str:
        """``"native"`` or ``"triton"``; see the constructor."""
        # Objects pickled before this option existed have no such attribute.
        return getattr(self, "_assignment_backend", _DEFAULT_ASSIGNMENT_BACKEND)

    @assignment_backend.setter
    def assignment_backend(self, value: str) -> None:
        if value not in _ASSIGNMENT_BACKENDS:
            raise ValueError(
                f"assignment_backend must be one of {_ASSIGNMENT_BACKENDS}, got {value!r}."
            )
        self._assignment_backend = value

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

    def _labels_inertia(
        self, X: torch.Tensor, return_inertia: bool = True
    ) -> tuple[torch.Tensor, float | None]:
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
            C = self._cluster_centers
            fused = self._fused_assign(X, C)
            if fused is not None:
                partial, labels = fused
                if not return_inertia:
                    return labels, None
                return labels, partial.add_(row_sq_norms(X)).sum().item()

            dist_sq = _pairwise_squared_distances(X, C)
            labels = dist_sq.argmin(dim=1)

            if not return_inertia:
                return labels, None

            # Reuse E-step distances: inertia = sum_i dist_sq[i, labels[i]]
            inertia = dist_sq.gather(1, labels.unsqueeze(1)).squeeze(1).sum().item()
            return labels, inertia

    def _fused_assign(
        self, X: torch.Tensor, C: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """``(partial, labels)`` from the fused kernel under ``"triton"``, else ``None``."""
        if self.assignment_backend != "triton":
            return None
        from . import _triton_assign as fused  # only "triton" imports Triton

        reason = fused.unsupported_reason(X, C)
        if reason is not None:
            raise RuntimeError(f'assignment_backend="triton" cannot run: {reason}.')
        return fused.nearest_centroid(X, C)

    def _check_input(self, X: torch.Tensor, predict: bool = False) -> None:
        """Validate ``X``; once fitted, its column count must match the centers."""
        if X.dim() != 2:
            raise ValueError("Input tensor must be 2D.")
        if X.layout is not torch.strided:
            raise ValueError("Input must be a dense strided torch.Tensor.")
        if not predict and X.shape[0] < self.n_clusters:
            raise ValueError(f"n_samples={X.shape[0]} should be >= n_clusters={self.n_clusters}.")
        if X.dtype != self.dtype:
            raise ValueError(f"Input tensor dtype {X.dtype} must match model dtype {self.dtype}.")
        if predict and self._cluster_centers is None:
            raise ValueError("Model has not been fitted yet.")
        if self._cluster_centers is not None and X.shape[1] != self._cluster_centers.shape[1]:
            raise ValueError(
                f"Input tensor dim {X.shape[1]} must match cluster centers dim {self._cluster_centers.shape[1]}."
            )

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
        batch_inertia: float | None,
        tol: float,
        max_no_improvement: int | None,
        inertia_convergence: _ConvergenceState,
    ) -> bool:
        with nvtx_range("_mini_batch_convergence", centers_new.device):
            if tol > 0.0:
                centers_squared_diff = ((centers_new - self._cluster_centers) ** 2).sum().item()
            else:
                centers_squared_diff = 0.0

            step = step + 1
            mean_inertia = None if batch_inertia is None else batch_inertia / batch_size

            if step == 1:
                if mean_inertia is not None and self.verbose:
                    print(f"Minibatch step {step}/{n_steps}: mean batch inertia: {mean_inertia}")
                return False

            if mean_inertia is not None:
                inertia_convergence.update_ewa(mean_inertia, batch_size, n_samples)
                if self.verbose:
                    print(
                        f"Minibatch step {step}/{n_steps}: mean batch inertia: "
                        f"{mean_inertia}, ewa inertia: {inertia_convergence.ewa}"
                    )

            if tol > 0.0 and centers_squared_diff <= tol:
                if self.verbose:
                    print(f"Converged (small centers change) at step {step}/{n_steps}")
                return True

            if mean_inertia is None:
                return False
            inertia_convergence.record_improvement()
            if inertia_convergence.is_stalled(max_no_improvement):
                if self.verbose:
                    print(f"Converged (lack of improvement in inertia) at step {step}/{n_steps}")
                return True

            return False

    def _under_used(
        self, centers: torch.Tensor, periodic: bool, row_errors: torch.Tensor
    ) -> torch.Tensor:
        """Mask of the clusters marked for reassignment.

        ``row_errors`` is the squared distance of each minibatch row to its center.
        """
        counts = self._counts
        under_used = counts < _AUTO_COUNT_FRAC * counts.mean()
        if not periodic:
            # A check forced by an empty cluster skips the pairwise search.
            return under_used
        cost, nearest = _removal_costs(centers, counts)
        # Mean error per row times mean rows per cluster: the error an average
        # cluster holds, on the same scale as the removal costs.
        mean_cluster_error = row_errors.mean() * counts.mean()
        cheap = cost < torch.maximum(
            _AUTO_COST_FRAC * cost.median(), _AUTO_GAIN_FRAC * mean_cluster_error
        )
        # Two centers that are each other's nearest have the same cost. Removing
        # one makes the other worth keeping, so the cheaper of a marked pair goes
        # first (lower index on a tie) and the other stays.
        index = torch.arange(len(cost), device=cost.device)
        nearest_goes_first = (cost[nearest] < cost) | ((cost[nearest] == cost) & (nearest < index))
        return under_used | (cheap & ~(cheap[nearest] & nearest_goes_first))

    def _maybe_reassign_centers(
        self, X: torch.Tensor, centers: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Move the centers of under-used clusters.

        A check runs when some cluster is empty, or every
        ``_REASSIGN_K * n_clusters`` rows (periodic). If ``reassign_centers``
        is set, the clusters that `_under_used` marks are candidates, at most
        ``_MAX_REASSIGN_FRAC * batch_size`` of them, and each one takes over
        half of a donor cluster (see `_split_donors`). The input ``centers``
        tensor is not modified, and ``self._counts`` is adjusted for the
        clusters that change.

        Parameters
        ----------
        X : torch.Tensor
            Minibatch, shape ``(batch_size, n_features)``.
        centers : torch.Tensor
            Proposed centers after `_minibatch_update` (read-only here).
        labels : torch.Tensor
            Cluster of each row of ``X`` under ``self._cluster_centers``.

        Returns
        -------
        torch.Tensor
            Centers after any reassignment; same object as input if none.
        """
        self._n_since_last_reassign += X.shape[0]
        periodic = self._n_since_last_reassign >= _REASSIGN_K * self.n_clusters
        if not periodic and not (self._counts == 0).any().item():
            return centers
        self._n_since_last_reassign = 0
        if not self.reassign_centers:
            return centers

        row_errors = (X - self._cluster_centers[labels]).square().sum(dim=1)
        to_reassign = self._under_used(centers, periodic, row_errors)
        # Donors: clusters that stay and have error in this minibatch.
        donor_weights = torch.zeros_like(self._counts).index_add_(0, labels, row_errors)
        donor_weights[to_reassign] = 0
        cap = min(int(_MAX_REASSIGN_FRAC * X.shape[0]), int((donor_weights > 0).sum().item()))

        n_to_reassign = int(to_reassign.sum().item())
        if n_to_reassign > cap:
            # Over the cap: the marked clusters with the largest weights stay.
            marked = torch.where(to_reassign)[0]
            keep = torch.topk(self._counts[marked], n_to_reassign - cap).indices
            to_reassign[marked[keep]] = False
            n_to_reassign = cap
        if n_to_reassign == 0:
            return centers
        if self.verbose:
            print(f"[MiniBatchKMeans] Reassigning {n_to_reassign} cluster centers.")

        out = centers.clone()
        donors = _sample_without_replacement(donor_weights, n_to_reassign, self.random_state)
        self._split_donors(X, labels, out, torch.where(to_reassign)[0], donors)
        return out

    def _split_donors(
        self,
        X: torch.Tensor,
        labels: torch.Tensor,
        centers: torch.Tensor,
        targets: torch.Tensor,
        donors: torch.Tensor,
    ) -> None:
        """Give each target center half of one donor cluster, in place.

        Donor ``centers[donors[i]]`` and target ``centers[targets[i]]`` end up on
        opposite sides of the donor's mean, ``_AUTO_SPLIT_STEP`` of the way to
        one of the donor's rows in ``X``, and share the donor's count. Every
        donor must have a row in ``X``.
        """
        # One random row per donor: the row of its cluster that comes last in a
        # random order of the minibatch. (A plain scatter with duplicate indices
        # would pick an arbitrary row on CUDA, and the last row in batch order
        # is an extreme one when the caller's rows are sorted.)
        order = torch.randperm(X.shape[0], device="cpu", generator=self.random_state).to(X.device)
        last = torch.zeros_like(self._counts, dtype=torch.int64)
        last.scatter_reduce_(
            0,
            labels[order],
            torch.arange(X.shape[0], device=X.device),
            reduce="amax",
            include_self=False,
        )
        mean = centers[donors]
        offset = _AUTO_SPLIT_STEP * (X[order[last[donors]]] - mean)
        centers[targets] = mean + offset
        centers[donors] = mean - offset
        half = self._counts[donors] / 2
        self._counts[donors] = half
        self._counts[targets] = half

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

            batch_weighted = torch.zeros((n_clusters, n_features), device=X.device, dtype=X.dtype)
            batch_weighted.index_add_(0, labels, X)
            batch_weights = torch.bincount(labels, minlength=n_clusters).to(X.dtype)

            mask = batch_weights > 0
            denom = (self._counts + batch_weights).unsqueeze(1).clamp(min=_DENOM_EPS)
            numerator = self._cluster_centers * self._counts.unsqueeze(1) + batch_weighted
            centers_new = torch.where(mask.unsqueeze(1), numerator / denom, self._cluster_centers)
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
            updated = self._maybe_reassign_centers(X, updated, labels)

            return updated, inertia

    def fit(
        self,
        X: torch.Tensor,
        *,
        batch_size: int = 1024,
        max_iter: int = 100,
        tol: float = 0.0,
        max_no_improvement: int | None = 10,
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
        # fit() trains from scratch: drop old centers so a new feature width is allowed.
        self._cluster_centers = None
        self._check_input(X)
        need_batch_inertia = self.verbose or (max_no_improvement is not None)
        device = self.device
        rng = self.random_state
        n_samples = X.shape[0]
        batch_size = min(batch_size, n_samples)
        inertia_convergence = _ConvergenceState()

        self._init_centroids(X)

        n_steps = (max_iter * n_samples) // batch_size
        dev = torch.device(device)
        use_staging = dev.type == "cuda" and X.device.type == "cpu"

        # Batch indices come from their own generator, seeded once from
        # ``random_state``. The prefetch thread draws them ahead of the
        # training loop, so they must not share a generator with the
        # reassignment code, or results would depend on thread timing.
        index_seed = int(torch.randint(0, 2**62, (1,), device="cpu", generator=rng).item())
        index_rng = torch.Generator(device="cpu").manual_seed(index_seed)

        prefetcher = (
            _BatchPrefetcher(X, batch_size, n_steps, index_rng, dev) if use_staging else None
        )
        try:
            for i in range(n_steps):
                if prefetcher is not None:
                    X_batch = prefetcher.next()
                else:
                    mb_idx = torch.randint(
                        0, n_samples, (batch_size,), device="cpu", generator=index_rng
                    )
                    X_batch = X[mb_idx].to(device)

                centers_new, batch_inertia = self._mini_batch_step(
                    X_batch,
                    return_inertia=need_batch_inertia,
                )
                if prefetcher is not None:
                    prefetcher.release()

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
        finally:
            if prefetcher is not None:
                prefetcher.close()

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
        max_no_improvement: int | None = 10,
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
            "reassign_centers": self.reassign_centers,
            "dtype_str": _dtype_to_str(self.dtype),
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
        assignment_backend: str = _DEFAULT_ASSIGNMENT_BACKEND,
        verbose: int = 0,
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
            it as the model device. Default: the ``map_location`` device (CPU),
            not the device the model was saved from.

        **assignment_backend** (`str`, default ``"native"``):
            Backend of the loaded model. Like ``device``, it describes the
            machine the model runs on, not the model, so it is not stored in
            the checkpoint: pass it again when you want the kernel.

        **verbose** (`int`, default ``0``):
            Verbosity of the loaded model.

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
        if version not in _READABLE_CHECKPOINT_VERSIONS:
            raise ValueError(
                f"Unsupported checkpoint format_version={version!r}; "
                f"expected one of {_READABLE_CHECKPOINT_VERSIONS}."
            )
        if device is not None:
            target_device = device
        elif payload["fitted"]:
            # Wherever torch.load put the tensors; works for every map_location form.
            target_device = payload["cluster_centers"].device
        elif isinstance(map_location, (str, torch.device)):
            target_device = torch.device(map_location)
        else:
            target_device = torch.device("cpu")
        dtype = _dtype_from_str(payload["dtype_str"])
        rng = torch.Generator(device="cpu")
        rng.set_state(payload["random_state"])
        if "reassign_centers" in payload:
            reassign_centers = bool(payload["reassign_centers"])
        else:
            # Version 1 stored a reassignment_ratio. 0 turned reassignment off;
            # any other value used a rule that this switch replaces.
            reassign_centers = bool(payload["reassignment_ratio"] != 0)
        inst = cls(
            n_clusters=int(payload["n_clusters"]),
            verbose=verbose,
            reassign_centers=reassign_centers,
            random_state=rng,
            device=target_device,
            dtype=dtype,
            assignment_backend=assignment_backend,
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
