r"""Exact structural knots for positive diagonal tree hiCAP problems.

A weighted tree-isotonic pass computes every mathematical zero-support
topology event. Coefficient-direction knots are handled separately by the
coefficient homotopy.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ._result import _real_array_snapshot
from ._quadratic import _make_quadratic_regression_loss
from ._problem import _diagonal_weighted_least_squares, _prepare_problem
from .proximal import (
    _LaminarQuadraticSolver,
    _positive_scalar,
    _zero_penalty_certificate,
)
from .tree_prox import LaminarGroupLinfProx


def _nonnegative_scalar(value: Any, name: str) -> float:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, float, np.number))
        or not np.isfinite(value)
        or float(value) < 0.0
    ):
        raise ValueError(f"{name} must be a finite nonnegative scalar")
    return float(value)


@dataclass(frozen=True)
class TreeTopologyPath:
    """Exact structural knots for a descendant-group tree penalty.

    ``activation_lambdas[j]`` is the penalty immediately below which input
    group (tree node) ``j`` is retained.  At an exact knot the minimal/tie
    convention is strict:
    ``topology_at(lambda)`` contains nodes with activation lambda greater than
    ``lambda``.  ``entering_groups[k]`` lists the tied groups that appear just
    below ``lambdas[k]``.  ``node_ids`` can optionally map each group position
    to an external tree-node identifier; ``entering_nodes`` and
    ``iter_node_events`` then expose those identifiers directly. Methods named
    ``topology`` and ``iter_events`` always return input-group positions;
    methods named ``tree_nodes`` or ``node_events`` apply the external mapping.

    Coefficients are optional because computing all activation thresholds and
    storing their event batches is near-linear, whereas materializing either
    every full topology or a dense coefficient vector at every knot is
    necessarily at least quadratic when there are order-p knots.
    """

    lambdas: np.ndarray
    activation_lambdas: np.ndarray
    entering_groups: tuple[tuple[int, ...], ...]
    node_ids: np.ndarray | None = None
    coefficients: np.ndarray | None = None
    intercepts: np.ndarray | None = None
    diagnostics: tuple[dict[str, Any], ...] = ()
    exact: bool = True
    status: str = "complete"
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        lambdas = _real_array_snapshot(self.lambdas, "lambdas")
        activation = _real_array_snapshot(self.activation_lambdas, "activation_lambdas")
        if (
            activation.ndim != 1
            or np.any(~np.isfinite(activation))
            or np.any(activation < 0.0)
        ):
            raise ValueError(
                "activation_lambdas must be a finite nonnegative vector"
            )
        expected_lambdas = np.r_[
            np.unique(activation[activation > 0.0])[::-1], 0.0
        ]
        if (
            lambdas.ndim != 1
            or lambdas.size == 0
            or np.any(~np.isfinite(lambdas))
            or np.any(lambdas < 0.0)
            or not np.array_equal(lambdas, expected_lambdas)
        ):
            raise ValueError(
                "lambdas must be the distinct positive activation values in "
                "descending order followed by zero"
            )
        entering = tuple(
            tuple(int(group) for group in value)
            for value in self.entering_groups
        )
        if len(entering) != lambdas.size:
            raise ValueError("entering_groups must align with lambdas")
        seen_groups: set[int] = set()
        for lam, group_batch in zip(lambdas, entering):
            if (
                len(set(group_batch)) != len(group_batch)
                or any(
                    group < 0 or group >= activation.size
                    for group in group_batch
                )
            ):
                raise ValueError(
                    "entering_groups must contain unique valid group indices"
                )
            if any(activation[group] != lam for group in group_batch):
                raise ValueError(
                    "each entering group must align with its activation lambda"
                )
            if seen_groups.intersection(group_batch):
                raise ValueError("an entering group may occur in only one batch")
            seen_groups.update(group_batch)
        expected_groups = set(np.flatnonzero(activation > 0.0).tolist())
        if seen_groups != expected_groups:
            raise ValueError(
                "entering_groups must contain every positive-activation group"
            )
        coefficients = self.coefficients
        intercepts = self.intercepts
        node_ids = self.node_ids
        if node_ids is not None:
            raw_node_ids = np.asarray(node_ids)
            if (
                raw_node_ids.ndim != 1
                or raw_node_ids.shape != activation.shape
                or raw_node_ids.dtype.kind not in "iu"
            ):
                raise ValueError(
                    "node_ids must be an integer vector aligned with groups"
                )
            node_ids = raw_node_ids.astype(np.intp, copy=True)
            if np.unique(node_ids).size != node_ids.size:
                raise ValueError("node_ids must be unique")
        if coefficients is not None:
            coefficients = _real_array_snapshot(coefficients, "coefficients")
            if coefficients.ndim != 2 or coefficients.shape[0] != lambdas.size:
                raise ValueError("coefficients must have one row per stored lambda")
            if not np.all(np.isfinite(coefficients)):
                raise ValueError("coefficients must contain only finite values")
        if intercepts is not None:
            intercepts = _real_array_snapshot(intercepts, "intercepts")
            if intercepts.shape != lambdas.shape or not np.all(np.isfinite(intercepts)):
                raise ValueError("intercepts must be finite and align with lambdas")
        if (coefficients is None) != (intercepts is None):
            raise ValueError("coefficients and intercepts must be stored together")
        diagnostics = tuple(self.diagnostics)
        if diagnostics and len(diagnostics) != lambdas.size:
            raise ValueError("diagnostics must have one entry per stored lambda")
        if node_ids is not None:
            node_ids.flags.writeable = False
        object.__setattr__(self, "lambdas", lambdas)
        object.__setattr__(self, "activation_lambdas", activation)
        object.__setattr__(self, "entering_groups", entering)
        object.__setattr__(self, "node_ids", node_ids)
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "intercepts", intercepts)
        object.__setattr__(self, "diagnostics", diagnostics)
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    @property
    def n_knots(self) -> int:
        """Number of distinct positive structural breakpoints."""
        return int(np.count_nonzero(self.lambdas > 0.0))

    @property
    def n_states(self) -> int:
        """Number of stored strict/tie-convention topology states."""
        return int(self.lambdas.size)

    @property
    def entering_nodes(self) -> tuple[tuple[int, ...], ...]:
        """Tied external node IDs entering immediately below each knot."""
        if self.node_ids is None:
            return self.entering_groups
        return tuple(
            tuple(int(self.node_ids[group]) for group in batch)
            for batch in self.entering_groups
        )

    def iter_events(self) -> Iterator[tuple[float, tuple[int, ...]]]:
        """Yield ``(lambda, entering_groups)`` batches in descending order.

        Applying the returned batches to a mutable active-node mask traverses
        the complete structural path in linear output space.  Groups in a
        batch become active immediately below the associated lambda.
        """
        for lam, entering in zip(self.lambdas, self.entering_groups):
            if lam > 0.0:
                yield float(lam), entering

    def iter_node_events(self) -> Iterator[tuple[float, tuple[int, ...]]]:
        """Yield structural event batches using external tree-node IDs."""
        for lam, entering in zip(self.lambdas, self.entering_nodes):
            if lam > 0.0:
                yield float(lam), entering

    def iter_pruning_events(self) -> Iterator[tuple[float, tuple[int, ...]]]:
        """Yield ``(lambda, leaving_groups)`` as lambda increases from zero.

        Initialize the active set with :meth:`topology_at` evaluated at zero.
        Groups whose activation value is exactly zero are inactive even at
        zero under the strict convention and therefore are never yielded.
        """
        for index in range(self.lambdas.size - 2, -1, -1):
            yield float(self.lambdas[index]), self.entering_groups[index]

    def iter_node_pruning_events(
        self,
    ) -> Iterator[tuple[float, tuple[int, ...]]]:
        """Yield external node-ID batches removed as lambda increases.

        Initialize the rendered tree with :meth:`tree_nodes_at` evaluated at
        zero; zero-activation nodes are absent from that initial topology.
        """
        entering_nodes = self.entering_nodes
        for index in range(self.lambdas.size - 2, -1, -1):
            yield float(self.lambdas[index]), entering_nodes[index]

    def iter_topologies(
        self, *, below: bool = False
    ) -> Iterator[tuple[float, tuple[int, ...]]]:
        """Yield full topology states at, or just below, every stored lambda.

        Full materialization can require quadratic output when there are
        order-p knots.  Use :meth:`iter_events` to update a tree incrementally
        when only the node changes are needed.
        """
        if not isinstance(below, (bool, np.bool_)):
            raise ValueError("below must be a boolean")
        active: set[int] = set()
        for lam, entering in zip(self.lambdas, self.entering_groups):
            if below and lam > 0.0:
                active.update(entering)
            yield float(lam), tuple(sorted(active))
            if not below and lam > 0.0:
                active.update(entering)

    def topology_at(self, lam: float, *, below: bool = False) -> tuple[int, ...]:
        """Return the exact retained-node topology at or just below ``lam``."""
        lam = _nonnegative_scalar(lam, "lam")
        if not isinstance(below, (bool, np.bool_)):
            raise ValueError("below must be a boolean")
        if below and lam > 0.0:
            active = self.activation_lambdas >= lam
        else:
            active = self.activation_lambdas > lam
        return tuple(int(node) for node in np.flatnonzero(active))

    def tree_nodes_at(
        self, lam: float, *, below: bool = False
    ) -> tuple[int, ...]:
        """Return retained external node IDs at or just below ``lam``."""
        groups = self.topology_at(lam, below=below)
        if self.node_ids is None:
            return groups
        return tuple(int(self.node_ids[group]) for group in groups)


def _tree_isotonic_activation_lambdas(
    parents: np.ndarray,
    masses: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Weighted decreasing tree isotonic regression via a leftist heap.

    ``parents`` must be in a child-before-parent topological order.  Each
    provisional block is a heap item keyed by its weighted mean.  At a parent,
    all child blocks whose mean exceeds the current parent-block mean are
    pooled.  A leftist heap makes child-heap melding and maximum deletion
    logarithmic; every block is inserted and removed at most once.
    """
    parents = np.asarray(parents, dtype=np.intp)
    masses = np.asarray(masses, dtype=float)
    weights = np.asarray(weights, dtype=float)
    n_nodes = parents.size
    if parents.ndim != 1:
        raise ValueError("isotonic parents must be a vector")
    if masses.shape != (n_nodes,) or weights.shape != (n_nodes,):
        raise ValueError("isotonic masses and weights must align with parents")
    if np.any(~np.isfinite(masses)) or np.any(masses < 0.0):
        raise ValueError("isotonic masses must be finite and nonnegative")
    if np.any(~np.isfinite(weights)) or np.any(weights <= 0.0):
        raise ValueError("isotonic weights must be finite and positive")
    for child, parent in enumerate(parents):
        if parent < -1 or parent >= n_nodes:
            raise ValueError("isotonic parent indices are out of range")
        if parent >= 0 and parent <= child:
            raise ValueError("isotonic parents must follow their children")

    maximum_mass = float(np.max(masses, initial=0.0))
    if maximum_mass == 0.0:
        return np.zeros(n_nodes, dtype=float)
    maximum_value = max(maximum_mass, float(np.max(weights)))
    # A pooled block contains at most n_nodes inputs. Keep ordinary arithmetic
    # unchanged, but prevent either sum from overflowing for extreme inputs.
    # One shared power-of-two scale preserves every block mean exactly unless
    # a small input loses bits in the subnormal range; reject that case.
    safe_addend = np.finfo(float).max / max(n_nodes, 1) / 2.0
    if maximum_value > safe_addend:
        exponent = int(np.frexp(maximum_value)[1])
        with np.errstate(under="ignore"):
            scaled_mass = np.ldexp(masses, -exponent)
            scaled_weight = np.ldexp(weights, -exponent)
        if (
            not np.array_equal(np.ldexp(scaled_mass, exponent), masses)
            or not np.array_equal(np.ldexp(scaled_weight, exponent), weights)
        ):
            raise ValueError(
                "isotonic pooling cannot safely represent this input range"
            )
        masses, weights = scaled_mass, scaled_weight

    children: list[list[int]] = [[] for _ in range(n_nodes)]
    roots: list[int] = []
    for child, parent in enumerate(parents):
        if parent < 0:
            roots.append(child)
        else:
            children[int(parent)].append(child)

    left = np.full(n_nodes, -1, dtype=np.intp)
    right = np.full(n_nodes, -1, dtype=np.intp)
    rank = np.ones(n_nodes, dtype=np.intp)
    block_weight = weights.copy()
    block_mass = masses.copy()
    member_head = np.arange(n_nodes, dtype=np.intp)
    member_tail = np.arange(n_nodes, dtype=np.intp)
    member_next = np.full(n_nodes, -1, dtype=np.intp)
    subtree_heap = np.full(n_nodes, -1, dtype=np.intp)

    def block_mean(block: int) -> float:
        # Provisional ratios can overflow even when subsequent pooling gives
        # a finite mean. Validate finalized blocks below, not these comparisons.
        return float(block_mass[block]) / float(block_weight[block])

    def heap_rank(node: int) -> int:
        return 0 if node < 0 else int(rank[node])

    def meld(first: int, second: int) -> int:
        if first < 0:
            return second
        if second < 0:
            return first
        if block_mean(first) < block_mean(second):
            first, second = second, first
        right[first] = meld(int(right[first]), second)
        if heap_rank(int(left[first])) < heap_rank(int(right[first])):
            left[first], right[first] = right[first], left[first]
        rank[first] = heap_rank(int(right[first])) + 1
        return first

    for node in range(n_nodes):
        heap = -1
        for child in children[node]:
            heap = meld(heap, int(subtree_heap[child]))
        while heap >= 0 and block_mean(heap) > block_mean(node):
            absorbed = heap
            heap = meld(int(left[absorbed]), int(right[absorbed]))
            left[absorbed] = -1
            right[absorbed] = -1
            rank[absorbed] = 1
            member_next[int(member_tail[node])] = member_head[absorbed]
            member_tail[node] = member_tail[absorbed]
            block_mass[node] += block_mass[absorbed]
            block_weight[node] += block_weight[absorbed]
        subtree_heap[node] = meld(heap, node)

    fitted = np.empty(n_nodes, dtype=float)
    for root in roots:
        heap_stack = [int(subtree_heap[root])]
        while heap_stack:
            block = heap_stack.pop()
            value = block_mean(block)
            if not np.isfinite(value) or (value == 0.0 and block_mass[block] > 0.0):
                raise ValueError(
                    "isotonic activation is outside the representable positive "
                    "floating-point range"
                )
            member = int(member_head[block])
            while member >= 0:
                fitted[member] = value
                member = int(member_next[member])
            if left[block] >= 0:
                heap_stack.append(int(left[block]))
            if right[block] >= 0:
                heap_stack.append(int(right[block]))
    return fitted


def _activation_event_arrays(
    activation: np.ndarray,
) -> tuple[np.ndarray, tuple[tuple[int, ...], ...]]:
    """Build compact descending event batches in ``O(m log m)`` time."""
    groups_by_activation: dict[float, list[int]] = {}
    for group, value in enumerate(np.asarray(activation, dtype=float)):
        if value > 0.0:
            groups_by_activation.setdefault(float(value), []).append(group)
    positive_knots = np.asarray(
        sorted(groups_by_activation, reverse=True), dtype=float
    )
    lambdas = np.r_[positive_knots, 0.0]
    entering = tuple(
        tuple(groups_by_activation[float(lam)]) for lam in positive_knots
    ) + ((),)
    return lambdas, entering


def _positive_node_weights(
    group_weights: float | Sequence[float] | None, n_nodes: int
) -> np.ndarray:
    if group_weights is None:
        weights = np.ones(n_nodes, dtype=float)
    else:
        raw = np.asarray(group_weights)
        if raw.ndim == 0:
            if raw.dtype.kind in "bc":
                raise ValueError("group_weights must be finite positive scalars")
            try:
                scalar = float(raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "group_weights must be finite positive scalars"
                ) from exc
            if not np.isfinite(scalar) or scalar <= 0.0:
                raise ValueError("group_weights must be finite and positive")
            weights = np.full(n_nodes, scalar, dtype=float)
        else:
            if (
                raw.ndim != 1
                or raw.shape != (n_nodes,)
                or raw.dtype.kind in "bc"
            ):
                raise ValueError(
                    f"group_weights must be a scalar or have shape ({n_nodes},)"
                )
            try:
                weights = raw.astype(float, copy=True)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "group_weights must be finite positive scalars"
                ) from exc
    if np.any(~np.isfinite(weights)) or np.any(weights <= 0.0):
        raise ValueError("group_weights must be finite and positive")
    return weights


def _child_before_parent_tree(
    parent_indices: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Validate a parent forest and return a child-before-parent reindexing."""
    raw_parents = np.asarray(parent_indices)
    if (
        raw_parents.ndim != 1
        or (raw_parents.size > 0 and raw_parents.dtype.kind not in "iu")
    ):
        raise ValueError("parent_indices must be an integer vector")
    n_nodes = raw_parents.size
    # Check before casting: uint64's largest value would otherwise wrap to
    # the valid root sentinel -1 in a signed index array.
    if (
        (raw_parents.dtype.kind == "i" and np.any(raw_parents < -1))
        or np.any(raw_parents >= n_nodes)
    ):
        raise ValueError("parent_indices contains an index out of range")
    parents = raw_parents.astype(np.intp, copy=True)
    if np.any(parents == np.arange(n_nodes)):
        raise ValueError("a tree node cannot be its own parent")

    remaining_children = np.bincount(
        parents[parents >= 0], minlength=n_nodes
    ).astype(np.intp, copy=False)
    queue = deque(int(node) for node in np.flatnonzero(remaining_children == 0))
    order: list[int] = []
    while queue:
        node = queue.popleft()
        order.append(node)
        parent = int(parents[node])
        if parent >= 0:
            remaining_children[parent] -= 1
            if remaining_children[parent] == 0:
                queue.append(parent)
    if len(order) != n_nodes:
        raise ValueError("parent_indices must define an acyclic rooted forest")

    source_order = np.asarray(order, dtype=np.intp)
    inverse = np.empty(n_nodes, dtype=np.intp)
    inverse[source_order] = np.arange(n_nodes)
    ordered_parents = np.full(n_nodes, -1, dtype=np.intp)
    for position, source in enumerate(source_order):
        parent = int(parents[source])
        if parent >= 0:
            ordered_parents[position] = inverse[parent]
    return source_order, ordered_parents


def tree_group_linf_exact_topology_path(
    linear_scores: Sequence[float],
    parent_indices: Sequence[int],
    *,
    group_weights: float | Sequence[float] | None = None,
    node_ids: Sequence[int] | None = None,
) -> TreeTopologyPath:
    r"""Compute exact structural knots from tree sufficient statistics.

    There is one coefficient and one descendant group per node.  For the
    diagonal quadratic

    ``0.5 * beta.T @ D @ beta - linear_scores.T @ beta``

    plus ``lambda * sum_v w_v ||beta[subtree(v)]||_inf``, the complete
    zero-support topology path depends on ``abs(linear_scores)`` and the tree,
    but not on the positive Gram diagonal ``D``.  Supplying scores directly
    avoids both a design matrix and explicit descendant-group memberships.

    ``parent_indices`` may use any node order and may describe a forest; roots
    have parent ``-1``.  ``node_ids`` optionally maps positions to unique
    integer external identifiers such as sklearn's original tree node numbers.
    For several sklearn trees, offset or otherwise encode the per-tree node IDs
    so they remain globally unique.

    Unsafe pooling ranges or unrepresentable positive activation values raise
    ``ValueError`` rather than returning an incorrectly certified path.
    """
    raw_scores = np.asarray(linear_scores)
    if (
        raw_scores.ndim != 1
        or raw_scores.dtype.kind == "c"
    ):
        raise ValueError("linear_scores must be a real vector")
    try:
        scores = raw_scores.astype(float, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("linear_scores must contain finite real values") from exc
    if np.any(~np.isfinite(scores)):
        raise ValueError("linear_scores must contain finite real values")

    source_order, ordered_parents = _child_before_parent_tree(parent_indices)
    if source_order.size != scores.size:
        raise ValueError("linear_scores and parent_indices must have equal length")
    weights = _positive_node_weights(group_weights, scores.size)
    activation_ordered = _tree_isotonic_activation_lambdas(
        ordered_parents,
        np.abs(scores[source_order]),
        weights[source_order],
    )
    activation = np.empty_like(activation_ordered)
    activation[source_order] = activation_ordered
    lambdas, entering = _activation_event_arrays(activation)
    if node_ids is None:
        external_node_ids = np.arange(scores.size, dtype=np.intp)
    else:
        external_node_ids = np.asarray(node_ids)
        if external_node_ids.shape == (0,):
            external_node_ids = np.empty(0, dtype=np.intp)
    return TreeTopologyPath(
        lambdas=lambdas,
        activation_lambdas=activation,
        entering_groups=entering,
        node_ids=external_node_ids,
        exact=True,
        status="complete",
        metadata={
            "problem": "diagonal_quadratic_from_scores",
            "ord": "inf",
            "path_scope": "exact_zero_support_tree_topology",
            "topology_knots_exact": True,
            "coefficient_knots_enumerated": False,
            "coefficients_materialized": False,
            "coefficient_points_certified": None,
            "coefficient_status": "not_materialized",
            "n_groups": scores.size,
            "n_features": scores.size,
            "n_uncovered_features": 0,
            "isotonic_algorithm": "leftist_heap_tree_pava",
            "isotonic_complexity": "O(n_nodes log n_nodes)",
            "descendant_groups_materialized": False,
            "exactness_condition": "caller_supplied_positive_diagonal_problem",
            "parent_indices": tuple(int(value) for value in parent_indices),
        },
    )


def laminar_group_linf_exact_topology_path(
    X: np.ndarray,
    y: np.ndarray,
    groups: Iterable[np.ndarray],
    *,
    sample_weight: np.ndarray | None = None,
    fit_intercept: bool = True,
    group_weights: float | Sequence[float] | None = None,
    assume_diagonal_gram: bool = False,
    include_coefficients: bool = False,
    solver_tolerance: float = 1e-10,
) -> TreeTopologyPath:
    r"""Compute every exact zero-support topology knot for a laminar penalty.

    For each group, its *atom* is the set of coordinates in that group but in
    none of its child groups.  Let ``Q_g`` be the sum of absolute quadratic
    scores over that atom and ``w_g`` its positive penalty weight.  Weighted
    decreasing tree-isotonic regression of ``Q_g / w_g`` gives an activation
    lambda for every group.  The exact retained-group topology is
    ``{g: activation[g] > lambda}``, and its distinct positive fitted values
    are all structural knots.

    This result requires a positive diagonal weighted Gram matrix and concerns
    exact zero support, not an operational nonzero coefficient threshold.
    Setting
    ``include_coefficients=False`` keeps the structural computation compact;
    enabling it evaluates the exact one-sweep coefficient solver at every
    structural knot.  Those coefficient rows use the strict, pre-entry
    topology at the knot; solve at an interior lambda immediately below it to
    obtain coefficients depicting the newly entered state.

    By default the full Gram matrix is formed and machine-scale correlations
    are treated as floating-point zero.  Thus exactness is for the accepted
    numerical diagonal model.  ``assume_diagonal_gram=True`` avoids the dense
    Gram calculation for large trees, conditional on the caller establishing
    orthogonality for these exact rows and weights.  Coordinates outside every
    group do not affect the group topology and remain unpenalized.
    """
    if not isinstance(fit_intercept, (bool, np.bool_)):
        raise ValueError("fit_intercept must be a boolean")
    if not isinstance(assume_diagonal_gram, (bool, np.bool_)):
        raise ValueError("assume_diagonal_gram must be a boolean")
    if not isinstance(include_coefficients, (bool, np.bool_)):
        raise ValueError("include_coefficients must be a boolean")
    solver_tolerance = _positive_scalar(solver_tolerance, "solver_tolerance")

    X_array = np.asarray(X)
    if X_array.ndim != 2 or X_array.shape[1] == 0:
        raise ValueError("X must be a nonempty two-dimensional matrix")
    # Shared validation also makes private group copies and normalizes weights.
    X_array, y_array, private_groups, initial_beta, weights, _ = _prepare_problem(
        X=X_array,
        y=y,
        groups=groups,
        beta_init=None,
        sample_weight=sample_weight,
        ord="inf",
    )
    normalized_weights = weights / float(weights.sum())
    if fit_intercept:
        x_mean = normalized_weights @ X_array
        y_mean = float(normalized_weights @ y_array)
        X_work = X_array - x_mean
        y_work = y_array - y_mean
    else:
        x_mean = np.zeros(X_array.shape[1], dtype=float)
        y_mean = 0.0
        X_work = X_array
        y_work = y_array
    quadratic = _make_quadratic_regression_loss(
        X_work,
        y_work,
        weights,
        assume_diagonal_gram=bool(assume_diagonal_gram),
    )
    if quadratic.diagonal is None or not np.all(quadratic.diagonal > 0.0):
        raise ValueError(
            "exact topology knots require a positive diagonal weighted Gram "
            "matrix"
        )
    solver: _LaminarQuadraticSolver | None = None
    if include_coefficients:
        solver = _LaminarQuadraticSolver(
            quadratic,
            private_groups,
            group_weights,
            max_iter=1,
            tolerance=solver_tolerance,
            restart=True,
        )
        prox = solver.euclidean_prox
    else:
        # Topology thresholds need only the compiled laminar hierarchy.  Do
        # not construct the coordinate-weighted coefficient prox as well.
        prox = LaminarGroupLinfProx(
            private_groups, X_array.shape[1], group_weights
        )
    if np.any(prox.group_weights <= 0.0):
        raise ValueError("exact topology knots require positive group weights")

    # In leaf-to-root order, the first group containing a coordinate owns that
    # coordinate's atom.  The laminar validator guarantees a membership chain.
    atom_owner = np.full(X_array.shape[1], -1, dtype=np.intp)
    for group_position, group in enumerate(prox.groups):
        unowned = atom_owner[group] < 0
        atom_owner[group[unowned]] = group_position
    atom_mass = np.zeros(len(prox.groups), dtype=float)
    covered = atom_owner >= 0
    np.add.at(
        atom_mass,
        atom_owner[covered],
        np.abs(quadratic.linear[covered]),
    )
    activation_ordered = _tree_isotonic_activation_lambdas(
        prox.parents, atom_mass, prox.group_weights
    )
    activation = np.empty_like(activation_ordered)
    activation[prox.source_order] = activation_ordered

    lambdas, entering = _activation_event_arrays(activation)

    coefficients: np.ndarray | None = None
    intercepts: np.ndarray | None = None
    diagnostics: tuple[dict[str, Any], ...] = ()
    coefficient_points_certified: bool | None = None
    if include_coefficients:
        if solver is None:  # pragma: no cover - guarded above
            raise RuntimeError("coefficient solver was not initialized")
        coefficient_rows: list[np.ndarray] = []
        intercept_values: list[float] = []
        diagnostic_values: list[dict[str, Any]] = []
        for lam in lambdas:
            if lam == 0.0:
                beta, raw_diagnostic = _diagonal_weighted_least_squares(
                    quadratic, initial_beta
                )
                diagnostic = dict(raw_diagnostic)
                diagnostic.update(
                    _zero_penalty_certificate(
                        quadratic, beta, solver_tolerance
                    )
                )
            else:
                point = solver.solve(float(lam), initial_beta)
                beta, diagnostic = point.beta, dict(point.diagnostic)
            intercept = y_mean - float(x_mean @ beta)
            diagnostic.update({"lambda": float(lam), "intercept": intercept})
            coefficient_rows.append(np.asarray(beta).copy())
            intercept_values.append(intercept)
            diagnostic_values.append(diagnostic)
        coefficients = np.vstack(coefficient_rows)
        intercepts = np.asarray(intercept_values)
        diagnostics = tuple(diagnostic_values)
        coefficient_points_certified = all(
            diagnostic.get("certified") is True
            for diagnostic in diagnostics
        )

    memberships = int(sum(group.size for group in prox.groups))
    n_uncovered_features = int(np.count_nonzero(~covered))
    numerical_diagonal_tolerance = (
        32.0 * np.finfo(float).eps * max(1, X_array.shape[1])
    )
    return TreeTopologyPath(
        lambdas=lambdas,
        activation_lambdas=activation,
        entering_groups=entering,
        node_ids=np.arange(len(prox.groups), dtype=np.intp),
        coefficients=coefficients,
        intercepts=intercepts,
        diagnostics=diagnostics,
        exact=True,
        status="complete",
        metadata={
            "problem": "regression",
            "ord": "inf",
            "fit_intercept": bool(fit_intercept),
            "path_scope": "exact_zero_support_group_topology",
            "topology_knots_exact": True,
            "coefficient_knots_enumerated": False,
            "coefficients_materialized": bool(include_coefficients),
            "coefficient_points_certified": coefficient_points_certified,
            "coefficient_status": (
                "not_materialized"
                if coefficient_points_certified is None
                else (
                    "certified"
                    if coefficient_points_certified
                    else "uncertified"
                )
            ),
            "coefficient_rows_at_knots": "strict_pre_entry_topology",
            "n_groups": len(prox.groups),
            "n_features": X_array.shape[1],
            "n_uncovered_features": n_uncovered_features,
            "n_group_memberships": memberships,
            "isotonic_algorithm": "leftist_heap_tree_pava",
            "isotonic_complexity": "O(n_groups log n_groups + memberships)",
            "quadratic_backend": quadratic.backend,
            "assumed_diagonal_gram": quadratic.assumed_diagonal_gram,
            "gram_max_off_diagonal": quadratic.max_off_diagonal,
            "gram_max_off_diagonal_correlation": (
                quadratic.max_off_diagonal_correlation
            ),
            "numerical_diagonal_correlation_tolerance": (
                numerical_diagonal_tolerance
            ),
            "exactness_condition": (
                "caller_asserted_diagonal_gram"
                if quadratic.assumed_diagonal_gram
                else "numerically_verified_diagonal_gram"
            ),
            "group_source_order": tuple(int(value) for value in prox.source_order),
        },
    )


__all__ = [
    "TreeTopologyPath",
    "laminar_group_linf_exact_topology_path",
    "tree_group_linf_exact_topology_path",
]
