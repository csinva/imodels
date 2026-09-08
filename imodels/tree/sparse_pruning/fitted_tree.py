"""hiCAP paths from sufficient statistics stored in a fitted CART tree."""
from __future__ import annotations

import hashlib
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import replace
from typing import Any

import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree._tree import TREE_LEAF, TREE_UNDEFINED
from sklearn.utils.validation import check_is_fitted

from .optimization._result import RegularizationPath
from .optimization.topology import TreeTopologyPath, tree_group_linf_exact_topology_path


_MEAN_REGRESSION_CRITERIA = {
    "squared_error",
    "friedman_mse",
    "poisson",
    # Compatibility with estimators serialized by older sklearn releases.
    "mse",
}

_FITTED_TREE_FINGERPRINT_VERSION = "sha256-v1"


def _update_fingerprint_array(
    digest: Any,
    name: str,
    values: Any,
    dtype: str,
) -> None:
    """Add one numeric array to a platform-independent tree fingerprint."""
    array = np.ascontiguousarray(np.asarray(values, dtype=dtype))
    digest.update(name.encode("utf-8"))
    digest.update(b"\0")
    digest.update(",".join(str(size) for size in array.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))


def _fitted_tree_fingerprint(estimator: DecisionTreeRegressor) -> str:
    """Return a stable fingerprint of the source fitted-tree state."""
    tree = estimator.tree_
    digest = hashlib.sha256()
    digest.update(b"imodels-fitted-regression-tree\0")
    digest.update(_FITTED_TREE_FINGERPRINT_VERSION.encode("ascii"))
    digest.update(b"\0")
    scalar_state = (
        str(estimator.criterion),
        int(estimator.n_outputs_),
        int(getattr(estimator, "n_features_in_", tree.n_features)),
        int(tree.node_count),
        int(tree.max_depth),
    )
    digest.update(repr(scalar_state).encode("utf-8"))
    digest.update(b"\0")
    _update_fingerprint_array(digest, "children_left", tree.children_left, "<i8")
    _update_fingerprint_array(digest, "children_right", tree.children_right, "<i8")
    _update_fingerprint_array(digest, "feature", tree.feature, "<i8")
    _update_fingerprint_array(digest, "threshold", tree.threshold, "<f8")
    _update_fingerprint_array(digest, "impurity", tree.impurity, "<f8")
    _update_fingerprint_array(
        digest, "n_node_samples", tree.n_node_samples, "<i8"
    )
    _update_fingerprint_array(
        digest,
        "weighted_n_node_samples",
        tree.weighted_n_node_samples,
        "<f8",
    )
    _update_fingerprint_array(digest, "value", tree.value, "<f8")
    missing_directions = getattr(tree, "missing_go_to_left", None)
    if missing_directions is None:
        digest.update(b"missing_go_to_left\0absent\0")
    else:
        _update_fingerprint_array(
            digest, "missing_go_to_left", missing_directions, "u1"
        )
    monotonic_constraints = getattr(estimator, "monotonic_cst", None)
    if monotonic_constraints is None:
        digest.update(b"monotonic_cst\0absent\0")
    else:
        _update_fingerprint_array(
            digest, "monotonic_cst", monotonic_constraints, "<i8"
        )
    return f"{_FITTED_TREE_FINGERPRINT_VERSION}:{digest.hexdigest()}"


def _internal_tree_nodes_and_parents(tree: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return internal sklearn node IDs and parent positions in preorder."""
    node_ids: list[int] = []
    parents: list[int] = []
    pending = [(0, -1)]
    while pending:
        node_id, parent_position = pending.pop()
        left = int(tree.children_left[node_id])
        right = int(tree.children_right[node_id])
        if left == TREE_LEAF or right == TREE_LEAF:
            if left == TREE_LEAF and right == TREE_LEAF:
                continue
            raise ValueError("a fitted internal tree node must have two children")
        position = len(node_ids)
        node_ids.append(int(node_id))
        parents.append(int(parent_position))
        pending.append((right, position))
        pending.append((left, position))
    return np.asarray(node_ids, dtype=np.intp), np.asarray(parents, dtype=np.intp)


def fitted_tree_linf_exact_topology_path(
    estimator: DecisionTreeRegressor,
    *,
    group_weights: float | Sequence[float] | None = None,
) -> TreeTopologyPath:
    r"""Compute the exact structural hiCAP path of a fitted regression tree.

    This is the scalable entry point for the standard local-stump tree
    problem.  It uses only statistics already stored in a fitted CART tree;
    neither the training rows, the stump design matrix, nor explicit
    descendant groups are constructed.

    For internal node ``v`` with child weights ``L_v, R_v``, child means
    ``mu_L, mu_R``, and root weight ``W``, the unnormalized local-stump
    sufficient statistics are

    ``D_vv = (L_v + R_v) / W`` and
    ``h_v = sqrt(L_v R_v) * (mu_R - mu_L) / W``.

    Local stumps are mutually orthogonal under the implicit fitting partition,
    so these scores define its normalized squared-loss hiCAP problem. With
    ordinary finite training data this is also the full local-stump design.
    Each returned event contains original sklearn node IDs through
    :meth:`TreeTopologyPath.iter_node_events`.

    The estimator must be a fitted, single-output ``DecisionTreeRegressor``
    whose criterion stores weighted means.  A path built this way is tied to
    the in-bag rows and weights used to fit that tree; it is not an OOB or
    held-out pruning path. With ``criterion="poisson"``, the result is still
    the exact path for a subsequent quadratic pruning objective, not for
    penalized Poisson deviance. Vector ``group_weights`` are aligned with the
    internal-node preorder exposed by ``path.node_ids``.

    On sklearn versions with native missing-value fitting, stored child
    statistics can differ from the partition obtained by later reapplying the
    fitted routing to the original NaN-containing matrix. This function is
    exact for the stored fitting partition. If the intended objective is the
    re-evaluated stump matrix, build that matrix explicitly and use
    ``laminar_group_linf_exact_topology_path`` instead.

    The fitted estimator and its criterion/monotonic parameters must remain
    unchanged after fitting; sklearn does not retain a separate immutable copy
    of all construction parameters on ``tree_``.

    Original node IDs refer to the unmodified fitted tree. Keep that tree (or
    a copy) when rendering multiple states, because in-place compaction can
    renumber nodes.
    """
    scores, _, parents, node_ids, metadata = _fitted_tree_statistics(estimator)
    path = tree_group_linf_exact_topology_path(
        scores, parents, group_weights=group_weights, node_ids=node_ids
    )
    return replace(path, metadata={**path.metadata, **metadata})


def _fitted_tree_statistics(estimator: DecisionTreeRegressor) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]
]:
    """Validate a fitted tree and extract statistics shared by both paths."""
    if not isinstance(estimator, DecisionTreeRegressor):
        raise TypeError("estimator must be a fitted DecisionTreeRegressor")
    check_is_fitted(estimator, attributes="tree_")
    if int(estimator.n_outputs_) != 1:
        raise ValueError("fitted-tree topology currently requires one output")
    criterion = str(estimator.criterion)
    if criterion not in _MEAN_REGRESSION_CRITERIA:
        raise ValueError(
            "fitted-tree sufficient statistics require a mean-based "
            f"regression criterion; got {criterion!r}"
        )
    monotonic_constraints = getattr(estimator, "monotonic_cst", None)
    if monotonic_constraints is not None and np.any(
        np.asarray(monotonic_constraints) != 0
    ):
        raise ValueError(
            "fitted-tree sufficient statistics do not support monotonic "
            "constraints because stored node values may be clipped"
        )

    tree = estimator.tree_
    node_ids, parents = _internal_tree_nodes_and_parents(tree)
    values = np.asarray(tree.value, dtype=float)
    if values.ndim != 3 or values.shape[1:] != (1, 1):
        raise ValueError("fitted-tree topology currently requires scalar values")
    node_weight = np.asarray(tree.weighted_n_node_samples, dtype=float)
    total_weight = float(node_weight[0])
    if not np.isfinite(total_weight) or total_weight <= 0.0:
        raise ValueError("the fitted tree must have positive finite root weight")

    scores = np.empty(node_ids.size, dtype=float)
    diagonal = np.empty(node_ids.size, dtype=float)
    sqrt_total_weight = np.sqrt(total_weight)
    maximum_weight_conservation_error = 0.0
    maximum_relative_weight_conservation_error = 0.0
    maximum_mean_conservation_error = 0.0
    for position, node_id in enumerate(node_ids):
        left = int(tree.children_left[node_id])
        right = int(tree.children_right[node_id])
        left_weight = float(node_weight[left])
        right_weight = float(node_weight[right])
        parent_weight = float(node_weight[node_id])
        if (
            not np.all(
                np.isfinite([left_weight, right_weight, parent_weight])
            )
            or min(left_weight, right_weight, parent_weight) <= 0.0
        ):
            raise ValueError("every fitted internal node must have positive weights")
        weight_conservation_error = abs(
            parent_weight - left_weight - right_weight
        )
        maximum_weight_conservation_error = max(
            maximum_weight_conservation_error, weight_conservation_error
        )
        weight_conservation_scale = max(
            abs(parent_weight),
            abs(left_weight) + abs(right_weight),
            np.finfo(float).tiny,
        )
        maximum_relative_weight_conservation_error = max(
            maximum_relative_weight_conservation_error,
            weight_conservation_error / weight_conservation_scale,
        )
        left_mean = float(values[left, 0, 0])
        right_mean = float(values[right, 0, 0])
        parent_mean = float(values[node_id, 0, 0])
        if not np.all(np.isfinite([left_mean, right_mean, parent_mean])):
            raise ValueError("the fitted tree must contain finite node values")
        child_weight = left_weight + right_weight
        if not np.isfinite(child_weight) or child_weight <= 0.0:
            raise ValueError("fitted child weights must have a finite positive sum")
        # Child weights are the direct partition of the node's training rows.
        # Use their sum in the sufficient statistics so harmless differences
        # between sklearn's separately accumulated parent and child totals do
        # not perturb the path.
        sqrt_left_weight = np.sqrt(left_weight)
        sqrt_right_weight = np.sqrt(right_weight)
        sqrt_child_weight = np.sqrt(child_weight)
        left_child_ratio = sqrt_left_weight / sqrt_child_weight
        right_child_ratio = sqrt_right_weight / sqrt_child_weight
        # Squaring only after multiplying the mean avoids losing a tiny child
        # fraction whose contribution to a very large mean is still finite.
        reconstructed_parent_mean = (
            (left_mean * left_child_ratio) * left_child_ratio
            + (right_mean * right_child_ratio) * right_child_ratio
        )
        maximum_mean_conservation_error = max(
            maximum_mean_conservation_error,
            abs(parent_mean - reconstructed_parent_mean),
        )
        # Algebraically this is sqrt(L * R) / W, but the ratio form is stable
        # when every sample weight is rescaled by a very large or small factor.
        parent_fraction = child_weight / total_weight
        left_root_ratio = sqrt_left_weight / sqrt_total_weight
        right_root_ratio = sqrt_right_weight / sqrt_total_weight
        larger_ratio = max(left_root_ratio, right_root_ratio)
        smaller_ratio = min(left_root_ratio, right_root_ratio)
        with np.errstate(over="ignore", invalid="ignore"):
            contrast = right_mean - left_mean
            score = (contrast * larger_ratio) * smaller_ratio
        if not np.isfinite(score):
            # Opposite extreme finite means can overflow during subtraction
            # even though scaling each term first is representable.
            with np.errstate(over="ignore", invalid="ignore"):
                score = (
                    (right_mean * larger_ratio) * smaller_ratio
                    - (left_mean * larger_ratio) * smaller_ratio
                )
        if not np.isfinite(score):
            raise ValueError("fitted node statistics produce a non-finite score")
        scores[position] = score
        diagonal[position] = parent_fraction

    # sklearn accumulates weighted parent and child totals separately. With
    # many rows or weights spanning many orders of magnitude, their roundoff
    # can exceed a fixed multiple of eps even for a valid fitted tree.
    root_sample_count = max(1, int(tree.n_node_samples[0]))
    relative_conservation_tolerance = max(
        4096.0 * np.finfo(float).eps,
        8.0 * root_sample_count * np.finfo(float).eps,
    )
    if (
        maximum_relative_weight_conservation_error
        > relative_conservation_tolerance
    ):
        raise ValueError(
            "tree child weights do not sum to their parent within floating "
            "point tolerance"
        )
    value_scale = max(1.0, float(np.max(np.abs(values))))
    mean_conservation_tolerance = max(
        4096.0 * np.finfo(float).eps,
        8.0 * root_sample_count * np.finfo(float).eps,
    ) * value_scale
    if maximum_mean_conservation_error > mean_conservation_tolerance:
        raise ValueError(
            "stored tree values are not weighted child means; the fitted-tree "
            "sufficient-statistics shortcut is unavailable"
        )

    metadata = {
        "problem": "fitted_regression_tree",
        "source": "fitted_tree_sufficient_statistics",
        "criterion": criterion,
        "tree_node_count": int(tree.node_count),
        "n_internal_nodes": int(node_ids.size),
        "tree_depth": int(tree.max_depth),
        "tree_root_weight": total_weight,
        "tree_node_ids": tuple(int(value) for value in node_ids),
        "fitted_tree_fingerprint": _fitted_tree_fingerprint(estimator),
        "fitted_tree_fingerprint_version": _FITTED_TREE_FINGERPRINT_VERSION,
        "gram_diagonal": tuple(float(value) for value in diagonal),
        "linear_scores": tuple(float(value) for value in scores),
        "maximum_weight_conservation_error": maximum_weight_conservation_error,
        "maximum_relative_weight_conservation_error": (
            maximum_relative_weight_conservation_error
        ),
        "relative_weight_conservation_tolerance": relative_conservation_tolerance,
        "maximum_mean_conservation_error": maximum_mean_conservation_error,
        "mean_conservation_tolerance": mean_conservation_tolerance,
        "training_design_materialized": False,
        "observation_count_after_fit_affects_path_cost": False,
        "exactness_condition": "fitted_tree_training_measure",
    }
    return scores, diagonal, parents, node_ids, metadata


def fitted_tree_linf_exact_coefficient_path(
    estimator: DecisionTreeRegressor,
    *,
    group_weights: float | Sequence[float] | None = None,
    tolerance: float = 1e-9,
    max_events: int = 10_000,
) -> RegularizationPath:
    r"""Trace every hiCAP coefficient-slope knot of a fitted regression tree.

    The objective is normalized squared loss plus
    ``lambda * sum_v group_weights[v] * ||beta[subtree(v)]||_inf`` in the
    **unnormalized** local-stump basis. Coefficient columns follow the internal
    node preorder in ``metadata['tree_node_ids']``. An unpenalized intercept
    equals the fitted root mean at every penalty value. The source estimator
    is never changed.

    Like :func:`fitted_tree_linf_exact_topology_path`, this function uses the
    stored fitting rows/weights implicitly. It does not apply to a held-out
    or OOB pruning objective. All criterion, conservation, and missing-value
    qualifications of that constructor also apply here.

    Sufficient-statistic extraction avoids the training design matrix. The
    initial coefficient homotopy nevertheless materializes descendant groups
    and rebuilds the active face after each event. Group memberships can be
    quadratic in tree depth; storing every coefficient at every knot requires
    ``O(n_internal_nodes * n_knots)`` output space. This is a full coefficient
    path, with a different cost from the near-linear structural path.

    ``exact=True`` means interval coverage and certificates passed through
    lambda zero, within numerical coefficient tolerance and the recorded
    event-boundary uncertainty. On an event limit or unresolved numerical
    event the result is a nonexact prefix. Check
    ``status`` and ``exact`` before relying on complete interpolation.
    """
    from .optimization.diagonal_homotopy import tree_group_linf_exact_coefficient_path

    scores, diagonal, parents, _, source = _fitted_tree_statistics(estimator)
    path = tree_group_linf_exact_coefficient_path(
        scores, diagonal, parents,
        group_weights=group_weights,
        tolerance=tolerance,
        max_events=max_events,
    )
    metadata = dict(source)
    metadata.update(path.metadata)
    metadata.update(
        {
            "problem": "fitted_regression_tree",
            "source": "fitted_tree_sufficient_statistics",
            "fit_intercept": True,
            "local_stump_normalization": "unnormalized",
            "exactness_condition": "fitted_tree_training_measure",
            "training_design_materialized": False,
            "coefficients_materialized": True,
            "coefficient_rows_at_knots": "continuous_full_coefficient_path",
            "coefficient_points_certified": path.metadata["point_solutions_certified"],
            "coefficient_status": path.status,
        }
    )
    intercept = float(estimator.tree_.value[0, 0, 0])
    return replace(
        path,
        intercepts=np.full(path.n_points, intercept, dtype=float),
        metadata=metadata,
    )


def materialize_fitted_tree_topology(
    estimator: DecisionTreeRegressor,
    path: TreeTopologyPath,
    lam: float,
    *,
    below: bool = False,
) -> DecisionTreeRegressor:
    """Return a plot-ready copy of ``estimator`` at one topology-path state.

    Inactive internal nodes become leaves on a deep copy; the input estimator
    is never mutated. Original sklearn node numbers remain valid (unreachable
    descendants are not physically compacted), which makes successive event
    batches easy to relate to a visualization. Consequently backing-array
    metadata such as ``tree_.node_count``, ``get_depth()``, and
    ``get_n_leaves()`` need not describe the reachable preview, and its
    serialized size remains that of the uncompressed backing tree.

    This helper materializes *topology*. Values displayed at the resulting
    leaves are the original fitted CART node values, not hiCAP-shrunken
    coefficients. Use an exact coefficient point solve when penalized values
    or predictions are required. The fitted-tree coefficient path can supply
    those coefficients without constructing the training design matrix.
    """
    if not isinstance(estimator, DecisionTreeRegressor):
        raise TypeError("estimator must be a fitted DecisionTreeRegressor")
    check_is_fitted(estimator, attributes="tree_")
    if not isinstance(path, TreeTopologyPath):
        raise TypeError("path must be a TreeTopologyPath")
    if path.node_ids is None:
        raise ValueError("path must carry fitted-tree node IDs")

    expected_fingerprint = path.metadata.get("fitted_tree_fingerprint")
    if not isinstance(expected_fingerprint, str):
        raise ValueError(
            "path must be produced by fitted_tree_linf_exact_topology_path"
        )
    observed_fingerprint = _fitted_tree_fingerprint(estimator)
    if expected_fingerprint != observed_fingerprint:
        raise ValueError(
            "path was computed from a different or subsequently modified "
            "fitted tree"
        )

    node_ids, parents = _internal_tree_nodes_and_parents(estimator.tree_)
    if not np.array_equal(path.node_ids, node_ids):
        raise ValueError("path does not match this fitted tree's internal nodes")
    active_node_ids = set(path.tree_nodes_at(lam, below=below))
    active_positions = np.asarray(
        [int(node_id) in active_node_ids for node_id in node_ids], dtype=bool
    )
    for position, parent in enumerate(parents):
        if (
            active_positions[position]
            and parent >= 0
            and not active_positions[int(parent)]
        ):
            raise ValueError("path topology is not ancestor-closed")

    rendered = deepcopy(estimator)
    tree = rendered.tree_
    for node_id in node_ids[~active_positions]:
        node_id = int(node_id)
        tree.children_left[node_id] = TREE_LEAF
        tree.children_right[node_id] = TREE_LEAF
        tree.feature[node_id] = TREE_UNDEFINED
        tree.threshold[node_id] = float(TREE_UNDEFINED)
        missing_directions = getattr(tree, "missing_go_to_left", None)
        if missing_directions is not None:
            missing_directions[node_id] = 0
    return rendered


__all__ = [
    "fitted_tree_linf_exact_coefficient_path",
    "fitted_tree_linf_exact_topology_path",
    "materialize_fitted_tree_topology",
]
