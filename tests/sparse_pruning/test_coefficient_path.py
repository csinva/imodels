"""Independent checks of the complete diagonal-tree coefficient homotopy."""
from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose
from sklearn.datasets import load_diabetes
from sklearn.tree import DecisionTreeRegressor

from imodels.tree.sparse_pruning.optimization.diagonal_homotopy import (
    tree_group_linf_exact_coefficient_path,
)
from imodels.tree.sparse_pruning.optimization.hicap import hicap_regression_path
from imodels.tree.sparse_pruning.fitted_tree import (
    fitted_tree_linf_exact_topology_path,
)
from imodels.tree.sparse_pruning.optimization.proximal import (
    laminar_group_linf_regression_path,
)
from imodels.tree.sparse_pruning.optimization.topology import (
    tree_group_linf_exact_topology_path,
)


def _descendant_groups(parents):
    """Build an oracle's explicit groups without assuming a node ordering."""
    parents = np.asarray(parents, dtype=int)
    groups = [[] for _ in parents]
    for node in range(parents.size):
        ancestor = node
        while ancestor >= 0:
            groups[ancestor].append(node)
            ancestor = int(parents[ancestor])
    return [np.asarray(group, dtype=int) for group in groups]


def _design_from_statistics(scores, diagonal):
    scores = np.asarray(scores, dtype=float)
    diagonal = np.asarray(diagonal, dtype=float)
    p = scores.size
    return np.diag(np.sqrt(p * diagonal)), np.sqrt(p) * scores / np.sqrt(diagonal)


def _assert_complete(path, scores, diagonal):
    assert path.exact, path.metadata
    assert path.status == "complete"
    assert path.lambdas[-1] == 0.0
    assert np.all(np.diff(path.lambdas) < 0.0)
    assert_allclose(path.coefficients[0], 0.0, atol=1e-10)
    assert_allclose(path.coefficients[-1], np.asarray(scores) / diagonal, atol=1e-9)


def test_two_node_path_contains_nonstructural_saturation_knot():
    scores = np.array([1.0, 2.0])
    diagonal = np.ones(2)
    parents = [-1, 0]
    path = tree_group_linf_exact_coefficient_path(scores, diagonal, parents)

    _assert_complete(path, scores, diagonal)
    assert_allclose(path.lambdas, [1.5, 0.5, 0.0], atol=1e-10)
    assert_allclose(path.coefficients, [[0, 0], [1, 1], [1, 2]], atol=1e-10)
    assert_allclose(path.at(1.0)[0], [0.5, 0.5], atol=1e-10)
    assert_allclose(path.at(0.25)[0], [1.0, 1.5], atol=1e-10)

    structural = tree_group_linf_exact_topology_path(scores, parents)
    assert not np.any(np.isclose(structural.lambdas, 0.5))
    assert structural.topology_at(0.75) == structural.topology_at(0.25)


def test_two_node_path_follows_activation_merge_and_saturation():
    scores = np.array([2.0, 1.0])
    diagonal = np.array([4.0, 1.0])
    path = tree_group_linf_exact_coefficient_path(scores, diagonal, [-1, 0])

    _assert_complete(path, scores, diagonal)
    assert_allclose(path.lambdas, [2.0, 1.0, 2.0 / 3.0, 0.25, 0.0], atol=1e-10)
    assert_allclose(
        path.coefficients,
        [[0, 0], [0.25, 0], [1.0 / 3, 1.0 / 3], [0.5, 0.5], [0.5, 1]],
        atol=1e-10,
    )


@pytest.mark.parametrize("score", [-3.0, 3.0])
@pytest.mark.parametrize("weight", [0.25, 2.0])
def test_weighted_single_node_path_is_analytic_soft_threshold(score, weight):
    curvature = 1.7
    path = tree_group_linf_exact_coefficient_path(
        [score], [curvature], [-1], group_weights=[weight]
    )
    _assert_complete(path, [score], np.array([curvature]))
    maximum = abs(score) / weight
    assert_allclose(path.lambdas, [maximum, 0.0], atol=1e-10)
    for fraction in [0.1, 0.5, 0.9]:
        lam = fraction * maximum
        expected = np.sign(score) * (abs(score) - weight * lam) / curvature
        assert_allclose(path.at(lam)[0], [expected], atol=1e-10)


@pytest.mark.parametrize("p", range(2, 8))
def test_small_random_tree_knots_and_coefficients_match_generic_hicap(p):
    rng = np.random.default_rng(400 + p)
    parents = np.r_[-1, [rng.integers(node) for node in range(1, p)]]
    diagonal = rng.uniform(0.3, 2.0, p)
    scores = rng.normal(size=p)
    X, y = _design_from_statistics(scores, diagonal)
    reference = hicap_regression_path(
        X, y, _descendant_groups(parents), fit_intercept=False, tolerance=1e-9
    )
    path = tree_group_linf_exact_coefficient_path(scores, diagonal, parents)

    _assert_complete(path, scores, diagonal)
    assert reference.exact, reference.metadata
    assert_allclose(path.lambdas, reference.lambdas, rtol=2e-7, atol=2e-9)
    assert_allclose(path.coefficients, reference.coefficients, rtol=2e-7, atol=2e-9)


@pytest.mark.parametrize("seed", range(4))
def test_weighted_path_intervals_match_exact_point_solver_and_green_knots(seed):
    rng = np.random.default_rng(seed + 730)
    p = 9
    parents = np.r_[-1, [rng.integers(node) for node in range(1, p)]]
    diagonal = rng.uniform(0.1, 3.0, p)
    scores = rng.normal(size=p)
    weights = rng.uniform(0.15, 2.0, p)
    path = tree_group_linf_exact_coefficient_path(
        scores, diagonal, parents, group_weights=weights
    )
    _assert_complete(path, scores, diagonal)

    queries = list(path.lambdas)
    for upper, lower in zip(path.lambdas[:-1], path.lambdas[1:]):
        queries.extend(lower + fraction * (upper - lower) for fraction in [0.25, 0.5, 0.75])
    queries = np.unique(queries)[::-1]
    X, y = _design_from_statistics(scores, diagonal)
    reference = laminar_group_linf_regression_path(
        X,
        y,
        _descendant_groups(parents),
        queries,
        fit_intercept=False,
        group_weights=weights,
        tol=1e-10,
    )
    assert reference.metadata["point_solutions_certified"]
    interpolated = np.vstack([path.at(float(lam))[0] for lam in queries])
    assert_allclose(interpolated, reference.coefficients, rtol=1e-8, atol=2e-9)

    structural = tree_group_linf_exact_topology_path(
        scores, parents, group_weights=weights
    )
    for knot in structural.lambdas[structural.lambdas > 0.0]:
        assert np.any(np.isclose(path.lambdas, knot, rtol=1e-8, atol=1e-10))


def test_tied_forest_batches_simultaneous_events():
    # Each component has the same saturation event, with opposite signs.
    scores = np.array([1.0, 2.0, -1.0, -2.0])
    diagonal = np.ones(4)
    path = tree_group_linf_exact_coefficient_path(
        scores, diagonal, [-1, 0, -1, 2]
    )
    _assert_complete(path, scores, diagonal)
    assert_allclose(path.lambdas, [1.5, 0.5, 0.0], atol=1e-10)
    assert_allclose(
        path.coefficients,
        [[0, 0, 0, 0], [1, 1, -1, -1], [1, 2, -1, -2]],
        atol=1e-10,
    )


def test_shared_saturation_and_structural_event_handles_one_ulp_roundoff():
    # The chain saturates its first coefficient at lambda .45, exactly when
    # the independent root activates. Computing the saturation boundary from
    # its affine face produces .45000000000000007 in float64; treating this as
    # a separate structural interval leaves no representable interior probe.
    scores = np.array([0.1, 1.0, 0.45])
    path = tree_group_linf_exact_coefficient_path(
        scores, np.ones(3), [-1, 0, -1]
    )
    _assert_complete(path, scores, np.ones(3))
    assert_allclose(path.lambdas, [0.55, 0.45, 0.0], atol=1e-15)
    assert_allclose(
        path.coefficients,
        [[0, 0, 0], [0.1, 0.1, 0], [0.1, 1.0, 0.45]],
        atol=1e-15,
    )
    assert_allclose(path.at(0.5)[0], [0.05, 0.05, 0.0], atol=1e-15)
    assert_allclose(path.at(0.2)[0], [0.1, 0.6, 0.25], atol=1e-15)


def test_arbitrary_node_order_preserves_entire_path():
    scores = np.array([2.0, -1.0, 0.0, 0.3, -0.6, 0.9])
    diagonal = np.array([4.0, 0.4, 0.7, 1.5, 0.9, 2.0])
    parents = np.array([-1, 0, 0, 1, -1, 4])
    weights = np.array([1.2, 0.7, 1.0, 0.4, 1.1, 0.8])
    reference = tree_group_linf_exact_coefficient_path(
        scores, diagonal, parents, group_weights=weights
    )
    order = np.array([3, 5, 1, 4, 2, 0])
    inverse = np.argsort(order)
    reordered_parents = np.array(
        [-1 if parents[node] < 0 else inverse[parents[node]] for node in order]
    )
    path = tree_group_linf_exact_coefficient_path(
        scores[order], diagonal[order], reordered_parents, group_weights=weights[order]
    )
    _assert_complete(path, scores[order], diagonal[order])
    assert reference.exact
    assert_allclose(path.lambdas, reference.lambdas, atol=1e-10)
    assert_allclose(path.coefficients[:, inverse], reference.coefficients, atol=1e-9)


def test_zero_score_ancestor_stays_zero_as_descendant_activates():
    path = tree_group_linf_exact_coefficient_path([0.0, 2.0], [3.0, 1.0], [-1, 0])
    _assert_complete(path, [0.0, 2.0], np.array([3.0, 1.0]))
    assert_allclose(path.lambdas, [1.0, 0.0], atol=1e-10)
    assert_allclose(path.at(0.5)[0], [0.0, 1.0], atol=1e-10)


def test_all_zero_scores_have_a_complete_constant_path():
    path = tree_group_linf_exact_coefficient_path([0, 0, 0], [1, 2, 3], [-1, 0, 0])
    _assert_complete(path, [0, 0, 0], np.array([1, 2, 3]))
    assert_allclose(path.lambdas, [0.0])
    assert_allclose(path.coefficients, [[0, 0, 0]])


def test_empty_forest_returns_one_empty_solution():
    path = tree_group_linf_exact_coefficient_path([], [], [])
    assert path.exact
    assert path.status == "complete"
    assert_allclose(path.lambdas, [0.0])
    assert path.coefficients.shape == (1, 0)
    assert path.at(0.0)[0].shape == (0,)


@pytest.mark.parametrize("scale", [1e-60, 1e60])
def test_common_quadratic_rescaling_preserves_coefficients_and_scales_knots(scale):
    scores = np.array([2.0, 1.0]) * scale
    diagonal = np.array([4.0, 1.0]) * scale
    path = tree_group_linf_exact_coefficient_path(scores, diagonal, [-1, 0])
    _assert_complete(path, scores, diagonal)
    assert_allclose(path.lambdas / scale, [2.0, 1.0, 2.0 / 3.0, 0.25, 0.0], atol=1e-10)
    assert_allclose(
        path.coefficients,
        [[0, 0], [0.25, 0], [1.0 / 3, 1.0 / 3], [0.5, 0.5], [0.5, 1]],
        atol=1e-10,
    )


def test_nearby_distinct_knots_do_not_skip_a_short_affine_interval():
    separation = 1e-7
    scores = np.array([1.0, 1.0 + separation])
    path = tree_group_linf_exact_coefficient_path(scores, np.ones(2), [-1, -1])
    _assert_complete(path, scores, np.ones(2))
    assert_allclose(path.lambdas, [1.0 + separation, 1.0, 0.0], rtol=0.0, atol=1e-11)
    assert_allclose(path.at(1.0 + separation / 2)[0], [0.0, separation / 2], atol=1e-11)


@pytest.mark.parametrize("separation", [1e-13, 1e-14, 1e-15])
def test_near_machine_precision_events_are_retained_or_fail_closed(separation):
    # A small coefficient discrepancy is not evidence that every knot was
    # enumerated. An unresolved narrow interval must not silently become an
    # exact path with just the two surrounding endpoints.
    scores = np.array([1.0, 1.0 + separation])
    path = tree_group_linf_exact_coefficient_path(scores, np.ones(2), [-1, -1])
    if not path.exact:
        assert path.status != "complete"
        return
    assert path.lambdas.size == 3
    assert_allclose(
        path.lambdas,
        [scores[1], scores[0], 0.0],
        rtol=0.0,
        atol=2 * np.finfo(float).eps,
    )
    midpoint = 0.5 * (scores[0] + scores[1])
    expected = np.maximum(scores - midpoint, 0.0)
    assert_allclose(
        path.at(midpoint)[0], expected, rtol=0.0, atol=2 * np.finfo(float).eps
    )


@pytest.mark.parametrize(
    "scores",
    [
        [1.0, np.nextafter(1.0, np.inf)],
        [0.5, np.nextafter(0.5, np.inf), 1.0],
    ],
)
def test_adjacent_float_structural_knots_are_both_retained(scores):
    # The third root fixes normalization at one, so the second case has no
    # representable interior point between two normalized event coordinates.
    scores = np.asarray(scores)
    path = tree_group_linf_exact_coefficient_path(
        scores, np.ones(scores.size), np.full(scores.size, -1)
    )
    _assert_complete(path, scores, np.ones(scores.size))
    expected_knots = np.r_[np.sort(scores)[::-1], 0.0]
    assert_allclose(path.lambdas, expected_knots, rtol=0.0, atol=0.0)
    expected_coefficients = np.maximum(scores[None, :] - expected_knots[:, None], 0.0)
    assert_allclose(
        path.coefficients,
        expected_coefficients,
        rtol=0.0,
        atol=2 * np.finfo(float).eps,
    )


def test_fully_tied_chain_has_one_affine_interval():
    p = 64
    path = tree_group_linf_exact_coefficient_path(
        np.ones(p), np.ones(p), np.arange(p) - 1
    )
    _assert_complete(path, np.ones(p), np.ones(p))
    assert_allclose(path.lambdas, [1.0, 0.0], atol=1e-10)
    assert_allclose(path.at(0.3)[0], np.full(p, 0.7), atol=1e-10)


def test_real_diabetes_tree_completes_despite_rounded_adjacent_face_boundaries():
    # In this fitted tree, equivalent formulas for an interior boundary differ
    # by 39 float64 ulps. Both complete coverage and accurate interpolation
    # matter; a blanket two/32-ulp adjacency cutoff stops a valid path early.
    dataset = load_diabetes()
    order = np.random.default_rng(0).permutation(dataset.target.size)
    X = dataset.data[order]
    y = dataset.target[order]
    y = (y - y.mean()) / y.std()
    tree = DecisionTreeRegressor(
        max_leaf_nodes=32, min_samples_leaf=2, random_state=0
    ).fit(X, y)
    structural = fitted_tree_linf_exact_topology_path(tree)
    scores = np.asarray(structural.metadata["linear_scores"])
    diagonal = np.asarray(structural.metadata["gram_diagonal"])
    parents = np.asarray(structural.metadata["parent_indices"])
    path = tree_group_linf_exact_coefficient_path(scores, diagonal, parents)
    _assert_complete(path, scores, diagonal)

    queries = np.unique(
        np.r_[path.lambdas, 0.5 * (path.lambdas[:-1] + path.lambdas[1:])]
    )[::-1]
    design, response = _design_from_statistics(scores, diagonal)
    reference = laminar_group_linf_regression_path(
        design,
        response,
        _descendant_groups(parents),
        queries,
        fit_intercept=False,
        tol=1e-10,
    )
    assert reference.metadata["point_solutions_certified"]
    assert_allclose(
        np.vstack([path.at(float(lam))[0] for lam in queries]),
        reference.coefficients,
        rtol=1e-8,
        atol=2e-9,
    )


@pytest.mark.parametrize(
    "scores, diagonal, parents, options",
    [
        ([1.0, 2.0], [1.0], [-1, 0], {}),
        ([1.0], [1.0], [-1, 0], {}),
        ([np.nan], [1.0], [-1], {}),
        ([np.inf], [1.0], [-1], {}),
        ([1.0j], [1.0], [-1], {}),
        ([[1.0]], [1.0], [-1], {}),
        ([1.0], [0.0], [-1], {}),
        ([1.0], [-1.0], [-1], {}),
        ([1.0], [np.inf], [-1], {}),
        ([1.0], [1.0j], [-1], {}),
        ([1.0], [1.0], [0], {}),
        ([1.0, 2.0], [1.0, 1.0], [1, 0], {}),
        ([1.0], [1.0], [2], {}),
        ([1.0], [1.0], [-1], {"group_weights": [0.0]}),
        ([1.0], [1.0], [-1], {"group_weights": [-1.0]}),
        ([1.0], [1.0], [-1], {"group_weights": [np.nan]}),
        ([1.0], [1.0], [-1], {"group_weights": [1.0j]}),
        ([1.0], [1.0], [-1], {"tolerance": 0.0}),
        ([1.0], [1.0], [-1], {"tolerance": np.nan}),
        ([1.0], [1.0], [-1], {"max_events": 0}),
        ([1.0], [1.0], [-1], {"max_events": 1.5}),
        ([1.0], [1.0], [-1], {"max_events": True}),
    ],
)
def test_invalid_inputs_are_rejected(scores, diagonal, parents, options):
    with pytest.raises(ValueError):
        tree_group_linf_exact_coefficient_path(scores, diagonal, parents, **options)


def test_oracle_numeric_range_failure_returns_an_explicit_nonexact_prefix(monkeypatch):
    from imodels.tree.sparse_pruning.optimization.tree_prox import LaminarGroupLinfProx

    def unavailable_point(*args, **kwargs):
        raise ValueError("positive group radius underflowed")

    monkeypatch.setattr(LaminarGroupLinfProx, "__call__", unavailable_point)
    path = tree_group_linf_exact_coefficient_path([1.0, 2.0], [1.0, 1.0], [-1, 0])
    assert not path.exact
    assert path.status == "numerical_failure"
    assert "underflowed" in path.metadata["failure_message"]
    assert not path.metadata["point_solutions_certified"]


def test_event_budget_exhaustion_does_not_claim_an_exact_complete_path():
    path = tree_group_linf_exact_coefficient_path(
        [2.0, 1.0], [4.0, 1.0], [-1, 0], max_events=1
    )
    assert not path.exact
    assert path.status != "complete"
    assert np.all(np.isfinite(path.coefficients))
