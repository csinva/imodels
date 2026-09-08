from copy import deepcopy

import numpy as np
import pytest
from scipy.optimize import LinearConstraint, minimize
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree._tree import TREE_LEAF

from imodels.importance.local_stumps import make_stumps, tree_feature_transform
from imodels.tree.sparse_pruning import (
    fitted_tree_linf_exact_topology_path,
    materialize_fitted_tree_topology,
)
from imodels.tree.sparse_pruning.optimization import (
    laminar_group_linf_exact_topology_path,
    laminar_group_linf_regression_path,
    tree_group_linf_exact_topology_path,
)
from imodels.tree.sparse_pruning.optimization.topology import (
    TreeTopologyPath,
    _tree_isotonic_activation_lambdas,
)


@pytest.mark.parametrize("field", ["X", "y", "sample_weight"])
def test_exact_topology_rejects_complex_inputs_before_float_conversion(field):
    inputs = dict(
        X=np.array([[1.0], [2.0], [3.0]]),
        y=np.array([1.0, 2.0, 3.0]),
        groups=[np.array([0])],
        sample_weight=np.ones(3),
    )
    inputs[field] = inputs[field].astype(complex) + 1j
    with pytest.raises(ValueError, match=f"{field} must contain real values"):
        laminar_group_linf_exact_topology_path(**inputs)


def test_unsigned_parent_index_cannot_wrap_to_the_root_sentinel():
    parents = np.array([np.iinfo(np.uint64).max], dtype=np.uint64)
    with pytest.raises(ValueError, match="index out of range"):
        tree_group_linf_exact_topology_path([1.0], parents)


def _tree_groups(parents):
    parents = np.asarray(parents, dtype=int)
    descendants = [[node] for node in range(parents.size)]
    for node in range(parents.size - 1, 0, -1):
        descendants[int(parents[node])].extend(descendants[node])
    return [np.asarray(values, dtype=int) for values in descendants]


def _child_before_parent(parents):
    """Reindex a parent-before-child tree into child-before-parent order."""
    parents = np.asarray(parents, dtype=int)
    order = np.arange(parents.size - 1, -1, -1)
    inverse = np.empty_like(order)
    inverse[order] = np.arange(order.size)
    reordered = np.full_like(parents, -1)
    for position, source in enumerate(order):
        parent = parents[source]
        if parent >= 0:
            reordered[position] = inverse[parent]
    return order, reordered


def test_leftist_tree_pava_matches_independent_quadratic_programs():
    rng = np.random.default_rng(91)
    for n_nodes in range(2, 13):
        for _ in range(8):
            original_parents = np.r_[
                -1, [rng.integers(node) for node in range(1, n_nodes)]
            ]
            order, parents = _child_before_parent(original_parents)
            weights = rng.uniform(0.2, 2.0, size=n_nodes)[order]
            observations = rng.uniform(0.0, 3.0, size=n_nodes)[order]
            masses = weights * observations

            actual = _tree_isotonic_activation_lambdas(
                parents, masses, weights
            )
            rows = np.zeros((n_nodes - 1, n_nodes))
            row = 0
            for child, parent in enumerate(parents):
                if parent >= 0:
                    rows[row, parent] = 1.0
                    rows[row, child] = -1.0
                    row += 1
            result = minimize(
                lambda value: 0.5
                * float(weights @ ((value - observations) ** 2)),
                observations,
                jac=lambda value: weights * (value - observations),
                constraints=LinearConstraint(rows, 0.0, np.inf),
                method="SLSQP",
                options={"ftol": 1e-13, "maxiter": 2_000},
            )

            assert result.success, result.message
            np.testing.assert_allclose(actual, result.x, atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize(
    "scores, weights, expected",
    [
        ([5e307, 1e308], [1e308, 1e308], 0.75),  # Weight sum overflows.
        ([1e308, 1.5e308], [1e308, 1e308], 1.25),  # Both sums overflow.
        ([1e308, 1.5e308], [1.0, 1.0], 1.25e308),  # Mass sum overflows.
        ([0.0, 1.0], [1.0, 1e-309], 1.0),  # Only the unpooled ratio overflows.
    ],
)
def test_exact_topology_safely_pools_extreme_finite_inputs(scores, weights, expected):
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        path = tree_group_linf_exact_topology_path(
            scores, [-1, 0], group_weights=weights
        )

    assert path.exact
    assert path.status == "complete"
    np.testing.assert_allclose(path.activation_lambdas, [expected, expected])
    assert path.entering_groups == ((0, 1), ())


def test_exact_topology_all_zero_scores_need_no_weight_pooling():
    path = tree_group_linf_exact_topology_path(
        [0.0, 0.0], [-1, 0], group_weights=[1e308, 1e-308]
    )
    assert path.exact
    np.testing.assert_array_equal(path.activation_lambdas, [0.0, 0.0])
    np.testing.assert_array_equal(path.lambdas, [0.0])


def test_exact_topology_keeps_representable_subnormal_activation():
    smallest = np.nextafter(0.0, 1.0)
    path = tree_group_linf_exact_topology_path([smallest], [-1])
    assert path.exact
    np.testing.assert_array_equal(path.activation_lambdas, [smallest])
    assert path.topology_at(0.0) == (0,)


@pytest.mark.parametrize(
    "scores, weights, parents",
    [
        ([np.nextafter(0.0, 1.0)], [2.0], [-1]),  # Positive knot rounds to zero.
        ([0.0, np.nextafter(0.0, 1.0)], [1.0, 1.0], [-1, 0]),  # Pool underflows.
        ([1.0], [1e-309], [-1]),  # Division overflows without large sums.
        ([1e308], [1e-308], [-1]),  # Knot overflows.
        ([1e-308, 1e308], [1e-308, 1e308], [-1, -1]),  # Unsafe shared scaling.
    ],
)
def test_exact_topology_rejects_unrepresentable_numeric_ranges(scores, weights, parents):
    with pytest.raises(ValueError, match="represent"):
        tree_group_linf_exact_topology_path(scores, parents, group_weights=weights)


@pytest.mark.parametrize("seed", range(5))
def test_exact_topology_knots_match_exact_diagonal_point_solutions(seed):
    rng = np.random.default_rng(seed)
    n_features = 9
    parents = np.r_[
        -1, [rng.integers(node) for node in range(1, n_features)]
    ]
    groups = _tree_groups(parents)
    diagonal = rng.uniform(0.1, 2.0, size=n_features)
    linear = rng.normal(size=n_features)
    group_weights = rng.uniform(0.2, 1.5, size=n_features)
    X = np.diag(np.sqrt(n_features * diagonal))
    y = linear * np.sqrt(n_features) / np.sqrt(diagonal)

    topology = laminar_group_linf_exact_topology_path(
        X,
        y,
        groups,
        fit_intercept=False,
        group_weights=group_weights,
        include_coefficients=True,
    )
    boundaries = np.unique(np.r_[topology.lambdas, topology.activation_lambdas])
    midpoints = 0.5 * (boundaries[:-1] + boundaries[1:])
    queries = np.unique(np.r_[midpoints, 0.0, topology.lambdas[0] * 1.1])[::-1]
    points = laminar_group_linf_regression_path(
        X,
        y,
        groups,
        queries,
        fit_intercept=False,
        group_weights=group_weights,
        tol=1e-11,
    )

    assert topology.exact
    assert topology.status == "complete"
    assert topology.metadata["topology_knots_exact"] is True
    assert topology.metadata["coefficient_points_certified"] is True
    assert topology.metadata["coefficient_status"] == "certified"
    assert topology.metadata["exactness_condition"] == (
        "numerically_verified_diagonal_gram"
    )
    assert topology.n_knots == np.unique(
        topology.activation_lambdas[topology.activation_lambdas > 0.0]
    ).size
    for lam, beta in zip(points.lambdas, points.coefficients):
        active = np.abs(beta) > 1e-9
        active_groups = tuple(
            group_number
            for group_number, group in enumerate(groups)
            if np.any(active[group])
        )
        assert active_groups == topology.topology_at(float(lam))

    knot_points = laminar_group_linf_regression_path(
        X,
        y,
        groups,
        topology.lambdas,
        fit_intercept=False,
        group_weights=group_weights,
        tol=1e-11,
    )
    np.testing.assert_allclose(
        topology.coefficients, knot_points.coefficients, atol=2e-11
    )
    assert list(topology.iter_topologies()) == [
        (float(lam), topology.topology_at(float(lam)))
        for lam in topology.lambdas
    ]
    assert list(topology.iter_topologies(below=True)) == [
        (float(lam), topology.topology_at(float(lam), below=True))
        for lam in topology.lambdas
    ]
    assert sum(len(batch) for _, batch in topology.iter_events()) == int(
        np.count_nonzero(topology.activation_lambdas > 0.0)
    )
    assert list(topology.iter_pruning_events()) == list(
        reversed(list(topology.iter_events()))
    )


def test_zero_score_ancestor_is_retained_by_descendant_topology():
    diagonal = np.ones(2)
    linear = np.array([0.0, 2.0])
    X = np.diag(np.sqrt(2.0 * diagonal))
    y = linear * np.sqrt(2.0)
    groups = [np.array([0, 1]), np.array([1])]

    path = laminar_group_linf_exact_topology_path(
        X, y, groups, fit_intercept=False, include_coefficients=True
    )

    np.testing.assert_allclose(path.activation_lambdas, [1.0, 1.0])
    knot = float(path.lambdas[0])
    assert path.topology_at(knot) == ()
    assert path.topology_at(knot, below=True) == (0, 1)
    assert path.entering_groups[0] == (0, 1)
    assert path.entering_nodes == path.entering_groups
    assert path.topology_at(0.5) == (0, 1)
    # The ancestor coefficient itself remains zero, but its group/node must be
    # retained in order to realize the active descendant.
    beta = laminar_group_linf_regression_path(
        X, y, groups, [0.5], fit_intercept=False
    ).coefficients[0]
    assert beta[0] == 0.0
    assert beta[1] > 0.0


def test_zero_activation_nodes_are_absent_from_initial_pruning_topology():
    path = tree_group_linf_exact_topology_path(
        [2.0, 0.0, 0.0], [-1, 0, 0], node_ids=[10, 20, 30]
    )

    assert path.tree_nodes_at(0.0) == (10,)
    assert list(path.iter_node_pruning_events()) == [(2.0, (10,))]


def test_exact_topology_rejects_nonpositive_group_weight():
    X = np.eye(2)
    y = np.ones(2)
    with pytest.raises(ValueError, match="positive group weights"):
        laminar_group_linf_exact_topology_path(
            X,
            y,
            [np.array([0, 1]), np.array([1])],
            fit_intercept=False,
            group_weights=[1.0, 0.0],
        )


def test_exact_group_topology_documents_uncovered_unpenalized_feature():
    X = np.eye(2) * np.sqrt(2.0)
    y = np.array([1.0, 2.0]) * np.sqrt(2.0)

    path = laminar_group_linf_exact_topology_path(
        X,
        y,
        [np.array([0])],
        fit_intercept=False,
        include_coefficients=True,
    )

    assert path.metadata["n_uncovered_features"] == 1
    assert path.topology_at(float(path.lambdas[0])) == ()
    assert path.coefficients[0, 1] != 0.0


def test_score_only_tree_path_supports_arbitrary_order_and_external_node_ids():
    parents = np.array([2, -1, -1, 2])
    scores = np.array([0.4, -0.8, 0.1, 0.2])
    groups = [
        np.array([0]),
        np.array([1]),
        np.array([0, 2, 3]),
        np.array([3]),
    ]
    X = np.eye(4) * 2.0
    y = scores * 2.0
    expected = laminar_group_linf_exact_topology_path(
        X, y, groups, fit_intercept=False
    )

    actual = tree_group_linf_exact_topology_path(
        scores, parents, node_ids=[10, 11, 12, 13]
    )

    np.testing.assert_allclose(
        actual.activation_lambdas, expected.activation_lambdas
    )
    assert actual.metadata["descendant_groups_materialized"] is False
    assert actual.tree_nodes_at(float(actual.lambdas[0]), below=True) == (
        11,
    )
    first_lambda, first_nodes = next(actual.iter_node_events())
    assert first_lambda == actual.lambdas[0]
    assert first_nodes == (11,)
    assert list(actual.iter_node_pruning_events()) == list(
        reversed(list(actual.iter_node_events()))
    )


def test_score_only_tree_path_rejects_complex_inputs():
    with pytest.raises(ValueError, match="real vector"):
        tree_group_linf_exact_topology_path([1.0 + 2.0j], [-1])
    with pytest.raises(ValueError, match="group_weights"):
        tree_group_linf_exact_topology_path(
            [1.0], [-1], group_weights=[1.0 + 2.0j]
        )


def test_fitted_tree_path_handles_a_no_split_tree():
    estimator = DecisionTreeRegressor(random_state=0).fit(
        np.zeros((8, 1)), np.ones(8)
    )

    path = fitted_tree_linf_exact_topology_path(estimator)

    np.testing.assert_array_equal(path.lambdas, [0.0])
    assert path.activation_lambdas.size == 0
    assert path.node_ids.size == 0
    assert path.n_knots == 0
    assert path.n_states == 1
    assert path.topology_at(0.0) == ()
    assert list(path.iter_node_events()) == []
    assert list(path.iter_node_pruning_events()) == []
    assert path.metadata["n_internal_nodes"] == 0

    empty = tree_group_linf_exact_topology_path([], [], node_ids=[])
    np.testing.assert_array_equal(empty.lambdas, [0.0])
    assert empty.tree_nodes_at(0.0) == ()


def test_materialized_fitted_tree_topologies_are_reachable_and_nonmutating():
    rng = np.random.default_rng(841)
    X = rng.normal(size=(160, 4))
    y = X[:, 0] * X[:, 1] - X[:, 2] + 0.1 * rng.normal(size=160)
    estimator = DecisionTreeRegressor(
        max_leaf_nodes=12, min_samples_leaf=3, random_state=3
    ).fit(X, y)
    original_left = estimator.tree_.children_left.copy()
    original_right = estimator.tree_.children_right.copy()
    path = fitted_tree_linf_exact_topology_path(estimator)

    for lam in path.lambdas:
        rendered = materialize_fitted_tree_topology(
            estimator, path, float(lam)
        )
        reachable_internal = []
        pending = [0]
        while pending:
            node_id = pending.pop()
            left = int(rendered.tree_.children_left[node_id])
            right = int(rendered.tree_.children_right[node_id])
            if left != TREE_LEAF:
                reachable_internal.append(node_id)
                pending.extend([left, right])
        assert tuple(sorted(reachable_internal)) == tuple(
            sorted(path.tree_nodes_at(float(lam)))
        )
        assert np.all(np.isfinite(rendered.predict(X)))

    np.testing.assert_array_equal(estimator.tree_.children_left, original_left)
    np.testing.assert_array_equal(estimator.tree_.children_right, original_right)


def test_materialized_topology_rejects_an_unrelated_same_shaped_tree():
    rng = np.random.default_rng(962)
    X = rng.normal(size=(1_000, 3))
    first = DecisionTreeRegressor(max_depth=3, random_state=0).fit(X, X[:, 0])
    second = DecisionTreeRegressor(max_depth=3, random_state=0).fit(X, X[:, 1])
    first_path = fitted_tree_linf_exact_topology_path(first)
    second_path = fitted_tree_linf_exact_topology_path(second)
    np.testing.assert_array_equal(first_path.node_ids, second_path.node_ids)
    assert first_path.metadata["fitted_tree_fingerprint"] != (
        second_path.metadata["fitted_tree_fingerprint"]
    )

    # A true copy remains compatible because the fingerprint binds fitted
    # state, rather than Python object identity.
    materialize_fitted_tree_topology(deepcopy(first), first_path, 0.0)
    with pytest.raises(ValueError, match="different or subsequently modified"):
        materialize_fitted_tree_topology(second, first_path, 0.0)


def test_fitted_tree_sufficient_statistics_match_weighted_stump_design():
    rng = np.random.default_rng(514)
    X = rng.normal(size=(120, 4))
    y = X[:, 0] * X[:, 1] - 0.4 * X[:, 2] + rng.normal(
        scale=0.1, size=120
    )
    sample_weight = rng.uniform(0.2, 2.0, size=120)
    estimator = DecisionTreeRegressor(
        max_leaf_nodes=12, min_samples_leaf=3, random_state=7
    ).fit(X, y, sample_weight=sample_weight)
    tree = estimator.tree_

    node_ids = []
    parents = []

    def visit(node_id, parent):
        if tree.children_left[node_id] < 0:
            return
        position = len(node_ids)
        node_ids.append(int(node_id))
        parents.append(parent)
        visit(int(tree.children_left[node_id]), position)
        visit(int(tree.children_right[node_id]), position)

    visit(0, -1)
    groups = _tree_groups(parents)
    stump_design = tree_feature_transform(make_stumps(tree), X)
    expected = laminar_group_linf_exact_topology_path(
        stump_design,
        y,
        groups,
        sample_weight=sample_weight,
        fit_intercept=True,
    )
    actual = fitted_tree_linf_exact_topology_path(estimator)

    np.testing.assert_array_equal(actual.node_ids, node_ids)
    np.testing.assert_allclose(
        actual.activation_lambdas,
        expected.activation_lambdas,
        atol=2e-14,
        rtol=2e-14,
    )
    normalized_weight = sample_weight / sample_weight.sum()
    expected_scores = stump_design.T @ (normalized_weight * y)
    expected_diagonal = np.einsum(
        "ij,i,ij->j", stump_design, normalized_weight, stump_design
    )
    np.testing.assert_allclose(actual.metadata["linear_scores"], expected_scores)
    np.testing.assert_allclose(actual.metadata["gram_diagonal"], expected_diagonal)
    assert actual.metadata["training_design_materialized"] is False


def test_fitted_tree_path_is_invariant_to_extreme_weight_rescaling():
    X = np.arange(24.0).reshape(-1, 1)
    y = np.where(X[:, 0] < 12.0, -1.0, 2.0)
    estimator = DecisionTreeRegressor(max_depth=3, random_state=0).fit(X, y)
    reference = fitted_tree_linf_exact_topology_path(estimator)

    for scale in (1e-200, 1e200):
        scaled = deepcopy(estimator)
        scaled.tree_.weighted_n_node_samples[:] *= scale
        observed = fitted_tree_linf_exact_topology_path(scaled)
        np.testing.assert_allclose(
            observed.activation_lambdas,
            reference.activation_lambdas,
            atol=0.0,
            rtol=5e-15,
        )
        np.testing.assert_allclose(
            observed.metadata["linear_scores"],
            reference.metadata["linear_scores"],
            atol=0.0,
            rtol=5e-15,
        )
        normalized_design = tree_feature_transform(
            make_stumps(scaled.tree_, normalize=True), X
        )
        np.testing.assert_allclose(
            normalized_design.T @ (scale * normalized_design),
            np.eye(normalized_design.shape[1]),
            atol=2e-15,
            rtol=2e-15,
        )


def test_fitted_tree_score_is_stable_under_a_large_response_translation():
    X = np.r_[np.zeros(10), np.ones(90)].reshape(-1, 1)
    baseline = 1e15
    contrast = np.spacing(baseline)
    y = np.r_[np.full(10, baseline), np.full(90, baseline + contrast)]
    estimator = DecisionTreeRegressor(max_depth=1, random_state=0).fit(X, y)

    path = fitted_tree_linf_exact_topology_path(estimator)

    assert path.metadata["linear_scores"][0] == pytest.approx(0.0375)
    assert path.n_knots == 1


def test_fitted_tree_score_preserves_tiny_weight_ratio_with_large_contrast():
    X = np.array([[0.0], [1.0]])
    y = np.array([1e162, 0.0])
    sample_weight = np.array([1e-300, 1e24])
    estimator = DecisionTreeRegressor(max_depth=1, random_state=0).fit(
        X, y, sample_weight=sample_weight
    )

    path = fitted_tree_linf_exact_topology_path(estimator)

    assert path.metadata["linear_scores"][0] == pytest.approx(-1.0)
    np.testing.assert_allclose(path.lambdas, [1.0, 0.0])


def test_fitted_tree_statistics_reject_non_mean_criterion():
    X = np.arange(20.0).reshape(-1, 1)
    y = np.sin(X[:, 0])
    estimator = DecisionTreeRegressor(
        criterion="absolute_error", max_depth=2, random_state=0
    ).fit(X, y)

    with pytest.raises(ValueError, match="mean-based"):
        fitted_tree_linf_exact_topology_path(estimator)


def test_fitted_tree_statistics_reject_monotonic_constraints():
    X = np.arange(30.0).reshape(-1, 1)
    y = -X[:, 0]
    try:
        estimator = DecisionTreeRegressor(
            max_depth=2, random_state=0, monotonic_cst=[1]
        ).fit(X, y)
    except TypeError:  # pragma: no cover - older supported sklearn releases
        pytest.skip("sklearn release has no monotonic tree constraints")

    with pytest.raises(ValueError, match="monotonic constraints"):
        fitted_tree_linf_exact_topology_path(estimator)


@pytest.mark.parametrize(
    "field", ["lambdas", "activation_lambdas", "coefficients", "intercepts"]
)
def test_topology_result_rejects_complex_arrays(field):
    values = dict(
        lambdas=np.array([1.0, 0.0]), activation_lambdas=np.array([1.0]),
        entering_groups=((0,), ()), coefficients=np.array([[0.0], [1.0]]),
        intercepts=np.zeros(2),
    )
    values[field] = values[field].astype(complex) + 1j
    with pytest.raises(ValueError, match="real numeric"):
        TreeTopologyPath(**values)
