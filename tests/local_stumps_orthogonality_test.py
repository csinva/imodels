import numpy as np
import pytest
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree._tree import TREE_LEAF

from imodels.importance.local_stumps import (
    LocalDecisionStump, make_stumps, tree_feature_transform,
)
from imodels.tree.sparse_pruning import (
    fitted_tree_linf_exact_coefficient_path,
    fitted_tree_linf_exact_topology_path,
)


def _internal_node_ids_preorder(tree):
    node_ids = []

    def visit(node_id):
        if node_id == TREE_LEAF or tree.feature[node_id] < 0:
            return
        node_ids.append(node_id)
        visit(int(tree.children_left[node_id]))
        visit(int(tree.children_right[node_id]))

    visit(0)
    return np.asarray(node_ids, dtype=int)


def test_training_tree_stumps_have_diagonal_weighted_gram():
    rng = np.random.default_rng(27)
    X = rng.normal(size=(96, 4))
    y = X[:, 0] * X[:, 1] + 0.2 * rng.normal(size=X.shape[0])
    weights = rng.uniform(0.2, 3.0, size=X.shape[0])
    estimator = DecisionTreeRegressor(
        max_leaf_nodes=12, min_samples_leaf=3, random_state=4
    ).fit(X, y, sample_weight=weights)
    tree = estimator.tree_
    node_ids = _internal_node_ids_preorder(tree)

    unnormalized = tree_feature_transform(make_stumps(tree), X)
    weighted_means = weights @ unnormalized
    weighted_gram = unnormalized.T @ (weights[:, None] * unnormalized)
    expected_diagonal = tree.weighted_n_node_samples[node_ids]

    np.testing.assert_allclose(weighted_means, 0.0, atol=2e-14)
    np.testing.assert_allclose(
        weighted_gram,
        np.diag(expected_diagonal),
        rtol=2e-14,
        atol=2e-14,
    )

    normalized = tree_feature_transform(make_stumps(tree, normalize=True), X)
    normalized_gram = normalized.T @ (weights[:, None] * normalized)
    np.testing.assert_allclose(
        normalized_gram,
        np.eye(node_ids.size),
        rtol=2e-14,
        atol=2e-14,
    )

    # Orthogonality is with respect to the measure that built the stump
    # values. Replacing nonuniform fitting weights by unit weights generally
    # breaks ancestor/descendant orthogonality.
    mismatched_gram = unnormalized.T @ unnormalized
    mismatched_off_diagonal = mismatched_gram - np.diag(
        np.diag(mismatched_gram)
    )
    assert np.max(np.abs(mismatched_off_diagonal)) > 1e-3


def test_training_tree_stumps_follow_sklearn_missing_value_routes():
    X = np.array(
        [[np.nan], [0.0], [1.0], [2.0], [3.0], [np.nan], [4.0], [5.0],
         [6.0], [7.0]]
    )
    y = np.array([4.0, -2.0, -1.0, 0.0, 1.0, 5.0, 2.0, 3.0, 4.0, 6.0])
    weights = np.array([0.5, 1.0, 2.0, 0.7, 1.3, 1.1, 0.8, 1.7, 0.9, 1.2])
    try:
        estimator = DecisionTreeRegressor(
            max_depth=3, random_state=3
        ).fit(X, y, sample_weight=weights)
    except ValueError:  # pragma: no cover - sklearn without native NaN routing
        pytest.skip("installed sklearn tree does not support missing values")
    tree = estimator.tree_
    node_ids = _internal_node_ids_preorder(tree)

    transformed = tree_feature_transform(make_stumps(tree), X)
    weighted_gram = transformed.T @ (weights[:, None] * transformed)

    np.testing.assert_allclose(weights @ transformed, 0.0, atol=3e-14)
    np.testing.assert_allclose(
        weighted_gram,
        np.diag(tree.weighted_n_node_samples[node_ids]),
        rtol=3e-14,
        atol=3e-14,
    )
    native_path = fitted_tree_linf_exact_topology_path(estimator)
    np.testing.assert_allclose(
        native_path.metadata["linear_scores"],
        transformed.T @ ((weights / weights.sum()) * y),
        rtol=3e-14,
        atol=3e-14,
    )


def _float32_midpoint_rows(base=4.0):
    lower = np.nextafter(np.float32(base), np.float32(np.inf))
    upper = np.nextafter(lower, np.float32(np.inf))
    midpoint = (float(lower) + float(upper)) / 2.0
    return np.array([[float(lower)], [midpoint], [float(upper)]]), midpoint


@pytest.mark.parametrize("base", [4.0, 32.0])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_sklearn_stumps_preserve_float32_routing_and_float64_thresholds(base, dtype):
    X, midpoint = _float32_midpoint_rows(base)
    X = X.astype(dtype)
    estimator = DecisionTreeRegressor(max_depth=1, random_state=0).fit(X, [0, 1, 1])
    assert estimator.tree_.threshold[0] == midpoint
    stumps = make_stumps(estimator.tree_)
    transformed = tree_feature_transform(stumps, X)

    np.testing.assert_array_equal(stumps[0](X), transformed[:, 0])
    np.testing.assert_allclose(transformed.mean(axis=0), 0.0, atol=3e-16)
    np.testing.assert_allclose(transformed.T @ transformed / len(X), [[1.0]])
    path = fitted_tree_linf_exact_coefficient_path(estimator)
    assert path.exact
    coefficients, intercept = path.at(0.0)
    np.testing.assert_allclose(
        intercept + transformed @ coefficients, estimator.predict(X), atol=3e-16
    )
    np.testing.assert_array_equal(estimator.predict(X), [0, 1, 1])


def test_descendant_stumps_share_float32_input_and_follow_ancestor_boundary(monkeypatch):
    boundary_rows, midpoint = _float32_midpoint_rows()
    X = np.column_stack((np.repeat(boundary_rows[:, 0], 2), np.tile([-1, 1], 3)))
    y = np.repeat([0, 10, 10], 2) + X[:, 1]
    estimator = DecisionTreeRegressor(max_depth=2, random_state=0).fit(X, y)
    assert estimator.tree_.feature[0] == 0
    assert estimator.tree_.threshold[0] == midpoint
    stumps = make_stumps(estimator.tree_)
    seen_inputs = []
    original_call = LocalDecisionStump.__call__

    def record_input(stump, data):
        seen_inputs.append(data)
        return original_call(stump, data)

    monkeypatch.setattr(LocalDecisionStump, "__call__", record_input)
    transformed = tree_feature_transform(stumps, X)
    assert len(seen_inputs) == 3
    assert all(data is seen_inputs[0] for data in seen_inputs)
    assert seen_inputs[0].dtype == np.float32
    node_ids = _internal_node_ids_preorder(estimator.tree_)
    np.testing.assert_allclose(transformed.mean(axis=0), 0.0, atol=3e-16)
    np.testing.assert_allclose(
        transformed.T @ transformed,
        np.diag(estimator.tree_.weighted_n_node_samples[node_ids]),
        atol=1e-15,
    )


def test_manual_stump_keeps_input_precision_in_mixed_tree_transform():
    X, midpoint = _float32_midpoint_rows()
    estimator = DecisionTreeRegressor(max_depth=1, random_state=0).fit(X, [0, 1, 1])
    derived = make_stumps(estimator.tree_)[0]
    manual = LocalDecisionStump(0, midpoint, 0.0, 1.0, [], [], [])
    transformed = tree_feature_transform([derived, manual], X)

    np.testing.assert_array_equal(manual(X), [0, 0, 1])
    np.testing.assert_array_equal(transformed[:, 1], manual(X))
    np.testing.assert_array_equal(transformed[:, 0], derived(X))
