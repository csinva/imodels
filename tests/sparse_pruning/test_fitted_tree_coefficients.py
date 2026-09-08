"""The fitted-tree wrapper must target the actual training stump objective."""

from copy import deepcopy

import numpy as np
import pytest
from sklearn.tree import DecisionTreeRegressor

from imodels.importance.local_stumps import make_stumps, tree_feature_transform
from imodels.tree.sparse_pruning import (
    fitted_tree_linf_exact_coefficient_path,
    fitted_tree_linf_exact_topology_path,
)
from imodels.tree.sparse_pruning.optimization import (
    laminar_group_linf_regression_path,
)


@pytest.mark.parametrize("weighted", [False, True])
def test_fitted_coefficient_path_matches_training_design_and_predictions(weighted):
    rng = np.random.default_rng(914)
    X = rng.normal(size=(96, 4))
    y = 4.0 + X[:, 0] * X[:, 1] - 0.6 * X[:, 2]
    weights = rng.uniform(0.2, 2.0, X.shape[0]) if weighted else None
    estimator = DecisionTreeRegressor(
        max_leaf_nodes=8, min_samples_leaf=4, random_state=7
    ).fit(X, y, sample_weight=weights)
    before = deepcopy(estimator.tree_.__getstate__())
    green = fitted_tree_linf_exact_topology_path(estimator)
    n_splits = green.node_ids.size
    group_weights = np.linspace(0.5, 1.5, n_splits)
    path = fitted_tree_linf_exact_coefficient_path(
        estimator, group_weights=group_weights, tolerance=1e-10
    )
    assert path.exact, path.metadata
    assert path.status == "complete"
    assert path.metadata["training_design_materialized"] is False
    assert path.metadata["local_stump_normalization"] == "unnormalized"
    np.testing.assert_array_equal(path.metadata["tree_node_ids"], green.node_ids)

    parents = np.asarray(green.metadata["parent_indices"])
    groups = [[node] for node in range(n_splits)]
    for child in range(n_splits - 1, -1, -1):
        if parents[child] >= 0:
            groups[parents[child]].extend(groups[child])
    groups = [np.asarray(group) for group in groups]
    np.testing.assert_allclose(path.penalties, [
        sum(w * np.max(np.abs(beta[group]))
            for w, group in zip(group_weights, groups))
        for beta in path.coefficients
    ])
    Z = tree_feature_transform(make_stumps(estimator.tree_), X)
    queries = np.unique(
        np.r_[path.lambdas, (path.lambdas[1:] + path.lambdas[:-1]) / 2]
    )[::-1]
    points = laminar_group_linf_regression_path(
        Z, y, groups, queries, sample_weight=weights,
        fit_intercept=True, group_weights=group_weights, tol=1e-10,
    )
    assert points.status == "complete"
    for lam, expected, intercept in zip(
        points.lambdas, points.coefficients, points.intercepts
    ):
        actual, actual_intercept = path.at(lam)
        np.testing.assert_allclose(actual, expected, rtol=2e-8, atol=2e-9)
        assert actual_intercept == pytest.approx(intercept, abs=2e-10)

    beta_zero, intercept_zero = path.at(0.0)
    np.testing.assert_allclose(
        intercept_zero + Z @ beta_zero, estimator.predict(X), atol=2e-10
    )
    assert np.all(path.intercepts == path.intercepts[0])
    after = estimator.tree_.__getstate__()
    np.testing.assert_array_equal(before["nodes"], after["nodes"])
    np.testing.assert_array_equal(before["values"], after["values"])


def test_fitted_coefficient_path_with_no_splits():
    estimator = DecisionTreeRegressor().fit(np.zeros((8, 2)), np.arange(8.0))
    path = fitted_tree_linf_exact_coefficient_path(estimator)
    assert path.exact
    assert path.status == "complete"
    np.testing.assert_array_equal(path.lambdas, [0.0])
    assert path.coefficients.shape == (1, 0)
    np.testing.assert_array_equal(path.intercepts, [3.5])
    beta, intercept = path.at(0.0)
    assert beta.size == 0
    assert intercept == 3.5


def test_fitted_coefficient_path_preserves_partial_status():
    rng = np.random.default_rng(47)
    X = rng.normal(size=(64, 3))
    estimator = DecisionTreeRegressor(max_leaf_nodes=8).fit(
        X, X[:, 0] + X[:, 1] ** 2
    )
    path = fitted_tree_linf_exact_coefficient_path(estimator, max_events=1)
    assert not path.exact
    assert path.status != "complete"
    assert path.intercepts.shape == path.lambdas.shape
    assert path.metadata["coefficient_status"] == path.status
    assert path.metadata["coefficient_points_certified"] is True
    assert path.metadata["coefficient_event_coverage_complete"] is False


def test_fitted_coefficient_path_rejects_non_mean_criterion():
    estimator = DecisionTreeRegressor(criterion="absolute_error").fit(
        np.arange(12.0)[:, None], np.arange(12.0)
    )
    with pytest.raises(ValueError, match="mean-based"):
        fitted_tree_linf_exact_coefficient_path(estimator)


def test_coefficient_wrapper_does_not_construct_a_redundant_topology_path(monkeypatch):
    from imodels.tree.sparse_pruning import fitted_tree

    def unexpected_topology_call(*args, **kwargs):
        pytest.fail("the coefficient solver already computes its structural thresholds")

    monkeypatch.setattr(
        fitted_tree, "fitted_tree_linf_exact_topology_path", unexpected_topology_call
    )
    tree = DecisionTreeRegressor(max_depth=2).fit(
        np.arange(16.0)[:, None], np.arange(16.0)
    )
    path = fitted_tree.fitted_tree_linf_exact_coefficient_path(tree)
    assert path.exact
    assert path.metadata["source"] == "fitted_tree_sufficient_statistics"
