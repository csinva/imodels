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


@pytest.mark.parametrize("n_classes", [2, 3])
@pytest.mark.parametrize("value_encoding", ["proportions", "counts"])
def test_fitted_classifier_coefficients_match_effectively_weighted_training_design(
    n_classes, value_encoding
):
    from scipy.special import expit, softmax
    from sklearn.tree import DecisionTreeClassifier
    from imodels.tree.sparse_pruning import (
        fitted_tree_linf_classification,
        fitted_tree_linf_classification_path,
        materialize_fitted_tree_topology,
    )
    from imodels.tree.sparse_pruning.optimization.classification import (
        laminar_group_linf_classification_path,
    )

    rng = np.random.default_rng(52)
    X = rng.normal(size=(120, 3))
    scores = X @ rng.normal(size=(3, n_classes)) + rng.normal(size=(120, n_classes))
    y = np.array([f"class-{k}" for k in scores.argmax(axis=1)])
    weights = rng.uniform(0.2, 2, len(X))
    class_weights = {f"class-{k}": 1 + 0.3 * k for k in range(n_classes)}
    tree = DecisionTreeClassifier(max_leaf_nodes=5, min_samples_leaf=4,
                                  class_weight=class_weights, random_state=4)
    tree.fit(X, y, sample_weight=weights)
    # sklearn has stored both weighted counts and proportions across releases.
    values = tree.tree_.value
    values[:] /= values.sum(axis=2, keepdims=True)
    if value_encoding == "counts":
        values[:] *= tree.tree_.weighted_n_node_samples[:, None, None]
    before = deepcopy(tree.tree_.__getstate__())
    topology = fitted_tree_linf_exact_topology_path(tree)
    parents = topology.metadata["parent_indices"]
    groups = [[node] for node in range(len(parents))]
    for child in range(len(parents) - 1, -1, -1):
        if parents[child] >= 0:
            groups[parents[child]].extend(groups[child])
    group_weights = np.linspace(0.7, 1.4, len(groups))
    lambdas = topology.lambdas[0] * np.array([1.5, 0.5, 0.1])
    path = fitted_tree_linf_classification_path(
        tree, lambdas, group_weights=group_weights, tol=1e-9, max_iter=20_000)
    Z = tree_feature_transform(make_stumps(tree.tree_), X)
    effective = weights * np.array([class_weights[label] for label in y])
    explicit = laminar_group_linf_classification_path(
        Z, y, groups, lambdas, sample_weight=effective, group_weights=group_weights,
        tol=1e-9, max_iter=20_000)
    assert path.status == explicit.status == "complete"
    assert not path.exact
    assert path.metadata["training_design_materialized"] is False
    assert path.metadata["local_stump_normalization"] == "unnormalized"
    np.testing.assert_array_equal(path.metadata["classes"], tree.classes_)
    np.testing.assert_array_equal(path.metadata["tree_node_ids"], topology.node_ids)
    for index, lam in enumerate(lambdas):
        beta, intercept = path.at(lam)
        predicted = Z @ beta + intercept
        expected = Z @ explicit.coefficients[index] + explicit.intercepts[index]
        if n_classes == 2:
            predicted, expected = expit(predicted), expit(expected)
        else:
            predicted, expected = softmax(predicted, axis=1), softmax(expected, axis=1)
        np.testing.assert_allclose(predicted, expected, atol=3e-6)
        ranges = np.abs(beta) if n_classes == 2 else np.ptp(beta, axis=1)
        assert path.penalties[index] == pytest.approx(sum(
            weight * np.max(ranges[group]) for weight, group in zip(group_weights, groups)
        ), abs=2e-10)
    beta, info = fitted_tree_linf_classification(
        tree, lambdas[-1], group_weights=group_weights, tol=1e-9, return_info=True,
        beta_init=path.coefficients[-1], intercept_init=path.intercepts[-1])
    new_X = rng.normal(size=(12, X.shape[1]))
    logits = tree_feature_transform(make_stumps(tree.tree_), new_X) @ beta + info["intercept"]
    expected = (np.column_stack([expit(-logits), expit(logits)]) if n_classes == 2
                else softmax(logits, axis=1))
    indices = np.searchsorted(info["leaf_node_ids"], tree.apply(new_X))
    np.testing.assert_allclose(info["leaf_probabilities"][indices], expected, atol=2e-10)
    assert info["certified"]
    collapsed = materialize_fitted_tree_topology(tree, topology, topology.lambdas[0])
    root_probs = np.bincount(np.searchsorted(tree.classes_, y), weights=effective)
    root_probs /= root_probs.sum()
    np.testing.assert_allclose(collapsed.predict_proba(X), np.tile(root_probs, (len(X), 1)))
    np.testing.assert_array_equal(before["nodes"], tree.tree_.__getstate__()["nodes"])
    np.testing.assert_array_equal(before["values"], tree.tree_.__getstate__()["values"])


def test_fitted_classifier_root_only_path_has_finite_class_prior_logits():
    from scipy.special import softmax
    from sklearn.tree import DecisionTreeClassifier
    from imodels.tree.sparse_pruning import fitted_tree_linf_classification_path

    tree = DecisionTreeClassifier().fit(np.zeros((6, 1)), [0, 0, 0, 1, 1, 2])
    path = fitted_tree_linf_classification_path(tree, [0.2, 0.01], tol=1e-10)
    assert path.coefficients.shape == (2, 0, 3)
    np.testing.assert_allclose(softmax(path.intercepts, axis=1),
                               [[0.5, 1 / 3, 1 / 6]] * 2, atol=1e-9)
    assert path.status == "complete"
    with pytest.raises(ValueError):
        fitted_tree_linf_classification_path(tree, [0.0])


def test_fitted_classification_rejects_monotonic_constraints():
    import inspect
    from sklearn.tree import DecisionTreeClassifier
    from imodels.tree.sparse_pruning import fitted_tree_linf_classification_path

    if "monotonic_cst" not in inspect.signature(DecisionTreeClassifier).parameters:
        pytest.skip("this sklearn version does not expose monotonic constraints")
    tree = DecisionTreeClassifier(monotonic_cst=[1]).fit(
        np.arange(12.0)[:, None], [0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0])
    for call in [lambda: fitted_tree_linf_exact_topology_path(tree),
                 lambda: fitted_tree_linf_classification_path(tree, [0.1])]:
        with pytest.raises(ValueError, match="monotonic"):
            call()


def test_fitted_classification_checks_class_identity_and_effective_class_mass():
    from sklearn.tree import DecisionTreeClassifier
    from imodels.tree.sparse_pruning import (
        fitted_tree_linf_classification_path,
        materialize_fitted_tree_topology,
    )

    X, y = np.zeros((6, 1)), [0, 0, 0, 1, 1, 2]
    tree = DecisionTreeClassifier().fit(X, y)
    topology = fitted_tree_linf_exact_topology_path(tree)
    renamed = deepcopy(tree)
    renamed.classes_ = renamed.classes_[::-1]
    with pytest.raises(ValueError, match="different|modified"):
        materialize_fitted_tree_topology(renamed, topology, 0.0)
    zero_class = DecisionTreeClassifier().fit(X, y, sample_weight=[1, 1, 1, 1, 1, 0])
    for call in [lambda: fitted_tree_linf_exact_topology_path(zero_class),
                 lambda: fitted_tree_linf_classification_path(zero_class, [0.1])]:
        with pytest.raises(ValueError, match="positive effective"):
            call()
