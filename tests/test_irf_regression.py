"""Tests for iterative random forest regression."""

import importlib

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils.estimator_checks import check_estimator

from imodels import IRFRegressor
from imodels.tree.iterative_random_forest._forest import _WeightedForest, _WeightedTree

irf_module = importlib.import_module(
    "imodels.tree.iterative_random_forest.iterative_random_forest"
)


def _tree(**kwargs):
    params = dict(max_features=1, max_depth=None, min_samples_split=2,
                  min_samples_leaf=1, bootstrap=False, random_state=0,
                  task="regression")
    params.update(kwargs)
    return _WeightedTree(**params)


def _and_data(n=300, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.uniform(size=(n, 5))
    y = ((X[:, 0] > 0.5) & (X[:, 1] > 0.5)).astype(float) + 0.05 * rng.normal(size=n)
    return X, y


def test_all_features_matches_weighted_sklearn_regression_cart():
    rng = np.random.RandomState(8)
    X = rng.normal(size=(150, 4)).astype(np.float32)
    y = X[:, 0] * X[:, 1] + rng.normal(size=150)
    weights = np.linspace(0.5, 2, len(y))
    # Five-row leaves avoid one-row splits that several features make equally
    # well; sklearn credits such ties by rounding in its running sums.
    reference = DecisionTreeRegressor(max_depth=3, min_samples_leaf=5,
                                      random_state=0).fit(X, y, sample_weight=weights)
    actual = _tree(max_features=4, max_depth=3, min_samples_leaf=5).fit(
        X, y, np.full(4, 0.25), weights)
    assert_allclose(actual.predict(X), reference.predict(X))
    assert_allclose(actual.feature_importances_, reference.feature_importances_)


def test_importance_is_residual_sum_of_squares_decrease():
    X = np.arange(4, dtype=np.float32).reshape(-1, 1)
    y = np.array([0.0, 0.0, 1.0, 1.0])
    tree = _tree().fit(X, y, np.ones(1), np.ones(4))
    # Total sum of squares 4 * 0.25 = 1, removed entirely by one split.
    assert_allclose(tree.raw_feature_importances_, [1.0])
    assert_allclose(tree.predict(X), y)


def test_bootstrap_duplicates_count_toward_regression_node_size():
    X = np.arange(4, dtype=np.float32).reshape(-1, 1)
    y = np.array([0.0, 1.0, 2.0, 3.0])
    # Seed 0 draws rows [0, 3, 1, 0]: four draws but three distinct rows.
    regression = _tree(bootstrap=True, min_samples_split=4).fit(
        X, y, np.ones(1), np.ones(4))
    assert_array_equal(regression.bootstrap_indices_, [0, 3, 1, 0])
    assert regression.n_node_samples_[0] == 4
    assert regression.node_count_ > 1
    classification = _tree(bootstrap=True, min_samples_split=4,
                           task="classification").fit(
        X, (y > 1).astype(int), np.ones(1), np.ones(4), n_classes=2)
    assert classification.n_node_samples_[0] == 3
    assert classification.node_count_ == 1


def test_regression_rejects_zero_gain_splits_like_r():
    X = np.tile([[0, 0], [0, 1], [1, 0], [1, 1]], (5, 1)).astype(np.float32)
    y = np.logical_xor(X[:, 0], X[:, 1]).astype(float)
    tree = _tree(max_features=2).fit(X, y, np.ones(2) / 2, np.ones(len(y)))
    assert tree.node_count_ == 1


def test_forest_predict_averages_leaf_means_and_has_no_probabilities():
    X, y = _and_data(80)
    forest = _WeightedForest(n_estimators=5, max_features=None, random_state=0,
                             task="regression").fit(X, y, feature_weights=np.ones(5))
    expected = np.mean([tree.predict(X) for tree in forest.estimators_], axis=0)
    assert_allclose(forest.predict(X), expected)
    with pytest.raises(AttributeError):
        forest.predict_proba(X)


def test_default_mtry_matches_r_regression():
    X, y = _and_data(40)
    X = np.hstack([X, X[:, :4]])  # nine features
    model = IRFRegressor(n_estimators=2, n_iterations=1, n_bootstraps=0,
                         random_state=0).fit(X, y)
    assert model.forest_.max_features_ == 3


def test_recovers_and_interaction_with_leaf_threshold():
    X, y = _and_data()
    model = IRFRegressor(n_estimators=30, n_iterations=3, n_bootstraps=5,
                         leaf_threshold=0.5, random_state=0).fit(X, y)
    assert model.interactions_[0] == (0, 1)
    assert model.interaction_stability_[(0, 1)] == 1.0
    assert set(np.argsort(model.feature_importances_)[-2:]) == {0, 1}
    assert model.score(X, y) > 0.8


def test_leaf_threshold_filters_paths_and_outer_bootstrap_is_unstratified(monkeypatch):
    X, y = _and_data(60)
    captured, strata = [], []
    original_resample = irf_module.resample

    def record_resample(*args, stratify=None, **kwargs):
        strata.append(stratify)
        return original_resample(*args, stratify=stratify, **kwargs)

    def record_rit(paths, masses, **kwargs):
        captured.append(len(paths))
        return set()

    monkeypatch.setattr(irf_module, "resample", record_resample)
    monkeypatch.setattr(irf_module, "_random_intersection_trees", record_rit)
    params = dict(n_estimators=3, n_iterations=1, n_bootstraps=2, random_state=0)
    IRFRegressor(leaf_threshold=y.max() + 1, **params).fit(X, y)
    assert captured == [0, 0]
    IRFRegressor(**params).fit(X, y)
    assert all(count > 0 for count in captured[2:])
    assert strata == [None] * 4


def test_invalid_leaf_threshold():
    X, y = _and_data(20)
    with pytest.raises(ValueError, match="leaf_threshold"):
        IRFRegressor(leaf_threshold="high").fit(X, y)


def test_sklearn_estimator_contract_without_row_bootstrapping():
    check_estimator(IRFRegressor(n_estimators=5, n_iterations=2, n_bootstraps=0,
                                 bootstrap=False, min_samples_split=2,
                                 random_state=0))


@pytest.mark.parametrize("scale, offset", [(1.0, 0.0), (1e-8, 0.0), (1e8, 0.0),
                                           (1.0, 1e8), (1e-8, 1.0)])
def test_splits_do_not_depend_on_target_scale_or_offset(scale, offset):
    X = np.linspace(0, 1, 40, dtype=np.float32).reshape(-1, 1)
    y = offset + scale * (X[:, 0] > 0.5)
    tree = _tree().fit(X, y, np.ones(1), np.ones(40))
    assert tree.node_count_ == 3
    assert_allclose(tree.predict(X), y, rtol=0, atol=1e-6 * scale)
    # Total sum of squares, 40 * 0.25 * scale**2, is removed by the split.
    assert_allclose(tree.raw_feature_importances_, [10 * scale ** 2], rtol=1e-6)


def test_deep_node_with_tiny_spread_still_splits():
    # The second split's node has spread 1e-9 of the root's, which is below
    # sklearn's absolute impurity tolerance unless the node is rescaled.
    X = np.linspace(0, 1, 40, dtype=np.float32).reshape(-1, 1)
    y = (X[:, 0] > 0.5) + 1e-9 * (X[:, 0] > 0.75)
    tree = _tree().fit(X, y, np.ones(1), np.ones(40))
    assert tree.node_count_ == 5
    assert_array_equal(tree.predict(X), y)


def test_weak_first_split_that_exposes_xor_is_kept():
    X = np.tile([[0, 0], [0, 1], [1, 0], [1, 1]], (5, 1)).astype(np.float32)
    y = np.logical_xor(X[:, 0], X[:, 1]).astype(float) + 1e-6 * X[:, 0]
    tree = _tree(max_features=2).fit(X, y, np.ones(2) / 2, np.ones(len(y)))
    assert tree.node_count_ == 7
    assert_allclose(tree.predict(X), y, rtol=0, atol=1e-12)
