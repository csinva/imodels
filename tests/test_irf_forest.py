"""Tests for the statistical behavior of the weighted CART backend."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier

from imodels.tree.iterative_random_forest._forest import _WeightedForest, _WeightedTree


def _tree(**kwargs):
    params = dict(max_features=1, max_depth=None, min_samples_split=2,
                  min_samples_leaf=1, bootstrap=False, random_state=0)
    params.update(kwargs)
    return _WeightedTree(**params)


def test_feature_weights_control_node_candidate_probability_and_exclude_zero():
    # All columns predict perfectly: with mtry=1 the selected feature identifies
    # the candidate draw, without confounding split quality.
    X = np.tile([[0, 0, 0], [1, 1, 1]], (20, 1))
    y = X[:, 0]
    forest = _WeightedForest(n_estimators=300, max_features=1, max_depth=1,
                             bootstrap=False, random_state=4).fit(
        X, y, feature_weights=np.array([0.9, 0.1, 0.0])
    )
    roots = np.array([tree.feature_[0] for tree in forest.estimators_])
    assert not np.any(roots == 2)
    assert 0.84 < np.mean(roots == 0) < 0.96


def test_candidates_are_resampled_at_each_node_and_zero_gain_xor_splits_survive():
    X = np.tile([[0, 0], [0, 1], [1, 0], [1, 1]], (12, 1))
    y = np.logical_xor(X[:, 0], X[:, 1]).astype(int)
    forest = _WeightedForest(n_estimators=40, max_features=1,
                             bootstrap=False, random_state=3).fit(
        X, y, feature_weights=np.array([0.5, 0.5])
    )
    complete = [tree for tree in forest.estimators_
                if np.array_equal(np.argmax(tree.predict_proba(X), axis=1), y)]
    assert complete, "A feature subset fixed once per tree cannot represent XOR with mtry=1."
    assert set(complete[0].feature_[complete[0].feature_ >= 0]) == {0, 1}


def test_all_features_matches_weighted_sklearn_cart():
    X, y = make_classification(n_samples=180, n_features=5, n_informative=3,
                                n_redundant=0, random_state=8)
    X = X.astype(np.float32)
    weights = np.linspace(0.4, 2, len(y))
    reference = DecisionTreeClassifier(max_depth=3, min_samples_leaf=3,
                                       random_state=2).fit(X, y, sample_weight=weights)
    actual = _tree(max_features=5, max_depth=3, min_samples_leaf=3).fit(
        X, y, np.full(5, 0.2), weights, n_classes=2
    )
    assert_allclose(actual.predict_proba(X), reference.predict_proba(X))
    assert_allclose(actual.feature_importances_, reference.feature_importances_)


def test_min_samples_leaf_counts_distinct_inbag_rows_not_multiplicity():
    X = np.arange(4, dtype=np.float32).reshape(-1, 1)
    y = np.array([0, 0, 1, 1])
    tree = _tree(bootstrap=True, min_samples_leaf=2).fit(
        X, y, np.ones(1), np.ones(4), n_classes=2
    )
    assert_array_equal(tree.bootstrap_indices_, [0, 3, 1, 0])
    assert tree.n_node_samples_[0] == 3
    assert tree.weighted_n_node_samples_[0] == 4
    assert tree.node_count_ == 1


def test_reference_path_mass_includes_all_rows_and_preserves_fractional_weights():
    X = np.arange(4, dtype=np.float32).reshape(-1, 1)
    weights = np.array([0.25, 0.5, 1, 2])
    tree = _tree(bootstrap=True).fit(
        X, np.zeros(4, dtype=int), np.ones(1), weights, n_classes=2
    )
    # Inner bootstrap [0, 3, 1, 0] excludes observation 2 and duplicates 0.
    assert tree.terminal_paths()[0][2] == 3.0
    assert tree.terminal_paths(X, sample_weight=weights)[0][2] == 3.75
    assert tree.terminal_paths(X)[0][2] == 4


def test_forest_gini_importance_aggregates_raw_decrease_before_normalizing():
    rng = np.random.RandomState(5)
    X = rng.normal(size=(50, 3)).astype(np.float32)
    y = ((X[:, 0] + X[:, 1]) > 0).astype(int)
    weights = np.exp(np.linspace(-1, 2, len(y)))
    forest = _WeightedForest(n_estimators=15, max_features=1, max_depth=1,
                             bootstrap=True, random_state=7).fit(
        X, y, feature_weights=np.ones(3) / 3, sample_weight=weights
    )
    expected = np.zeros(3)
    for tree in forest.estimators_:
        for node in np.flatnonzero(tree.feature_ >= 0):
            left, right = tree.children_left_[node], tree.children_right_[node]
            mass, impurity = tree.weighted_n_node_samples_, tree.impurity_
            decrease = mass[node] * impurity[node] - (
                mass[left] * impurity[left] + mass[right] * impurity[right]
            )
            expected[tree.feature_[node]] += decrease
    expected /= expected.sum()
    assert_allclose(forest.feature_importances_, expected)
    per_tree_normalized = np.mean([tree.feature_importances_
                                   for tree in forest.estimators_], axis=0)
    assert not np.allclose(expected, per_tree_normalized, atol=0.02)


def _adjacent_pair_whose_midpoint_rounds_up():
    # Adjacent float32 values whose float64 midpoint rounds to the upper value
    # when cast to float32 (round-half-to-even). The magnitude keeps their gap
    # above sklearn's 1e-7 constant-feature tolerance.
    value = np.float32(1000.0)
    for _ in range(64):
        upper = np.nextafter(value, np.float32(np.inf))
        midpoint = (np.float64(value) + np.float64(upper)) / 2
        if np.float32(midpoint) == upper:
            return value, upper
        value = upper
    raise AssertionError("no rounding-up pair found")


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_node_partition_uses_float64_threshold_like_apply(task):
    # Must hold on NumPy 1.x as well as 2.x: NumPy 1 compares a float32 array
    # with a float64 scalar in float32.
    lower, upper = _adjacent_pair_whose_midpoint_rounds_up()
    X = np.array([[lower], [upper]] * 3, dtype=np.float32)
    y = np.array([0, 1] * 3, dtype=float if task == "regression" else int)
    # With a float32 comparison the left child received every row and repeated
    # the same split without end. Cap depth so a regression fails, not hangs.
    tree = _tree(max_depth=4, task=task).fit(X, y, np.ones(1), np.ones(6), n_classes=2)
    assert tree.node_count_ == 3
    # Rows used to grow each leaf are the rows apply() routes there.
    routed = np.bincount(tree.apply(X), minlength=tree.node_count_)
    assert_array_equal(routed[tree.is_leaf_], tree.n_node_samples_[tree.is_leaf_])
    assert_array_equal(tree.predict(X), y)


def test_tied_leaf_classes_are_broken_at_random_for_rit_paths():
    # Identical inputs cannot be split, so each tree is one leaf with a 50/50 tie.
    X = np.zeros((4, 2), dtype=np.float32)
    y = np.array([0, 1, 0, 1])
    labels = [
        _tree(random_state=seed).fit(X, y, np.ones(2) / 2, np.ones(4),
                                     n_classes=2).terminal_paths(X)[0][1]
        for seed in range(200)
    ]
    assert 0.35 < np.mean(labels) < 0.65


def test_untied_leaf_classes_follow_the_majority():
    X = np.zeros((3, 1), dtype=np.float32)
    tree = _tree().fit(X, np.array([0, 1, 1]), np.ones(1), np.ones(3), n_classes=2)
    assert tree.terminal_paths(X)[0][1] == 1


def test_fractional_max_features_rounds_down_like_r():
    X = np.zeros((4, 9), dtype=np.float32)
    y = np.array([0, 1, 0, 1])
    forest = _WeightedForest(n_estimators=1, max_features=1 / 3,
                             random_state=0).fit(X, y, feature_weights=np.ones(9))
    assert forest.max_features_ == 3


def test_near_tie_is_not_treated_as_tie():
    # R compares class masses exactly; a slightly heavier class always wins.
    X = np.zeros((2, 1), dtype=np.float32)
    labels = {
        _tree(random_state=seed).fit(X, np.array([0, 1]), np.ones(1),
                                     np.array([1.0, 1.0 + 5e-13]),
                                     n_classes=2).terminal_paths(X)[0][1]
        for seed in range(50)
    }
    assert labels == {1}
