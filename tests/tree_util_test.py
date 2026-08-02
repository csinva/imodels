"""Tests for the tree measurement helpers in imodels.util.tree."""

import numpy as np
import pytest
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from imodels.util.tree import (_validate_feature_costs,
                               calculate_mean_depth_of_points_in_tree,
                               calculate_mean_unique_calls_in_ensemble)


def _data(binary=False):
    rng = np.random.RandomState(0)
    X = rng.randn(300, 4)
    if binary:
        X = (X > 0).astype(int)
    return X, (X[:, 0] > 0).astype(int)


def test_feature_costs_are_validated():
    """Invalid feature costs must be rejected, not silently used

    The non-negativity check was written as a bare comparison, so it did
    nothing and negative costs produced negative depths.
    """
    X, y = _data()
    tree = DecisionTreeClassifier(max_depth=4, random_state=0).fit(X, y)

    with pytest.raises(ValueError, match='non-negative'):
        calculate_mean_depth_of_points_in_tree(
            tree, X, feature_costs=np.array([-5.0, 1.0, 1.0, 1.0]))

    with pytest.raises(ValueError, match='features'):
        _validate_feature_costs(np.ones(2), 4)

    # valid costs still work, and scale the depth
    plain = calculate_mean_depth_of_points_in_tree(tree, X)
    weighted = calculate_mean_depth_of_points_in_tree(
        tree, X, feature_costs=np.array([2.0, 1.0, 1.0, 1.0]))
    assert plain > 0 and weighted >= plain


@pytest.mark.parametrize('ensemble', [
    RandomForestClassifier(n_estimators=5, random_state=0),
    GradientBoostingClassifier(n_estimators=5, random_state=0),
], ids=['RandomForest', 'GradientBoosting'])
def test_mean_unique_calls_in_ensemble(ensemble):
    """This measurement worked for neither of its two code paths

    Passing X raised UnboundLocalError because the feature count was only read
    on the branch taken when X was None, and that branch then called .flatten()
    on estimators_, which is a list for forests.
    """
    X, y = _data(binary=True)
    ensemble.fit(X, y)

    with_X = calculate_mean_unique_calls_in_ensemble(ensemble, X=X)
    assert with_X > 0

    without_X = calculate_mean_unique_calls_in_ensemble(ensemble, X=None)
    assert without_X > 0

    # costs are validated on this path too
    with pytest.raises(ValueError, match='non-negative'):
        calculate_mean_unique_calls_in_ensemble(
            ensemble, X=X, feature_costs=np.array([-1.0, 1.0, 1.0, 1.0]))
