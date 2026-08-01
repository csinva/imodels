"""Tests for cost-complexity-pruned trees."""

import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from imodels import DecisionTreeCCPClassifier, DecisionTreeCCPRegressor


def _data():
    rng = np.random.RandomState(0)
    X = rng.randn(400, 4)
    y = ((X[:, 0] > 0) ^ (rng.rand(400) < 0.25)).astype(int)
    return X, y


def _classifier():
    # a fresh inner estimator each time: two models must not share one object
    return DecisionTreeCCPClassifier(
        estimator_=DecisionTreeClassifier(random_state=0), desired_complexity=4)


def test_sample_weight_is_used():
    """sample_weight was accepted but never reached the underlying fit

    https://github.com/csinva/imodels/issues/89: DecisionTreeCCP took a
    sample_weight argument and dropped it, so weighted fits silently matched
    unweighted ones.
    """
    X, y = _data()
    weights = np.where(X[:, 1] > 0, 25.0, 1.0)

    unweighted = _classifier().fit(X, y)
    weighted = _classifier().fit(X, y, sample_weight=weights)

    assert not np.allclose(unweighted.predict_proba(X), weighted.predict_proba(X))
    # the fitted tree records the weights it was given
    assert np.isclose(
        weighted.estimator_.tree_.weighted_n_node_samples[0], weights.sum())

    # integer weights must match repeating the rows
    int_weights = np.where(np.arange(len(y)) % 2 == 0, 3.0, 1.0)
    by_weight = _classifier().fit(X, y, sample_weight=int_weights)
    by_duplication = _classifier().fit(
        np.repeat(X, int_weights.astype(int), axis=0),
        np.repeat(y, int_weights.astype(int)))
    assert np.array_equal(by_weight.estimator_.tree_.feature,
                          by_duplication.estimator_.tree_.feature)


def test_sample_weight_is_used_by_the_regressor():
    X, y = _data()
    rng = np.random.RandomState(1)
    y = X[:, 0] + 0.5 * rng.randn(len(X))
    weights = np.where(X[:, 1] > 0, 25.0, 1.0)

    def regressor():
        return DecisionTreeCCPRegressor(
            estimator_=DecisionTreeRegressor(random_state=0), desired_complexity=4)

    assert not np.allclose(regressor().fit(X, y).predict(X),
                           regressor().fit(X, y, sample_weight=weights).predict(X))
