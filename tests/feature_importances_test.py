"""Tests for MDI feature importances on tree-based models (issue #127)."""

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

import imodels

N_SAMPLES = 300


def _data():
    rng = np.random.RandomState(0)
    X = rng.randn(N_SAMPLES, 4)
    # only the first two features carry signal
    y = ((X[:, 0] > 0) | (X[:, 1] > 1)).astype(int)
    return X, y


def _models():
    return {
        'HSTreeClassifier': imodels.HSTreeClassifier(),
        'HSTreeClassifierCV': imodels.HSTreeClassifierCV(cv=3),
        'FIGSClassifierCV': imodels.FIGSClassifierCV(
            n_rules_list=[3], n_trees_list=[2], cv=2),
        'DecisionTreeCCPClassifier': imodels.DecisionTreeCCPClassifier(
            estimator_=DecisionTreeClassifier(random_state=0), desired_complexity=4),
        'TaoTreeClassifier': imodels.TaoTreeClassifier(),
    }


@pytest.mark.parametrize('model_name', sorted(_models()))
def test_feature_importances(model_name):
    """Importances look like sklearn's: non-negative, summing to 1, informative"""
    X, y = _data()
    model = _models()[model_name].fit(X, y)

    importances = model.feature_importances_
    assert importances.shape == (4,)
    assert (importances >= 0).all()
    assert np.isclose(importances.sum(), 1)

    # the two signal-carrying features should dominate the noise ones
    assert importances[:2].sum() > importances[2:].sum()


def test_shrinkage_matches_its_estimator():
    """Shrinkage doesn't change tree structure, so MDI matches the plain tree"""
    X, y = _data()
    shrunk = imodels.HSTreeClassifier(
        DecisionTreeClassifier(max_leaf_nodes=8, random_state=0),
        reg_param=100).fit(X, y)
    plain = DecisionTreeClassifier(max_leaf_nodes=8, random_state=0).fit(X, y)

    assert np.allclose(shrunk.feature_importances_, plain.feature_importances_)


def test_shrunk_forest_importances():
    """Wrapping an ensemble delegates to the ensemble's own importances"""
    X, y = _data()
    model = imodels.HSTreeClassifier(
        RandomForestClassifier(n_estimators=5, random_state=0)).fit(X, y)

    assert np.isclose(model.feature_importances_.sum(), 1)
    assert model.feature_importances_[:2].sum() > 0.5


def test_regressors_expose_importances():
    X, y = _data()
    y_continuous = X[:, 0] + 0.1 * np.random.RandomState(1).randn(N_SAMPLES)

    for model in [imodels.HSTreeRegressor(),
                  imodels.DecisionTreeCCPRegressor(
                      estimator_=DecisionTreeRegressor(random_state=0),
                      desired_complexity=4)]:
        importances = model.fit(X, y_continuous).feature_importances_
        assert np.isclose(importances.sum(), 1)
        assert np.argmax(importances) == 0
