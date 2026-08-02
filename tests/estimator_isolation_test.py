"""Wrapper models must not modify the estimator they were handed.

HSTree and DecisionTreeCCP take a base estimator and fit it in place. Because
shrinkage and pruning rewrite the tree, two wrappers built on one estimator
silently became the same model -- the second refit and reshrank the very tree
the first was still pointing at.
"""

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils.validation import check_is_fitted

import imodels


@pytest.fixture
def data():
    rng = np.random.RandomState(0)
    X = rng.randn(200, 4)
    y = (X[:, 0] > 0).astype(int)
    return X, y


def _base(regression=False):
    tree = DecisionTreeRegressor if regression else DecisionTreeClassifier
    return tree(max_leaf_nodes=8, random_state=0)


WRAPPERS = [
    ('HSTreeClassifier', {}),
    ('HSTreeClassifierCV', {}),
    ('DecisionTreeCCPClassifier', {'desired_complexity': 4}),
]


@pytest.mark.parametrize('name,kwargs', WRAPPERS, ids=[w[0] for w in WRAPPERS])
def test_fit_leaves_the_given_estimator_unfitted(name, kwargs, data):
    """sklearn estimators must not modify their __init__ arguments"""
    X, y = data
    base = _base()
    getattr(imodels, name)(estimator_=base, **kwargs).fit(X, y)

    with pytest.raises(NotFittedError):
        check_is_fitted(base)


def test_two_models_can_share_one_base_estimator(data):
    """The consequence: sharing an estimator collapsed both models into one

    Both wrappers ended up holding the same tree object, so the second fit
    overwrote the first and reg_param stopped mattering entirely.
    """
    X, y = data
    base = _base()

    light = imodels.HSTreeClassifier(estimator_=base, reg_param=0.1).fit(X, y)
    heavy = imodels.HSTreeClassifier(estimator_=base, reg_param=500.0).fit(X, y)

    assert light.estimator_ is not heavy.estimator_
    assert not np.allclose(light.predict_proba(X), heavy.predict_proba(X))


def test_ccp_models_can_share_one_base_estimator(data):
    X, y = data
    base = _base()

    coarse = imodels.DecisionTreeCCPClassifier(
        estimator_=base, desired_complexity=2).fit(X, y)
    fine = imodels.DecisionTreeCCPClassifier(
        estimator_=base, desired_complexity=8).fit(X, y)

    assert coarse.estimator_ is not fine.estimator_
    assert coarse.estimator_.tree_.node_count <= fine.estimator_.tree_.node_count


def test_regressor_wrappers_too(data):
    X, y = data
    y = y.astype(float)
    base = _base(regression=True)
    imodels.HSTreeRegressor(estimator_=base).fit(X, y)
    imodels.DecisionTreeCCPRegressor(
        estimator_=base, desired_complexity=4).fit(X, y)

    with pytest.raises(NotFittedError):
        check_is_fitted(base)


def test_refitting_a_wrapper_still_works(data):
    """Copying at fit time must not break an ordinary refit"""
    X, y = data
    model = imodels.HSTreeClassifier(estimator_=_base(), reg_param=1.0)
    first = model.fit(X, y).predict_proba(X)
    second = model.fit(X, y).predict_proba(X)
    assert np.allclose(first, second)
