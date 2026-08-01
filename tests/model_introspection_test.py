"""Checks that the introspection APIs (get_rules, apply) agree across all models.

get_rules and apply both need to find the trees or rules behind a fitted model.
These tests pin that they stay consistent with each other and with the methods
models advertise, which is easy to break when a model is added or changed.
"""

import contextlib
import io

import numpy as np
import pandas as pd
import pytest

import imodels
from imodels.util.apply import apply_leaves
from imodels.util.get_rules import CORE_COLUMNS
from tests.model_configs import (BINARY_INPUT_MODELS, EXCLUDED_MODELS,
                                 MODEL_KWARGS)

N_SAMPLES = 150
FEATURE_NAMES = list('abcd')

MODELS = [m for m in imodels.ESTIMATORS if m.__name__ not in EXCLUDED_MODELS]
IDS = [m.__name__ for m in MODELS]

# SLIPPER boosts single-rule estimators rather than trees, so it inherits an
# apply method from BoostedRulesClassifier that cannot work; it reports that.
NO_LEAVES_DESPITE_METHOD = {'SlipperClassifier'}


def _data(model_type):
    rng = np.random.RandomState(0)
    X = pd.DataFrame(rng.randn(N_SAMPLES, 4), columns=FEATURE_NAMES)
    if model_type.__name__ in BINARY_INPUT_MODELS:
        X = (X > 0).astype(int)
    if model_type in imodels.CLASSIFIERS:
        y = (X.iloc[:, 0] > 0).astype(int)
    else:
        y = X.iloc[:, 0] + 0.01 * rng.randn(N_SAMPLES)
    return X, y


def _fit(model_type):
    X, y = _data(model_type)
    model = model_type(**MODEL_KWARGS.get(model_type.__name__, {}))
    with contextlib.redirect_stdout(io.StringIO()):  # some models print
        model.fit(X, y)
    return model, X, y


def _supported(fn):
    """Whether an introspection function works, as opposed to saying it can't."""
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            fn()
        return True
    except ValueError:
        return False


@pytest.mark.parametrize('model_type', MODELS, ids=IDS)
def test_get_rules_method_matches_function(model_type):
    """A model has .get_rules() exactly when imodels.get_rules supports it"""
    model, X, y = _fit(model_type)
    works = _supported(lambda: imodels.get_rules(model))
    assert hasattr(model, 'get_rules') == works


@pytest.mark.parametrize('model_type', MODELS, ids=IDS)
def test_apply_method_matches_function(model_type):
    """A model has .apply() exactly when leaf membership is available for it"""
    model, X, y = _fit(model_type)
    if model_type.__name__ in NO_LEAVES_DESPITE_METHOD:
        pytest.skip('inherits apply but is not built from trees')
    works = _supported(lambda: apply_leaves(model, X))
    assert hasattr(model, 'apply') == works


@pytest.mark.parametrize('model_type', MODELS, ids=IDS)
def test_introspection_never_fails_unexpectedly(model_type):
    """These APIs either work or raise ValueError -- never anything else"""
    model, X, y = _fit(model_type)
    for fn in (lambda: imodels.get_rules(model), lambda: apply_leaves(model, X)):
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                fn()
        except ValueError:
            pass  # the documented "not supported" signal


def test_regressors_support_leaf_membership():
    """Tree-based regressors were broken while their classifiers worked"""
    rng = np.random.RandomState(0)
    X = rng.randn(N_SAMPLES, 4)
    y = X[:, 0] + 0.1 * rng.randn(N_SAMPLES)

    figs = imodels.FIGSRegressor(max_rules=5).fit(X, y)
    assert figs.apply(X).shape[0] == N_SAMPLES

    from sklearn.tree import DecisionTreeRegressor
    ccp = imodels.DecisionTreeCCPRegressor(
        estimator_=DecisionTreeRegressor(random_state=0),
        desired_complexity=4).fit(X, y)
    assert ccp.apply(X).shape == (N_SAMPLES,)
    assert list(ccp.get_rules().columns[:len(CORE_COLUMNS)]) == CORE_COLUMNS


def test_multiclass_leaf_membership():
    """Leaf membership should work for more than two classes"""
    from sklearn.datasets import load_iris
    X, y = load_iris(return_X_y=True)
    model = imodels.FIGSClassifier(max_rules=6).fit(X, y)
    assert model.apply(X).shape[0] == len(y)


def test_apply_accepts_missing_values():
    """fit and predict accept NaN for sklearn trees, so apply must too"""
    from sklearn.tree import DecisionTreeClassifier

    rng = np.random.RandomState(0)
    X = rng.randn(120, 3)
    y = (X[:, 0] > 0).astype(int)
    X[::9, 1] = np.nan

    model = imodels.HSTreeClassifier(
        DecisionTreeClassifier(max_leaf_nodes=6)).fit(X, y)
    model.predict(X)
    assert model.apply(X).shape == (120,)


def test_boosted_rules_report_their_weights():
    """A boosted ensemble votes by weight, so the weights must be reported"""
    rng = np.random.RandomState(0)
    X = pd.DataFrame(rng.randn(300, 3), columns=list('abc'))
    y = (X['a'] + 0.8 * rng.randn(300) > 0).astype(int)

    model = imodels.BoostedRulesClassifier(n_estimators=4).fit(X, y)
    rules = model.get_rules()
    assert 'weight' in rules.columns
    assert rules.groupby('tree')['weight'].nunique().eq(1).all()

    # each tree's reported weight is the one the model gives it, so a caller can
    # weigh the rules the way the model does (how they are combined is the
    # boosting algorithm's business, and varies by sklearn version)
    reported = rules.groupby('tree')['weight'].first().to_numpy()
    assert np.allclose(reported, model.estimator_weights_[:len(reported)])

    # a single tree carries no weights
    assert 'weight' not in imodels.GreedyTreeClassifier(
        max_leaf_nodes=4).fit(X, y).get_rules().columns


def test_models_do_not_share_mutable_defaults():
    """Default arguments must not be shared between instances"""
    first, second = imodels.TaoTreeClassifier(), imodels.TaoTreeClassifier()
    assert first.model_args is not second.model_args
    first.model_args['max_leaf_nodes'] = 999
    assert second.model_args['max_leaf_nodes'] != 999
