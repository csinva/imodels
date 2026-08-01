"""Multiclass support across classifiers (issue #93)."""

import contextlib
import io

import numpy as np
import pytest

import imodels
from tests.model_configs import BINARY_INPUT_MODELS, EXCLUDED_MODELS, MODEL_KWARGS

# verified to produce one probability column per class
MULTICLASS_MODELS = [
    'BoostedRulesClassifier', 'SLIMClassifier', 'TaoTreeClassifier',
    'FIGSClassifier', 'FIGSClassifierCV', 'HSTreeClassifier',
    'HSTreeClassifierCV', 'GreedyTreeClassifier', 'DecisionTreeCCPClassifier',
]

# binary-only: these must say so rather than silently collapsing the target
BINARY_ONLY_MODELS = [
    'BayesianRuleListClassifier', 'RuleFitClassifier', 'FPLassoClassifier',
    'GreedyRuleListClassifier', 'SkopeRulesClassifier', 'C45TreeClassifier',
    'OneRClassifier', 'FPSkopeClassifier', 'TreeGAMClassifier',
    'SlipperClassifier',
]


def _data(model_name, n_classes):
    rng = np.random.RandomState(0)
    X = rng.randn(150, 4)
    if model_name in BINARY_INPUT_MODELS:
        X = (X > 0).astype(int)
    if n_classes == 3:
        y = (X[:, 0] > 0).astype(int) + (X[:, 1] > 0).astype(int)
    else:
        y = (X[:, 0] > 0).astype(int)
    return X, y


def _fit(model_name, n_classes):
    X, y = _data(model_name, n_classes)
    model = getattr(imodels, model_name)(**MODEL_KWARGS.get(model_name, {}))
    with contextlib.redirect_stdout(io.StringIO()):
        model.fit(X, y)
    return model, X, y


@pytest.mark.parametrize('model_name', MULTICLASS_MODELS)
def test_multiclass_models(model_name):
    """These give one probability column per class"""
    model, X, y = _fit(model_name, n_classes=3)
    probs = model.predict_proba(X)
    assert probs.shape == (len(X), 3)
    assert np.allclose(probs.sum(axis=1), 1)
    assert set(np.unique(model.predict(X))) <= set(np.unique(y))


@pytest.mark.parametrize('model_name', BINARY_ONLY_MODELS)
def test_binary_only_models_reject_multiclass(model_name):
    """Binary-only models must raise, not silently treat y as binary"""
    X, y = _data(model_name, n_classes=3)
    model = getattr(imodels, model_name)(**MODEL_KWARGS.get(model_name, {}))
    with pytest.raises(ValueError, match='binary|multiclass'):
        with contextlib.redirect_stdout(io.StringIO()):
            model.fit(X, y)


@pytest.mark.parametrize('model_name', BINARY_ONLY_MODELS)
def test_binary_only_models_still_fit_binary(model_name):
    """The guard must not disturb the binary case"""
    model, X, y = _fit(model_name, n_classes=2)
    probs = model.predict_proba(X)
    assert probs.shape == (len(X), 2)


def test_every_classifier_is_accounted_for():
    """A new classifier must be classified as multiclass or binary-only"""
    registered = {m.__name__ for m in imodels.CLASSIFIERS} - set(EXCLUDED_MODELS)
    covered = set(MULTICLASS_MODELS) | set(BINARY_ONLY_MODELS)
    # AutoInterpretable wraps a grid search, so it follows whatever it selects
    assert registered - covered <= {'AutoInterpretableClassifier'}
