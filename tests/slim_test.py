"""Tests for SLIM's integer coefficients."""

import warnings

import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from imodels import SLIMClassifier, SLIMRegressor


def test_warns_when_rounding_destroys_the_model():
    """Integer coefficients collapse on unscaled data, and should say so

    On breast_cancer the feature scales span five orders of magnitude, so
    rounding the fitted coefficients to integers zeroes most of them and the
    classifier scores below chance. Nothing said why.
    """
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        raw = SLIMClassifier().fit(X_train, y_train)
    messages = [str(w.message) for w in caught]
    assert any('Rounding coefficients to integers' in m for m in messages), messages
    assert any('scale X first' in m or 'StandardScaler' in m for m in messages)

    # scaling is what the warning recommends, and it works
    scaler = StandardScaler().fit(X_train)
    with warnings.catch_warnings(record=True) as caught_scaled:
        warnings.simplefilter('always')
        scaled = SLIMClassifier().fit(scaler.transform(X_train), y_train)
    assert not any('Rounding coefficients to integers' in str(w.message)
                   for w in caught_scaled), 'should not warn once scaled'

    raw_auc = roc_auc_score(y_test, raw.predict_proba(X_test)[:, 1])
    scaled_auc = roc_auc_score(
        y_test, scaled.predict_proba(scaler.transform(X_test))[:, 1])
    assert scaled_auc > 0.9 and scaled_auc > raw_auc


def test_no_warning_when_coefficients_survive():
    """Data already on a sensible scale fits without complaint"""
    rng = np.random.RandomState(0)
    X = rng.randn(200, 4)
    y = 3 * X[:, 0] - 2 * X[:, 1] + 0.1 * rng.randn(200)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        model = SLIMRegressor().fit(X, y)
    assert not any('Rounding coefficients to integers' in str(w.message)
                   for w in caught)
    assert np.count_nonzero(model.model_.coef_) > 0
