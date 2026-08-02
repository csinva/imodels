"""Predicting with columns in a different order must not silently mislead.

Models index X by feature position, so a DataFrame whose columns are in a
different order than at fit produces predictions for the wrong features.
sklearn raises in that case; so should imodels.
"""

import contextlib
import io

import numpy as np
import pandas as pd
import pytest

import imodels
from tests.model_configs import (BINARY_INPUT_MODELS, EXCLUDED_MODELS,
                                 MODEL_KWARGS)

FEATURE_NAMES = ['a', 'b', 'c', 'd']
MODELS = [m for m in imodels.ESTIMATORS if m.__name__ not in EXCLUDED_MODELS]
IDS = [m.__name__ for m in MODELS]


def _fit(model_type):
    rng = np.random.RandomState(0)
    X = pd.DataFrame(rng.randn(200, 4), columns=FEATURE_NAMES)
    if model_type.__name__ in BINARY_INPUT_MODELS:
        X = (X > 0).astype(int)
    if model_type in imodels.CLASSIFIERS:
        y = (X['a'] > 0).astype(int)
    else:
        y = X['a'] + 0.01 * rng.randn(200)
    model = model_type(**MODEL_KWARGS.get(model_type.__name__, {}))
    with contextlib.redirect_stdout(io.StringIO()):
        model.fit(X, y)
    return model, X


@pytest.mark.parametrize('model_type', MODELS, ids=IDS)
def test_reordered_columns_raise(model_type):
    """Reordering the columns must raise, not quietly change the answer"""
    model, X = _fit(model_type)
    shuffled = X[FEATURE_NAMES[::-1]]

    with pytest.raises(ValueError):
        with contextlib.redirect_stdout(io.StringIO()):
            model.predict(shuffled)


@pytest.mark.parametrize('model_type', MODELS, ids=IDS)
def test_matching_columns_and_arrays_still_work(model_type):
    """The check must not disturb ordinary use"""
    model, X = _fit(model_type)
    with contextlib.redirect_stdout(io.StringIO()):
        from_frame = np.asarray(model.predict(X), dtype=float)
        from_array = np.asarray(model.predict(X.values), dtype=float)

    assert from_frame.shape[0] == len(X)
    # an unnamed array carries no column order to check, and must be unchanged
    assert np.allclose(from_frame, from_array)


def test_wrong_number_of_columns_raises():
    model, X = _fit(imodels.FIGSClassifier)
    with pytest.raises(ValueError, match='features'):
        model.predict(X[['a', 'b']])
