"""Tests for the shared transforms."""

import numpy as np
import pandas as pd

from imodels.util.transforms import CorrelationScreenTransformer, FriedScale, Winsorizer


def _correlated(seed, pair):
    rng = np.random.RandomState(seed)
    X = rng.randn(30, 4)
    X[:, pair[1]] = X[:, pair[0]]
    return X


def test_refit_forgets_the_previous_fit():
    """Refitting must screen the new data only

    The sets of correlated features were built in __init__ and appended to by
    fit, so refitting kept the previous data's correlations and zeroed columns
    that are not correlated in the new data.
    """
    first, second = _correlated(0, (0, 1)), _correlated(1, (2, 3))

    refit = CorrelationScreenTransformer()
    refit.fit(first)
    refit.fit(second)
    fresh = CorrelationScreenTransformer().fit(second)

    assert refit.correlated_feature_sets == fresh.correlated_feature_sets

    zeroed = lambda out: [j for j in range(4) if np.all(np.asarray(out)[:, j] == 0)]
    assert zeroed(refit.transform(second)) == zeroed(fresh.transform(second))


def test_output_type_follows_input_type():
    """A numpy array in gives a numpy array out

    The conversion back used `==` rather than `=`, so it silently did nothing.
    """
    X = _correlated(0, (0, 1))
    transformer = CorrelationScreenTransformer().fit(X)

    assert isinstance(transformer.transform(X), np.ndarray)
    assert isinstance(transformer.transform(pd.DataFrame(X)), pd.DataFrame)


def test_correlated_columns_are_screened():
    """The transformer still does its job: keep the first, zero the rest"""
    X = _correlated(0, (0, 1))
    out = np.asarray(CorrelationScreenTransformer().fit_transform(X))

    assert np.all(out[:, 1] == 0)
    assert np.allclose(out[:, 0], X[:, 0])
    assert np.allclose(out[:, 2], X[:, 2])


def test_winsorizer_and_friedscale_round_trip():
    """Winsorizing clips to the requested quantiles; scaling standardises"""
    rng = np.random.RandomState(0)
    X = rng.randn(500, 3)
    X[0, 0] = 100.0  # an outlier to clip

    winsorizer = Winsorizer(trim_quantile=0.05)
    winsorizer.train(X)
    trimmed = winsorizer.trim(X)
    assert trimmed[:, 0].max() < 100.0
    assert trimmed.min() >= np.percentile(X, 5, axis=0).min() - 1e-9

    scaler = FriedScale(winsorizer)
    scaler.train(X)
    scaled = scaler.scale(X)
    assert scaled.shape == X.shape
    # binary columns are left alone, continuous ones are rescaled
    binary = np.c_[X[:, :2], (X[:, 2] > 0).astype(float)]
    scaler_binary = FriedScale(None)
    scaler_binary.train(binary)
    assert scaler_binary.scale_multipliers[2] == 1
