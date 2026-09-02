"""Tests for GPGamRegressor, the binned additive Gaussian-process GAM."""

import numpy as np
import pytest
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from imodels import GPGamRegressor
from imodels.algebraic.gp_gam import GPGamRegressor as _GPGam


FAST = dict(schedule=False, n_bins=12, n_pairs=2, pair_bins=6, n_steps=25)


def _additive_data(n=400, seed=0):
    """y is additive in two features, with a third that is pure noise."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 3)
    y = 2.5 * X[:, 0] + np.sin(2.0 * X[:, 1]) + rng.randn(n) * 0.15
    return X, y


class TestGPGamRegressor:
    def test_fits_an_additive_signal(self):
        X, y = _additive_data()
        Xtr, Xte, ytr, yte = train_test_split(X, y, random_state=0)
        model = _GPGam(**FAST).fit(Xtr, ytr)
        assert r2_score(yte, model.predict(Xte)) > 0.9

    def test_beats_a_linear_model_on_a_nonlinear_signal(self):
        rng = np.random.RandomState(3)
        X = rng.uniform(-3, 3, size=(400, 2))
        y = np.sin(1.5 * X[:, 0]) + 0.5 * X[:, 1] ** 2 + rng.randn(400) * 0.1
        Xtr, Xte, ytr, yte = train_test_split(X, y, random_state=0)
        gam = _GPGam(**FAST).fit(Xtr, ytr)
        ridge = RidgeCV().fit(Xtr, ytr)
        assert r2_score(yte, gam.predict(Xte)) > r2_score(yte, ridge.predict(Xte))

    def test_recovers_a_pairwise_interaction(self):
        """The screener should rank the true interacting pair first."""
        rng = np.random.RandomState(5)
        X = rng.randn(600, 4)
        y = X[:, 0] + 2.0 * X[:, 1] * X[:, 2] + rng.randn(600) * 0.1
        model = _GPGam(schedule=False, n_bins=10, n_pairs=1, pair_bins=6,
                       n_steps=25).fit(X, y)
        assert model.interaction_terms() == [(1, 2)]

    def test_prediction_is_deterministic(self):
        """No splits, seeds or sampling anywhere: two fits must agree exactly."""
        X, y = _additive_data(n=300, seed=1)
        a = _GPGam(**FAST).fit(X, y).predict(X)
        b = _GPGam(**FAST).fit(X, y).predict(X)
        np.testing.assert_allclose(a, b)

    def test_shape_function_matches_the_model(self):
        X, y = _additive_data(n=300, seed=2)
        model = _GPGam(**FAST).fit(X, y)
        grid, values = model.shape_function(0)
        assert grid.shape == values.shape
        assert np.all(np.diff(grid) >= 0)
        # feature 0 carries a strong linear effect, feature 2 is noise
        assert np.ptp(values) > np.ptp(model.shape_function(2)[1])

    def test_shape_function_reports_uncertainty(self):
        """The GP supplies a posterior band for each curve."""
        X, y = _additive_data(n=400, seed=6)
        model = _GPGam(**FAST).fit(X, y)
        grid, values = model.shape_function(0)
        grid2, values2, std = model.shape_function(0, return_std=True)
        np.testing.assert_allclose(grid, grid2)
        np.testing.assert_allclose(values, values2)
        assert std.shape == values.shape
        assert np.all(np.isfinite(std)) and np.all(std >= 0)

    def test_uncertainty_is_small_next_to_a_strong_effect(self):
        """A band wider than the curve itself would say the curve means nothing."""
        X, y = _additive_data(n=600, seed=7)
        model = _GPGam(**FAST).fit(X, y)
        values, std = model.shape_function(0, return_std=True)[1:]
        assert 2 * std.mean() < np.ptp(values)

    def test_log_target_rule_handles_skewed_positive_targets(self):
        rng = np.random.RandomState(7)
        X = rng.randn(400, 2)
        y = np.exp(1.2 * X[:, 0] + rng.randn(400) * 0.2)      # right-skewed, positive
        model = _GPGam(**FAST).fit(X, y)
        assert model.log_target_ is True
        assert np.all(model.predict(X) > 0)

    def test_constant_feature_is_dropped(self):
        X, y = _additive_data(n=200, seed=4)
        X = np.column_stack([X, np.ones(len(X))])
        model = _GPGam(**FAST).fit(X, y)
        assert 3 not in model.edges_
        assert model.predict(X).shape == (len(X),)

    def test_raises_when_every_feature_is_constant(self):
        X = np.ones((50, 2))
        y = np.arange(50, dtype=float)
        with pytest.raises(ValueError):
            _GPGam(**FAST).fit(X, y)

    def test_exposed_in_the_package_namespace(self):
        assert GPGamRegressor is _GPGam
