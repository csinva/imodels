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

    def test_explicit_parameters_beat_the_schedule(self):
        """A value passed to the constructor must not be overridden by the schedule."""
        rng = np.random.RandomState(8)
        X = rng.randn(1200, 4)                       # over the 1000-row threshold
        y = X[:, 0] + X[:, 1] * X[:, 2] + rng.randn(1200) * 0.3
        model = _GPGam(n_pairs=2, n_steps=15).fit(X, y)
        assert len(model.interaction_terms()) == 2
        assert model.n_steps == 15

    def test_learned_lengthscales_are_reported(self):
        """The shared lengthscales are learned and named in kernel_weights."""
        X, y = _additive_data(n=400, seed=9)
        model = _GPGam(schedule=False, n_bins=16, n_pairs=0, n_steps=20).fit(X, y)
        assert hasattr(model, "scales_learned_")
        s0, s1 = model.scales_learned_
        assert 0.005 <= s0 <= 2.0 and 0.005 <= s1 <= 2.0
        names = list(model.kernel_weights(0))
        assert names[0].startswith("matern-") and names[1].startswith("rbf-")

    def test_interactions_beyond_48_are_backfit(self):
        """Above 48 selected pairs the extra ones are backfit and appear in the model."""
        rng = np.random.RandomState(10)
        X = rng.randn(3200, 12)
        y = X[:, 0] * X[:, 1] + np.sin(X[:, 2]) + rng.randn(3200) * 0.3
        model = _GPGam(schedule=False, n_bins=8, n_pairs=52, pair_res=(5, 4), n_steps=15,
                       sweeps=1).fit(X, y)
        assert len(model.interaction_terms()) == 52
        assert np.all(np.isfinite(model.predict(X[:50])))

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


# ----------------------------------------------------------------------
# regressions from the 2026-10 bug sweep (one test per finding)
# ----------------------------------------------------------------------
NOPAIR = dict(schedule=False, n_bins=12, n_pairs=0, n_steps=25)


class TestGPGamRegressions:
    def test_uneven_levels_predict_their_own_value(self):
        """Finding 1: a few unevenly spaced levels must not borrow a neighbour's value."""
        rng = np.random.RandomState(0)
        lv = rng.randint(0, 3, 600)
        X = np.c_[np.array([0.1, 0.2, 5.0])[lv], rng.randn(600)]
        y = np.array([0.0, 10.0, 5.0])[lv] + rng.randn(600)
        pred = _GPGam(**NOPAIR).fit(X, y).predict(X)
        for k, target in enumerate([0.0, 10.0, 5.0]):
            assert abs(pred[lv == k].mean() - target) < 0.5

    def test_tied_minimum_gets_its_own_bin(self):
        """Finding 2: a zero-inflated feature has no empty bin and no leak from the tie."""
        rng = np.random.RandomState(0)
        x0 = np.where(rng.rand(800) < 0.7, 0.0, rng.exponential(size=800))
        X = np.c_[x0, rng.randn(800)]
        y = 3 * (x0 == 0) + X[:, 1] + rng.randn(800) * 0.2
        model = _GPGam(schedule=False, n_bins=32, n_pairs=0, n_steps=25).fit(X, y)
        counts = np.bincount(np.searchsorted(model.edges_[0], x0, side="right"),
                             minlength=len(model.edges_[0]) + 1)
        assert counts.min() > 0
        pred = model.predict(X)
        assert abs(pred[x0 > 0].mean() - y[x0 > 0].mean()) < 0.15
        assert r2_score(y, pred) > 0.97

    def test_bins_are_never_empty(self):
        rng = np.random.RandomState(1)
        x = np.r_[np.zeros(300), np.full(200, 7.0), rng.rand(100) * 7]
        edges = _GPGam()._bin_edges(x, 16)
        counts = np.bincount(np.searchsorted(edges, x, side="right"), minlength=len(edges) + 1)
        assert counts.min() > 0

    def test_several_scales_with_learned_scales_raise(self):
        """Finding 3: a clear error instead of a broadcasting crash."""
        X, y = _additive_data(n=100)
        with pytest.raises(ValueError, match="learn_scales"):
            _GPGam(**dict(FAST, scales=(0.05, 0.2))).fit(X, y)
        with pytest.raises(ValueError, match="learn_scales"):
            _GPGam(**dict(FAST, rbf_scales=(0.1, 0.5))).fit(X, y)
        model = _GPGam(**dict(FAST, scales=(0.05, 0.2), learn_scales=False)).fit(X, y)
        assert len(model.kernel_weights(0)) == 3

    def test_numpy_refit_drops_stale_feature_names(self):
        """Finding 4."""
        pd = pytest.importorskip("pandas")
        X, y = _additive_data(n=100)
        model = _GPGam(**FAST).fit(pd.DataFrame(X, columns=list("abc")), y)
        model.fit(X, y)
        assert not hasattr(model, "feature_names_in_")
        model.predict(pd.DataFrame(X, columns=list("xyz")))

    def test_feature_names_argument(self):
        """Finding 5: validated, stored as feature_names_, never overrides the columns."""
        pd = pytest.importorskip("pandas")
        X, y = _additive_data(n=100)
        with pytest.raises(ValueError, match="feature_names"):
            _GPGam(**FAST).fit(X, y, feature_names=["a"])
        model = _GPGam(**FAST).fit(X, y, feature_names=["a", "b", "c"])
        assert model.feature_names_ == ["a", "b", "c"]
        assert not hasattr(model, "feature_names_in_")
        df = pd.DataFrame(X, columns=list("abc"))
        model = _GPGam(**FAST).fit(df, y, feature_names=list("pqr"))
        assert list(model.feature_names_in_) == list("abc")
        model.predict(df)

    def test_refit_clears_learned_scales(self):
        """Finding 6: kernel names describe the kernels of the latest fit."""
        X, y = _additive_data(n=200)
        model = _GPGam(**FAST).fit(X, y)
        assert hasattr(model, "scales_learned_")
        model.set_params(learn_scales=False).fit(X, y)
        assert not hasattr(model, "scales_learned_")
        assert list(model.kernel_weights(0)) == ["matern-0.05", "rbf-0.25"]

    def test_schedule_honours_explicit_values_and_arrays(self):
        """Finding 7: array-valued parameters work; an explicit default-valued one wins."""
        from sklearn.base import clone
        X, y = _additive_data(n=200)
        model = _GPGam(pair_res=np.array([6, 4]), n_pairs=1, n_steps=10).fit(X, y)
        assert len(model.interaction_terms()) == 1
        model = _GPGam(n_bins=64, n_pairs=0, n_steps=5)
        assert clone(model).get_params()["n_bins"] == 64
        model.fit(X, y)
        assert model._p("n_bins") == 64            # not the schedule's 96
        model = _GPGam(n_pairs=0, n_steps=5).fit(X, y)
        assert model._p("n_bins") == 96            # left at None: the schedule decides
        model = _GPGam(schedule=False, n_pairs=0, n_steps=5).fit(X, y)
        assert model._p("n_bins") == 64            # documented fallback

    @pytest.mark.parametrize("bad", [
        dict(n_bins=1), dict(scales=()), dict(scales=(), rbf_scales=(), learn_scales=False),
        dict(log_target="yes"), dict(lr=-1.0), dict(sweeps=-1), dict(n_pairs=-1),
        dict(pair_bins=1), dict(pair_res=()), dict(pair_scales=()), dict(n_steps=0),
        dict(scales=(-0.1,)), dict(tau=-1.0), dict(noise_init=0.0),
    ])
    def test_invalid_parameters_raise(self, bad):
        """Finding 8: a ValueError naming the parameter."""
        X, y = _additive_data(n=60)
        with pytest.raises(ValueError, match=list(bad)[0]):
            _GPGam(**dict(FAST, **bad)).fit(X, y)

    def test_numpy_bool_log_target(self):
        """Finding 8: np.True_ means True, not 'auto'."""
        rng = np.random.RandomState(0)
        X = rng.randn(100, 2)
        y = 5.0 + X[:, 0] + 0.1 * rng.randn(100)          # positive and unskewed
        assert _GPGam(**dict(FAST, log_target=np.True_)).fit(X, y).log_target_ is True
        assert _GPGam(**dict(FAST, log_target=np.False_)).fit(X, y).log_target_ is False

    def test_single_sample_raises(self):
        """Finding 9."""
        with pytest.raises(ValueError, match="at least 2 samples"):
            _GPGam(**FAST).fit(np.ones((1, 3)), np.ones(1))

    def test_feature_lookup(self):
        """Finding 10: range check and lookup by column name."""
        pd = pytest.importorskip("pandas")
        X, y = _additive_data(n=150)
        df = pd.DataFrame(X, columns=list("abc"))
        model = _GPGam(**FAST).fit(df, y)
        np.testing.assert_allclose(model.shape_function("b")[1], model.shape_function(1)[1])
        assert model.kernel_weights("a") == model.kernel_weights(0)
        for bad in (10, -1):
            with pytest.raises(ValueError, match="out of range"):
                model.shape_function(bad)
            with pytest.raises(ValueError, match="out of range"):
                model.kernel_weights(bad)
        with pytest.raises(ValueError, match="unknown feature name"):
            model.shape_function("zz")

    def test_constant_target_has_no_warnings(self):
        """Finding 11: the log-target heuristic is skipped for a constant y."""
        import warnings
        X, _ = _additive_data(n=100)
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            model = _GPGam(**FAST).fit(X, np.full(100, 3.0))
        assert model.log_target_ is False
        np.testing.assert_allclose(model.predict(X), 3.0)
