"""Regression checks for wrappers that aren't in the imodels.REGRESSORS registry.

Every registered regressor is exercised by model_api_test.py; this file covers the
meta-estimators that need another model passed in, and so can't be auto-instantiated.
"""

from functools import partial

import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

from imodels import DistilledRegressor

N_SAMPLES = 20
N_FEATURES = 5


class TestClassRegression:
    def setup_method(self):
        rng = np.random.RandomState(13)
        self.X_regression = rng.randn(N_SAMPLES, N_FEATURES)
        self.y_regression = self.X_regression[:, 0] + \
            rng.randn(N_SAMPLES) * 0.01

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_regression(self):
        """Meta-estimators fit an easy, nearly linear target"""
        for model_type in [
            partial(
                DistilledRegressor,
                teacher=RandomForestRegressor(n_estimators=3, random_state=0),
                student=DecisionTreeRegressor(random_state=0),
            ),
        ]:
            m = model_type()
            m.fit(self.X_regression, self.y_regression)

            preds = np.asarray(m.predict(self.X_regression), dtype=float)
            assert preds.size == N_SAMPLES, "predictions are right size"

            mse = np.mean(np.square(preds - self.y_regression))
            assert mse < np.var(self.y_regression), \
                f"mse {mse:0.3f} is no better than predicting the mean"


def test_marginal_shrinkage_single_feature():
    """The marginal model should fit data with one feature

    Stacking the per-feature coefficients used a bare squeeze(), which collapses
    a single coefficient to a scalar, so X @ coef raised
    "Input operand 1 does not have enough dimensions".
    """
    from imodels import MarginalShrinkageLinearModelRegressor

    rng = np.random.RandomState(0)
    X = rng.randn(200, 1)
    y = 2 * X[:, 0] + 0.1 * rng.randn(200)

    model = MarginalShrinkageLinearModelRegressor().fit(X, y)
    assert np.shape(model.coef_marginal_) == (1,)
    assert model.predict(X).shape == (200,)
    assert np.mean((model.predict(X) - y) ** 2) < np.var(y)
