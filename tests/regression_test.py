import numpy as np
import pytest

from imodels import RuleFitRegressor, SLIMRegressor, GreedyTreeRegressor


class TestClassRegression:
    def setup(self):
        np.random.seed(13)
        self.n = 10
        self.p = 10
        self.X_regression = np.random.randn(self.n, self.p)
        self.y_regression = self.X_regression[:, 0] + np.random.randn(self.n) * 0.01

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_regression(self):
        '''Test imodels on basic binary classification task
        '''
        for model_type in [RuleFitRegressor, SLIMRegressor, GreedyTreeRegressor]:
            if model_type == RuleFitRegressor:
                m = model_type(include_linear=False, max_rules=3)
            else:
                m = model_type()
            m.fit(self.X_regression, self.y_regression)

            preds = m.predict(self.X_regression)
            assert preds.size == self.n, 'predictions are right size'

            mse = np.mean(np.square(preds - self.y_regression))
            assert mse < 1, 'mse less than 1'
