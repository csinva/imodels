import numpy as np
from sklearn.base import RegressorMixin, clone
from sklearn.linear_model.base import LinearRegression

from bartpy.sklearnmodel import SklearnModel


class ResidualBART(SklearnModel):

    def __init__(self,
                 base_estimator: RegressorMixin = None,
                 **kwargs):

        if base_estimator is not None:
            self.base_estimator = clone(base_estimator)
        else:
            base_estimator = LinearRegression()
        self.base_estimator = base_estimator
        super().__init__(**kwargs)

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'ResidualBART':
        self.base_estimator.fit(X, y)
        SklearnModel.fit(self, X, y - self.base_estimator.predict(X))
        return self

    def predict(self, X: np.ndarray=None) -> np.ndarray:
        if X is None:
            X = self.data.X
        sm_prediction = self.base_estimator.predict(X)
        bart_prediction = SklearnModel.predict(self, X)
        return sm_prediction + bart_prediction
