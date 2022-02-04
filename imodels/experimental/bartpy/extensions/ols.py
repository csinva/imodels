from typing import Type

import numpy as np

from bartpy.sklearnmodel import SklearnModel


class OLS(SklearnModel):

    def __init__(self,
                 stat_model: Type,
                 **kwargs):
        self.stat_model = stat_model
        self.stat_model_fit = None
        super().__init__(**kwargs)

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'OLS':
        self.stat_model_fit = self.stat_model(y, X).fit()
        SklearnModel.fit(self, X, self.stat_model_fit.resid)
        return self

    def predict(self, X: np.ndarray=None) -> np.ndarray:
        if X is None:
            X = self.data.X
        sm_prediction = self.stat_model_fit.predict(X)
        bart_prediction = SklearnModel.predict(self, X)
        return sm_prediction + bart_prediction
