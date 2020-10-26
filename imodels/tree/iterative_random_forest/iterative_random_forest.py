import numpy as np
from irf.ensemble import wrf
from sklearn.base import BaseEstimator


class IRFClassifier(BaseEstimator):
    def __init__(self):
        self.model = wrf()
        self.predict = self.model.predict
        self.predict_proba = self.model.predict_proba

    def fit(self, X, y, lambda_reg=0.1, sample_weight=None):
        '''fit a linear model with integer coefficient and L1 regularization
        
        Params
        ------
        sample_weight: np.ndarray (n,)
            weight for each individual sample
        '''

        if 'pandas' in str(type(X)):
            X = X.values
        if 'pandas' in str(type(y)):
            y = y.values
        assert type(X) == np.ndarray, 'inputs should be ndarrays'
        assert type(y) == np.ndarray, 'inputs should be ndarrays'

        self.model.fit(X, y, keep_record=False)
