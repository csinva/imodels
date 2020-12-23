'''
Wrapper for sparse, integer linear models.

minimizes norm(X * w - y, 2) + lambda_reg * norm(w, 1)

with integer coefficients in w
'''
import cvxpy as cp  # package for optimization
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression, Lasso


class SLIMRegressor(BaseEstimator):
    '''Sparse integer linear model
    '''

    def __init__(self):
        self.model = LinearRegression()
        self.predict = self.model.predict

    def fit(self, X, y, lambda_reg=10, sample_weight=None):
        '''fit a linear model with integer coefficient and L1 regularization.
        In case the optimization fails, fit lasso and round coefs.
        
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

        # declare the integer-valued optimization variable
        w = cp.Variable(X.shape[1], integer=True)

        # set up the minimization problem
        residuals = X @ w - y
        if sample_weight is not None:
            # print('shapes', residuals.shape, sample_weight.shape)
            residuals = cp.multiply(sample_weight, residuals)
        
        try:
            mse = cp.sum_squares(residuals)
            l1_penalty = lambda_reg * cp.norm(w, 1)
            obj = cp.Minimize(mse + l1_penalty)
            prob = cp.Problem(obj)

            # solve the problem using an appropriate solver
            prob.solve()
            self.model.coef_ = w.value.astype(np.int)
            self.model.intercept_ = 0

        except:
            m = Lasso(alpha=lambda_reg)
            m.fit(X, y, sample_weight=sample_weight)
            self.model.coef_ = np.round(m.coef_).astype(np.int)
            self.model.intercept_ = m.intercept_

    def predict_proba(self, X):
        preds = self.predict(X)
        preds_proba = np.array([1 / (1 + np.exp(-y)) for y in preds])
        return np.vstack((1 - preds_proba, preds_proba)).transpose()
