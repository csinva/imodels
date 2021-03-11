'''
Wrapper for sparse, integer linear models.

minimizes norm(X * w - y, 2) + lambda_reg * norm(w, 1)

with integer coefficients in w

Requires installation of a solver for mixed-integer linear programs, e.g. gurobi, mosek, or cplex
'''
import cvxpy as cp  # package for optimization
import numpy as np
import warnings
from cvxpy.error import SolverError
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression, Lasso, LogisticRegression


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
        lambda_reg: float
            weight for sparsity penalty
        sample_weight: np.ndarray (n,), optional
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
            self.model.coef_ = w.value.astype(int)
            self.model.intercept_ = 0

        except SolverError as e:
            warnings.warn("gurobi, mosek, or cplex solver required for mixed-integer linear "
                          "regression. rounding non-integer coefficients instead")
            m = Lasso(alpha=lambda_reg)
            m.fit(X, y, sample_weight=sample_weight)
            self.model.coef_ = np.round(m.coef_).astype(int)
            self.model.intercept_ = m.intercept_
        return self

    def predict_proba(self, X):
        '''Converts predicted continuous output to probabilities using softmax
        '''
        preds = self.predict(X)
        preds_proba = np.array([1 / (1 + np.exp(-y)) for y in preds])
        return np.vstack((1 - preds_proba, preds_proba)).transpose()


class SLIMClassifier(BaseEstimator):

    def __init__(self):
        self.model = LogisticRegression()
        self.predict = self.model.predict
        self.predict_proba = self.model.predict_proba

    def fit(self, X, y, lambda_reg=1, sample_weight=None):
        '''fit a logistic model with integer coefficient and L1 regularization.
        In case the optimization fails, fit lasso and round coefs.
        
        Params
        ------
        lambda_reg: float
            weight for sparsity penalty
        sample_weight: np.ndarray (n,), optional
            weight for each individual sample
        '''
        if 'pandas' in str(type(X)):
            X = X.values
        if 'pandas' in str(type(y)):
            y = y.values
        assert type(X) == np.ndarray, 'inputs should be ndarrays'
        assert type(y) == np.ndarray, 'inputs should be ndarrays'
        self.model.classes_ = np.unique(y)

        # declare the integer-valued optimization variable
        w = cp.Variable(X.shape[1], integer=True)

        # set up the minimization problem
        logits = -X @ w
        residuals = cp.multiply(1 - y, logits) - cp.logistic(logits)
        if sample_weight is not None:
            residuals = cp.multiply(sample_weight, residuals)

        try:
            celoss = -cp.sum(residuals)
            l1_penalty = lambda_reg * cp.norm(w, 1)
            obj = cp.Minimize(celoss + l1_penalty)
            prob = cp.Problem(obj)

            # solve the problem using an appropriate solver
            prob.solve()
            self.model.coef_ = np.array([w.value.astype(int)])
            self.model.intercept_ = 0

        except SolverError as e:
            warnings.warn("mosek solver required for mixed-integer logistic regression. "
                          "rounding non-integer coefficients instead")
            m = LogisticRegression(C=1 / lambda_reg)
            m.fit(X, y, sample_weight=sample_weight)
            self.model.coef_ = np.round(m.coef_).astype(int)
            self.model.intercept_ = m.intercept_

        return self
