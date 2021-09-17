'''
Wrapper for sparse, integer linear models.

minimizes norm(X * w - y, 2) + alpha * norm(w, 1)

with integer coefficients in w

Requires installation of a solver for mixed-integer linear programs, e.g. gurobi, mosek, or cplex
'''
import cvxpy as cp  # package for optimization
import numpy as np
import warnings
from cvxpy.error import SolverError
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.linear_model import LinearRegression, Lasso, LogisticRegression
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class SLIMRegressor(BaseEstimator, RegressorMixin):
    '''Sparse integer linear model
    Params
    ------
    alpha: float
        weight for sparsity penalty
    '''

    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def fit(self, X, y, sample_weight=None):
        '''fit a linear model with integer coefficient and L1 regularization.
        In case the optimization fails, fit lasso and round coefs.
        
        Params
        ------
        sample_weight: np.ndarray (n,), optional
            weight for each individual sample
        '''
        X, y = check_X_y(X, y)
        self.n_features_in_ = X.shape[1]
        self.model_ = LinearRegression()

        # declare the integer-valued optimization variable
        w = cp.Variable(X.shape[1], integer=True)

        # set up the minimization problem
        residuals = X @ w - y
        if sample_weight is not None:
            residuals = cp.multiply(sample_weight, residuals)

        try:
            mse = cp.sum_squares(residuals)
            l1_penalty = self.alpha * cp.norm(w, 1)
            obj = cp.Minimize(mse + l1_penalty)
            prob = cp.Problem(obj)

            # solve the problem using an appropriate solver
            prob.solve()
            self.model_.coef_ = w.value.astype(int)
            self.model_.intercept_ = 0

        except SolverError as e:
            warnings.warn("gurobi, mosek, or cplex solver required for sparse integer linear "
                          "regression. Rounding non-integer coefficients instead.")
            m = Lasso(alpha=self.alpha)
            m.fit(X, y, sample_weight=sample_weight)
            self.model_.coef_ = np.round(m.coef_).astype(int)
            self.model_.intercept_ = m.intercept_
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        return self.model_.predict(X)


class SLIMClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, alpha=1):
        '''Model is initialized during fitting

        Params
        ------
        alpha: float
            weight for sparsity penalty
        '''
        self.alpha = alpha

    def fit(self, X, y, sample_weight=None):
        '''fit a logistic model with integer coefficient and L1 regularization.
        In case the optimization fails, fit lasso and round coefs.
        
        Params
        ------
        sample_weight: np.ndarray (n,), optional
            weight for each individual sample
        '''
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        self.n_features_in_ = X.shape[1]
        self.classes_, y = np.unique(y, return_inverse=True)  # deals with str inputs
        self.model_ = LogisticRegression()
        self.model_.classes_ = self.classes_

        # declare the integer-valued optimization variable
        w = cp.Variable(X.shape[1], integer=True)

        # set up the minimization problem
        logits = -X @ w
        residuals = cp.multiply(1 - y, logits) - cp.logistic(logits)
        if sample_weight is not None:
            residuals = cp.multiply(sample_weight, residuals)

        try:
            celoss = -cp.sum(residuals)
            l1_penalty = self.alpha * cp.norm(w, 1)
            obj = cp.Minimize(celoss + l1_penalty)
            prob = cp.Problem(obj)

            # solve the problem using an appropriate solver
            prob.solve()
            self.model_.coef_ = np.array([w.value.astype(int)])
            self.model_.intercept_ = 0

        except SolverError as e:
            warnings.warn("mosek solver required for mixed-integer logistic regression. "
                          "rounding non-integer coefficients instead")
            m = LogisticRegression(C=1 / self.alpha)
            m.fit(X, y, sample_weight=sample_weight)
            self.model_.coef_ = np.round(m.coef_).astype(int)
            self.model_.intercept_ = m.intercept_

        return self

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        return self.model_.predict(X)

    def predict_proba(self, X):
        check_is_fitted(self)
        X = check_array(X)
        return self.model_.predict_proba(X)
