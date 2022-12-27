import copy
from abc import ABC, abstractmethod
import warnings
from functools import partial

import numpy as np
import scipy as sp

from sklearn.linear_model import RidgeCV, Ridge, LogisticRegression, \
    HuberRegressor
from sklearn.metrics import log_loss, mean_squared_error


class _PartialPredictionModelBase(ABC):
    """
    An interface for partial prediction models, objects that make use of a
    block partitioned data object, fits a regression or classification model
    on all the data, and for each block k, applies the model on a modified copy
    of the data (by either imputing the mean of each feature in block k or
    imputing the mean of each feature not in block k.)

    Parameters
    ----------
    estimator: scikit estimator object
        The regression or classification model used to obtain predictions.

    """

    def __init__(self, estimator):
        self.estimator = copy.deepcopy(estimator)
        self.n_blocks = None
        self._partial_preds = dict({})
        self._full_preds = None
        self.is_fitted = False

    def fit(self, train_blocked_data, y_train, test_blocked_data, y_test=None,
            mode="keep_k"):
        """
        Fit the partial predictions model. The regression or classification
        model is fit on the training data, and then the predictions are made
        and scored on the test data. Test and training data can be the same.

        Parameters
        ----------
        train_blocked_data: BlockPartitionedData object
            Training covariate data
        y_train: ndarray of shape (n_samples, n_targets)
            Training response data
        test_blocked_data: BlockPartitionedData object
            Test covariate data
        y_test: ndarray of shape (n_samples, n_targets)
            Training response data
        mode: string in {"keep_k", "keep_rest"}
            Mode for the method. "keep_k" imputes the mean of each feature not
            in block k, "keep_rest" imputes the mean of each feature in block k

        """
        self.n_blocks = train_blocked_data.n_blocks
        self._fit_model(train_blocked_data, y_train)
        self._full_preds = \
            self._fit_full_predictions(test_blocked_data, y_test)
        for k in range(self.n_blocks):
            self._partial_preds[k] = \
                self._fit_partial_predictions(k, mode, test_blocked_data,
                                              y_test)
        self.is_fitted = True

    @abstractmethod
    def _fit_model(self, train_blocked_data, y_train):
        """
        Fit the regression or classification model on all the data.
        """
        pass

    @abstractmethod
    def _fit_full_predictions(self, test_blocked_data, y_test=None):
        """
        Calculate the predictions made by model on all the data.
        """
        pass

    @abstractmethod
    def _fit_partial_predictions(self, k, mode, test_blocked_data,
                                 y_test=None):
        """
        Calculate the predictions made by the model on modified copies of the
        data.
        """
        pass

    def get_full_predictions(self):
        """
        Get the predictions made by model on all the data.
        """
        return self._full_preds

    def get_partial_predictions(self, k):
        """
        Get the predictions made by the model on modified copies of the data.
        """
        return self._partial_preds[k]

    def get_all_partial_predictions(self):
        return self._partial_preds


class GenericPPM(_PartialPredictionModelBase, ABC):
    """
    Partial prediction model for arbitrary estimators. May be slow.
    """

    def __init__(self, estimator):
        super().__init__(estimator)

    def _fit_model(self, train_blocked_data, y_train):
        self.estimator.fit(train_blocked_data.get_all_data(), y_train)

    def _fit_full_predictions(self, test_blocked_data, y_test=None):
        pred_func = self._get_pred_func()
        return pred_func(test_blocked_data.get_all_data())

    def _fit_partial_predictions(self, k, mode, test_blocked_data,
                                 y_test=None):
        pred_func = self._get_pred_func()
        modified_data = test_blocked_data.get_modified_data(k, mode)
        return pred_func(modified_data)

    def _get_pred_func(self):
        if hasattr(self.estimator, "predict_proba"):
            pred_func = self.estimator.predict_proba
        else:
            pred_func = self.estimator.predict
        return pred_func


class GlmPPM(_PartialPredictionModelBase, ABC):
    """
    PPM class for GLM estimator. The GLM estimator is assumed to have a single
    regularization hyperparameter accessible as a named attribute called either
    "alpha" or "C". When fitting, the PPM class will select this hyperparameter
    using efficient approximate leave-one-out calculations.

    Parameters
    ----------
    estimator: scikit estimator object
        The regression or classification model used to obtain predictions.
    loo: bool
        Flag for whether to also use LOO calculations for making predictions.
    alpha_grid: ndarray of shape (n_alphas, )
        The grid of alpha values for hyperparameter optimization.
    inv_link_fn: function
        The inverse of the GLM link function.
    l_dot: function
        The first derivative of the log likelihood (with respect to the linear
        predictor), as a function of the linear predictor and the response y.
    l_doubledot: function
        The second derivative of the log likelihood (with respect to the linear
        predictor), as a function of the linear predictor and the true response
        y.
    r_doubledot: function
        The second derivative of the regularizer with respect to each
        coefficient. We assume that the regularizer is separable and symmetric
        with respect to the coefficients.
    hyperparameter_scorer: function
        The function used to evaluate different hyperparameter values.
        Typically, this is the loglikelihood as a function of the linear
        predictor and the true response y.
    trim: float
        The amount by which to trim predicted probabilities away from 0 and 1.
        This helps to stabilize some loss calculations.
    """

    def __init__(self, estimator, loo=True, alpha_grid=np.logspace(-4, 4, 10),
                 inv_link_fn=lambda a: a, l_dot=lambda a, b: b - a,
                 l_doubledot=lambda a, b: 1, r_doubledot=lambda a: 1,
                 hyperparameter_scorer=mean_squared_error,
                 trim=None):
        super().__init__(estimator)
        self.loo = loo
        self.alpha_grid = alpha_grid
        self.inv_link_fn = inv_link_fn
        self.l_dot = l_dot
        self.l_doubledot = l_doubledot
        self.r_doubledot = r_doubledot
        self.trim = trim
        self.hyperparameter_scorer = hyperparameter_scorer
        self.alpha_ = None
        self.loo_coefficients_ = None
        self.coefficients_ = None
        self._intercept_pred = None

    def _fit_model(self, train_blocked_data, y_train):
        # Compute regularization hyperparameter using approximate LOOCV
        X_train = train_blocked_data.get_all_data()
        if isinstance(self.estimator, Ridge):
            cv = RidgeCV(alphas=self.alpha_grid)
            cv.fit(X_train, y_train)
            self.alpha_ = cv.alpha_
        else:
            self.alpha_ = self._get_aloocv_alpha(X_train, y_train)
        # Fit the model on the training set and compute the coefficients
        if self.loo:
            self.loo_coefficients_ = \
                self._fit_loo_coefficients(X_train, y_train, self.alpha_)
        else:
            self.coefficients_ = \
                self._fit_coefficients(X_train, y_train, self.alpha_)

    def _fit_full_predictions(self, test_blocked_data, y_test=None):
        X_test = test_blocked_data.get_all_data()
        if self.loo:
            preds = _get_preds(X_test, self.loo_coefficients_,
                               self.inv_link_fn)
        else:
            preds = _get_preds(X_test, self.coefficients_,
                               self.inv_link_fn)
        return _trim_values(preds, self.trim)

    def _fit_partial_predictions(self, k, mode, test_blocked_data,
                                 y_test=None):
        if mode == "keep_k":
            block_indices = test_blocked_data.get_block_indices(k)
            data_block = test_blocked_data.get_block(k)
        elif mode == "keep_rest":
            block_indices = test_blocked_data.get_all_except_block_indices(k)
            data_block = test_blocked_data.get_all_except_block(k)
        else:
            raise ValueError("Invalid mode")
        if len(block_indices) == 0: # If empty block
            return self.intercept_pred
        else:
            if self.loo:
                coefs = self.loo_coefficients_[:, block_indices]
                intercept = self.loo_coefficients_[:, -1]
            else:
                coefs = self.coefficients_[block_indices]
                intercept = self.coefficients_[-1]
            return _trim_values(_get_preds(data_block, coefs, self.inv_link_fn,
                                           intercept), self.trim)

    @property
    def intercept_pred(self):
        if self._intercept_pred is None:
            self._intercept_pred = \
                _trim_values(self.inv_link_fn(self.coefficients_[-1]),
                             self.trim)
        return self._intercept_pred

    def _fit_coefficients(self, X, y, alpha):
        _set_alpha(self.estimator, alpha)
        self.estimator.fit(X, y)
        self.coefficients_ = _extract_coef_and_intercept(self.estimator)
        return self.coefficients_

    def _fit_loo_coefficients(self, X, y, alpha):
        """
        Get the coefficient (and intercept) for each LOO model. Since we fit
        one model for each sample, this gives an ndarray of shape (n_samples,
        n_features + 1)
        """
        orig_coef_ = self._fit_coefficients(X, y, alpha)
        X1 = np.hstack([X, np.ones((X.shape[0], 1))])
        orig_preds = _get_preds(X, orig_coef_, self.inv_link_fn)
        l_doubledot_vals = self.l_doubledot(y, orig_preds)
        J = X1.T * l_doubledot_vals @ X1
        if self.r_doubledot is not None:
            r_doubledot_vals = self.r_doubledot(orig_coef_) * \
                               np.ones_like(orig_coef_)
            r_doubledot_vals[-1] = 0 # Do not penalize constant term
            reg_curvature = np.diag(r_doubledot_vals)
            J += alpha * reg_curvature
        normal_eqn_mat = np.linalg.inv(J) @ X1.T
        h_vals = np.sum(X1.T * normal_eqn_mat, axis=0) * l_doubledot_vals
        loo_coef_ = orig_coef_[:, np.newaxis] + \
                    normal_eqn_mat * self.l_dot(y, orig_preds) / (1 - h_vals)
        return loo_coef_.T

    def _get_aloocv_alpha(self, X, y):
        cv_scores = np.zeros_like(self.alpha_grid)
        for i, alpha in enumerate(self.alpha_grid):
            loo_coef_ = self._fit_loo_coefficients(X, y, alpha)
            X1 = np.hstack([X, np.ones((X.shape[0], 1))])
            sample_scores = np.sum(loo_coef_ * X1, axis=1)
            preds = _trim_values(self.inv_link_fn(sample_scores), self.trim)
            cv_scores[i] = self.hyperparameter_scorer(y, preds)
        self.alpha_ = self.alpha_grid[np.argmin(cv_scores)]
        return self.alpha_


class RidgePPM(GlmPPM, ABC):
    """
    PPM class that uses ridge as the estimator.

    Parameters
    ----------
    loo: bool
        Flag for whether to also use LOO calculations for making predictions.
    alpha_grid: ndarray of shape (n_alphas, )
        The grid of alpha values for hyperparameter optimization.
    **kwargs
        Other Parameters are passed on to Ridge().
    """

    def __init__(self, loo=True, alpha_grid=np.logspace(-5, 5, 100), **kwargs):
        super().__init__(Ridge(**kwargs), loo, alpha_grid)

    def set_alphas(self, alphas="default", blocked_data=None, y=None):
        full_data = blocked_data.get_all_data()
        if alphas == "default":
            alphas = get_alpha_grid(full_data, y)
        else:
            alphas = alphas
        self.alpha_grid = alphas


class LogisticPPM(GlmPPM, ABC):
    """
    PPM class that uses logistic regression as the estimator.

    Parameters
    ----------
    loo: bool
        Flag for whether to also use LOO calculations for making predictions.
    alpha_grid: ndarray of shape (n_alphas, )
        The grid of alpha values for hyperparameter optimization.
    max_iter: int
        The maximum number of iterations for the LogisticRegression solver.
    trim: float
        The amount by which to trim predicted probabilities away from 0 and 1.
        This helps to stabilize some loss calculations.
    **kwargs
        Other Parameters are passed on to LogisticRegression().
    """

    def __init__(self, loo=True, alpha_grid=np.logspace(-4, 4, 10),
                 max_iter=1000, trim=0.01, **kwargs):
        super().__init__(LogisticRegression(max_iter=max_iter, **kwargs),
                         loo, alpha_grid,
                         inv_link_fn=sp.special.expit,
                         l_doubledot=lambda a, b: b * (1 - b),
                         hyperparameter_scorer=log_loss,
                         trim=trim)


class RobustPPM(GlmPPM, ABC):
    """
    PPM class that uses Huber robust regression as the estimator.

    Parameters
    ----------
    loo: bool
        Flag for whether to also use LOO calculations for making predictions.
    alpha_grid: ndarray of shape (n_alphas, )
        The grid of alpha values for hyperparameter optimization.
    epsilon: float
        The robustness parameter for Huber regression. The smaller the epsilon,
        the more robust it is to outliers. Epsilon must be in the range
        [1, inf).
    **kwargs
        Other Parameters are passed on to LogisticRegression().
    """
    def __init__(self, loo=True, alpha_grid=np.linspace(0.01, 3, 100),
                 epsilon=1.35, **kwargs):
        loss_fn = partial(huber_loss, epsilon=epsilon)
        l_dot = lambda a, b: (b - a) / (1 + ((a - b) / epsilon) ** 2) ** 0.5
        l_doubledot=lambda a, b: (1 + (((a - b) / epsilon) ** 2)) ** (-1.5)
        super().__init__(
            HuberRegressor(**kwargs), loo, alpha_grid,
            l_dot=l_dot,
            l_doubledot=l_doubledot,
            hyperparameter_scorer=loss_fn)


def _trim_values(values, trim=None):
    if trim is not None:
        assert 0 < trim < 0.5, "Limit must be between 0 and 0.5"
        return np.clip(values, trim, 1 - trim)
    else:
        return values


def _extract_coef_and_intercept(estimator):
    """
    Get the coefficient vector and intercept from a GLM estimator
    """
    coef_ = estimator.coef_
    intercept_ = estimator.intercept_
    if coef_.ndim > 1:  # For classifer estimators
        coef_ = coef_.ravel()
        intercept_ = intercept_[0]
    augmented_coef_ = np.append(coef_, intercept_)
    return augmented_coef_


def _set_alpha(estimator, alpha):
    if hasattr(estimator, "alpha"):
        estimator.set_params(alpha=alpha)
    elif hasattr(estimator, "C"):
        estimator.set_params(C=1/alpha)
    else:
        warnings.warn("Estimator has no regularization parameter.")


def _get_preds(data_block, coefs, inv_link_fn, intercept=None):
    if coefs.ndim > 1: # LOO predictions
        if coefs.shape[1] == (data_block.shape[1] + 1):
            intercept = coefs[:, -1]
            coefs = coefs[:, :-1]
        lin_preds = np.sum(data_block * coefs, axis=1) + intercept
    else:
        if len(coefs) == (data_block.shape[1] + 1):
            intercept = coefs[-1]
            coefs = coefs[:-1]
        lin_preds = data_block @ coefs + intercept
    return inv_link_fn(lin_preds)


def huber_loss(y, preds, epsilon=1.35):
    total_loss = 0
    for i in range(len(y)):
        sample_absolute_error = np.abs(y[i] - preds[i])
        if sample_absolute_error < epsilon:
            total_loss += 0.5 * ((y[i] - preds[i]) ** 2)
        else:
            sample_robust_loss = epsilon * sample_absolute_error - 0.5 * \
                                 epsilon ** 2
            total_loss += sample_robust_loss
    return total_loss / len(y)


def get_alpha_grid(X, y, start=-5, stop=5, num=100):
    X = X - X.mean(axis=0)
    y = y - y.mean(axis=0)
    sigma_sq_ = np.linalg.norm(y, axis=0) ** 2 / X.shape[0]
    X_var_ = np.linalg.norm(X, axis=0) ** 2
    alpha_opts_ = (X_var_[:, np.newaxis] / (X.T @ y)) ** 2 * sigma_sq_
    base = np.max(alpha_opts_)
    alphas = np.logspace(start, stop, num=num) * base
    return alphas