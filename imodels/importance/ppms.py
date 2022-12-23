import copy
from abc import ABC, abstractmethod
import warnings
from functools import partial

import numpy as np
import scipy as sp
from skimage.metrics import mean_squared_error

from sklearn.linear_model import RidgeCV, LogisticRegressionCV, Ridge, \
    LogisticRegression, HuberRegressor
from sklearn.metrics import log_loss


class PartialPredictionModelBase(ABC):
    """
    An interface for partial prediction models, objects that make use of a
    block partitioned data object, fits a regression or classification model
    on all the data, and for each block k, applies the model on a modified copy
    of the data (by either imputing the mean of each feature in block k or
    imputing the mean of each feature not in block k.)

    Parameters
    ----------
    estimator: scikit estimator object
        The regression or classification model used to obtain predictions

    """

    def __init__(self, estimator):
        self.estimator = estimator
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
        self._full_preds = self._fit_full_predictions(test_blocked_data,
                                                      y_test)
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
    def _fit_partial_predictions(self, k, mode, test_blocked_data, y_test=None):
        """
        Calculate the predictions made by the model on modified copies of the
        data.
        """
        pass

    def get_partial_predictions(self, k):
        """
        Get the predictions made by the model on modified copies of the data.
        """
        return self._partial_preds[k]

    def get_full_predictions(self):
        """
        Get the predictions made by model on all the data.
        """
        return self._full_preds


class GenericPPM(PartialPredictionModelBase, ABC):
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

    def _fit_partial_predictions(self, k, mode, test_blocked_data, y_test=None):
        pred_func = self._get_pred_func()
        modified_data = test_blocked_data.get_modified_data(k, mode)
        return pred_func(modified_data)

    def _get_pred_func(self):
        if hasattr(self.estimator, "predict_proba"):
            pred_func = self.estimator.predict_proba
        else:
            pred_func = self.estimator.predict
        return pred_func


class GlmPPM(PartialPredictionModelBase, ABC):
    """
    PPM class for GLM predictors. Not fully implemented yet.
    """

    def __init__(self, estimator, alpha_grid=np.logspace(-4, 4, 10),
                 link_fn=lambda a: a, l_dot=lambda a, b: b - a,
                 l_doubledot=lambda a, b: 1, r_doubledot=lambda a: 1,
                 hyperparameter_scorer=mean_squared_error,
                 trim=None):
        super().__init__(estimator)
        self.aloo_calculator = _GlmAlooCalculator(copy.deepcopy(self.estimator), alpha_grid, link_fn=link_fn,
                                                  l_dot=l_dot, l_doubledot=l_doubledot, r_doubledot=r_doubledot,
                                                  hyperparameter_scorer=hyperparameter_scorer, trim=trim)
        self.trim = trim

    def _fit_model(self, train_blocked_data, y_train):
        self.alpha_ = self.aloo_calculator.get_aloocv_alpha(train_blocked_data.get_all_data(), y_train)
        if hasattr(self.estimator, "alpha"):
            self.estimator.set_params(alpha=self.alpha_)
        elif hasattr(self.estimator, "C"):
            self.estimator.set_params(C=1 / self.alpha_)
        else:
            warnings.warn("Estimator has no regularization parameter.")


class RidgePPM(PartialPredictionModelBase, ABC):
    """
    PPM class with ridge as the regression model (default).
    """

    def __init__(self, **kwargs):
        super().__init__(estimator=RidgeCV(**kwargs))

    def _fit_model(self, train_blocked_data, y_train):
        self.estimator.fit(train_blocked_data.get_all_data(), y_train)

    def _fit_full_predictions(self, test_blocked_data, y_test=None):
        return self.estimator.predict(test_blocked_data.get_all_data())

    def _fit_partial_predictions(self, k, mode, test_blocked_data,
                                 y_test=None):
        if mode == "keep_k":
            # Instead of modifying the data, we subset only those features
            # in the block to reduce computational cost. This is equivalent
            # to imputing the mean if the data is centered
            col_indices = test_blocked_data.get_block_indices(k)
            reduced_data = test_blocked_data.get_block(k)
        elif mode == "keep_rest":
            col_indices = test_blocked_data.get_all_except_block_indices(k)
            reduced_data = test_blocked_data.get_all_except_block(k)
        else:
            raise ValueError("Invalid mode")
        partial_coef_ = np.append(self.estimator.coef_,
                                  self.estimator.intercept_)
        partial_preds = reduced_data @ self.estimator.coef_[col_indices] + \
                        self.estimator.intercept_
        return partial_preds, partial_coef_

    def set_alphas(self, alphas="default", blocked_data=None, y=None):
        full_data = blocked_data.get_all_data()
        if alphas == "default":
            alphas = get_alpha_grid(full_data, y)
        else:
            alphas = alphas
        self.estimator = RidgeCV(alphas=alphas)


class LogisticPPM(PartialPredictionModelBase, ABC):

    def __init__(self, loo_model_selection=True, alphas=np.logspace(-4, 4, 10),
                 trim=0.01, **kwargs):
        if loo_model_selection:
            self.alphas = alphas
            super().__init__(estimator=LogisticRegression(**kwargs))
        else:
            super().__init__(estimator=LogisticRegressionCV(alphas, **kwargs))
        self.loo_model_selection = loo_model_selection
        self.trim = trim

    def _fit_model(self, train_blocked_data, y_train):
        if self.loo_model_selection:
            aloo_calculator = _GlmAlooCalculator(copy.deepcopy(self.estimator), self.alphas, link_fn=sp.special.expit,
                                                 l_doubledot=lambda a, b: b * (1 - b), hyperparameter_scorer=log_loss,
                                                 trim=self.trim)
            alpha_ = aloo_calculator.get_aloocv_alpha(train_blocked_data.get_all_data(), y_train)
            self.estimator.set_params(C=1 / alpha_)
            self.estimator.fit(train_blocked_data.get_all_data(), y_train)
        else:
            self.estimator.fit(train_blocked_data.get_all_data(), y_train)

    def _fit_full_predictions(self, test_blocked_data, y_test=None):
        return self._trim_values(self.estimator.predict_proba(test_blocked_data.get_all_data()))

    def _fit_partial_predictions(self, k, mode, test_blocked_data, y_test=None):
        if mode == "keep_k":
            col_indices = test_blocked_data.get_block_indices(k)
            reduced_data = test_blocked_data.get_block(k)
        elif mode == "keep_rest":
            col_indices = test_blocked_data.get_all_except_block_indices(k)
            reduced_data = test_blocked_data.get_all_except_block(k)
        else:
            raise ValueError("Invalid mode")
        coef_, intercept_ = _extract_coef_and_intercept(self.estimator)
        reduced_coef_ = coef_[col_indices]
        return self._trim_values(sp.special.expit(reduced_data @ reduced_coef_ + intercept_)), reduced_coef_

    def _trim_values(self, values):
        if self.trim is not None:
            assert 0 < self.trim < 0.5, "Limit must be between 0 and 0.5"
            return np.clip(values, self.trim, 1 - self.trim)
        else:
            return values


class GenericLOOPPM(PartialPredictionModelBase, ABC):
    """
    PPM class that fits (approximate) leave-one-out predictions for a GLM
    estimator.

    Parameters
    ----------
    estimator: scikit estimator object
        The regression or classification model used to obtain predictions
    alpha_grid: ndarray
        The values of the regularization hyperparameter alpha to try.
    link_fn: function
        Vectorized GLM link function
    l_dot: function
        Vectorized first derivative of the link function
    l_doubledot: function
        Vectorized second derivative of the link function
    r_doubledot: function
        Vectorized second derivative of the regularizer function
    hyperparameter_scorer: function
        Vectorized function to compute loss used to select the regularization
        hyperparameter alpha.
    trim: float or None
        The amount by which to trim the predictitons (used when estimator is
        a classifier model as extreme predicted probabilities may be unstable.)
    fixed_intercept: bool
        If True, use

    """

    def __init__(self, estimator, alpha_grid=np.logspace(-4, 4, 10),
                 link_fn=lambda a: a, l_dot=lambda a, b: b - a,
                 l_doubledot=lambda a, b: 1, r_doubledot=lambda a: 1,
                 hyperparameter_scorer=mean_squared_error,
                 trim=None, fixed_intercept=True):

        super().__init__(estimator)
        self.aloo_calculator = \
            _GlmAlooCalculator(copy.deepcopy(self.estimator), alpha_grid,
                               link_fn=link_fn, l_dot=l_dot,
                               l_doubledot=l_doubledot,
                               r_doubledot=r_doubledot,
                               hyperparameter_scorer=hyperparameter_scorer,
                               trim=trim)
        self.trim = trim
        self.fixed_intercept = fixed_intercept

    def _fit_model(self, train_blocked_data, y_train):
        self.alpha_ = self.aloo_calculator. \
            get_aloocv_alpha(train_blocked_data.get_all_data(), y_train)
        if hasattr(self.estimator, "alpha"):
            self.estimator.set_params(alpha=self.alpha_)
        elif hasattr(self.estimator, "C"):
            self.estimator.set_params(C=1 / self.alpha_)
        else:
            warnings.warn("Estimator has no regularization parameter.")

    def _fit_full_predictions(self, test_blocked_data, y_test=None):
        if y_test is None:
            raise ValueError("Need to supply y_test for LOO")
        X1 = np.hstack([test_blocked_data.get_all_data(),
                        np.ones((test_blocked_data.n_samples, 1))])
        fitted_parameters = self.aloo_calculator \
            .get_aloo_fitted_parameters(test_blocked_data.get_all_data(),
                                        y_test, self.alpha_, cache=True)
        full_preds = self.aloo_calculator.score_to_pred(
            np.sum(fitted_parameters.T * X1, axis=1))

        return full_preds

    def _fit_partial_predictions(self, k, mode, test_blocked_data,
                                 y_test=None):
        if y_test is None:
            raise ValueError("Need to supply y_test for LOO")
        if mode == "keep_k":
            col_indices = test_blocked_data.get_block_indices(k)
            reduced_data = test_blocked_data.get_block(k)
        elif mode == "keep_rest":
            col_indices = test_blocked_data.get_all_except_block_indices(k)
            reduced_data = test_blocked_data.get_all_except_block(k)
        else:
            raise ValueError("Invalid mode")
        reduced_data1 = np.hstack([reduced_data,
                                   np.ones((test_blocked_data.n_samples, 1))])
        col_indices = np.append(col_indices, -1)
        if self.fixed_intercept and len(col_indices) == 1:
            _, intercept = _extract_coef_and_intercept(self.aloo_calculator.estimator)
            return np.repeat(self.aloo_calculator.score_to_pred(intercept), len(y_test)), [
                intercept]  # returning learnt intercept for null model
        else:
            fitted_parameters = self.aloo_calculator.get_aloo_fitted_parameters()
            reduced_parameters = fitted_parameters.T[:, col_indices]
            partial_preds = self.aloo_calculator.score_to_pred(
                np.sum(reduced_parameters * reduced_data1, axis=1))
            return partial_preds, reduced_parameters  # returning learnt parameters for partial model

    def _trim_values(self, values):
        if self.trim is not None:
            assert 0 < self.trim < 0.5, "Limit must be between 0 and 0.5"
            return np.clip(values, self.trim, 1 - self.trim)
        else:
            return values


class RidgeLOOPPM(GenericLOOPPM, ABC):
    def __init__(self, alpha_grid=np.logspace(-5, 5, 100),
                 fixed_intercept=True, **kwargs):
        super().__init__(Ridge(**kwargs), alpha_grid,
                         fixed_intercept=fixed_intercept)

    def set_alphas(self, alphas="default", blocked_data=None, y=None):
        full_data = blocked_data.get_all_data()
        if alphas == "default":
            alphas = get_alpha_grid(full_data, y)
        else:
            alphas = alphas
        self.aloo_calculator.alpha_grid = alphas


class RobustLOOPPM(GenericLOOPPM, ABC):
    def __init__(self, alpha_grid=np.linspace(0.01, 3, 100),
                 fixed_intercept=True, epsilon=1.35, **kwargs):
        loss_fn = partial(huber_loss, epsilon=epsilon)
        l_dot = lambda a, b: (b - a) / (1 + ((a - b) / epsilon) ** 2) ** 0.5
        l_doubledot=lambda a, b: (1 + (((a - b) / epsilon) ** 2)) ** (-1.5)
        super().__init__(
            HuberRegressor(**kwargs), alpha_grid, l_dot=l_dot,
            l_doubledot=l_doubledot,
            hyperparameter_scorer=loss_fn, fixed_intercept=fixed_intercept)


class LogisticLOOPPM(GenericLOOPPM, ABC):

    def __init__(self, alpha_grid=np.logspace(-4, 4, 10),
                 fixed_intercept=True, **kwargs):
        super().__init__(LogisticRegression(**kwargs), alpha_grid,
                         link_fn=sp.special.expit,
                         l_doubledot=lambda a, b: b * (1 - b),
                         hyperparameter_scorer=log_loss,
                         trim=0.01, fixed_intercept=fixed_intercept)


class _GlmAlooCalculator:
    """
    Class to perform approximate leave-one-out calculations
    """

    def __init__(self, estimator, alpha_grid=np.logspace(-4, 4, 10),
                 link_fn=lambda a: a, l_dot=lambda a, b: b - a,
                 l_doubledot=lambda a, b: 1, r_doubledot=lambda a: 1,
                 hyperparameter_scorer=mean_squared_error,
                 trim=None):
        super().__init__()
        self.estimator = estimator
        self.alpha_grid = alpha_grid
        self.link_fn = link_fn
        self.l_dot = l_dot
        self.l_doubledot = l_doubledot
        self.r_doubledot = r_doubledot
        self.trim = trim
        self.hyperparameter_scorer = hyperparameter_scorer
        self.alpha_ = None
        self.loo_fitted_parameters = None

    def get_aloo_fitted_parameters(self, X=None, y=None, alpha=None,
                                   cache=False):
        """
        Get the coefficient (and intercept) for each LOO model. Since we fit
        one model for each sample, this gives an ndarray of shape (n_samples,
        n_features + 1)
        """
        if self.loo_fitted_parameters is not None:
            return self.loo_fitted_parameters
        else:
            if hasattr(self.estimator, "alpha"):
                self.estimator.set_params(alpha=alpha)
            elif hasattr(self.estimator, "C"):
                self.estimator.set_params(C=1 / alpha)
            else:
                alpha = 0
            estimator = copy.deepcopy(self.estimator)
            estimator.fit(X, y)
            X1 = np.hstack([X, np.ones((X.shape[0], 1))])
            augmented_coef_ = _extract_coef_and_intercept(estimator,
                                                          merge=True)
            orig_preds = self.link_fn(X1 @ augmented_coef_)
            l_doubledot_vals = self.l_doubledot(y, orig_preds)
            J = X1.T * l_doubledot_vals @ X1
            if self.r_doubledot is not None:
                r_doubledot_vals = self.r_doubledot(augmented_coef_) * \
                                   np.ones_like(augmented_coef_)
                r_doubledot_vals[-1] = 0 # Do not penalize constant term
                reg_curvature = np.diag(r_doubledot_vals)
                J += alpha * reg_curvature
            normal_eqn_mat = np.linalg.inv(J) @ X1.T
            h_vals = np.sum(X1.T * normal_eqn_mat, axis=0) * l_doubledot_vals
            loo_fitted_parameters = \
                augmented_coef_[:, np.newaxis] + \
                normal_eqn_mat * self.l_dot(y, orig_preds) / (1 - h_vals)
            if cache:
                self.loo_fitted_parameters = loo_fitted_parameters
                self.estimator = estimator
            return loo_fitted_parameters

    def score_to_pred(self, score):
        return self._trim_values(self.link_fn(score))

    def get_aloocv_alpha(self, X, y):
        cv_scores = np.zeros_like(self.alpha_grid)
        for i, alpha in enumerate(self.alpha_grid):
            loo_fitted_parameters = self.get_aloo_fitted_parameters(X, y,
                                                                    alpha)
            X1 = np.hstack([X, np.ones((X.shape[0], 1))])
            preds = self.score_to_pred(np.sum(loo_fitted_parameters.T * X1,
                                              axis=1))
            cv_scores[i] = self.hyperparameter_scorer(y, preds)
        self.alpha_ = self.alpha_grid[np.argmin(cv_scores)]
        return self.alpha_

    def _trim_values(self, values):
        if self.trim is not None:
            assert 0 < self.trim < 0.5, "Limit must be between 0 and 0.5"
            return np.clip(values, self.trim, 1 - self.trim)
        else:
            return values


def _extract_coef_and_intercept(estimator, merge=False):
    """
    Get the coefficient vector and intercept from a GLM estimator
    """
    coef_ = estimator.coef_
    intercept_ = estimator.intercept_
    if coef_.ndim > 1:  # For classifer estimators
        coef_ = coef_.ravel()
        intercept_ = intercept_[0]
    if merge:
        augmented_coef_ = np.append(coef_, intercept_)
        return augmented_coef_
    else:
        return coef_, intercept_


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