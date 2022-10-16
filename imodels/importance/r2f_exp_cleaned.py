import copy
from abc import ABC, abstractmethod

import numpy as np
import scipy as sp
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble._forest import _generate_unsampled_indices, _generate_sample_indices
from sklearn.linear_model import RidgeCV, LogisticRegressionCV, Ridge, LogisticRegression
from sklearn.metrics import roc_auc_score, mean_squared_error, log_loss
from sklearn.preprocessing import OneHotEncoder

from imodels.importance.representation_cleaned import TreeTransformer, IdentityTransformer, CompositeTransformer


def GMDI_pipeline(X, y, fit, regression=True, mode="keep_k", 
                  partial_prediction_model="auto", scoring_fn="auto",
                  include_raw=True, drop_features=True, oob=False):

    p = X.shape[1]
    fit = copy.deepcopy(fit)
    if include_raw:
        tree_transformers = [CompositeTransformer([TreeTransformer(p, tree_model),
                                                    IdentityTransformer(p)], adj_std="max", drop_features = drop_features)
                            for tree_model in fit.estimators_]
    else:
        tree_transformers = [TreeTransformer(p, tree_model) for tree_model in fit.estimators_]

    if partial_prediction_model == "auto":
        if regression:
            partial_prediction_model = RidgeLOOPPM()
        else:
            partial_prediction_model = LogisticLOOPPM(max_iter=1000)
    if scoring_fn == "auto":
        if regression:
            def r2_score(y_true, y_pred):
                numerator = ((y_true - y_pred) ** 2).sum(axis=0, dtype=np.float64)
                denominator = ((y_true - np.mean(y_true, axis=0)) ** 2).sum(axis=0, dtype=np.float64)
                return 1 - numerator / denominator
            scoring_fn = r2_score
        else:
            scoring_fn = roc_auc_score
    if not regression:
        if len(np.unique(y)) > 2:
            y = OneHotEncoder().fit_transform(y.reshape(-1, 1)).toarray()
    
    gmdi = GMDIEnsemble(tree_transformers, partial_prediction_model, scoring_fn, mode, oob)
    scores = gmdi.get_scores(X, y)
    
    results = pd.DataFrame(data={'importance': scores})

    if isinstance(X, pd.DataFrame):
        results.index = X.columns
    results.index.name = 'var'
    results.reset_index(inplace=True)

    return results


class GMDI:

    def __init__(self, transformer, partial_prediction_model, scoring_fn, mode="keep_k", oob=False):
        self.transformer = transformer
        self.partial_prediction_model = partial_prediction_model
        self.scoring_fn = scoring_fn
        self.mode = mode
        self.n_features = None
        self._scores = None
        self.is_fitted = False
        self.oob = oob

    def _fit_importance_scores(self, X, y):
        if self.oob:
            n_samples = len(y)
            train_indices = _generate_sample_indices(self.transformer.estimator.random_state, n_samples, n_samples)
            test_indices = _generate_unsampled_indices(self.transformer.estimator.random_state, n_samples, n_samples)
            all_data = self.transformer.transform(X)
            train_blocked_data, test_blocked_data = all_data.train_test_split(train_indices, test_indices)
            y_train = y[train_indices]
            y_test = y[test_indices]
        else:
            blocked_data = self.transformer.transform(X)
            train_blocked_data = blocked_data
            test_blocked_data = blocked_data
            y_train = y
            y_test = y
        if train_blocked_data.get_all_data().shape[1] == 0: #checking if learnt representation is empty
            self._scores = np.zeros(X.shape[1])
            for k in range(X.shape[1]):
                self._scores[k] = np.NaN
        else:
            self.partial_prediction_model.fit(train_blocked_data, y_train, test_blocked_data, y_test, self.mode)
            self.n_features = self.partial_prediction_model.n_blocks
            self._scores = np.zeros(self.n_features)
            if self.mode == "keep_k":
                for k in range(self.n_features): #checking if there are any stumps associated to the feature in the tree
                    partial_preds = self.partial_prediction_model.get_partial_predictions(k)
                    self._scores[k] = self.scoring_fn(y_test, partial_preds)
            elif self.mode == "keep_rest":
                full_preds = self.partial_prediction_model.get_full_predictions()
                full_score = self.scoring_fn(y_test, full_preds)
                for k in range(self.n_features):
                    partial_preds = self.partial_prediction_model.get_partial_predictions(k)
                    self._scores[k] = full_score - self.scoring_fn(y_test, partial_preds)
            self.is_fitted = True

    def get_scores(self, X=None, y=None):
        if self.is_fitted:
            pass
        else:
            if X is None or y is None:
                raise ValueError("Not yet fitted. Need X and y as inputs.")
            else:
                self._fit_importance_scores(X, y)
        return self._scores


class GMDIEnsemble:

    def __init__(self, transformers, partial_prediction_model, scoring_fn, mode="keep_k", oob=False):
        self.n_transformers = len(transformers)
        self.gmdi_objects = [GMDI(transformer, copy.deepcopy(partial_prediction_model), scoring_fn, mode, oob)
                             for transformer in transformers]
        self.oob = oob
        self.scoring_fn = scoring_fn
        self.mode = mode
        self.n_features = None
        self._scores = None
        self.is_fitted = False

    def _fit_importance_scores(self, X, y):
        assert X.shape[0] == len(y)
        # n_samples = len(y)
        scores = []
        for gmdi_object in self.gmdi_objects:
            # if self.subsetting_scheme is None:
            #     sample_indices = list(range(n_samples))
            # else:
            #     estimator = gmdi_object.transformer.estimator
            #     if self.subsetting_scheme == "oob":
            #         sample_indices = _generate_unsampled_indices(estimator.random_state, n_samples, n_samples)
            #     elif self.subsetting_scheme == "inbag":
            #         sample_indices = _generate_sample_indices(estimator.random_state, n_samples, n_samples)
            #     else:
            #         raise ValueError("Unsupported subsetting scheme")
            scores.append(gmdi_object.get_scores(X, y))
        self._scores = np.nanmean(scores, axis=0)
        self.is_fitted = True
        self.n_features = self.gmdi_objects[0].n_features

    def get_scores(self, X=None, y=None):
        if self.is_fitted:
            pass
        else:
            if X is None or y is None:
                raise ValueError("Not yet fitted. Need X and y as inputs.")
            else:
                self._fit_importance_scores(X, y)
        return self._scores


class PartialPredictionModelBase(ABC):

    def __init__(self, estimator):
        self.estimator = estimator
        self.n_blocks = None
        self._partial_preds = dict({})
        self._full_preds = None
        self.is_fitted = False

    def fit(self, train_blocked_data, y_train, test_blocked_data, y_test=None, mode="keep_k"):
        self.n_blocks = train_blocked_data.n_blocks
        self._fit_model(train_blocked_data, y_train)
        self._full_preds = self._fit_full_predictions(test_blocked_data, y_test)
        for k in range(self.n_blocks):
            self._partial_preds[k] = self._fit_partial_predictions(k, mode, test_blocked_data, y_test)
        self.is_fitted = True

    @abstractmethod
    def _fit_model(self, train_blocked_data, y_train):
        pass

    @abstractmethod
    def _fit_full_predictions(self, test_blocked_data, y_test=None):
        pass

    @abstractmethod
    def _fit_partial_predictions(self, k, mode, test_blocked_data, y_test=None):
        pass

    def get_partial_predictions(self, k):
        return self._partial_preds[k]

    def get_full_predictions(self):
        return self._full_preds


class GenericPPM(PartialPredictionModelBase, ABC):
    """
    Partial prediction model logic for arbitrary estimators. May be slow.
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
    PPM class for GLM predictors. Not implemented yet.
    """
    pass


class RidgePPM(PartialPredictionModelBase, ABC):
    """
    PPM class for ridge (default).
    """

    def __init__(self, **kwargs):
        super().__init__(estimator=RidgeCV(**kwargs))

    def _fit_model(self, train_blocked_data, y_train):
        self.estimator.fit(train_blocked_data.get_all_data(), y_train)

    def _fit_full_predictions(self, test_blocked_data, y_test=None):
        return self.estimator.predict(test_blocked_data.get_all_data())

    def _fit_partial_predictions(self, k, mode, test_blocked_data, y_test=None):
        if mode == "keep_k":
            col_indices = test_blocked_data.get_block_indices(k)
            reduced_data = test_blocked_data.get_block(k)
        elif mode == "keep_rest":
            col_indices = test_blocked_data.get_all_except_block_indices(k)
            reduced_data = test_blocked_data.get_all_except_block(k)
        else:
            raise ValueError("Invalid mode")
        return reduced_data @ self.estimator.coef_[col_indices] + self.estimator.intercept_

    def set_alphas(self, alphas="default", blocked_data=None, y=None):
        full_data = blocked_data.get_all_data()
        if alphas == "default":
            alphas = get_alpha_grid(full_data, y)
        else:
            alphas = alphas
        self.estimator = RidgeCV(alphas=alphas)


class LogisticPPM(PartialPredictionModelBase, ABC):

    def __init__(self, loo_model_selection=True, alphas=np.logspace(-4, 4, 10), trim=0.01,
                 **kwargs):
        if loo_model_selection:
            self.alphas = alphas
            super().__init__(estimator=LogisticRegression(**kwargs))
        else:
            super().__init__(estimator=LogisticRegressionCV(alphas, **kwargs))
        self.loo_model_selection = loo_model_selection
        self.trim = trim

    def _fit_model(self, train_blocked_data, y_train):
        if self.loo_model_selection:
            aloo_calculator = GlmAlooCalculator(copy.deepcopy(self.estimator), self.alphas, link_fn=sp.special.expit,
                                                l_doubledot=lambda a, b: b * (1-b), hyperparameter_scorer=log_loss,
                                                trim=self.trim)
            alpha_ = aloo_calculator.get_aloocv_alpha(train_blocked_data.get_all_data(), y_train)
            self.estimator.set_params(C=1/alpha_)
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
        coef_, intercept_ = extract_coef_and_intercept(self.estimator)
        reduced_coef_ = coef_[col_indices]
        return self._trim_values(sp.special.expit(reduced_data @ reduced_coef_ + intercept_))

    def _trim_values(self, values):
        if self.trim is not None:
            assert 0 < self.trim < 0.5, "Limit must be between 0 and 0.5"
            return np.clip(values, self.trim, 1 - self.trim)
        else:
            return values


class GenericLOOPPM(PartialPredictionModelBase, ABC):

    def __init__(self, estimator, alpha_grid=np.logspace(-4, 4, 10), link_fn=lambda a: a, l_dot=lambda a, b: b-a,
                 l_doubledot=lambda a, b: 1, r_doubledot=lambda a: 1, hyperparameter_scorer=mean_squared_error,
                 trim=None, fixed_intercept=True):
        super().__init__(estimator)
        self.aloo_calculator = GlmAlooCalculator(copy.deepcopy(self.estimator), alpha_grid, link_fn=link_fn,
                                                 l_doubledot=l_doubledot, r_doubledot=r_doubledot,
                                                 hyperparameter_scorer=hyperparameter_scorer, trim=trim)
        self.trim = trim
        self.fixed_intercept = fixed_intercept
        self.multi_target = False

    def _fit_model(self, train_blocked_data, y_train):
        if y_train.ndim == 1:
            self.alpha_ = self.aloo_calculator.get_aloocv_alpha(train_blocked_data.get_all_data(), y_train)
            if hasattr(self.estimator, "alpha"):
                self.estimator.set_params(alpha=self.alpha_)
            elif hasattr(self.estimator, "C"):
                self.estimator.set_params(C=1/self.alpha_)
        else:
            self.multi_target = True
            self.n_targets = y_train.shape[1]
            self.alphas_ = np.zeros(self.n_targets)
            self.estimators = [copy.deepcopy(self.estimator) for i in range(self.n_targets)]
            self.aloo_calculators = [copy.deepcopy(self.aloo_calculator) for i in range(self.n_targets)]
            for j in range(self.n_targets):
                self.alphas_[j] = self.aloo_calculator.get_aloocv_alpha(train_blocked_data.get_all_data(),
                                                                        y_train[:, j])
                if hasattr(self.estimator, "alpha"):
                    self.estimators[j].set_params(alpha=self.alphas_[j])
                elif hasattr(self.estimator, "C"):
                    self.estimators[j].set_params(C=1/self.alphas_[j])

    def _fit_full_predictions(self, test_blocked_data, y_test=None):
        if y_test is None:
            raise ValueError("Need to supply y_test for LOO")
        X1 = np.hstack([test_blocked_data.get_all_data(), np.ones((test_blocked_data.n_samples, 1))])
        if not self.multi_target:
            fitted_parameters = self.aloo_calculator.get_aloo_fitted_parameters(test_blocked_data.get_all_data(),
                                                                                y_test, self.alpha_, cache=True)
            return self.aloo_calculator.score_to_pred(np.sum(fitted_parameters.T * X1, axis=1))
        else:
            full_preds = np.empty_like(y_test)
            for j in range(self.n_targets):
                aloo_calculator = self.aloo_calculators[j]
                fitted_parameters = aloo_calculator.get_aloo_fitted_parameters(test_blocked_data.get_all_data(),
                                                                               y_test[:, j], self.alphas_[j],
                                                                               cache=True)
                full_preds[:, j] = self.aloo_calculator.score_to_pred(np.sum(fitted_parameters.T * X1, axis=1))
            return full_preds
                # full_preds, partial_preds = self._fit_single_target(train_blocked_data, y_train[:, j],
                #                                                     test_blocked_data, alpha_, mode=mode)
                # self.alpha_[j] = self.aloo_calculator.score_to_pred(np.sum(fitted_parameters.T * X1, axis=1))
                # self._full_preds[:, j] = full_preds
                # for k in range(self.n_blocks):
                #     self._partial_preds[k][:, j] = partial_preds[k]

    def _fit_partial_predictions(self, k, mode, test_blocked_data, y_test=None):
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
        reduced_data1 = np.hstack([reduced_data, np.ones((test_blocked_data.n_samples, 1))])
        col_indices = np.append(col_indices, -1)
        if not self.multi_target:
            if self.fixed_intercept and len(col_indices) == 1:
                _, intercept = extract_coef_and_intercept(self.aloo_calculator.estimator)
                return np.repeat(self.aloo_calculator.score_to_pred(intercept), len(y_test))
            else:
                fitted_parameters = self.aloo_calculator.get_aloo_fitted_parameters()
                reduced_parameters = fitted_parameters.T[:, col_indices]
                return self.aloo_calculator.score_to_pred(np.sum(reduced_parameters * reduced_data1, axis=1))
        else:
            partial_preds = np.empty_like(y_test)
            for j in range(self.n_targets):
                aloo_calculator = self.aloo_calculators[j]
                if self.fixed_intercept and len(col_indices) == 1:
                    _, intercept = extract_coef_and_intercept(self.aloo_calculator.estimator)
                    partial_preds[:, j] = np.repeat(aloo_calculator.score_to_pred(intercept), len(y_test))
                else:
                    fitted_parameters = aloo_calculator.get_aloo_fitted_parameters()
                    reduced_parameters = fitted_parameters.T[:, col_indices]
                    partial_preds[:, j] = aloo_calculator.score_to_pred(np.sum(reduced_parameters * reduced_data1,
                                                                               axis=1))
                return partial_preds

    def _trim_values(self, values):
        if self.trim is not None:
            assert 0 < self.trim < 0.5, "Limit must be between 0 and 0.5"
            return np.clip(values, self.trim, 1 - self.trim)
        else:
            return values


class RidgeLOOPPM(GenericLOOPPM, ABC):
    def __init__(self, alpha_grid=np.logspace(-5, 5, 100), fixed_intercept=True, **kwargs):
        super().__init__(Ridge(**kwargs), alpha_grid, fixed_intercept=fixed_intercept)
        
    def set_alphas(self, alphas="default", blocked_data=None, y=None):
        full_data = blocked_data.get_all_data()
        if alphas == "default":
            alphas = get_alpha_grid(full_data, y)
        else:
            alphas = alphas
        self.alpha_grid = alphas


class LogisticLOOPPM(GenericLOOPPM, ABC):

    def __init__(self, alpha_grid=np.logspace(-4, 4, 10), fixed_intercept=True, **kwargs):
        super().__init__(LogisticRegression(**kwargs), alpha_grid, link_fn=sp.special.expit,
                         l_doubledot=lambda a, b: b * (1-b), hyperparameter_scorer=log_loss, 
                         trim=0.01, fixed_intercept=fixed_intercept)


def get_alpha_grid(X, y, start=-5, stop=5, num=100):
    X = X - X.mean(axis=0)
    y = y - y.mean(axis=0)
    sigma_sq_ = np.linalg.norm(y, axis=0) ** 2 / X.shape[0]
    X_var_ = np.linalg.norm(X, axis=0) ** 2
    alpha_opts_ = (X_var_[:, np.newaxis] / (X.T @ y)) ** 2 * sigma_sq_
    base = np.max(alpha_opts_)
    alphas = np.logspace(start, stop, num=num) * base
    return alphas


class GlmAlooCalculator:

    def __init__(self, estimator, alpha_grid=np.logspace(-4, 4, 10), link_fn=lambda a: a, l_dot=lambda a, b: b-a,
                 l_doubledot=lambda a, b: 1, r_doubledot=lambda a: 1, hyperparameter_scorer=mean_squared_error,
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

    def get_aloo_fitted_parameters(self, X=None, y=None, alpha=None, cache=False):
        if self.loo_fitted_parameters is not None:
            return self.loo_fitted_parameters
        else:
            if hasattr(self.estimator, "alpha"):
                self.estimator.set_params(alpha=alpha)
            elif hasattr(self.estimator, "C"):
                self.estimator.set_params(C=1/alpha)
            else:
                alpha = 0
                # raise ValueError("Estimator has no regularization parameter.")
            estimator = copy.deepcopy(self.estimator)
            estimator.fit(X, y)
            X1 = np.hstack([X, np.ones((X.shape[0], 1))])
            augmented_coef_ = extract_coef_and_intercept(estimator, merge=True)
            orig_preds = self.link_fn(X1 @ augmented_coef_)
            l_doubledot_vals = self.l_doubledot(y, orig_preds)
            J = X1.T * l_doubledot_vals @ X1
            if self.r_doubledot is not None:
                r_doubledot_vals = self.r_doubledot(augmented_coef_) * np.ones_like(augmented_coef_)
                r_doubledot_vals[-1] = 0
                reg_curvature = np.diag(r_doubledot_vals)
                J += alpha * reg_curvature
            normal_eqn_mat = np.linalg.inv(J) @ X1.T
            h_vals = np.sum(X1.T * normal_eqn_mat, axis=0) * l_doubledot_vals
            loo_fitted_parameters = augmented_coef_[:, np.newaxis] + normal_eqn_mat * self.l_dot(y, orig_preds) / (1 - h_vals)
            if cache:
                self.loo_fitted_parameters = loo_fitted_parameters
                self.estimator = estimator
            return loo_fitted_parameters

    def score_to_pred(self, score):
        return self._trim_values(self.link_fn(score))

    def get_aloocv_alpha(self, X, y, return_cv=False):
        cv_scores = np.zeros_like(self.alpha_grid)
        for i, alpha in enumerate(self.alpha_grid):
            loo_fitted_parameters = self.get_aloo_fitted_parameters(X, y, alpha)
            X1 = np.hstack([X, np.ones((X.shape[0], 1))])
            preds = self.score_to_pred(np.sum(loo_fitted_parameters.T * X1, axis=1))
            cv_scores[i] = self.hyperparameter_scorer(y, preds)
        self.alpha_ = self.alpha_grid[np.argmin(cv_scores)]
        if return_cv:
            return self.alpha_, cv_scores
        else:
            return self.alpha_

    def _trim_values(self, values):
        if self.trim is not None:
            assert 0 < self.trim < 0.5, "Limit must be between 0 and 0.5"
            return np.clip(values, self.trim, 1 - self.trim)
        else:
            return values


def extract_coef_and_intercept(estimator, merge=False):
    coef_ = estimator.coef_
    intercept_ = estimator.intercept_
    if coef_.ndim > 1:
        coef_ = coef_.ravel()
        intercept_ = intercept_[0]
    if merge:
        augmented_coef_ = np.append(coef_, intercept_)
        return augmented_coef_
    else:
        return coef_, intercept_