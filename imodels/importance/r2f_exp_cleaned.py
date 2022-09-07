import copy
from abc import ABC, abstractmethod

import numpy as np
import scipy as sp
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble._forest import _generate_unsampled_indices, _generate_sample_indices
from sklearn.linear_model import RidgeCV, LogisticRegressionCV, Ridge, LogisticRegression
from sklearn.metrics import roc_auc_score, r2_score, mean_squared_error, log_loss
from sklearn.preprocessing import OneHotEncoder

from imodels.importance.representation_cleaned import TreeTransformer, IdentityTransformer, CompositeTransformer


def GMDI_pipeline(X, y, fit, regression=True, mode="keep_k", 
                  partial_prediction_model="auto", scoring_fn="auto", 
                  include_raw=True, subsetting_scheme=None):

    p = X.shape[1]
    if include_raw:
        tree_transformers = [CompositeTransformer([TreeTransformer(p, tree_model, data=X),
                                                    IdentityTransformer(p)], adj_std="max")
                            for tree_model in fit.estimators_]
    else:
        tree_transformers = [TreeTransformer(p, tree_model, data=X) for tree_model in fit.estimators_]

    if partial_prediction_model == "auto":
        if regression:
            partial_prediction_model = RidgeLOOPPM()
        else:
            partial_prediction_model = LogisticLOOPPM()
    if scoring_fn == "auto":
        if regression:
            scoring_fn = r2_score
        else:
            scoring_fn = roc_auc_score
    if not regression:
        if len(np.unique(y)) > 2:
            y = OneHotEncoder().fit_transform(y.reshape(-1, 1)).toarray()
    
    gmdi = GMDIEnsemble(tree_transformers, partial_prediction_model, scoring_fn, mode, subsetting_scheme)
    scores = gmdi.get_scores(X, y)
    
    results = pd.DataFrame(data={'importance': scores})

    if isinstance(X, pd.DataFrame):
        results.index = X.columns
    results.index.name = 'var'
    results.reset_index(inplace=True)

    return results


class GMDI:

    def __init__(self, transformer, partial_prediction_model, scoring_fn, mode="keep_k"):
        self.transformer = transformer
        self.partial_prediction_model = partial_prediction_model
        self.scoring_fn = scoring_fn
        self.mode = mode
        self.n_features = None
        self._scores = None
        self.is_fitted = False

    def _fit_importance_scores(self, X, y):
        blocked_data = self.transformer.transform(X)
        self.partial_prediction_model.fit(blocked_data, y)
        self.n_features = self.partial_prediction_model.n_blocks
        self._scores = np.zeros(self.n_features)
        if self.mode == "keep_k":
            for k in range(self.n_features):
                partial_preds = self.partial_prediction_model.get_partial_predictions(k)
                self._scores[k] = self.scoring_fn(y, partial_preds)
        elif self.mode == "keep_rest":
            full_preds = self.partial_prediction_model.get_full_predictions()
            full_score = self.scoring_fn(y, full_preds)
            for k in range(self.n_features):
                partial_preds = self.partial_prediction_model.get_partial_predictions(k, mode="keep_rest")
                self._scores[k] = full_score - self.scoring_fn(y, partial_preds)
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

    def __init__(self, transformers, partial_prediction_model, scoring_fn, mode="keep_k", subsetting_scheme=None):
        self.n_transformers = len(transformers)
        self.gmdi_objects = [GMDI(transformer, partial_prediction_model, scoring_fn, mode)
                             for transformer in transformers]
        self.subsetting_scheme = subsetting_scheme
        self.scoring_fn = scoring_fn
        self.mode = mode
        self.n_features = None
        self._scores = None
        self.is_fitted = False

    def _fit_importance_scores(self, X, y):
        assert X.shape[0] == len(y)
        n_samples = len(y)
        scores = []
        for gmdi_object in self.gmdi_objects:
            if self.subsetting_scheme is None:
                sample_indices = list(range(n_samples))
            else:
                estimator = gmdi_object.transformer.estimator
                if self.subsetting_scheme == "oob":
                    sample_indices = _generate_unsampled_indices(estimator.random_state, n_samples, n_samples)
                elif self.subsetting_scheme == "inbag":
                    sample_indices = _generate_sample_indices(estimator.random_state, n_samples, n_samples)
                else:
                    raise ValueError("Unsupported subsetting scheme")
            scores.append(gmdi_object.get_scores(X[sample_indices, :], y[sample_indices]))
        self._scores = np.mean(scores, axis=0)
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

    def __init__(self):
        self.n_blocks = None
        self._partial_preds = dict({})
        self._full_preds = None
        self.is_fitted = False

    @abstractmethod
    def fit(self, blocked_data, y, mode="keep_k"):
        pass

    def get_partial_predictions(self, k):
        return self._partial_preds[k]

    def get_full_predictions(self):
        return self._full_preds


class GenericPPM(PartialPredictionModelBase, ABC):

    def __init__(self, estimator):
        super().__init__()
        self.estimator = estimator

    def fit(self, blocked_data, y, mode="keep_k"):
        self.n_blocks = blocked_data.n_blocks
        full_data = blocked_data.get_all_data()
        self.estimator.fit(full_data, y)
        if hasattr(self.estimator, "predict_proba"):
            pred_func = self.estimator.predict_proba
        else:
            pred_func = self.estimator.predict
        self._full_preds = pred_func(full_data)
        for k in range(self.n_blocks):
            modified_data = blocked_data.get_modified_data(k, mode)
            self._partial_preds[k] = pred_func(modified_data)


class RidgePPM(GenericPPM, ABC):

    def __init__(self, **kwargs):
        super().__init__(estimator=RidgeCV(**kwargs))

    def set_alphas(self, alphas="default", blocked_data=None, y=None):
        full_data = blocked_data.get_all_data()
        if alphas == "default":
            alphas = get_alpha_grid(full_data, y)
        else:
            alphas = alphas
        self.estimator = RidgeCV(alphas=alphas)


class GenericLOOPPM(PartialPredictionModelBase, ABC):

    def __init__(self, estimator, alpha_grid=np.logspace(-5, 5, 20), link_fn = lambda a: a, l_dot=lambda a, b: b-a,
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

    def _get_loo_fitted_parameters(self, X, y, coef_, alpha=0, constant_term=True):
        orig_preds = self.link_fn(X @ coef_)
        l_doubledot_vals = self.l_doubledot(y, orig_preds)
        J = X.T * l_doubledot_vals @ X
        if self.r_doubledot is not None:
            r_doubledot_vals = self.r_doubledot(coef_) * np.ones_like(coef_)
            if constant_term:
                r_doubledot_vals[-1] = 0
            reg_curvature = np.diag(r_doubledot_vals)
            J += alpha * reg_curvature
        normal_eqn_mat = np.linalg.inv(J) @ X.T
        h_vals = np.sum(X.T * normal_eqn_mat, axis=0) * l_doubledot_vals
        loo_fitted_parameters = coef_[:, np.newaxis] + normal_eqn_mat * self.l_dot(y, orig_preds) / (1 - h_vals)
        return loo_fitted_parameters

    def _fit_single_target(self, blocked_data, y, alpha=None, partial_preds=True, mode="keep_k"):
        full_data = blocked_data.get_all_data()
        augmented_data = np.hstack([full_data, np.ones((full_data.shape[0], 1))]) # Tag on constant feature vector
        estimator = copy.deepcopy(self.estimator)
        if hasattr(estimator, "alpha"):
            estimator.set_params(alpha=alpha)
        elif hasattr(estimator, "C"):
            estimator.set_params(C=1/alpha)
        else:
            alpha = 0
        estimator.fit(full_data, y)
        coef_ = estimator.coef_
        if coef_.ndim > 1:
            augmented_coef_ = np.concatenate([coef_.ravel(), estimator.intercept_])
        else:
            augmented_coef_ = np.array(list(coef_) + [estimator.intercept_])
        loo_fitted_parameters = self._get_loo_fitted_parameters(augmented_data, y, augmented_coef_, alpha)
        full_preds = self._trim_values(self.link_fn(np.sum(loo_fitted_parameters.T * augmented_data, axis=1)))
        if partial_preds:
            partial_preds = dict({})
            for k in range(self.n_blocks):
                modified_data = blocked_data.get_modified_data(k, mode)
                modified_data = np.hstack([modified_data, np.ones((modified_data.shape[0], 1))])
                partial_preds[k] = self._trim_values(self.link_fn(np.sum(loo_fitted_parameters.T * modified_data,
                                                                         axis=1)))
            return full_preds, partial_preds
        else:
            return full_preds

    def fit(self, blocked_data, y, mode="keep_k"):
        self.n_blocks = blocked_data.n_blocks
        if y.ndim > 1:
            self.alpha_ = np.empty(y.shape[1])
            self._full_preds = np.empty_like(y)
            for k in range(self.n_blocks):
                self._partial_preds[k] = np.empty_like(y)
            for j in range(y.shape[1]):
                alpha_ = self._fit_hyperparameter(blocked_data, y[:, j])
                full_preds, partial_preds = self._fit_single_target(blocked_data, y[:, j], alpha_, mode)
                self.alpha_[j] = alpha_
                self._full_preds[:, j] = full_preds
                for k in range(self.n_blocks):
                    self._partial_preds[k][:, j] = partial_preds[k]
        else:
            alpha_ = self._fit_hyperparameter(blocked_data, y)
            self._full_preds, self._partial_preds = self._fit_single_target(blocked_data, y, alpha_, mode=mode)
            self.alpha_ = alpha_

    def _fit_hyperparameter(self, blocked_data, y):
        cv_scores = np.zeros_like(self.alpha_grid)
        for i, alpha in enumerate(self.alpha_grid):
            full_preds = self._fit_single_target(blocked_data, y, alpha, partial_preds=False)
            cv_scores[i] = self.hyperparameter_scorer(y, full_preds)
        alpha_ = self.alpha_grid[np.argmin(cv_scores)]
        return alpha_

    def _trim_values(self, values):
        if self.trim is not None:
            assert 0 < self.trim < 0.5, "Limit must be between 0 and 0.5"
            return np.clip(values, self.trim, 1 - self.trim)
        else:
            return values


class RidgeLOOPPM(GenericLOOPPM, ABC):

    def __init__(self, **kwargs):
        super().__init__(Ridge(**kwargs))

    def set_alphas(self, alphas="default", blocked_data=None, y=None):
        full_data = blocked_data.get_all_data()
        if alphas == "default":
            alphas = get_alpha_grid(full_data, y)
        else:
            alphas = alphas
        self.alpha_grid = alphas


class LogisticLOOPPM(GenericLOOPPM, ABC):

    def __init__(self, **kwargs):
        super().__init__(LogisticRegression(**kwargs), link_fn=sp.special.expit,
                         l_doubledot=lambda a, b: b * (1-b), hyperparameter_scorer=log_loss, trim=0.01)


def get_alpha_grid(X, y, start=-10, stop=10, num=50):
    X = X - X.mean(axis=0)
    y = y - y.mean(axis=0)
    sigma_sq_ = np.linalg.norm(y, axis=0) ** 2 / X.shape[0]
    X_var_ = np.linalg.norm(X, axis=0) ** 2
    alpha_opts_ = (X_var_[:, np.newaxis] / (X.T @ y)) ** 2 * sigma_sq_
    base = np.max(alpha_opts_)
    alphas = np.logspace(start, stop, num=num) * base
    return alphas
