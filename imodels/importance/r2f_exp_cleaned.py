from abc import ABC, abstractmethod

import numpy as np
from sklearn.ensemble._forest import _generate_unsampled_indices, _generate_sample_indices
from sklearn.linear_model import RidgeCV, LogisticRegressionCV
from sklearn.base import RegressorMixin, ClassifierMixin


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
        self._scores = np.mean(axis=0)
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

    def __init__(self, estimator, l_dot=lambda a, b: b-a, l_doubledot=lambda a, b: 1, r_doubledot=lambda a: 1):
        super().__init__()
        self.estimator = estimator
        self.l_dot = l_dot
        self.l_doubledot = l_doubledot
        self.r_doubledot = r_doubledot

    def get_loo_fitted_parameters(self, X, y, coef_, alpha=0, constant_term=True):
        linear_predictor_vals = X @ coef_
        l_doubledot_vals = self.l_doubledot(y, linear_predictor_vals)
        J = X.T * l_doubledot_vals @ X
        if self.r_doubledot is not None:
            r_doubledot_vals = self.r_doubledot(coef_) * np.ones_like(coef_)
            if constant_term:
                r_doubledot_vals[-1] = 0
            reg_curvature = np.diag(r_doubledot_vals)
            J += alpha * reg_curvature
        normal_eqn_mat = np.linalg.inv(J) @ X.T
        h_vals = np.sum(X.T * normal_eqn_mat, axis=0) * l_doubledot_vals
        loo_fitted_parameters = coef_[:, np.newaxis] - normal_eqn_mat * h_vals * self.l_dot(y, linear_predictor_vals)
        return loo_fitted_parameters

    def fit(self, blocked_data, y, mode="keep_k"):
        self.n_blocks = blocked_data.n_blocks
        full_data = blocked_data.get_all_data()
        self.estimator.fit(full_data, y)
        if hasattr(self.estimator, "alpha_"):
            alpha = self.estimator.alpha_
        elif hasattr(self.estimator, "C_"):
            alpha = 1 / np.mean(self.estimator.C_)
        else:
            alpha = 0
        augmented_data = np.hstack([full_data, np.ones((full_data.shape[0], 1))]) # Tag on constant feature vector
        augmented_coef_ = np.array(list(self.estimator.coef_) + [self.estimator.intercept_])
        loo_fitted_parameters = self.get_loo_fitted_parameters(augmented_data, y, augmented_coef_, alpha)
        self._full_preds = np.sum(loo_fitted_parameters.T * augmented_data, axis=1)
        for k in range(self.n_blocks):
            modified_data = blocked_data.get_modified_data(k, mode)
            modified_data = np.hstack([modified_data, np.ones((modified_data.shape[0], 1))])
            self._partial_preds[k] = np.sum(loo_fitted_parameters.T * modified_data, axis=1)


class RidgeLOOPPM(GenericLOOPPM, ABC):

    def __init__(self, **kwargs):
        super().__init__(RidgeCV(**kwargs))

    def set_alphas(self, alphas="default", blocked_data=None, y=None):
        full_data = blocked_data.get_all_data()
        if alphas == "default":
            alphas = get_alpha_grid(full_data, y)
        else:
            alphas = alphas
        self.estimator = RidgeCV(alphas=alphas)


class LogisticLOOPPM(GenericLOOPPM, ABC):

    def __init__(self, **kwargs):
        l_doubledot = lambda a: a * (1-a)
        super().__init__(LogisticRegressionCV(**kwargs), l_doubledot=l_doubledot)




# class RidgeLOOPPM(PartialPredictionModelBase, ABC):
#
#     def __init__(self, alphas="default"):
#         super().__init__()
#         self.alphas = alphas
#
#     def fit(self, blocked_data, y, mode="keep_k"):
#         full_data = blocked_data.get_all_data()
#         if self.alphas == "default":
#             alphas = get_alpha_grid(full_data, y)
#         else:
#             alphas = self.alphas
#         ridge_model = RidgeCV(alphas=alphas)
#         ridge_model.fit(full_data, y)
#         augmented_data = np.hstack([full_data, np.ones((full_data.shape[0], 1))])
#         G = augmented_data.T @ augmented_data + ridge_model.alpha_ * np.diag([1] * full_data.shape[1] + [0])
#         normal_eqn_mat = np.linalg.inv(G) @ augmented_data.T
#         h_vals = np.sum(augmented_data * normal_eqn_mat.T, axis=1)
#         full_model_residuals = y - ridge_model.predict(full_data)
#         coef_hat_loo_diff = - normal_eqn_mat * full_model_residuals / (1 - h_vals)
#         full_pred_loo_diff = np.sum(coef_hat_loo_diff.T * full_data, axis=1)
#         self._full_preds = ridge_model.predict(full_data) - full_pred_loo_diff
#         for k in range(self.n_blocks):
#             modified_data = blocked_data.get_modified_data(k, mode)
#             partial_pred_loo_diff = np.sum(coef_hat_loo_diff.T * modified_data, axis=1)
#             self._partial_preds[k] = ridge_model.predict(modified_data) + partial_pred_loo_diff


def get_alpha_grid(X, y, start=-10, stop=10, num=50):
    X = X - X.mean(axis=0)
    y = y - y.mean(axis=0)
    sigma_sq_ = np.linalg.norm(y, axis=0) ** 2 / X.shape[0]
    X_var_ = np.linalg.norm(X, axis=0) ** 2
    alpha_opts_ = (X_var_[:, np.newaxis] / (X.T @ y)) ** 2 * sigma_sq_
    base = np.max(alpha_opts_)
    alphas = np.logspace(start, stop, num=num) * base
    return alphas
