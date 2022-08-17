from abc import ABC, abstractmethod

import numpy as np
from sklearn.ensemble._forest import _generate_unsampled_indices, _generate_sample_indices
from sklearn.linear_model import RidgeCV


class GMDI:

    def __init__(self, transformer, partial_prediction_model, scoring_fn):
        self.transformer = transformer
        self.partial_prediction_model = partial_prediction_model
        self.scoring_fn = scoring_fn
        self.n_features = None
        self._scores = None
        self.is_fitted = False

    def fit_transformer(self, X, y=None, refit=False):
        pass

    def fit_importance_scores(self, X, y):
        if not self.transformer.is_fitted:
            raise AttributeError("Transformer not fitted yet")
        blocked_data = self.transformer.transform(X)
        self.partial_prediction_model.fit(blocked_data, y)
        self.n_features = self.partial_prediction_model.n_blocks
        self._scores = np.zeros(self.n_features)
        for k in range(self.n_features):
            partial_preds = self.partial_prediction_model.get_partial_predictions(k)
            self._scores[k] = self.scoring_fn(y, partial_preds)
        self.is_fitted = True

    def get_scores(self):
        if self.is_fitted:
            return self._scores
        else:
            raise AttributeError("Scores not fitted yet.")


class GMDIEnsemble:

    def __init__(self, transformers, partial_prediction_model, scoring_fn):

        self.n_features = None
        self.n_transformers = len(transformers)
        self.gmdi_objects = [GMDI(transformer, partial_prediction_model, scoring_fn) for transformer in transformers]
        self.is_fitted = False

    def fit_transformers(self, X, y=None, refit=False):
        for gmdi_object in self.gmdi_objects:
            gmdi_object.fit_transformer(X, y)

    def fit_importance_scores(self, X, y, subsetting_scheme=None):
        assert X.shape[0] == len(y)
        n_samples = len(y)
        for gmdi_object in self.gmdi_objects:
            if subsetting_scheme is None:
                sample_indices = list(range(n_samples))
            else:
                estimator = gmdi_object.transformer.estimator
                if subsetting_scheme == "oob":
                    sample_indices = _generate_unsampled_indices(estimator.random_state, n_samples, n_samples)
                elif subsetting_scheme == "inbag":
                    sample_indices = _generate_sample_indices(estimator.random_state, n_samples, n_samples)
                else:
                    raise ValueError("Unsupported subsetting scheme")
            X = X[sample_indices, :]
            y = y[sample_indices]
            gmdi_object.fit_importance_scores(X, y)
        self.is_fitted = True
        self.n_features = self.gmdi_objects[0].n_features

    def get_scores(self):
        if self.is_fitted:
            scores = np.zeros(len(self.n_features))
            for gmdi_object in self.gmdi_objects:
                for k in range(self.n_features):
                    scores += gmdi_object.get_scores()
            scores /= self.n_transformers
        else:
            raise AttributeError("Scores not fitted yet.")


class PartialPredictionModelBase(ABC):

    def __init__(self):
        self.n_blocks = None
        self._partial_preds = dict({})
        self.is_fitted = False

    @abstractmethod
    def fit(self, blocked_data, y):
        self.n_blocks = blocked_data.n_blocks

    def get_partial_predictions(self, k):
        return self._partial_preds[k]


class RidgePPM(PartialPredictionModelBase, ABC):

    def __init__(self, alphas="default"):
        super().__init__()
        self.alphas = alphas

    def fit(self, blocked_data, y):
        full_data = blocked_data.get_all_data()
        if self.alphas == "default":
            alphas = get_alpha_grid(full_data, y)
        else:
            alphas = self.alphas
        ridge_model = RidgeCV(alphas=alphas)
        ridge_model.fit(full_data, y)
        for k in range(self.n_blocks):
            modified_data = blocked_data.get_modified_data(k)
            self._partial_preds[k] = ridge_model.predict(modified_data)


class RidgeLOOPPM(PartialPredictionModelBase, ABC):

    def __init__(self, alphas="default"):
        super().__init__()
        self.alphas = alphas

    def fit(self, blocked_data, y):
        full_data = blocked_data.get_all_data()
        if self.alphas == "default":
            alphas = get_alpha_grid(full_data, y)
        else:
            alphas = self.alphas
        ridge_model = RidgeCV(alphas=alphas)
        ridge_model.fit(full_data, y)
        for k in range(self.n_blocks):
            modified_data = blocked_data.get_modified_data(k)
            augmented_data = np.hstack([full_data, np.ones((full_data.shape[0], 1))])
            G = augmented_data.T @ augmented_data + ridge_model.alpha_ * np.diag([1] * full_data.shape[1] + [0])
            normal_eqn_mat = np.linalg.inv(G) @ augmented_data.T
            h_vals = np.diag(augmented_data @ normal_eqn_mat)
            full_model_residuals = y - ridge_model.predict(full_data)
            self._partial_preds[k] = ridge_model.predict(modified_data) - \
                                     modified_data @ normal_eqn_mat * full_model_residuals / (1 - h_vals)


class LOOPartialPredictionModelBase(PartialPredictionModelBase, ABC):

    def __init__(self, loss_fn, loss_fn_der, loss_fn_der2, regularizer, regularizer_der, regularizer_der2):
        pass


def get_alpha_grid(X, y, start=-10, stop=10, num=50):
    X = X - X.mean(axis=0)
    y = y - y.mean(axis=0)
    sigma_sq_ = np.linalg.norm(y, axis=0) ** 2 / X.shape[0]
    X_var_ = np.linalg.norm(X, axis=0) ** 2
    alpha_opts_ = (X_var_[:, np.newaxis] / (X.T @ y)) ** 2 * sigma_sq_
    base = np.max(alpha_opts_)
    alphas = np.logspace(start, stop, num=num) * base
    return alphas
