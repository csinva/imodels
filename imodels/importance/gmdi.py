import copy

import numpy as np

import pandas as pd
from sklearn.ensemble._forest import _generate_unsampled_indices, _generate_sample_indices
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder

from imodels.importance.block_transformers import \
    TreeTransformer, IdentityTransformer, CompositeTransformer, \
    BlockPartitionedData
from imodels.importance.ppms import RidgeLOOPPM, LogisticLOOPPM


def GMDI_pipeline(X, y, fit, regression=True, mode="keep_k",
                  partial_prediction_model="auto", scoring_fn="auto",
                  include_raw=True, drop_features=True, oob=False, training=False, center=True):
    p = X.shape[1]
    fit = copy.deepcopy(fit)
    if include_raw:
        tree_transformers = [CompositeTransformer([TreeTransformer(p, tree_model),
                                                   IdentityTransformer(p)], adj_std="max", drop_features=drop_features)
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

    gmdi = GMDIEnsemble(tree_transformers, partial_prediction_model, scoring_fn, mode, oob, training, center,
                        include_raw, drop_features)
    scores = gmdi.get_scores(X, y)

    results = pd.DataFrame(data={'importance': scores})

    if isinstance(X, pd.DataFrame):
        results.index = X.columns
    results.index.name = 'var'
    results.reset_index(inplace=True)

    return results


class GMDI:

    def __init__(self, transformer, partial_prediction_model, scoring_fn,
                 mode="keep_k", training=False, center=True,
                 include_raw=True, drop_features=True):
        self.transformer = transformer
        self.partial_prediction_model = partial_prediction_model
        self.scoring_fn = scoring_fn
        self.mode = mode
        self.n_features = None
        self._scores = None
        self.is_fitted = False
        self.training = training
        self.center = center
        self.include_raw = include_raw
        self.drop_features = drop_features

    def get_scores(self, X=None, y=None):
        """
        Get feature importance scores. If no data is supplied, then check
        if the scores are already fitted, if so return them, else throw and
        error. If data is supplied, fit the scores before returning them.

        Parameters
        ----------
        X
        y

        Returns
        -------
        scores: dict of {int, ndarray} pairs
            A dictionary with feature indices as the keys and arrays of
            gmdi feature importance scores as the values
        """
        if X is None or y is None:
            if self.is_fitted:
                pass
            else:
                raise ValueError("Not yet fitted. Need X and y as inputs.")
        else:
            self._fit_importance_scores(X, y)
        return self._scores

    def _fit_importance_scores(self, X, y):
        blocked_data = self.transformer.transform(X, center=self.center)
        self.n_features = blocked_data.n_blocks
        if self.include_raw and self.drop_features == False:
            data_blocks = []
            min_adj_factor = np.nanmin(np.concatenate(self.transformer.all_adj_factors, axis=0))
            for k in range(self.n_features):
                if blocked_data.get_block(k).shape[1] == 1 and X[:, [k]].std() > 0.0:  # only contains raw feature
                    data_blocks.append(
                        blocked_data.get_all_data()[:, blocked_data.get_block_indices(k)] * min_adj_factor / X[:,
                                                                                                             [k]].std())
                else:
                    data_blocks.append(blocked_data.get_all_data()[:, blocked_data.get_block_indices(k)])
            blocked_data = BlockPartitionedData(data_blocks)
        if self.oob:
            if self.training:
                train_blocked_data, test_blocked_data, y_train, y_test = self._train_test_split(blocked_data, y)
                test_blocked_data = copy.deepcopy(train_blocked_data)
                y_test = copy.deepcopy(y_train)
            else:
                train_blocked_data, test_blocked_data, y_train, y_test = self._train_test_split(blocked_data, y)
        else:
            train_blocked_data = test_blocked_data = blocked_data
            y_train = y_test = y
        if train_blocked_data.get_all_data().shape[1] == 0:  # checking if learnt representation is empty
            self._scores = np.zeros(X.shape[1])
            for k in range(X.shape[1]):
                self._scores[k] = np.NaN
        else:
            if y.ndim == 1:
                full_preds, partial_preds, partial_params = self._fit_one_target(train_blocked_data, y_train,
                                                                                 test_blocked_data, y_test)
            else:
                full_preds_list = []
                partial_preds_list = []
                partial_params = None
                for j in range(y.shape[1]):
                    yj_train = y_train[:, j]
                    yj_test = y_test[:, j]
                    full_preds_j, partial_preds_j, _ = self._fit_one_target(train_blocked_data, yj_train,
                                                                            test_blocked_data, yj_test,
                                                                            multitarget=True)
                    full_preds_list.append(full_preds_j)
                    partial_preds_list.append(partial_preds_j)
                full_preds = np.array(full_preds_list).T
                partial_preds = dict()
                for k in range(self.n_features):
                    partial_preds[k] = np.array([partial_preds_j[k] for partial_preds_j in partial_preds_list]).T
            self._score_partial_predictions(full_preds, partial_params, partial_preds, y_test)
        self.is_fitted = True

    def _fit_one_target(self, train_blocked_data, y_train, test_blocked_data, y_test, multitarget=False):
        partial_preds = dict()
        partial_params = dict()
        if multitarget:
            ppm = copy.deepcopy(self.partial_prediction_model)
        else:
            ppm = self.partial_prediction_model
        ppm.fit(train_blocked_data, y_train, test_blocked_data, y_test, self.mode)
        full_preds = ppm.get_full_predictions()
        for k in range(self.n_features):
            partial_preds[k], partial_params[k] = ppm.get_partial_predictions(k)
        return full_preds, partial_preds, partial_params

    def _score_partial_predictions(self, full_preds, partial_params, partial_preds, y_test):
        self._scores = np.zeros(self.n_features)
        if self.mode == "keep_k":
            for k in range(self.n_features):
                if self.scoring_fn == "mdi_oob":
                    if partial_params is None:
                        raise ValueError("scoring_fn='mdi_oob' has not been implemented for multi-task y.")
                    if len(partial_params[k]) == 1:  # only intercept model
                        self._scores[k] = np.dot(y_test, partial_preds[k] - partial_params[k]) / len(y_test)
                    elif partial_params[k].ndim == 1:  # partial prediction model without LOO
                        self._scores[k] = np.dot(y_test, partial_preds[k] - partial_params[k][-1]) / len(y_test)
                    else:  # LOO partial prediction model
                        self._scores[k] = np.dot(y_test, partial_preds[k] - partial_params[k][:, -1]) / len(y_test)
                else:
                    self._scores[k] = self.scoring_fn(y_test, partial_preds[k])
        elif self.mode == "keep_rest":
            full_score = self.scoring_fn(y_test, full_preds)
            for k in range(self.n_features):
                self._scores[k] = full_score - self.scoring_fn(y_test, partial_preds[k])

    def _train_test_split(self, blocked_data, y):
        n_samples = len(y)
        train_indices = _generate_sample_indices(self.transformer.estimator.random_state, n_samples, n_samples)
        test_indices = _generate_unsampled_indices(self.transformer.estimator.random_state, n_samples, n_samples)
        train_blocked_data, test_blocked_data = blocked_data.train_test_split(train_indices, test_indices)
        if y.ndim > 1:
            y_train = y[train_indices, :]
            y_test = y[test_indices, :]
        else:
            y_train = y[train_indices]
            y_test = y[test_indices]
        return train_blocked_data, test_blocked_data, y_train, y_test


class GMDIEnsemble:

    def __init__(self, transformers, partial_prediction_model, scoring_fn,
                 mode="keep_k", oob=False, training=False,
                 center=True, include_raw=True, drop_features=True):
        self.n_transformers = len(transformers)
        self.gmdi_objects = [
            GMDI(transformer, copy.deepcopy(partial_prediction_model),
                 scoring_fn, mode, oob, training, center,
                 include_raw, drop_features)
            for transformer in transformers]
        self.oob = oob
        self.training = training
        self.scoring_fn = scoring_fn
        self.mode = mode
        self.n_features = None
        self._scores = None
        self.is_fitted = False
        self.include_raw = include_raw
        self.drop_features = drop_features

    def _fit_importance_scores(self, X, y):
        assert X.shape[0] == len(y)
        # n_samples = len(y)
        scores = []
        for gmdi_object in self.gmdi_objects:
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
