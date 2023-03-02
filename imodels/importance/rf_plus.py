import copy
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import r2_score, roc_auc_score, log_loss
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder

from imodels.importance.block_transformers_new import GmdiDefaultTransformer, TreeTransformer, \
    CompositeTransformer, IdentityTransformer
from imodels.importance.ppms_new import PartialPredictionModelBase, GlmClassifierPPM, \
    RidgeRegressorPPM, LogisticClassifierPPM
from imodels.importance.gmdi_new import ForestGMDI, \
    _get_default_sample_split, _validate_sample_split, _get_sample_split_data


class _RandomForestPlus(BaseEstimator):

    def __init__(self, rf_model=None, prediction_model=None, sample_split="auto",
                 include_raw=True, drop_features=True, add_transformers=None,
                 center=True, normalize=False):
        assert sample_split in ["loo", "oob", "inbag", "auto", None]
        super().__init__()
        if isinstance(self, RegressorMixin):
            self._task = "regression"
        elif isinstance(self, ClassifierMixin):
            self._task = "classification"
        else:
            raise ValueError("Unknown task.")
        if rf_model is None:
            if self._task == "regression":
                rf_model = RandomForestRegressor()
            elif self._task == "classification":
                rf_model = RandomForestClassifier()
        if prediction_model is None:
            if self._task == "regression":
                prediction_model = RidgeRegressorPPM()
            elif self._task == "classification":
                prediction_model = LogisticClassifierPPM()
        self.rf_model = copy.deepcopy(rf_model)
        self.prediction_model = copy.deepcopy(prediction_model)
        self.include_raw = include_raw
        self.drop_features = drop_features
        self.add_transformers = add_transformers
        self.center = center
        self.normalize = normalize
        self._is_ppm = isinstance(prediction_model, PartialPredictionModelBase)
        self.sample_split = _get_default_sample_split(sample_split, prediction_model, self._is_ppm)
        _validate_sample_split(self.sample_split, prediction_model, self._is_ppm)

    def fit(self, X, y, sample_weight=None, **kwargs):
        self.transformers_ = []
        self.estimators_ = []
        self._tree_random_states = []
        self.prediction_score_ = None
        self.gmdi_ = None
        self.gmdi_scores_ = None
        self._n_samples_train = X.shape[0]

        # fit random forest
        n_samples = X.shape[0]
        self.rf_model.fit(X, y, sample_weight=sample_weight)
        # onehot encode multiclass response for GlmClassiferPPM
        if isinstance(self.prediction_model, GlmClassifierPPM):
            self._multi_class = False
            if len(np.unique(y)) > 2:
                self._multi_class = True
                self._y_encoder = OneHotEncoder()
                y = self._y_encoder.fit_transform(y.reshape(-1, 1)).toarray()
        # fit model for each tree
        all_full_preds = []
        for tree_model in self.rf_model.estimators_:
            # get transformer
            if self.add_transformers is None:
                if self.include_raw:
                    transformer = GmdiDefaultTransformer(tree_model, drop_features=self.drop_features)
                else:
                    transformer = TreeTransformer(tree_model)
            else:
                if self.include_raw:
                    base_transformer_list = [TreeTransformer(tree_model), IdentityTransformer()]
                else:
                    base_transformer_list = [TreeTransformer(tree_model)]
                transformer = CompositeTransformer(base_transformer_list + self.add_transformers,
                                                   drop_features=self.drop_features)
            # fit transformer
            blocked_data = transformer.fit_transform(X, center=self.center, normalize=self.normalize)
            # do sample split
            train_blocked_data, test_blocked_data, y_train, y_test, test_indices = \
                _get_sample_split_data(blocked_data, y, tree_model.random_state, self.sample_split)
            # fit prediction model
            if train_blocked_data.get_all_data().shape[1] != 0:  # if tree has >= 1 split
                self.prediction_model.fit(train_blocked_data.get_all_data(), y_train, **kwargs)
                self.estimators_.append(copy.deepcopy(self.prediction_model))
                self.transformers_.append(copy.deepcopy(transformer))
                self._tree_random_states.append(tree_model.random_state)

                # get full predictions
                pred_func = self._get_pred_func()
                full_preds = pred_func(test_blocked_data.get_all_data())
                full_preds_n = np.empty(n_samples) if full_preds.ndim == 1\
                    else np.empty((n_samples, full_preds.shape[1]))
                full_preds_n[:] = np.nan
                full_preds_n[test_indices] = full_preds
                all_full_preds.append(full_preds_n)

        # compute prediction accuracy on internal sample split
        full_preds = np.nanmean(all_full_preds, axis=0)
        if self._task == "regression":
            pred_score = r2_score(y, full_preds)
            pred_score_name = "r2"
        elif self._task == "classification":
            if full_preds.shape[1] == 2:
                pred_score = roc_auc_score(y, full_preds[:, 1], multi_class="ovr")
            else:
                pred_score = roc_auc_score(y, full_preds, multi_class="ovr")
            pred_score_name = "auroc"
        self.prediction_score_ = pd.DataFrame({pred_score_name: [pred_score]})
        self._full_preds = full_preds

    def predict(self, X):
        X = check_array(X)
        check_is_fitted(self, "estimators_")
        if self._task == "regression":
            predictions = 0
            for estimator, transformer in zip(self.estimators_, self.transformers_):
                blocked_data = transformer.transform(X, center=self.center, normalize=self.normalize)
                predictions += estimator.predict(blocked_data.get_all_data())
            predictions = predictions / len(self.estimators_)
        elif self._task == "classification":
            prob_predictions = self.predict_proba(X)
            if prob_predictions.ndim == 1:
                prob_predictions = np.stack([1-prob_predictions, prob_predictions], axis=1)
            predictions = self.rf_model.classes_[np.argmax(prob_predictions, axis=1)]
        return predictions

    def predict_proba(self, X):
        X = check_array(X)
        check_is_fitted(self, "estimators_")
        if not hasattr(self.estimators_[0], "predict_proba"):
            raise AttributeError("'{}' object has no attribute 'predict_proba'".format(
                self.estimators_[0].__class__.__name__)
            )
        predictions = 0
        for estimator, transformer in zip(self.estimators_, self.transformers_):
            blocked_data = transformer.transform(X, center=self.center, normalize=self.normalize)
            predictions += estimator.predict_proba(blocked_data.get_all_data())
        predictions = predictions / len(self.estimators_)
        return predictions

    def get_gmdi_scores(self, X=None, y=None,
                        scoring_fns="auto", sample_split="inherit", mode="keep_k"):
        if X is None or y is None:
            if self.gmdi_scores_ is None:
                raise ValueError("Need X and y as inputs.")
        else:
            # get defaults
            if sample_split == "inherit":
                sample_split = self.sample_split
            if X.shape[0] != self._n_samples_train and sample_split is not None:
                raise ValueError("Set sample_split=None to fit GMDI on non-training X and y. "
                                 "To use other sample_split schemes, input the training X and y data.")
            if scoring_fns == "auto":
                scoring_fns = {"importance": _fast_r2_score} if self._task == "regression" \
                    else {"importance": _neg_log_loss}
            # onehot encode if multi-class for GlmClassiferPPM
            if isinstance(self.prediction_model, GlmClassifierPPM):
                if self._multi_class:
                    y = self._y_encoder.transform(y.reshape(-1, 1)).toarray()
            # compute GMDI for forest
            gmdi_obj = ForestGMDI(estimators=self.estimators_,
                                  transformers=self.transformers_,
                                  scoring_fns=scoring_fns,
                                  sample_split=sample_split,
                                  tree_random_states=self._tree_random_states,
                                  mode=mode,
                                  task=self._task,
                                  center=self.center,
                                  normalize=self.normalize)
            self.gmdi_ = gmdi_obj
            self.gmdi_scores_ = gmdi_obj.get_scores(X, y)
        return self.gmdi_scores_

    def get_gmdi_stability_scores(self, B=10, metrics="auto"):
        if self.gmdi_ is None:
            raise ValueError("Need to compute gmdi scores first using self.get_gmdi_scores(X, y)")
        return self.gmdi_.get_stability_scores(B=B, metrics=metrics)

    def _get_pred_func(self):
        if hasattr(self.prediction_model, "predict_proba_loo"):
            pred_func = self.prediction_model.predict_proba_loo
        elif hasattr(self.prediction_model, "predict_loo"):
            pred_func = self.prediction_model.predict_loo
        elif hasattr(self.prediction_model, "predict_proba"):
            pred_func = self.prediction_model.predict_proba
        else:
            pred_func = self.prediction_model.predict
        return pred_func


class RandomForestPlusRegressor(_RandomForestPlus, RegressorMixin):
    ...


class RandomForestPlusClassifier(_RandomForestPlus, ClassifierMixin):
    ...


def _fast_r2_score(y_true, y_pred, multiclass=False):
    numerator = ((y_true - y_pred) ** 2).sum(axis=0, dtype=np.float64)
    denominator = ((y_true - np.mean(y_true, axis=0)) ** 2). \
        sum(axis=0, dtype=np.float64)
    if multiclass:
        return np.mean(1 - numerator / denominator)
    else:
        return 1 - numerator / denominator


def _neg_log_loss(y_true, y_pred):
    return -log_loss(y_true, y_pred)