from copy import deepcopy
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.linear_model import ElasticNetCV, LinearRegression, RidgeCV, LassoCV, LogisticRegressionCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_array
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_X_y
from sklearn.utils.validation import _check_sample_weight
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, AdaBoostClassifier, AdaBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm
from sklearn.multioutput import MultiOutputRegressor, MultiOutputClassifier
from collections import defaultdict
import pandas as pd
import json
from sklearn.preprocessing import StandardScaler
from imodels.util.transforms import CorrelationScreenTransformer

import imodels
from interpret.glassbox import ExplainableBoostingClassifier, ExplainableBoostingRegressor

from sklearn.base import RegressorMixin, ClassifierMixin


# See notes on EBM in the docs
# main file: https://github.com/interpretml/interpret/blob/develop/python/interpret-core/interpret/glassbox/_ebm/_ebm.py
# merge ebms: https://github.com/interpretml/interpret/blob/develop/python/interpret-core/interpret/glassbox/_ebm/_merge_ebms.py#L280
# eval_terms: https://interpret.ml/docs/python/api/ExplainableBoostingRegressor.html#interpret.glassbox.ExplainableBoostingRegressor.eval_terms

class MultiTaskGAM(BaseEstimator):
    """EBM-based GAM that shares curves for predicting different outputs.
    - If only one target is given, we fit an EBM to predict each covariate
    - If multiple targets are given, we fit a an EBM to predict each target
    - If only one target is given and use_single_task_with_reweighting, we fit an EBM to predict the single target, then apply reweighting
    """

    def __init__(
        self,
        ebm_kwargs={'n_jobs': 1, 'max_rounds': 5000, },
        multitask=True,
        interactions=0.95,
        linear_penalty='ridge',
        onehot_prior=False,
        renormalize_features=False,
        use_normalize_feature_targets=False,
        use_internal_classifiers=False,
        fit_target_curves=True,
        use_correlation_screening_for_features=False,
        use_single_task_with_reweighting=False,
        fit_linear_frac: float = None,
        random_state=42,
    ):
        """
        Params
        ------
        Note: args override ebm_kwargs if there are duplicates
        one_hot_prior: bool
            If True and multitask, the linear model will be fit with a prior that the ebm
            features predicting the target should have coef 1
        renormalize_features: bool
            If True, renormalize the features before fitting the linear model
        use_normalize_feature_targets: bool
            whether to normalize the features used as targets for internal EBMs
            (does not apply to target columns)
            If input features are normalized already, this has no effect
        use_internal_classifiers: bool
            whether to use internal classifiers (as opposed to regressors)
        fit_target_curves: bool
            whether to fit an EBM to predict the target
        use_single_task_with_reweighting: bool
            fit an EBM to predict the single target, then apply linear reweighting
        use_correlation_screening_for_features: bool
            whether to use correlation screening for features
        fit_linear_frac: float
            If not None, the fraction of features to use for the linear model (the rest are used for the EBM)
        """
        self.ebm_kwargs = ebm_kwargs
        self.multitask = multitask
        self.linear_penalty = linear_penalty
        self.random_state = random_state
        self.interactions = interactions
        self.onehot_prior = onehot_prior
        self.use_normalize_feature_targets = use_normalize_feature_targets
        self.renormalize_features = renormalize_features
        self.use_internal_classifiers = use_internal_classifiers
        self.fit_target_curves = fit_target_curves
        self.use_single_task_with_reweighting = use_single_task_with_reweighting
        self.use_correlation_screening_for_features = use_correlation_screening_for_features
        self.fit_linear_frac = fit_linear_frac

        # override ebm_kwargs
        ebm_kwargs['random_state'] = random_state
        ebm_kwargs['interactions'] = interactions

    def fit(self, X, y, sample_weight=None):
        X, y = check_X_y(X, y, accept_sparse=False, multi_output=True)
        self.n_outputs_ = 1 if len(y.shape) == 1 else y.shape[1]
        if self.n_outputs_ > 1 and not self.fit_target_curves:
            raise ValueError(
                "fit_target_curves must be True when n_outputs > 1")
        if isinstance(self, ClassifierMixin):
            check_classification_targets(y)
            if self.n_outputs_ == 1:
                self.classes_, y = np.unique(y, return_inverse=True)
                if len(self.classes_) > 2:
                    raise ValueError(
                        "MultiTaskGAMClassifier currently only supports binary classification")
            elif self.n_outputs_ > 1:
                self.classes_ = [np.unique(y[:, i])
                                 for i in range(self.n_outputs_)]
                if any(len(c) > 2 for c in self.classes_):
                    raise ValueError(
                        "MultiTaskGAMClassifier currently only supports binary classification")
        sample_weight = _check_sample_weight(sample_weight, X, dtype=None)
        if self.use_single_task_with_reweighting:
            assert self.n_outputs_ == 1, "use_single_task_with_reweighting only works with one output"
            assert self.multitask, "use_single_task_with_reweighting only works with multitask"

        # just fit ebm normally
        if not self.multitask:
            if isinstance(self, ClassifierMixin):
                self.ebm_ = ExplainableBoostingClassifier(**self.ebm_kwargs)
            else:
                self.ebm_ = ExplainableBoostingRegressor(**self.ebm_kwargs)

            # fit
            if self.n_outputs_ > 1:
                if isinstance(self, ClassifierMixin):
                    self.ebm_multioutput_ = MultiOutputClassifier(self.ebm_)
                else:
                    self.ebm_multioutput_ = MultiOutputRegressor(self.ebm_)
                self.ebm_multioutput_.fit(X, y, sample_weight=sample_weight)
            else:
                self.ebm_.fit(X, y, sample_weight=sample_weight)
            return self

        # fit EBM(s)
        self.ebms_ = []
        num_samples, num_features = X.shape
        idxs_ebm, idxs_lin = self._split_data(num_samples)

        # fit EBM
        if self.use_single_task_with_reweighting:
            # fit an EBM to predict the single output
            self.ebms_.append(self._initialize_ebm_internal(y[idxs_ebm]))
            self.ebms_[-1].fit(X[idxs_ebm], y[idxs_ebm],
                               sample_weight=sample_weight[idxs_ebm])
        elif self.n_outputs_ == 1:
            # with 1 output, we fit an EBM to each feature
            for task_num in tqdm(range(num_features)):
                y_ = np.ascontiguousarray(X[idxs_ebm][:, task_num])
                X_ = deepcopy(X[idxs_ebm])
                X_[:, task_num] = 0
                self.ebms_.append(self._initialize_ebm_internal(y_))
                if isinstance(self, ClassifierMixin):
                    _, y_ = np.unique(y_, return_inverse=True)
                elif self.use_normalize_feature_targets:
                    y_ = StandardScaler().fit_transform(y_.reshape(-1, 1)).ravel()
                self.ebms_[task_num].fit(
                    X_, y_, sample_weight=sample_weight[idxs_ebm])

            # also fit an EBM to the target
            if self.fit_target_curves:
                self.ebms_.append(self._initialize_ebm_internal(y[idxs_ebm]))
                self.ebms_[num_features].fit(
                    X[idxs_ebm], y[idxs_ebm], sample_weight=sample_weight[idxs_ebm])
        elif self.n_outputs_ > 1:
            # with multiple outputs, we fit an EBM to each output
            for task_num in tqdm(range(self.n_outputs_)):
                self.ebms_.append(self._initialize_ebm_internal(y[idxs_ebm]))
                y_ = np.ascontiguousarray(y[idxs_ebm][:, task_num])
                self.ebms_[task_num].fit(
                    X[idxs_ebm], y_, sample_weight=sample_weight[idxs_ebm])

        # extract features from EBMs
        self.term_names_list_ = [
            ebm_.term_names_ for ebm_ in self.ebms_]
        self.term_names_ = sum(self.term_names_list_, [])
        feats = self._extract_ebm_features(X)
        if self.renormalize_features:
            self.scaler_ = StandardScaler()
            feats = self.scaler_.fit_transform(feats)
        if self.use_correlation_screening_for_features:
            self.correlation_screener_ = CorrelationScreenTransformer()
            feats = self.correlation_screener_.fit_transform(feats, y)
        feats[np.isinf(feats)] = 0

        # fit linear model
        self.lin_model = self._fit_linear_model(
            feats[idxs_lin], y[idxs_lin], sample_weight[idxs_lin])

        return self

    def _initialize_ebm_internal(self, y):
        if self.use_internal_classifiers and len(np.unique(y)) == 2:
            return ExplainableBoostingClassifier(**self.ebm_kwargs)
        else:
            return ExplainableBoostingRegressor(**self.ebm_kwargs)

    def _split_data(self, num_samples):
        '''Split data into EBM and linear model data
        '''
        if self.fit_linear_frac is not None:
            rng = np.random.RandomState(self.random_state)
            idxs_ebm = rng.choice(num_samples, int(
                num_samples * self.fit_linear_frac), replace=False)
            idxs_lin = np.array(
                [i for i in range(num_samples) if i not in idxs_ebm])
        else:
            idxs_ebm = np.arange(num_samples)
            idxs_lin = idxs_ebm
        assert len(idxs_ebm) > 0, f"No data for EBM! {self.fit_linear_frac=}"
        assert len(
            idxs_lin) > 0, f"No data for linear model! {self.fit_linear_frac=}"
        return idxs_ebm, idxs_lin

    def _fit_linear_model(self, feats, y, sample_weight):
        # fit a linear model to the features
        if isinstance(self, ClassifierMixin):
            lin_model = {
                'ridge': LogisticRegressionCV(penalty='l2'),
                'elasticnet': LogisticRegressionCV(penalty='elasticnet'),
                'lasso': LogisticRegressionCV(penalty='l1'),
            }[self.linear_penalty]
            if self.n_outputs_ > 1:
                lin_model = MultiOutputClassifier(lin_model)
        else:
            lin_model = {
                'ridge': RidgeCV(alphas=np.logspace(-2, 3, 7)),
                'elasticnet': ElasticNetCV(n_alphas=7),
                'lasso': LassoCV(n_alphas=7),
            }[self.linear_penalty]

        # onehot prior is a prior (for regression only) that
        # the ebm features predicting the target should have coef 1
        if not self.onehot_prior or isinstance(self, ClassifierMixin):
            lin_model.fit(feats, y, sample_weight=sample_weight)
        else:
            coef_prior_ = np.zeros((feats.shape[1], ))
            coef_prior_[:-len(self.term_names_list_[-1])] = 1
            preds_prior = feats @ coef_prior_
            residuals = y - preds_prior
            lin_model.fit(feats, residuals, sample_weight=sample_weight)
            lin_model.coef_ = lin_model.coef_ + coef_prior_
        return lin_model

    def _extract_ebm_features(self, X):
        '''
        Extract features by extracting all terms with EBM
        '''
        feats = np.empty((X.shape[0], len(self.term_names_)))
        offset = 0
        for ebm_num in range(len(self.ebms_)):
            n_features_ebm_num = len(self.term_names_list_[ebm_num])
            feats[:, offset: offset + n_features_ebm_num] = \
                self.ebms_[ebm_num].eval_terms(X)
            offset += n_features_ebm_num

        return feats

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X, accept_sparse=False)
        if hasattr(self, 'ebms_'):
            feats = self._extract_ebm_features(X)
            if hasattr(self, 'scaler_'):
                feats = self.scaler_.transform(feats)
            if hasattr(self, 'correlation_screener_'):
                feats = self.correlation_screener_.transform(feats)
            feats[np.isinf(feats)] = 0
            return self.lin_model.predict(feats)

        # multi-output without multitask learning
        elif hasattr(self, 'ebm_multioutput_'):
            return self.ebm_multioutput_.predict(X)

        # single-task standard
        elif hasattr(self, 'ebm_'):
            return self.ebm_.predict(X)

    def predict_proba(self, X):
        check_is_fitted(self)
        if hasattr(self, 'ebms_'):
            feats = self._extract_ebm_features(X)
            if hasattr(self, 'scaler_'):
                feats = self.scaler_.transform(feats)
            if hasattr(self, 'correlation_screener_'):
                feats = self.correlation_screener_.transform(feats)
            return self.lin_model.predict_proba(feats)

        # multi-output without multitask learning
        elif hasattr(self, 'ebm_multioutput_'):
            return self.ebm_multioutput_.predict_proba(X)

        # single-task standard
        elif hasattr(self, 'ebm_'):
            return self.ebm_.predict_proba(X)


class MultiTaskGAMRegressor(MultiTaskGAM, RegressorMixin):
    ...


class MultiTaskGAMClassifier(MultiTaskGAM, ClassifierMixin):
    ...
