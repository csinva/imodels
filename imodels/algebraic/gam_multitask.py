from copy import deepcopy
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.linear_model import ElasticNetCV, LinearRegression, RidgeCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_array
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_X_y
from sklearn.utils.validation import _check_sample_weight
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm
from collections import defaultdict

import imodels
from interpret.glassbox import ExplainableBoostingClassifier, ExplainableBoostingRegressor

from sklearn.base import RegressorMixin, ClassifierMixin


# See notes in this implementation:
# https://github.com/interpretml/interpret/blob/develop/python/interpret-core/interpret/glassbox/_ebm/_ebm.py
# merge ebms: https://github.com/interpretml/interpret/blob/develop/python/interpret-core/interpret/glassbox/_ebm/_merge_ebms.py#L280

class MultiTaskGAM(BaseEstimator):
    """Multi-task GAM classifier.
    """

    def __init__(
        self,
        ebm_kwargs={},
        multitask=True,
        random_state=42,

    ):
        """
        Params
        ------
        """
        self.ebm_kwargs = ebm_kwargs
        self.multitask = multitask
        self.random_state = random_state
        if not 'random_state' in ebm_kwargs:
            ebm_kwargs['random_state'] = random_state
        self.ebm_ = ExplainableBoostingRegressor(**(ebm_kwargs or {}))

        # self.ebm_ = ExplainableBoostingClassifier(**(ebm_kwargs or {}))

    def fit(self, X, y, sample_weight=None):
        X, y = check_X_y(X, y, accept_sparse=False, multi_output=False)
        if isinstance(self, ClassifierMixin):
            check_classification_targets(y)
            self.classes_, y = np.unique(y, return_inverse=True)
        sample_weight = _check_sample_weight(sample_weight, X, dtype=None)

        # just fit normal ebm
        if not self.multitask:
            self.ebm_.fit(X, y, sample_weight=sample_weight)
            return self

        # fit EBM to each column of X
        self.ebms_ = defaultdict(list)
        num_features = X.shape[1]
        for task_num in tqdm(range(num_features)):
            self.ebms_[task_num] = deepcopy(self.ebm_)
            y_ = np.ascontiguousarray(X[:, task_num])
            X_ = deepcopy(X)
            X_[:, task_num] = 0
            self.ebms_[task_num].fit(X_, y_, sample_weight=sample_weight)

        # finally, fit EBM to the target
        self.ebms_[num_features] = deepcopy(self.ebm_)
        self.ebms_[num_features].fit(X, y, sample_weight=sample_weight)

        # extract features
        feats = self.extract_ebm_features(X)

        # fit a linear model to the features
        self.lin_model = RidgeCV(alphas=np.logspace(-2, 3, 7))
        self.lin_model.fit(feats, y)
        return self

    def extract_ebm_features(self, X):
        '''
        Extract features by predicting each feature with each EBM
        This is a hack for now, ideally would just extract curves
        '''
        num_features = X.shape[1]
        num_outputs = num_features + 1
        feats = np.zeros((X.shape[0], num_features * num_outputs))
        for feat_num in range(num_features):
            X_ = np.zeros_like(X)
            X_[:, feat_num] = X[:, feat_num]

            # extract feature curve from each EBM for feat_num
            for task_num in range(num_outputs):
                ebm = self.ebms_[task_num]
                feats[:, feat_num * num_outputs +
                      task_num] = ebm.predict(X_) - ebm.intercept_
        return feats

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X, accept_sparse=False)
        if hasattr(self, 'ebms_'):
            feats = self.extract_ebm_features(X)
            return self.lin_model.predict(feats)
        else:
            return self.ebm_.predict(X)

    def predict_proba(self, X):
        check_is_fitted(self)
        X = check_array(X, accept_sparse=False)
        return self.ebm_.predict_proba(X)


class MultiTaskGAMRegressor(MultiTaskGAM, RegressorMixin):
    ...


class MultiTaskGAMClassifier(MultiTaskGAM, ClassifierMixin):
    ...


if __name__ == "__main__":
    # X, y, feature_names = imodels.get_clean_dataset("heart")
    X, y, feature_names = imodels.get_clean_dataset("bike_sharing")
    # X, y, feature_names = imodels.get_clean_dataset("diabetes")

    # remove some features to speed things up
    # X = X[:, :3]
    X, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    # gam = MultiTaskGAMClassifier(
    kwargs = dict(
        random_state=42,
    )
    results = defaultdict(list)
    for gam in tqdm([
            MultiTaskGAMRegressor(multitask=False),
            MultiTaskGAMRegressor(multitask=True),
    ]):
        np.random.seed(42)
        results["model_name"].append(gam)
        print('Fitting', results['model_name'][-1])
        gam.fit(X, y_train)

        # check roc auc score
        # y_pred = gam.predict_proba(X_test)[:, 1]
        # print(
        #     "train roc:",
        #     roc_auc_score(y_train, gam.predict_proba(X)[:, 1]).round(3),
        # )
        # print("test roc:", round(roc_auc_score(y_test, y_pred), 3))
        # print("test acc:", round(accuracy_score(y_test, gam.predict(X_test)), 3))
        # print('\t(imb:', np.mean(y_test).round(3), ')')
        results['test_corr'].append(np.corrcoef(
            y_test, gam.predict(X_test))[0, 1].round(3))
        results['test_r2'].append(gam.score(X_test, y_test).round(3))
        if hasattr(gam, 'lin_model'):
            print('lin model coef', gam.lin_model.coef_)

        # print('test corr', np.corrcoef(
        # y_test, gam.predict(X_test))[0, 1].round(3))
        # print('test r2', gam.score(X_test, y_test).round(3))

        # print(
        #     "accs",
        #     accuracy_score(y_train, gam.predict(X)).round(3),
        #     accuracy_score(y_test, gam.predict(X_test)).round(3),
        #     "imb",
        #     np.mean(y_train).round(3),
        #     np.mean(y_test).round(3),
        # )

        # # print(gam.estimators_)

    # don't round strings
    with pd.option_context(
        "display.max_rows", None, "display.max_columns", None, "display.width", 1000
    ):
        print(pd.DataFrame(results).round(3))
