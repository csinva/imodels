from copy import deepcopy
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.linear_model import ElasticNetCV, LinearRegression, RidgeCV, LassoCV
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
from collections import defaultdict
import pandas as pd
import json

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
        ebm_kwargs={'interactions': 0, 'n_jobs': 1},
        multitask=True,
        linear_penalty='ridge',
        random_state=42,
    ):
        """
        Params
        ------
        """
        self.ebm_kwargs = ebm_kwargs
        self.multitask = multitask
        self.linear_penalty = linear_penalty
        self.random_state = random_state
        if not 'random_state' in ebm_kwargs:
            ebm_kwargs['random_state'] = random_state
        self.ebm_ = ExplainableBoostingRegressor(**(ebm_kwargs or {}))

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
        self.ebms_ = []
        num_features = X.shape[1]
        for task_num in tqdm(range(num_features)):
            self.ebms_.append(deepcopy(self.ebm_))
            y_ = np.ascontiguousarray(X[:, task_num])
            X_ = deepcopy(X)
            X_[:, task_num] = 0
            self.ebms_[task_num].fit(X_, y_, sample_weight=sample_weight)

        # finally, fit EBM to the target
        self.ebms_.append(deepcopy(self.ebm_))
        self.ebms_[num_features].fit(X, y, sample_weight=sample_weight)

        # extract features
        feats = self._extract_ebm_features(X)

        # fit a linear model to the features
        if self.linear_penalty == 'ridge':
            self.lin_model = RidgeCV(alphas=np.logspace(-2, 3, 7))
        elif self.linear_penalty == 'elasticnet':
            self.lin_model = ElasticNetCV(n_alphas=7)
        elif self.linear_penalty == 'lasso':
            self.lin_model = LassoCV(n_alphas=7)

        self.lin_model.fit(feats, y)
        return self

    def _extract_ebm_features(self, X):
        '''
        Extract features by predicting each feature with each EBM
        Note: this doesn't currently handle interactions
        '''
        num_ebms = X.shape[1] + 1
        num_features = X.shape[1]
        feats = np.zeros((X.shape[0], num_ebms * num_features))
        for ebm_num in range(num_ebms):
            # see eval_terms function: https://interpret.ml/docs/python/api/ExplainableBoostingRegressor.html#interpret.glassbox.ExplainableBoostingRegressor.eval_terms
            feats[:, ebm_num * num_features: (ebm_num + 1) * num_features] = \
                self.ebms_[ebm_num].eval_terms(X)

        return feats

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X, accept_sparse=False)
        if hasattr(self, 'ebms_'):
            feats = self._extract_ebm_features(X)
            return self.lin_model.predict(feats)
        else:
            return self.ebm_.predict(X)

    # def predict_proba(self, X):
    #     check_is_fitted(self)
    #     X = check_array(X, accept_sparse=False)
    #     return self.ebm_.predict_proba(X)


class MultiTaskGAMRegressor(MultiTaskGAM, RegressorMixin):
    ...


class MultiTaskGAMClassifier(MultiTaskGAM, ClassifierMixin):
    ...


def test_multitask_extraction():
    X, y, feature_names = imodels.get_clean_dataset("california_housing")
    # X, y, feature_names = imodels.get_clean_dataset("bike_sharing")

    # remove some features to speed things up
    X = X[:10]
    y = y[:10]
    X, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    # unit test
    gam = MultiTaskGAMRegressor(multitask=False)
    gam.fit(X, y_train)
    ebm = gam.ebm_
    # print('feature_names_in', ebm.feature_names_in_)
    gam2 = MultiTaskGAMRegressor(multitask=True)
    gam2.fit(X, y_train)
    preds_orig = gam.predict(X_test)
    assert np.allclose(preds_orig, gam2.ebms_[-1].predict(X_test))

    # extracted curves should sum to original predictions
    feats_extracted = gam2._extract_ebm_features(X_test)
    num_samples = X_test.shape[0]
    num_features = X_test.shape[1]
    num_ebms = num_features + 1
    feats_extracted_target = feats_extracted[:, -num_features:]
    assert feats_extracted_target.shape == (num_samples, num_features)
    preds_extracted_target = np.sum(feats_extracted_target, axis=1) + \
        gam2.ebms_[-1].intercept_
    diff = preds_extracted_target - preds_orig
    assert np.allclose(preds_extracted_target, preds_orig), diff
    print('Tests pass successfully')


if __name__ == "__main__":
    test_multitask_extraction()
    # X, y, feature_names = imodels.get_clean_dataset("heart")
    X, y, feature_names = imodels.get_clean_dataset("bike_sharing")
    # X, y, feature_names = imodels.get_clean_dataset("diabetes")

    # remove some features to speed things up
    X = X[:, :3]
    X, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    kwargs = dict(
        random_state=42,
    )
    results = defaultdict(list)
    for gam in tqdm([
            # AdaBoostRegressor(estimator=MultiTaskGAMRegressor(
        # multitask=True), n_estimators=2),
        MultiTaskGAMRegressor(multitask=False),
            # MultiTaskGAMRegressor(multitask=True),
        # ExplainableBoostingRegressor(n_jobs=1, interactions=0)
    ]):
        np.random.seed(42)
        results["model_name"].append(gam)
        print('Fitting', results['model_name'][-1])
        gam.fit(X, y_train)
        results['test_corr'].append(np.corrcoef(
            y_test, gam.predict(X_test))[0, 1].round(3))
        results['test_r2'].append(gam.score(X_test, y_test).round(3))
        if hasattr(gam, 'lin_model'):
            print('lin model coef', gam.lin_model.coef_)

    # don't round strings
    with pd.option_context(
        "display.max_rows", None, "display.max_columns", None, "display.width", 1000
    ):
        print(pd.DataFrame(results).round(3))
