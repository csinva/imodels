from copy import deepcopy
from typing import List

import numpy as np
from sklearn import datasets
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier

from imodels.tree.hierarchical_shrinkage import HSTreeRegressor, HSTreeClassifier
from imodels.util.tree import compute_tree_complexity


class DecisionTreeCCPClassifier(DecisionTreeClassifier):
    def __init__(self, estimator_: BaseEstimator, desired_complexity: int = 1, complexity_measure='max_rules', *args,
                 **kwargs):
        self.desired_complexity = desired_complexity
        # print('est', estimator_)
        self.estimator_ = estimator_
        self.complexity_measure = complexity_measure

    def _get_alpha(self, X, y, sample_weight=None, *args, **kwargs):
        path = self.estimator_.cost_complexity_pruning_path(X, y)
        ccp_alphas, impurities = path.ccp_alphas, path.impurities
        complexities = {}
        low = 0
        high = len(ccp_alphas) - 1
        cur = 0
        while low <= high:
            cur = (high + low) // 2
            est_params = self.estimator_.get_params()
            est_params['ccp_alpha'] = ccp_alphas[cur]
            copied_estimator = deepcopy(self.estimator_).set_params(**est_params)
            copied_estimator.fit(X, y)
            if self._get_complexity(copied_estimator, self.complexity_measure) < self.desired_complexity:
                high = cur - 1
            elif self._get_complexity(copied_estimator, self.complexity_measure) > self.desired_complexity:
                low = cur + 1
            else:
                break
        self.alpha = ccp_alphas[cur]

        # for alpha in ccp_alphas:
        #    est_params = self.estimator_.get_params()
        #    est_params['ccp_alpha'] = alpha
        #    copied_estimator =  deepcopy(self.estimator_).set_params(**est_params)
        #    copied_estimator.fit(X, y)
        #    complexities[alpha] = self._get_complexity(copied_estimator,self.complexity_measure)
        # closest_alpha, closest_leaves = min(complexities.items(), key=lambda x: abs(self.desired_complexity - x[1]))
        # self.alpha = closest_alpha

    def fit(self, X, y, sample_weight=None, *args, **kwargs):
        params_for_fitting = self.estimator_.get_params()
        self._get_alpha(X, y, sample_weight, *args, **kwargs)
        params_for_fitting['ccp_alpha'] = self.alpha
        self.estimator_.set_params(**params_for_fitting)
        self.estimator_.fit(X, y, *args, **kwargs)

    def _get_complexity(self, BaseEstimator, complexity_measure):
        return compute_tree_complexity(BaseEstimator.tree_, complexity_measure)

    def predict_proba(self, *args, **kwargs):
        if hasattr(self.estimator_, 'predict_proba'):
            return self.estimator_.predict_proba(*args, **kwargs)
        else:
            return NotImplemented

    def predict(self, X, *args, **kwargs):
        return self.estimator_.predict(X, *args, **kwargs)

    def score(self, *args, **kwargs):
        if hasattr(self.estimator_, 'score'):
            return self.estimator_.score(*args, **kwargs)
        else:
            return NotImplemented


class DecisionTreeCCPRegressor(BaseEstimator):

    def __init__(self, estimator_: BaseEstimator, desired_complexity: int = 1, complexity_measure='max_rules', *args,
                 **kwargs):
        self.desired_complexity = desired_complexity
        # print('est', estimator_)
        self.estimator_ = estimator_
        self.alpha = 0.0
        self.complexity_measure = complexity_measure

    def _get_alpha(self, X, y, sample_weight=None):
        path = self.estimator_.cost_complexity_pruning_path(X, y)
        ccp_alphas, impurities = path.ccp_alphas, path.impurities
        complexities = {}
        low = 0
        high = len(ccp_alphas) - 1
        cur = 0
        while low <= high:
            cur = (high + low) // 2
            est_params = self.estimator_.get_params()
            est_params['ccp_alpha'] = ccp_alphas[cur]
            copied_estimator = deepcopy(self.estimator_).set_params(**est_params)
            copied_estimator.fit(X, y)
            if self._get_complexity(copied_estimator, self.complexity_measure) < self.desired_complexity:
                high = cur - 1
            elif self._get_complexity(copied_estimator, self.complexity_measure) > self.desired_complexity:
                low = cur + 1
            else:
                break
        self.alpha = ccp_alphas[cur]

    #  path = self.estimator_.cost_complexity_pruning_path(X,y)
    #  ccp_alphas, impurities = path.ccp_alphas, path.impurities
    #  complexities = {}
    #  for alpha in ccp_alphas:
    #      est_params = self.estimator_.get_params()
    #      est_params['ccp_alpha'] = alpha
    #      copied_estimator =  deepcopy(self.estimator_).set_params(**est_params)
    #      copied_estimator.fit(X, y)
    #      complexities[alpha] = self._get_complexity(copied_estimator,self.complexity_measure)
    #  closest_alpha, closest_leaves = min(complexities.items(), key=lambda x: abs(self.desired_complexity - x[1]))
    #  self.alpha = closest_alpha

    def fit(self, X, y, sample_weight=None):
        params_for_fitting = self.estimator_.get_params()
        self._get_alpha(X, y, sample_weight)
        params_for_fitting['ccp_alpha'] = self.alpha
        self.estimator_.set_params(**params_for_fitting)
        self.estimator_.fit(X, y)

    def _get_complexity(self, BaseEstimator, complexity_measure):
        return compute_tree_complexity(BaseEstimator.tree_, self.complexity_measure)

    def predict(self, X, *args, **kwargs):
        return self.estimator_.predict(X, *args, **kwargs)

    def score(self, *args, **kwargs):
        if hasattr(self.estimator_, 'score'):
            return self.estimator_.score(*args, **kwargs)
        else:
            return NotImplemented


class HSDecisionTreeCCPRegressorCV(HSTreeRegressor):
    def __init__(self, estimator_: BaseEstimator, reg_param_list: List[float] = [0.1, 1, 10, 50, 100, 500],
                 desired_complexity: int = 1, cv: int = 3, scoring=None, *args, **kwargs):
        super().__init__(estimator_=estimator_, reg_param=None)
        self.reg_param_list = np.array(reg_param_list)
        self.cv = cv
        self.scoring = scoring
        self.desired_complexity = desired_complexity

    def fit(self, X, y, sample_weight=None, *args, **kwargs):
        m = DecisionTreeCCPRegressor(self.estimator_, desired_complexity=self.desired_complexity)
        m.fit(X, y, sample_weight, *args, **kwargs)
        self.scores_ = []
        for reg_param in self.reg_param_list:
            est = HSTreeRegressor(deepcopy(m.estimator_), reg_param)
            cv_scores = cross_val_score(est, X, y, cv=self.cv, scoring=self.scoring)
            self.scores_.append(np.mean(cv_scores))
        self.reg_param = self.reg_param_list[np.argmax(self.scores_)]
        super().fit(X=X, y=y)


class HSDecisionTreeCCPClassifierCV(HSTreeClassifier):
    def __init__(self, estimator_: BaseEstimator, reg_param_list: List[float] = [0.1, 1, 10, 50, 100, 500],
                 desired_complexity: int = 1, cv: int = 3, scoring=None, *args, **kwargs):
        super().__init__(estimator_=estimator_, reg_param=None)
        self.reg_param_list = np.array(reg_param_list)
        self.cv = cv
        self.scoring = scoring
        self.desired_complexity = desired_complexity

    def fit(self, X, y, sample_weight=None, *args, **kwargs):
        m = DecisionTreeCCPClassifier(self.estimator_, desired_complexity=self.desired_complexity)
        m.fit(X, y, sample_weight, *args, **kwargs)
        self.scores_ = []
        for reg_param in self.reg_param_list:
            est = HSTreeClassifier(deepcopy(m.estimator_), reg_param)
            cv_scores = cross_val_score(est, X, y, cv=self.cv, scoring=self.scoring)
            self.scores_.append(np.mean(cv_scores))
        self.reg_param = self.reg_param_list[np.argmax(self.scores_)]
        super().fit(X=X, y=y)


if __name__ == '__main__':
    m = DecisionTreeCCPClassifier(estimator_=DecisionTreeClassifier(random_state=1), desired_complexity=10,
                                  complexity_measure='max_leaf_nodes')
    # X,y = make_friedman1() #For regression
    X, y = datasets.load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42)
    m.fit(X_train, y_train)
    m.predict(X_test)
    print(m.score(X_test, y_test))

    m = HSDecisionTreeCCPClassifierCV(estimator_=DecisionTreeClassifier(random_state=1), desired_complexity=10,
                                       reg_param_list=[0.0, 0.1, 1.0, 5.0, 10.0, 25.0, 50.0, 100.0])
    m.fit(X_train, y_train)
    print(m.score(X_test, y_test))
