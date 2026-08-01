from copy import deepcopy
from typing import List

import numpy as np
from sklearn import datasets
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier

from imodels.tree.hierarchical_shrinkage import HSTreeRegressor, HSTreeClassifier
from imodels.util.tree import compute_tree_complexity


class DecisionTreeCCPClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, estimator_: BaseEstimator, desired_complexity: int = 1, complexity_measure='max_rules', *args,
                 **kwargs):
        self.desired_complexity = desired_complexity
        self.estimator_ = estimator_
        self.complexity_measure = complexity_measure

    def get_params(self, deep=True):
        # defined explicitly because __init__ takes *args/**kwargs, which sklearn's
        # automatic parameter introspection rejects
        return {
            "estimator_": self.estimator_,
            "desired_complexity": self.desired_complexity,
            "complexity_measure": self.complexity_measure,
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def _copy_fitted_attributes(self):
        """Mirror the wrapped estimator's fitted sklearn attributes onto self."""
        for attr in ("classes_", "n_features_in_", "feature_names_in_"):
            if hasattr(self.estimator_, attr):
                setattr(self, attr, getattr(self.estimator_, attr))

    def _get_alpha(self, X, y, sample_weight=None, *args, **kwargs):
        path = self.estimator_.cost_complexity_pruning_path(
            X, y, sample_weight=sample_weight)
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
            copied_estimator.fit(X, y, sample_weight=sample_weight)
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
        self.estimator_.fit(X, y, *args, sample_weight=sample_weight, **kwargs)
        self._copy_fitted_attributes()
        return self

    def get_rules(self, feature_names=None):
        """Return this model's rules as a DataFrame (see imodels.get_rules)."""
        from imodels.util.get_rules import get_rules
        return get_rules(self, feature_names=feature_names)

    @property
    def feature_importances_(self):
        """Mean decrease in impurity of the pruned tree, as in sklearn."""
        return self.estimator_.feature_importances_

    def _get_complexity(self, BaseEstimator, complexity_measure):
        return compute_tree_complexity(BaseEstimator.tree_, complexity_measure)

    def predict_proba(self, X, *args, **kwargs):
        if hasattr(self.estimator_, 'predict_proba'):
            return self.estimator_.predict_proba(X, *args, **kwargs)
        else:
            return NotImplemented

    def predict(self, X, *args, **kwargs):
        return self.estimator_.predict(X, *args, **kwargs)

    def score(self, X, y, *args, **kwargs):
        if hasattr(self.estimator_, 'score'):
            return self.estimator_.score(X, y, *args, **kwargs)
        else:
            return NotImplemented


class DecisionTreeCCPRegressor(BaseEstimator):

    def __init__(self, estimator_: BaseEstimator, desired_complexity: int = 1, complexity_measure='max_rules', *args,
                 **kwargs):
        self.desired_complexity = desired_complexity
        self.estimator_ = estimator_
        self.alpha = 0.0
        self.complexity_measure = complexity_measure

    def get_params(self, deep=True):
        # defined explicitly because __init__ takes *args/**kwargs, which sklearn's
        # automatic parameter introspection rejects
        return {
            "estimator_": self.estimator_,
            "desired_complexity": self.desired_complexity,
            "complexity_measure": self.complexity_measure,
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def _copy_fitted_attributes(self):
        """Mirror the wrapped estimator's fitted sklearn attributes onto self."""
        for attr in ("n_features_in_", "feature_names_in_"):
            if hasattr(self.estimator_, attr):
                setattr(self, attr, getattr(self.estimator_, attr))

    def _get_alpha(self, X, y, sample_weight=None):
        path = self.estimator_.cost_complexity_pruning_path(
            X, y, sample_weight=sample_weight)
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
            copied_estimator.fit(X, y, sample_weight=sample_weight)
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
        self.estimator_.fit(X, y, sample_weight=sample_weight)
        self._copy_fitted_attributes()
        return self

    @property
    def feature_importances_(self):
        """Mean decrease in impurity of the pruned tree, as in sklearn."""
        return self.estimator_.feature_importances_

    def _get_complexity(self, BaseEstimator, complexity_measure):
        return compute_tree_complexity(BaseEstimator.tree_, self.complexity_measure)

    def predict(self, X, *args, **kwargs):
        return self.estimator_.predict(X, *args, **kwargs)

    def score(self, X, y, *args, **kwargs):
        if hasattr(self.estimator_, 'score'):
            return self.estimator_.score(X, y, *args, **kwargs)
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
        return super().fit(X=X, y=y)


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
        return super().fit(X=X, y=y)


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
