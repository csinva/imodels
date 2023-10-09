from copy import deepcopy
from functools import partial

import numpy as np
import sklearn
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.base import BaseEstimator, ClassifierMixin, MetaEstimatorMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils.multiclass import check_classification_targets, unique_labels
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from imodels.rule_set.rule_set import RuleSet
from imodels.rule_set.slipper_util import SlipperBaseEstimator
from imodels.util.arguments import check_fit_arguments
from imodels.util.convert import tree_to_code, tree_to_rules, dict_to_rule
from imodels.util.rule import Rule, get_feature_dict, replace_feature_name


class BoostedRulesClassifier(AdaBoostClassifier):
    '''An easy-interpretable classifier optimizing simple logical rules.

    Params
    ------
    estimator: object with fit and predict methods
        Defaults to DecisionTreeClassifier with AdaBoost.
        For SLIPPER, should pass estimator=imodels.SlipperBaseEstimator
    '''

    def __init__(
        self,
        estimator=DecisionTreeClassifier(max_depth=1),
        *,
        n_estimators=15,
        learning_rate=1.0,
        random_state=None,
    ):
        try: # sklearn version >= 1.2
            super().__init__(
                estimator=estimator,
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                random_state=random_state,
            )
        except: # sklearn version < 1.2
            super().__init__(
                base_estimator=estimator,
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                random_state=random_state,
            )
            self.estimator = estimator


    def fit(self, X, y, feature_names=None, **kwargs):
        X, y, feature_names = check_fit_arguments(self, X, y, feature_names)
        super().fit(X, y, **kwargs)
        self.complexity_ = len(self.estimators_)

class BoostedRulesRegressor(AdaBoostRegressor):
    '''An easy-interpretable regressor optimizing simple logical rules.

    Params
    ------
    estimator: object with fit and predict methods
        Defaults to DecisionTreeRegressor with AdaBoost.
    '''

    def __init__(
        self,
        estimator=DecisionTreeRegressor(max_depth=1),
        *,
        n_estimators=15,
        learning_rate=1.0,
        random_state=13,
    ):
        try: # sklearn version >= 1.2
            super().__init__(
                estimator=estimator,
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                random_state=random_state,
            )
        except: # sklearn version < 1.2
            super().__init__(
                base_estimator=estimator,
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                random_state=random_state,
            )
            self.estimator = estimator

    def fit(self, X, y, feature_names=None, **kwargs):
        X, y, feature_names = check_fit_arguments(self, X, y, feature_names)
        super().fit(X, y, **kwargs)
        self.complexity_ = len(self.estimators_)


if __name__ == '__main__':
    np.random.seed(13)
    X, Y = sklearn.datasets.load_breast_cancer(as_frame=True, return_X_y=True)
    model = BoostedRulesClassifier(estimator=DecisionTreeClassifier)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3)
    model.fit(X_train, y_train, feature_names=X_train.columns)
    y_pred = model.predict(X_test)
    acc = model.score(X_test, y_test)
    print('acc', acc, 'complexity', model.complexity_)
    print(model)
