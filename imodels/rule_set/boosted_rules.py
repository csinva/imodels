
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from imodels.util.arguments import check_fit_arguments
from imodels.util.introspection import RuleInspectionMixin


class BoostedRulesClassifier(RuleInspectionMixin, AdaBoostClassifier):
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
        classes = self.classes_  # super().fit overwrites this with the encoded labels
        names_in = getattr(self, 'feature_names_in_', None)
        super().fit(X, y, **kwargs)
        self.classes_ = classes
        if names_in is not None:  # super().fit strips this when passed a plain array
            self.feature_names_in_ = names_in
        self.complexity_ = len(self.estimators_)
        return self


class BoostedRulesRegressor(RuleInspectionMixin, AdaBoostRegressor):
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
        names_in = getattr(self, 'feature_names_in_', None)
        super().fit(X, y, **kwargs)
        if names_in is not None:  # super().fit strips this when passed a plain array
            self.feature_names_in_ = names_in
        self.complexity_ = len(self.estimators_)
        return self
