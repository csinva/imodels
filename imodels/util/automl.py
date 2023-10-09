from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin, ClassifierMixin
from imodels import (
    RuleFitClassifier,
    TreeGAMClassifier,
    FIGSClassifier,
    HSTreeClassifier,
    RuleFitRegressor,
    TreeGAMRegressor,
    FIGSRegressor,
    HSTreeRegressor,
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression, ElasticNet, Ridge
import imodels
from sklearn.model_selection import GridSearchCV, train_test_split
import numpy as np
from sklearn.pipeline import Pipeline


class AutoInterpretableModel(BaseEstimator):
    """Automatically fit and select a classifier that is interpretable.
    Note that all preprocessing should be done beforehand.
    This is basically a wrapper around GridSearchCV, with some preselected models.
    """

    def __init__(self, param_grid=None, refit=True):
        if param_grid is None:
            if isinstance(self, ClassifierMixin):
                self.param_grid = self.PARAM_GRID_DEFAULT_CLASSIFICATION
            elif isinstance(self, RegressorMixin):
                self.param_grid = self.PARAM_GRID_DEFAULT_REGRESSION
        else:
            self.param_grid = param_grid
        self.refit = refit

    def fit(self, X, y, cv=5):
        self.pipe_ = Pipeline([("est", BaseEstimator())]
                              )  # Placeholder Estimator
        if isinstance(self, ClassifierMixin):
            scoring = "roc_auc"
        elif isinstance(self, RegressorMixin):
            scoring = "r2"
        self.est_ = GridSearchCV(
            self.pipe_, self.param_grid, scoring=scoring, cv=cv, refit=self.refit)
        self.est_.fit(X, y)
        return self

    def predict(self, X):
        return self.est_.predict(X)

    def predict_proba(self, X):
        return self.est_.predict_proba(X)

    def score(self, X, y):
        return self.est_.score(X, y)

    PARAM_GRID_LINEAR_CLASSIFICATION = [
        {
            "est": [
                LogisticRegression(
                    solver="saga", penalty="elasticnet", max_iter=100, random_state=42)
            ],
            "est__C": [0.1, 1, 10],
            "est__l1_ratio": [0, 0.5, 1],
        },
    ]

    PARAM_GRID_DEFAULT_CLASSIFICATION = [
        {
            "est": [DecisionTreeClassifier(random_state=42)],
            "est__max_leaf_nodes": [2, 5, 10],
        },
        {
            "est": [RuleFitClassifier(random_state=42)],
            "est__max_rules": [10, 100],
            "est__n_estimators": [20],
        },
        {
            "est": [TreeGAMClassifier(random_state=42)],
            "est__n_boosting_rounds": [10, 100],
        },
        {
            "est": [HSTreeClassifier(random_state=42)],
            "est__max_leaf_nodes": [5, 10],
        },
        {
            "est": [FIGSClassifier(random_state=42)],
            "est__max_rules": [5, 10],
        },
    ] + PARAM_GRID_LINEAR_CLASSIFICATION

    PARAM_GRID_LINEAR_REGRESSION = [
        {
            "est": [
                ElasticNet(max_iter=100, random_state=42)
            ],
            "est__alpha": [0.1, 1, 10],
            "est__l1_ratio": [0.5, 1],
        },
        {
            "est": [
                Ridge(max_iter=100, random_state=42)
            ],
            "est__alpha": [0, 0.1, 1, 10],
        },
    ]

    PARAM_GRID_DEFAULT_REGRESSION = [
        {
            "est": [DecisionTreeRegressor()],
            "est__max_leaf_nodes": [2, 5, 10],
        },
        {
            "est": [HSTreeRegressor()],
            "est__max_leaf_nodes": [5, 10],
        },

        {
            "est": [RuleFitRegressor()],
            "est__max_rules": [10, 100],
            "est__n_estimators": [20],
        },
        {
            "est": [TreeGAMRegressor()],
            "est__n_boosting_rounds": [10, 100],
        },
        {
            "est": [FIGSRegressor()],
            "est__max_rules": [5, 10],
        },
    ] + PARAM_GRID_LINEAR_REGRESSION


class AutoInterpretableClassifier(AutoInterpretableModel, ClassifierMixin):
    ...


class AutoInterpretableRegressor(AutoInterpretableModel, RegressorMixin):
    ...


if __name__ == "__main__":
    X, y, feature_names = imodels.get_clean_dataset("heart")

    print("shapes", X.shape, y.shape, "nunique", np.unique(y).size)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=42, test_size=0.2
    )

    m = AutoInterpretableClassifier()
    # m = AutoInterpretableRegressor()
    m.fit(X_train, y_train)

    print("best params", m.est_.best_params_)
    print("best score", m.est_.best_score_)
    print("best estimator", m.est_.best_estimator_)
    print("best estimator params", m.est_.best_estimator_.get_params())
