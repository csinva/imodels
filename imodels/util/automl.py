from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin, ClassifierMixin
from imodels import (
    RuleFitClassifier,
    TreeGAMClassifier,
    FIGSClassifier,
    HSTreeClassifier,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import imodels
from sklearn.model_selection import GridSearchCV, train_test_split
import numpy as np
from sklearn.pipeline import Pipeline


class AutoInterpretableClassifier(BaseEstimator, ClassifierMixin):
    """Automatically fit and select a classifier that is interpretable.
    Note that all preprocessing should be done beforehand.
    This is basically a wrapper around GridSearchCV, with some preselected models.
    """

    PARAM_GRID_DEFAULT = [
        {
            "est": [DecisionTreeClassifier()],
            "est__max_leaf_nodes": [2, 5, 10],
        },
        {
            "est": [LogisticRegression()],
            "est__C": [0.1, 1, 10],
            "est__penalty": ["l1", "l2", "elasticnet"],
        },
        {
            "est": [RuleFitClassifier()],
            "est__max_rules": [10, 100],
            "est__n_estimators": [20],
        },
        {
            "est": [TreeGAMClassifier()],
            "est__n_boosting_rounds": [10, 100],
        },
        {
            "est": [HSTreeClassifier()],
            "est__max_leaf_nodes": [5, 10],
        },
        {
            "est": [FIGSClassifier()],
            "est__max_rules": [5, 10],
        },
    ]

    def __init__(self, param_grid=None):
        if param_grid is None:
            self.param_grid_ = self.PARAM_GRID_DEFAULT
        else:
            self.param_grid_ = param_grid

    def fit(self, X, y):
        self.pipe_ = Pipeline([("est", BaseEstimator())])  # Placeholder Estimator
        self.est_ = GridSearchCV(self.pipe_, self.param_grid_, scoring="roc_auc")
        self.est_.fit(X, y)
        return self

    def predict(self, X):
        return self.est_.predict(X)

    def predict_proba(self, X):
        return self.est_.predict_proba(X)

    def score(self, X, y):
        return self.est_.score(X, y)


if __name__ == "__main__":
    X, y, feature_names = imodels.get_clean_dataset("heart")

    print("shapes", X.shape, y.shape, "nunique", np.unique(y).size)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=42, test_size=0.2
    )

    m = AutoInterpretableClassifier()
    m.fit(X_train, y_train)

    print("best params", m.est_.best_params_)
    print("best score", m.est_.best_score_)
    print("best estimator", m.est_.best_estimator_)
    print("best estimator params", m.est_.best_estimator_.get_params())
