from copy import deepcopy
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_array
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import check_X_y
from sklearn.utils.validation import check_random_state
from sklearn.utils.validation import column_or_1d
from sklearn.utils.validation import check_consistent_length
from sklearn.utils.validation import _check_sample_weight
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm


class TreeGAMClassifier(BaseEstimator):
    """Tree-based GAM classifier.
    Uses cyclical boosting to fit a GAM with small trees.
    Simplified version of the explainable boosting machine described in https://github.com/interpretml/interpret
    """

    def __init__(
        self,
        max_leaf_nodes=3,
        n_boosting_rounds=20,
        random_state=None,
    ):
        self.max_leaf_nodes = max_leaf_nodes
        self.random_state = random_state
        self.n_boosting_rounds = n_boosting_rounds

    def fit(self, X, y, sample_weight=None):
        X, y = check_X_y(X, y, accept_sparse=False, multi_output=False)
        check_classification_targets(y)
        sample_weight = _check_sample_weight(sample_weight, X, dtype=None)

        # cycle through features and fit a tree to each one
        ests = []
        for boosting_round in tqdm(range(self.n_boosting_rounds)):
            for feature_num in range(X.shape[1]):
                X_ = np.zeros_like(X)
                X_[:, feature_num] = X[:, feature_num]
                est = DecisionTreeRegressor(
                    max_leaf_nodes=self.max_leaf_nodes,
                    random_state=self.random_state,
                )
                est.fit(X_, y, sample_weight=sample_weight)
                if not est.tree_.feature[0] == feature_num:
                    # failed to split on this feature
                    continue
                ests.append(est)
                y = y - est.predict(X)

        self.est_ = GradientBoostingRegressor()
        self.est_.fit(X, y)
        self.est_.n_estimators_ = len(ests)
        self.est_.estimators_ = np.array(ests).reshape(-1, 1)

        return self

    def predict_proba(self, X):
        X = check_array(X, accept_sparse=False, dtype=None)
        check_is_fitted(self)
        probs1 = self.est_.predict(X)
        return np.array([1 - probs1, probs1]).T

    def predict(self, X):
        X = check_array(X, accept_sparse=False, dtype=None)
        check_is_fitted(self)
        return (self.est_.predict(X) > 0.5).astype(int)


if __name__ == "__main__":
    breast = load_breast_cancer()
    feature_names = list(breast.feature_names)
    X, y = pd.DataFrame(breast.data, columns=feature_names), breast.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    gam = TreeGAMClassifier(max_leaf_nodes=2)
    gam.fit(X_train, y_train)

    # check roc auc score
    y_pred = gam.predict_proba(X_test)[:, 1]
    print(
        "train roc auc score:", roc_auc_score(y_train, gam.predict_proba(X_train)[:, 1])
    )
    print("test roc auc score:", roc_auc_score(y_test, y_pred))
