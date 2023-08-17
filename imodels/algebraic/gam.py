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

import imodels


class TreeGAMClassifier(BaseEstimator):
    """Tree-based GAM classifier.
    Uses cyclical boosting to fit a GAM with small trees.
    Simplified version of the explainable boosting machine described in https://github.com/interpretml/interpret
    Only works for binary classification.
    """

    def __init__(
        self,
        max_leaf_nodes=3,
        n_boosting_rounds=100,
        random_state=None,
    ):
        self.max_leaf_nodes = max_leaf_nodes
        self.random_state = random_state
        self.n_boosting_rounds = n_boosting_rounds

    def fit(self, X, y, sample_weight=None, learning_rate=0.01):
        X, y = check_X_y(X, y, accept_sparse=False, multi_output=False)
        check_classification_targets(y)
        sample_weight = _check_sample_weight(sample_weight, X, dtype=None)

        # cycle through features and fit a tree to each one
        self.estimators_ = []
        self.learning_rate = learning_rate
        self.bias_ = np.mean(y)
        residuals = y - self.bias_
        for boosting_round in tqdm(range(self.n_boosting_rounds)):
            for feature_num in range(X.shape[1]):
                X_ = np.zeros_like(X)
                X_[:, feature_num] = X[:, feature_num]
                est = DecisionTreeRegressor(
                    max_leaf_nodes=self.max_leaf_nodes,
                    random_state=self.random_state,
                )
                est.fit(X_, residuals, sample_weight=sample_weight)
                succesfully_split_on_feature = np.all(
                    (est.tree_.feature[0] == feature_num) | (est.tree_.feature[0] == -2)
                )
                if not succesfully_split_on_feature:
                    continue
                self.estimators_.append(est)
                residuals = residuals - self.learning_rate * est.predict(X)
        return self

    def predict_proba(self, X):
        X = check_array(X, accept_sparse=False, dtype=None)
        check_is_fitted(self)
        probs1 = np.ones(X.shape[0]) * self.bias_
        for est in self.estimators_:
            probs1 += self.learning_rate * est.predict(X)
        probs1 = np.clip(probs1, a_min=0, a_max=1)
        return np.array([1 - probs1, probs1]).T

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


if __name__ == "__main__":
    X, y, feature_names = imodels.get_clean_dataset("heart")
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    gam = TreeGAMClassifier()
    gam.fit(X_train, y_train)

    # check roc auc score
    y_pred = gam.predict_proba(X_test)[:, 1]
    print(
        "train roc auc score:", roc_auc_score(y_train, gam.predict_proba(X_train)[:, 1])
    )
    print("test roc auc score:", roc_auc_score(y_test, y_pred))
    print(
        "accs",
        accuracy_score(y_train, gam.predict(X_train)),
        accuracy_score(y_test, gam.predict(X_test)),
        "imb",
        np.mean(y_train),
        np.mean(y_test),
    )
