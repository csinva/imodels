from copy import deepcopy
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_array
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_X_y
from sklearn.utils.validation import _check_sample_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm

import imodels

from sklearn.base import RegressorMixin, ClassifierMixin


class TreeGAMSimple(BaseEstimator):
    """Tree-based GAM classifier.
    Uses cyclical boosting to fit a GAM with small trees.
    Simplified version of the explainable boosting machine described in https://github.com/interpretml/interpret
    Only works for binary classification.
    Fits a scalar bias to the mean.
    """

    def __init__(
        self,
        n_boosting_rounds=100,
        max_leaf_nodes=3,
        learning_rate: float = 0.01,
        boosting_strategy="cyclic",
        validation_frac=0.15,
        random_state=None,
    ):
        """
        Params
        ------
        n_boosting_rounds : int
            Number of boosting rounds for the cyclic boosting.
        max_leaf_nodes : int
            Maximum number of leaf nodes for the trees in the cyclic boosting.
        learning_rate: float
            Learning rate for the cyclic boosting.
        boosting_strategy : str ["cyclic", "greedy"]
            Whether to use cyclic boosting (cycle over features) or greedy boosting (select best feature at each step)
        validation_frac: float
            Fraction of data to use for early stopping.
        random_state : int
            Random seed.
        """
        self.n_boosting_rounds = n_boosting_rounds
        self.max_leaf_nodes = max_leaf_nodes
        self.learning_rate = learning_rate
        self.boosting_strategy = boosting_strategy
        self.validation_frac = validation_frac
        self.random_state = random_state

    def fit(self, X, y, sample_weight=None):
        X, y = check_X_y(X, y, accept_sparse=False, multi_output=False)
        if isinstance(self, ClassifierMixin):
            check_classification_targets(y)
            self.classes_, y = np.unique(y, return_inverse=True)

        sample_weight = _check_sample_weight(sample_weight, X, dtype=None)

        # split into train and validation for early stopping
        (
            X_train,
            X_val,
            y_train,
            y_val,
            sample_weight_train,
            sample_weight_val,
        ) = train_test_split(
            X,
            y,
            sample_weight,
            test_size=self.validation_frac,
            random_state=self.random_state,
            stratify=y if isinstance(self, ClassifierMixin) else None,
        )

        self.estimators_ = []
        self.bias_ = np.mean(y)

        self._cyclic_boost(
            X_train,
            y_train,
            sample_weight_train,
            X_val,
            y_val,
            sample_weight_val,
        )

        self.mse_val_ = self._calc_mse(X_val, y_val, sample_weight_val)

        return self

    def _cyclic_boost(
        self, X_train, y_train, sample_weight_train, X_val, y_val, sample_weight_val
    ):
        """Apply cyclic boosting, storing trees in self.estimators_"""

        residuals_train = y_train - self.predict_proba(X_train)[:, 1]
        mse_val = self._calc_mse(X_val, y_val, sample_weight_val)
        for _ in range(self.n_boosting_rounds):
            boosting_round_ests = []
            boosting_round_mses = []
            for feature_num in range(X_train.shape[1]):
                X_ = np.zeros_like(X_train)
                X_[:, feature_num] = X_train[:, feature_num]
                est = DecisionTreeRegressor(
                    max_leaf_nodes=self.max_leaf_nodes,
                    random_state=self.random_state,
                )
                est.fit(X_, residuals_train, sample_weight=sample_weight_train)
                succesfully_split_on_feature = np.all(
                    (est.tree_.feature[0] == feature_num) | (
                        est.tree_.feature[0] == -2)
                )
                if not succesfully_split_on_feature:
                    continue
                self.estimators_.append(est)
                residuals_train_new = (
                    residuals_train - self.learning_rate * est.predict(X_train)
                )
                if self.boosting_strategy == "cyclic":
                    residuals_train = residuals_train_new
                elif self.boosting_strategy == "greedy":
                    mse_train_new = self._calc_mse(
                        X_train, y_train, sample_weight_train
                    )
                    # don't add each estimator for greedy
                    boosting_round_ests.append(
                        deepcopy(self.estimators_.pop()))
                    boosting_round_mses.append(mse_train_new)

            if self.boosting_strategy == "greedy":
                best_est = boosting_round_ests[np.argmin(boosting_round_mses)]
                self.estimators_.append(best_est)
                residuals_train = (
                    residuals_train - self.learning_rate *
                    best_est.predict(X_train)
                )

            # early stopping if validation error does not decrease
            mse_val_new = self._calc_mse(X_val, y_val, sample_weight_val)
            if mse_val_new >= mse_val:
                return
            else:
                mse_val = mse_val_new

    def predict_proba(self, X):
        X = check_array(X, accept_sparse=False, dtype=None)
        check_is_fitted(self)
        probs1 = np.ones(X.shape[0]) * self.bias_
        for i, est in enumerate(self.estimators_):
            probs1 += est.predict(X) * self.learning_rate
        probs1 = np.clip(probs1, a_min=0, a_max=1)
        return np.array([1 - probs1, probs1]).T

    def predict(self, X):
        if isinstance(self, RegressorMixin):
            return self.predict_proba(X)[:, 1]
        elif isinstance(self, ClassifierMixin):
            return np.argmax(self.predict_proba(X), axis=1)

    def _calc_mse(self, X, y, sample_weight=None):
        return np.average(
            np.square(y - self.predict_proba(X)[:, 1]),
            weights=sample_weight,
        )


class TreeGAMSimpleRegressor(TreeGAMSimple, RegressorMixin):
    ...


class TreeGAMSimpleClassifier(TreeGAMSimple, ClassifierMixin):
    ...


if __name__ == "__main__":
    X, y, feature_names = imodels.get_clean_dataset("heart")
    X, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    gam = TreeGAMSimpleClassifier(
        boosting_strategy="cyclic",
        random_state=42,
        learning_rate=0.1,
        max_leaf_nodes=3,
        n_boosting_rounds=100,
    )
    gam.fit(X, y_train)

    # check roc auc score
    y_pred = gam.predict_proba(X_test)[:, 1]
    print(
        "train roc:",
        roc_auc_score(y_train, gam.predict_proba(X)[:, 1]).round(3),
    )
    print("test roc:", roc_auc_score(y_test, y_pred).round(3))
    print("test acc:", accuracy_score(y_test, gam.predict(X_test)).round(3))
    print('\t(imb:', np.mean(y_test).round(3), ')')
