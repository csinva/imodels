from copy import deepcopy
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.linear_model import ElasticNetCV, LinearRegression, RidgeCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_array
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_X_y
from sklearn.utils.validation import _check_sample_weight
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm

import imodels

from sklearn.base import RegressorMixin, ClassifierMixin


class TreeGAM(BaseEstimator):
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
        reg_param=0.0,
        learning_rate: float = 0.01,
        n_boosting_rounds_marginal=0,
        max_leaf_nodes_marginal=2,
        reg_param_marginal=0.0,
        fit_linear_marginal=None,
        select_linear_marginal=False,
        decay_rate_towards_marginal=1.0,
        fit_posthoc_tree_coefs=None,
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
        reg_param : float
            Regularization parameter for the cyclic boosting.
        learning_rate: float
            Learning rate for the cyclic boosting.
        n_boosting_rounds_marginal : int
            Number of boosting rounds for the marginal boosting.
        max_leaf_nodes_marginal : int
            Maximum number of leaf nodes for the trees in the marginal boosting.
        reg_param_marginal : float
            Regularization parameter for the marginal boosting.
        fit_linear_marginal : str [None, "None", "ridge", "NNLS"]
            Whether to fit a linear model to the marginal effects.
            NNLS for non-negative least squares
            ridge for ridge regression
            None for no linear model
        select_linear_marginal: bool
            Whether to restrict features to those with non-negative coefficients in the linear model.
            Requires that fit_linear_marginal is NNLS.
        decay_rate_towards_marginal: float, [0, 1]
            Decay rate for regularizing each shape function towards the marginal shape function after each step
            1 means no decay, 0 means only use marginal effects
            shape = (1 - decay_rate_towards_marginal) * shape + decay_rate_towards_marginal * marginal_shape
            The way this is implemented is by keeping track of how many times to multiply decay_rate_towards_marginal for each cyclic estimator
        fit_posthoc_tree_coefs: str [None, "ridge"]
            Whether to fit a linear model to the tree coefficients after fitting the cyclic boosting.
        boosting_strategy : str ["cyclic", "greedy"]
            Whether to use cyclic boosting (cycle over features) or greedy boosting (select best feature at each step)
        validation_frac: float
            Fraction of data to use for early stopping.
        random_state : int
            Random seed.
        """
        self.n_boosting_rounds = n_boosting_rounds
        self.max_leaf_nodes = max_leaf_nodes
        self.reg_param = reg_param
        self.learning_rate = learning_rate
        self.max_leaf_nodes_marginal = max_leaf_nodes_marginal
        self.reg_param_marginal = reg_param_marginal
        self.n_boosting_rounds_marginal = n_boosting_rounds_marginal
        self.fit_linear_marginal = fit_linear_marginal
        self.select_linear_marginal = select_linear_marginal
        self.decay_rate_towards_marginal = decay_rate_towards_marginal
        self.fit_posthoc_tree_coefs = fit_posthoc_tree_coefs
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

        self.estimators_marginal = []
        self.estimators_ = []
        self.bias_ = np.mean(y)

        if self.n_boosting_rounds_marginal > 0:
            self._marginal_fit(
                X_train,
                y_train,
                sample_weight_train,
            )

        if self.n_boosting_rounds > 0:
            self._cyclic_boost(
                X_train,
                y_train,
                sample_weight_train,
                X_val,
                y_val,
                sample_weight_val,
            )

        if self.fit_posthoc_tree_coefs is not None:
            self._fit_posthoc_tree_coefs(X_train, y_train, sample_weight_train)

        self.mse_val_ = self._calc_mse(X_val, y_val, sample_weight_val)

        return self

    def _marginal_fit(
        self,
        X_train,
        y_train,
        sample_weight_train,
    ):
        """Fit a gbdt estimator for each feature independently.
        Store in self.estimators_marginal"""
        residuals_train = y_train - self.predict_proba(X_train)[:, 1]
        p = X_train.shape[1]
        for feature_num in range(p):
            X_ = np.zeros_like(X_train)
            X_[:, feature_num] = X_train[:, feature_num]
            est = GradientBoostingRegressor(
                max_leaf_nodes=self.max_leaf_nodes_marginal,
                random_state=self.random_state,
                n_estimators=self.n_boosting_rounds_marginal,
            )
            est.fit(X_, residuals_train, sample_weight=sample_weight_train)
            if self.reg_param_marginal > 0:
                est = imodels.HSTreeRegressor(
                    est, reg_param=self.reg_param_marginal)
            self.estimators_marginal.append(est)

        if (
            self.fit_linear_marginal is not None
            and not self.fit_linear_marginal == "None"
        ):
            if self.fit_linear_marginal.lower() == "ridge":
                linear_marginal = RidgeCV(fit_intercept=False)
            elif self.fit_linear_marginal == "NNLS":
                linear_marginal = LinearRegression(
                    fit_intercept=False, positive=True)
            linear_marginal.fit(
                np.array([est.predict(X_train)
                         for est in self.estimators_marginal]).T,
                residuals_train,
                sample_weight_train,
            )
            self.marginal_coef_ = linear_marginal.coef_
            self.lin = linear_marginal

        else:
            self.marginal_coef_ = np.ones(p) / p

    def _cyclic_boost(
        self, X_train, y_train, sample_weight_train, X_val, y_val, sample_weight_val
    ):
        """Apply cyclic boosting, storing trees in self.estimators_"""

        residuals_train = y_train - self.predict_proba(X_train)[:, 1]
        mse_val = self._calc_mse(X_val, y_val, sample_weight_val)
        self.decay_coef_towards_marginal_ = []
        for _ in range(self.n_boosting_rounds):
            boosting_round_ests = []
            boosting_round_mses = []
            feature_nums = np.arange(X_train.shape[1])
            if self.select_linear_marginal:
                assert (
                    self.fit_linear_marginal == "NNLS"
                    and self.n_boosting_rounds_marginal > 0
                ), "select_linear_marginal requires fit_linear_marginal to be NNLS and for n_boosting_rounds_marginal > 0"
                feature_nums = np.where(self.marginal_coef_ > 0)[0]
            for feature_num in feature_nums:
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
                if self.reg_param > 0:
                    est = imodels.HSTreeRegressor(
                        est, reg_param=self.reg_param)
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

            # decay marginal effects
            if self.decay_rate_towards_marginal < 1.0:
                new_decay_coefs = [self.decay_rate_towards_marginal] * (
                    len(self.estimators_) -
                    len(self.decay_coef_towards_marginal_)
                )
                # print(self.decay_coef_towards_marginal_)
                # print('new_decay_coefs', new_decay_coefs)
                self.decay_coef_towards_marginal_ = [
                    x * self.decay_rate_towards_marginal
                    for x in self.decay_coef_towards_marginal_
                ] + new_decay_coefs
                # print(self.decay_coef_towards_marginal_)

            # early stopping if validation error does not decrease
            mse_val_new = self._calc_mse(X_val, y_val, sample_weight_val)
            if mse_val_new >= mse_val:
                # print("early stop!")
                return
            else:
                mse_val = mse_val_new

    def _fit_posthoc_tree_coefs(self, X, y, sample_weight=None):
        # extract predictions from each tree
        X_pred_tree = np.array([est.predict(X) for est in self.estimators_]).T
        print('shapes', X.shape, X_pred_tree.shape,
              y.shape, len(self.estimators_))

        coef_prior = np.ones(len(self.estimators_)) * self.learning_rate
        y = y - self.bias_ - X_pred_tree @ coef_prior

        if self.fit_posthoc_tree_coefs.lower() == "ridge":
            m = RidgeCV(fit_intercept=False)
        elif self.fit_posthoc_tree_coefs.lower() == "nnls":
            m = LinearRegression(fit_intercept=False, positive=True)
        elif self.fit_posthoc_tree_coefs.lower() == "elasticnet":
            m = ElasticNetCV(fit_intercept=False, positive=True)

        m.fit(X_pred_tree, y, sample_weight=sample_weight)
        self.cyclic_coef_ = m.coef_ + coef_prior

    def predict_proba(self, X, marginal_only=False):
        """
        Params
        ------
        marginal_only: bool
            If True, only use the marginal effects.
        """
        X = check_array(X, accept_sparse=False, dtype=None)
        check_is_fitted(self)
        probs1 = np.ones(X.shape[0]) * self.bias_

        # marginal prediction
        for i, est in enumerate(self.estimators_marginal):
            probs1 += est.predict(X) * self.marginal_coef_[i]

        # cyclic coefs prediction
        if not marginal_only:
            if not hasattr(self, "cyclic_coef_"):
                cyclic_coef_ = np.ones(
                    len(self.estimators_)) * self.learning_rate
            else:
                cyclic_coef_ = self.cyclic_coef_
                # print('coef', cyclic_coef_)

            if self.decay_rate_towards_marginal < 1.0:
                for i, est in enumerate(self.estimators_):
                    if i < len(self.decay_coef_towards_marginal_):
                        probs1 += (
                            cyclic_coef_[i]
                            * self.decay_coef_towards_marginal_[i]
                            * est.predict(X)
                        )
                    else:
                        probs1 += cyclic_coef_[i] * est.predict(X)
            else:
                for i, est in enumerate(self.estimators_):
                    probs1 += cyclic_coef_[i] * est.predict(X)
        probs1 = np.clip(probs1, a_min=0, a_max=1)
        return np.array([1 - probs1, probs1]).T

    def predict(self, X, marginal_only=False):
        if isinstance(self, RegressorMixin):
            return self.predict_proba(X, marginal_only=marginal_only)[:, 1]
        elif isinstance(self, ClassifierMixin):
            return np.argmax(self.predict_proba(X, marginal_only=marginal_only), axis=1)

    def _calc_mse(self, X, y, sample_weight=None):
        return np.average(
            np.square(y - self.predict_proba(X)[:, 1]),
            weights=sample_weight,
        )


class TreeGAMRegressor(TreeGAM, RegressorMixin):
    ...


class TreeGAMClassifier(TreeGAM, ClassifierMixin):
    ...


if __name__ == "__main__":
    X, y, feature_names = imodels.get_clean_dataset("heart")
    X, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    gam = TreeGAMClassifier(
        boosting_strategy="cyclic",
        random_state=42,
        learning_rate=0.1,
        max_leaf_nodes=3,
        # select_linear_marginal=True,
        # fit_linear_marginal="NNLS",
        # n_boosting_rounds_marginal=3,
        # decay_rate_towards_marginal=0,
        fit_posthoc_tree_coefs="elasticnet",
        n_boosting_rounds=100,
    )
    gam.fit(X, y_train)

    # check roc auc score
    y_pred = gam.predict_proba(X_test)[:, 1]
    # print(
    #     "train roc:",
    #     roc_auc_score(y_train, gam.predict_proba(X)[:, 1]).round(3),
    # )
    print("test roc:", roc_auc_score(y_test, y_pred).round(3))
    print("test acc:", accuracy_score(y_test, gam.predict(X_test)).round(3))
    print('\t(imb:', np.mean(y_test).round(3), ')')
    # print(
    #     "accs",
    #     accuracy_score(y_train, gam.predict(X)).round(3),
    #     accuracy_score(y_test, gam.predict(X_test)).round(3),
    #     "imb",
    #     np.mean(y_train).round(3),
    #     np.mean(y_test).round(3),
    # )

    # # print(gam.estimators_)
