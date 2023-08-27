from copy import deepcopy
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_X_y
from sklearn.utils.validation import _check_sample_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm

import imodels

from sklearn.base import RegressorMixin, ClassifierMixin


class MarginalShrinkageLinearModel(BaseEstimator):
    """Linear model that shrinks towards the marginal effects of each feature."""

    def __init__(
        self,
        random_state=None,
        est_marginal_name="ridge",
        est_main_name="ridge",
        marginal_only=False,
    ):
        """
        Params
        ------
        random_state : int
            Random seed
        est_marginal_name : str
            Name of estimator to use for marginal effects
        est_main_name : str
            Name of estimator to use for main effects
        marginal_only : bool
            If True, only fit marginal effects (marginal regression)
        """
        self.random_state = random_state
        self.est_marginal_name = est_marginal_name
        self.est_main_name = est_main_name
        self.marginal_only = marginal_only

    def fit(self, X, y, sample_weight=None):
        # checks
        X, y = check_X_y(X, y, accept_sparse=False, multi_output=False)
        sample_weight = _check_sample_weight(sample_weight, X, dtype=None)
        if isinstance(self, ClassifierMixin):
            check_classification_targets(y)
            self.classes_, y = np.unique(y, return_inverse=True)

        # assume X and y are centered
        assert np.allclose(np.mean(X, axis=0), 0), "X must be centered"
        assert np.allclose(np.mean(y), 0), "y must be centered"

        # initialize marginal estimator
        est_marginal = self._get_est_from_name(self.est_marginal_name)

        # fit marginal estimator to each feature
        self.coef_marginal_ = []
        for i in range(X.shape[1]):
            est_marginal.fit(X[:, i].reshape(-1, 1), y, sample_weight=sample_weight)
            self.coef_marginal_.append(deepcopy(est_marginal.coef_))
        self.coef_marginal_ = np.vstack(self.coef_marginal_).squeeze()

        # evenly divide effects among features
        self.coef_marginal_ /= X.shape[1]

        # fit main estimator (predicting residuals is the same as setting a prior over coef_marginal)
        preds_marginal = X @ self.coef_marginal_
        print("preds_marginal", preds_marginal)
        residuals = y - preds_marginal
        self.est_main_ = self._get_est_from_name(self.est_main_name)
        self.est_main_.fit(X, residuals, sample_weight=sample_weight)

        # add back coef after fitting residuals
        self.est_main_.coef_ = self.est_main_.coef_ + self.coef_marginal_

        if self.marginal_only:
            self.est_main_.coef_ = self.coef_marginal_

        return self

    def _get_est_from_name(self, est_name):
        return {
            "ridge": RidgeCV(
                fit_intercept=False,
            ),
            "ols": LinearRegression(
                fit_intercept=False,
            ),
        }[est_name]

    def predict_proba(self, X):
        return self.est_main_.predict_proba(X)

    def predict(self, X):
        return self.est_main_.predict(X)


class MarginalShrinkageLinearModelRegressor(
    MarginalShrinkageLinearModel, RegressorMixin
):
    ...


# class MarginalShrinkageLinearModelClassifier(
#     MarginalShrinkageLinearModel, ClassifierMixin
# ):
#     ...


if __name__ == "__main__":
    # X, y, feature_names = imodels.get_clean_dataset("heart")
    X, y, feature_names = imodels.get_clean_dataset(
        **imodels.util.data_util.DSET_KWARGS["california_housing"]
    )
    print("shapes", X.shape, y.shape, "nunique", np.unique(y).size)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    # center X and y
    X_train = X_train - np.mean(X_train, axis=0)
    X_test = X_test - np.mean(X_test, axis=0)
    y_train = y_train - np.mean(y_train)
    y_test = y_test - np.mean(y_test)

    coefs = []
    for m in [
        MarginalShrinkageLinearModelRegressor(random_state=42, marginal_only=True),
        MarginalShrinkageLinearModelRegressor(random_state=42),
        RidgeCV(fit_intercept=False),
    ]:
        print(m)
        m.fit(X_train, y_train)

        # check roc auc score
        if isinstance(m, ClassifierMixin):
            y_pred = m.predict_proba(X_test)[:, 1]
            print(
                "train roc:",
                roc_auc_score(y_train, m.predict_proba(X_train)[:, 1]).round(3),
            )
            print("test roc:", roc_auc_score(y_test, y_pred).round(3))
            print(
                "accs",
                accuracy_score(y_train, m.predict(X_train)).round(3),
                accuracy_score(y_test, m.predict(X_test)).round(3),
                "imb",
                np.mean(y_train).round(3),
                np.mean(y_test).round(3),
            )
        else:
            y_pred = m.predict(X_test)
            print(
                "train mse:",
                np.mean((y_train - m.predict(X_train)) ** 2).round(3),
            )
            print("test mse:", np.mean((y_test - y_pred) ** 2).round(3))
            print(
                "r2",
                m.score(X_train, y_train).round(3),
                m.score(X_test, y_test).round(3),
            )
        if isinstance(m, MarginalShrinkageLinearModelRegressor):
            coefs.append(deepcopy(m.est_main_.coef_))
        else:
            coefs.append(deepcopy(m.coef_))

    diffs = pd.DataFrame({str(i): coefs[i] for i in range(len(coefs))})
    diffs["diff 0 - 1"] = diffs["0"] - diffs["1"]
    diffs["diff 1 - 2"] = diffs["1"] - diffs["2"]
    print(diffs)
