from copy import deepcopy
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import RidgeCV, ElasticNet, ElasticNetCV
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_X_y, check_array, _check_sample_weight
from imodels.util.arguments import check_predict_X, set_feature_names_in
from sklearn.preprocessing import StandardScaler

from sklearn.base import RegressorMixin, ClassifierMixin
from sklearn.utils.validation import check_is_fitted


class MarginalShrinkageLinearModel(BaseEstimator):
    """Linear model that shrinks towards the marginal effects of each feature."""

    def __str__(self):
        return (
            repr(self)
            .replace("MarginalShrinkageLinearModel", "MSLM")
            .replace("Regressor", "Reg")
            .replace("Classifier", "Clf")
        )

    def __init__(
        self,
        est_marginal_name="ridge",
        est_main_name="ridge",
        marginal_divide_by_d=True,
        marginal_sign_constraint=False,
        alphas=np.logspace(-3, 5, num=9).tolist(),
        elasticnet_ratio=0.5,
        random_state=None,
    ):
        """
        Params
        ------
        est_marginal_name : str
            Name of estimator to use for marginal effects (marginal regression)
            If "None", then assume marginal effects are zero (standard Ridge)
        est_main_name : str
            Name of estimator to use for main effects
            If "None", then assume marginal effects are zero (standard Ridge)
            "ridge", "lasso", "elasticnet"
        marginal_divide_by_d : bool
            If True, then divide marginal effects by n_features
        marginal_sign_constraint : bool
            If True, then constrain main effects to be same sign as marginal effects
        alphas: Tuple[float]
            Alphas to try for regularized regression (only main, not marginal)
        elasticnet_ratio : float
            If using elasticnet, Ratio of l1 to l2 penalty for elastic net
        random_state : int
            Random seed
        """
        self.random_state = random_state
        self.est_marginal_name = est_marginal_name
        self.est_main_name = est_main_name
        self.marginal_divide_by_d = marginal_divide_by_d
        self.marginal_sign_constraint = marginal_sign_constraint
        self.elasticnet_ratio = elasticnet_ratio
        if alphas is None:
            alphas = np.logspace(-3, 5, num=9).tolist()
        elif isinstance(alphas, float) or isinstance(alphas, int):
            alphas = [alphas]
        self.alphas = alphas

    def fit(self, X, y, sample_weight=None):
        # checks
        set_feature_names_in(self, X)
        X, y = check_X_y(X, y, accept_sparse=False, multi_output=False)
        self.n_features_in_ = X.shape[1]
        sample_weight = _check_sample_weight(sample_weight, X, dtype=None)
        if isinstance(self, ClassifierMixin):
            check_classification_targets(y)
            self.classes_, y = np.unique(y, return_inverse=True)

        # preprocess X and y
        self.scalar_X_ = StandardScaler()
        X = self.scalar_X_.fit_transform(X)

        if isinstance(self, RegressorMixin):
            self.scalar_y_ = StandardScaler()
            y = self.scalar_y_.fit_transform(y.reshape(-1, 1)).squeeze()

        # fit marginal
        self.coef_marginal_ = self._fit_marginal(X, y, sample_weight)

        # fit main
        self.est_main_ = self._fit_main(
            X, y, sample_weight, self.coef_marginal_)

        return self

    def _fit_marginal(self, X, y, sample_weight):
        # initialize marginal estimator
        ALPHAS_MARGINAL = np.logspace(-1, 3, num=5).tolist()
        est_marginal = self._get_est_from_name(
            self.est_marginal_name,
            alphas=ALPHAS_MARGINAL,
            marginal_sign_constraint=False,
        )

        # fit marginal estimator to each feature
        if est_marginal is None:
            coef_marginal_ = np.zeros(X.shape[1])
        else:
            coef_marginal_ = []
            for i in range(X.shape[1]):
                est_marginal.fit(X[:, i].reshape(-1, 1), y,
                                 sample_weight=sample_weight)
                coef_marginal_.append(deepcopy(est_marginal.coef_))
            # squeeze() alone collapses a single feature's coefficient to a
            # scalar, which then cannot be multiplied with X
            coef_marginal_ = np.vstack(coef_marginal_).squeeze(axis=1)

        # evenly divide effects among features
        if self.marginal_divide_by_d:
            coef_marginal_ /= X.shape[1]

        return coef_marginal_

    def _fit_main(self, X, y, sample_weight, coef_marginal_):
        # constrain main effects to be same sign as marginal effects by flipping sign
        # of X appropriately and refitting with a non-negative least squares
        est_main_ = self._get_est_from_name(
            self.est_main_name,
            alphas=self.alphas,
            marginal_sign_constraint=self.marginal_sign_constraint,
        )

        if self.marginal_sign_constraint:
            assert self.est_marginal_name is not None, "must have marginal effects"
            coef_signs = np.sign(coef_marginal_)
            X = X * coef_signs
            est_main_.fit(X, y, sample_weight=sample_weight)
            est_main_.coef_ = est_main_.coef_ * coef_signs
            # check that signs do not disagree
            coef_final_signs = np.sign(est_main_.coef_)
            assert np.all(
                (coef_final_signs == coef_signs) | (coef_final_signs == 0)
            ), "signs should agree but" + str(np.sign(est_main_.coef_), coef_signs)
        elif est_main_ is None:
            # fit dummy clf and override coefs
            est_main_ = ElasticNetCV(fit_intercept=False)
            est_main_.fit(X[:5], y[:5])
            est_main_.coef_ = coef_marginal_
        else:
            # fit main estimator
            # predicting residuals is the same as setting a prior over coef_marginal
            # because we do solve change of variables ridge(prior = coef = coef - coef_marginal)
            preds_marginal = X @ coef_marginal_
            residuals = y - preds_marginal
            est_main_.fit(X, residuals, sample_weight=sample_weight)
            est_main_.coef_ = est_main_.coef_ + coef_marginal_
        return est_main_

    def _get_est_from_name(self, est_name, alphas, marginal_sign_constraint):
        L1_RATIOS = {
            "ridge": 1e-6,
            "lasso": 1,
            "elasticnet": self.elasticnet_ratio,
        }
        if est_name not in L1_RATIOS:
            return None
        else:
            if est_name == "ridge" and not marginal_sign_constraint:
                # this implementation is better than ElasticNetCV with l1_ratio close to 0
                return RidgeCV(
                    alphas=alphas,
                    fit_intercept=False,
                )
            return ElasticNetCV(
                l1_ratio=L1_RATIOS[est_name],
                alphas=alphas,
                max_iter=10000,
                fit_intercept=False,
                positive=bool(marginal_sign_constraint),
            )

    def predict_proba(self, X):
        check_is_fitted(self, 'est_main_')
        check_predict_X(self, X)
        X = self.scalar_X_.transform(X)
        return self.est_main_.predict_proba(X)

    def predict(self, X):
        check_is_fitted(self, 'est_main_')
        check_predict_X(self, X)
        X = self.scalar_X_.transform(X)
        pred = self.est_main_.predict(X)
        return self.scalar_y_.inverse_transform(pred.reshape(-1, 1)).squeeze()


class MarginalShrinkageLinearModelRegressor(
    MarginalShrinkageLinearModel, RegressorMixin
):
    ...


# class MarginalShrinkageLinearModelClassifier(
#     MarginalShrinkageLinearModel, ClassifierMixin
# ):
#     ...


class MarginalLinearModel(BaseEstimator):
    """Linear model that only fits marginal effects of each feature.
    """

    def __init__(self, alpha=1.0, l1_ratio=0.5, max_iter=1000, random_state=None):
        '''Arguments are passed to sklearn.linear_model.ElasticNet
        '''
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X, y, sample_weight=None):
        # checks
        X, y = check_X_y(X, y, accept_sparse=False, multi_output=False)
        sample_weight = _check_sample_weight(sample_weight, X, dtype=None)
        if isinstance(self, ClassifierMixin):
            check_classification_targets(y)
            self.classes_, y = np.unique(y, return_inverse=True)

        # fit marginal estimator to each feature
        coef_marginal_ = []
        for i in range(X.shape[1]):
            est_marginal = ElasticNet(alpha=self.alpha, l1_ratio=self.l1_ratio,
                                      max_iter=self.max_iter, random_state=self.random_state)
            est_marginal.fit(X[:, i].reshape(-1, 1), y,
                             sample_weight=sample_weight)
            coef_marginal_.append(deepcopy(est_marginal.coef_))
        coef_marginal_ = np.vstack(coef_marginal_).squeeze()

        self.coef_ = coef_marginal_ / X.shape[1]
        self.alpha_ = self.alpha

        return self

    def predict_proba(self, X):
        X = check_array(X, accept_sparse=False, dtype=None)
        return X @ self.coef_

    def predict(self, X):
        probs = self.predict_proba(X)
        if isinstance(self, ClassifierMixin):
            return np.argmax(probs, axis=1)
        else:
            return probs


class MarginalLinearRegressor(MarginalLinearModel, RegressorMixin):
    ...


class MarginalLinearClassifier(MarginalLinearModel, ClassifierMixin):
    ...


# if __name__ == '__main__':
#     X, y = imodels.get_clean_dataset('heart')
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, random_state=42, test_size=0.2)
#     m = MarginalLinearModelRegressor()

#     m.fit(X_train, y_train)
#     print(m.coef_)
#     print(m.predict(X_test))
#     print(m.score(X_test, y_test))
