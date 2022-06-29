import copy
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import rankdata, kendalltau
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV, LinearRegression, LassoLarsIC, LogisticRegressionCV, \
    TheilSenRegressor, QuantileRegressor, Lasso, Ridge,HuberRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble._forest import _generate_unsampled_indices
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import f_regression
import sklearn.metrics as metrics
import statsmodels.api as sm

from imodels.importance.representation import TreeTransformer
from imodels.importance.LassoICc import LassoLarsICc


class R2FExp:
    """
    Class to compute R2F feature importance values.


    :param estimator: scikit-learn estimator,
        default=RandomForestRegressor(n_estimators=100, min_samples_leaf=5, max_features=0.33)
        The scikit-learn tree or tree ensemble estimator object

    :param max_components_type: {"auto", "median_splits", "max_splits", "nsamples", "nstumps", "min_nsamples_nstumps",
        "min_fracnsamples_nstumps"} or int, default="auto"
        Method for choosing the max number of components for PCA transformer for each sub-representation corresponding
        to an original feature:
            - If "auto", then same settings are used as "min_fracnsamples_nstumps"
            - If "median_splits", then max_components is alpha * median number of splits on the original feature
              among trees in the estimator
            - If "max_splits", then max_components is alpha * maximum number of splits on the original feature among
              trees in the estimator
            - If "nsamples", then max_components is alpha * n_samples
            - If "nstumps", then max_components is alpha * n_stumps
            - If "min_nsamples_nstumps", then max_components is alpha * min(n_samples, n_stumps), where n_stumps is
              total number of local decision stumps splitting on that feature in the ensemble
            - If "min_fracnsamples_nstumps", then max_components is min(alpha * n_samples, n_stumps), where n_stumps is
              total number of local decision stumps splitting on that feature in the ensemble
            - If int, then max_components is the given integer

    :param alpha: float, default=0.5
        Parameter for adjusting the max number of components for PCA.

    :param normalize: bool, default=False
        Flag. If set to True, then divide the nonzero function values for each local decision stump by
        sqrt(n_samples in node) so that the vector of function values on the training set has unit norm. If False,
        then do not divide, so that the vector of function values on the training set has norm equal to n_samples
        in node.

    :param random_state: int, default=None
        Random seed for sample splitting

    :param criterion: {"aic", "bic", "cv"}, default="bic"
        Criterion used for lasso model selection

    :param refit: bool, default=True
        If True, refit OLS after doing lasso model selection to compute r2 values, if not, compute r2 values from the
        lasso model

    :param add_raw: bool, default=True
        If true, concatenate X_k with the learnt representation from PCA

    :param n_splits: int, default=10
        The number of splits to use to compute r2f values
    """

    def __init__(self, estimator=None, max_components_type="auto", alpha=0.5, normalize=False, random_state=None,
                 use_noise_variance=True, scorer=None, treelet=False,
                 criterion="bic", refit=True, add_raw=True, split_data=True, val_size=0.5, n_splits=10,
                 rank_by_p_val=False, pca=True, normalize_raw=False):

        if estimator is None:
            self.estimator = RandomForestRegressor(n_estimators=100, min_samples_leaf=5, max_features=0.33)
        else:
            self.estimator = estimator
        if scorer is None:
            self.scorer = LassoScorer(criterion=self.criterion, refit=self.refit)
        else:
            self.scorer = copy.deepcopy(scorer)

        self.max_components_type = max_components_type
        self.alpha = alpha
        self.normalize = normalize
        self.random_state = random_state
        self.criterion = criterion
        self.refit = refit
        self.add_raw = add_raw
        self.split_data = split_data
        self.rank_by_p_val = rank_by_p_val
        self.val_size = val_size
        self.n_splits = n_splits
        self.use_noise_variance = use_noise_variance
        self.pca = pca
        self.treelet = treelet
        self.normalize_raw = normalize_raw

    def get_importance_scores(self, X, y, sample_weight=None, diagnostics=False):
        """
        Compute R2F feature importance values.

        :param X: array-like of shape (n_samples, n_features)
            Covariate data matrix
        :param y: array-like of shape (n_samples,)
            Vector of responses
        :param sample_weight: array-like of shape (n_samples,) or None, default=None
            Sample weights to use in fitting the tree ensemble for feature learning
        :param diagnostics: bool
            If False, return only the r2f values. If True, also return the r2 values, number of stumps, and
            number of PCs chosen in lasso model selection for each feature and over each split
        :return:
            r2f_values: array-like of shape (n_features,)
                The computed r2f values
            r_squared: array-like of shape (n_splits, n_features)
                The r2 values (for each feature) over each run
            n_stumps: array-like of shape (n_splits, n_features)
                The number of stumps in the ensemble (splitting on each feature) over each run
            n_components_chosen: array-like of shape (n_splits, n_features)
                The number of PCs chosen (for each feature) over each run
        """
        n_features = X.shape[1]
        r_squared = np.zeros((self.n_splits, n_features))
        n_components_chosen = np.zeros((self.n_splits, n_features))
        n_stumps = np.zeros((self.n_splits, n_features))

        if sample_weight is None:
            sample_weight = np.ones_like(y)
        for i in range(self.n_splits):
            if self.random_state is not None:
                random_state = self.random_state + i
            else:
                random_state = None
            if self.split_data:
                X_train, X_val, y_train, y_val, sample_weight_train, sample_weight_val = \
                    train_test_split(X, y, sample_weight, test_size=self.val_size, random_state=random_state)
                tree_transformer = self._feature_learning_one_split(X_train, y_train)
                if self.rank_by_p_val == False:
                    r_squared[i, :], n_stumps[i, :], n_components_chosen[i, :] = \
                        self._model_selection_r2_one_split(tree_transformer, X_val, y_val)
                else:
                    r_squared[i, :], n_stumps[i, :], n_components_chosen[i, :] = \
                        self._model_selection_pval_one_split(tree_transformer, X_val, y_val)
            else:
                tree_transformer = self._feature_learning_one_split(X, y)
                r_squared[i, :], n_stumps[i, :], n_components_chosen[i, :] = \
                    self._model_selection_r2_one_split(tree_transformer, X, y)
        r2f_values = np.mean(r_squared, axis=0)
        if diagnostics:
            return r2f_values, r_squared, n_stumps, n_components_chosen
        else:
            return r2f_values

    def _feature_learning_one_split(self, X_train, y_train, sample_weight=None):
        """
        Step 1 and 2 of r2f: Fit the RF (or other tree ensemble) and learn feature representations from it,
        storing the information in the TreeTransformer class
        """
        if self.max_components_type == "auto":
            max_components_type = "min_fracnsamples_nstumps"
        else:
            max_components_type = self.max_components_type
        estimator = copy.deepcopy(self.estimator)
        estimator.fit(X_train, y_train, sample_weight=sample_weight)
        tree_transformer = TreeTransformer(estimator=estimator, max_components_type=max_components_type,
                                           add_raw=self.add_raw, treelet=self.treelet,
                                           alpha=self.alpha, normalize=self.normalize, pca=self.pca,
                                           normalize_raw=self.normalize_raw)
        tree_transformer.fit(X_train)
        return tree_transformer

    def _model_selection_r2_one_split(self, tree_transformer, X_val, y_val):
        """
        Step 3 of r2f: Do lasso model selection and compute r2 values
        """
        n_features = X_val.shape[1]
        r_squared = np.zeros(n_features)
        n_components_chosen = np.zeros(n_features)
        n_stumps = np.zeros(n_features)
        for k in range(n_features):
            X_transformed = tree_transformer.transform_one_feature(X_val, k)
            n_stumps[k] = len(tree_transformer.get_stumps_for_feature(k))
            if X_transformed is None:
                r_squared[k] = 0
                n_components_chosen[k] = 0
            else:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    self.scorer.fit(X_transformed, y_val)
                    n_components_chosen[k] = self.scorer.get_model_size()
                    r_squared[k] = self.scorer.get_score()

        return r_squared, n_stumps, n_components_chosen

    def _model_selection_pval_one_split(self, tree_transformer, X_val, y_val):
        """
        Step 3 of r2f: Do lasso model selection and rank by p-values
        """
        n_features = X_val.shape[1]
        r_squared = np.zeros(n_features)
        n_components_chosen = np.zeros(n_features)
        n_stumps = np.zeros(n_features)
        for k in range(n_features):
            X_transformed = tree_transformer.transform_one_feature(X_val, k)
            n_stumps[k] = len(tree_transformer.get_stumps_for_feature(k))
            if X_transformed is None:
                r_squared[k] = 0
                n_components_chosen[k] = 0
            else:
                y_val_centered = y_val - np.mean(y_val)
                if self.add_raw:
                    X_transformed = np.hstack([X_val[:, [k]] - np.mean(X_val[:, k]), X_transformed])
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    OLS_for_k = sm.OLS(y_val_centered, X_transformed).fit(cov_type="HC0")
                    if np.isnan(OLS_for_k.f_pvalue) == False:
                        r_squared[k] = 1.0 - OLS_for_k.f_pvalue
                        n_components_chosen[k] = X_transformed.shape[1]
                    else:
                        r_squared[k] = 0.0
                        n_components_chosen[k] = X_transformed.shape[1]
        return r_squared, n_stumps, n_components_chosen


class GeneralizedMDI:
    """
    Class to compute generalized MDI importance values.


    :param estimator: scikit-learn estimator,
        default=RandomForestRegressor(n_estimators=100, min_samples_leaf=5, max_features=0.33)
        The scikit-learn tree or tree ensemble estimator object

    :param max_components_type: {"auto", "median_splits", "max_splits", "nsamples", "nstumps", "min_nsamples_nstumps",
        "min_fracnsamples_nstumps"} or int, default="auto"
        Method for choosing the max number of components for PCA transformer for each sub-representation corresponding
        to an original feature:
            - If "auto", then same settings are used as "min_fracnsamples_nstumps"
            - If "median_splits", then max_components is alpha * median number of splits on the original feature
              among trees in the estimator
            - If "max_splits", then max_components is alpha * maximum number of splits on the original feature among
              trees in the estimator
            - If "nsamples", then max_components is alpha * n_samples
            - If "nstumps", then max_components is alpha * n_stumps
            - If "min_nsamples_nstumps", then max_components is alpha * min(n_samples, n_stumps), where n_stumps is
              total number of local decision stumps splitting on that feature in the ensemble
            - If "min_fracnsamples_nstumps", then max_components is min(alpha * n_samples, n_stumps), where n_stumps is
              total number of local decision stumps splitting on that feature in the ensemble
            - If int, then max_components is the given integer

    :param alpha: float, default=0.5
        Parameter for adjusting the max number of components for PCA.

    :param normalize: bool, default=False
        Flag. If set to True, then divide the nonzero function values for each local decision stump by
        sqrt(n_samples in node) so that the vector of function values on the training set has unit norm. If False,
        then do not divide, so that the vector of function values on the training set has norm equal to n_samples
        in node.

    :param random_state: int, default=None
        Random seed for sample splitting

    :param criterion: {"aic", "bic", "cv"}, default="bic"
        Criterion used for lasso model selection

    :param refit: bool, default=True
        If True, refit OLS after doing lasso model selection to compute r2 values, if not, compute r2 values from the
        lasso model

    :param add_raw: bool, default=True
        If true, concatenate X_k with the learnt representation from PCA

    :param n_splits: int, default=10
        The number of splits to use to compute r2f values
    """

    def __init__(self, estimator=None, scorer=None, normalize=False, add_raw=True, refit=True,
                 criterion="aic_c", random_state=None, normalize_raw=False):

        if estimator is None:
            self.estimator = RandomForestRegressor(n_estimators=100, min_samples_leaf=5, max_features=0.33,
                                                   random_state=random_state)
        else:
            self.estimator = copy.deepcopy(estimator)
        self.normalize = normalize
        self.add_raw = add_raw
        self.refit = refit
        self.criterion = criterion
        self.normalize_raw = normalize_raw
        if scorer is None:
            self.scorer = LassoScorer(criterion=self.criterion, refit=self.refit)
        else:
            self.scorer = copy.deepcopy(scorer)

    def get_importance_scores(self, X, y, sample_weight=None, diagnostics=False):
        """
        Compute R2F feature importance values.

        :param X: array-like of shape (n_samples, n_features)
            Covariate data matrix
        :param y: array-like of shape (n_samples,)
            Vector of responses
        :param sample_weight: array-like of shape (n_samples,) or None, default=None
            Sample weights to use in fitting the tree ensemble for feature learning
        :param diagnostics: bool
            If False, return only the r2f values. If True, also return the r2 values, number of stumps, and
            number of PCs chosen in lasso model selection for each feature and over each split
        :return:
            r2f_values: array-like of shape (n_features,)
                The computed r2f values
            r_squared: array-like of shape (n_splits, n_features)
                The r2 values (for each feature) over each run
            n_stumps: array-like of shape (n_splits, n_features)
                The number of stumps in the ensemble (splitting on each feature) over each run
            n_components_chosen: array-like of shape (n_splits, n_features)
                The number of PCs chosen (for each feature) over each run
        """
        n_samples, n_features = X.shape
        n_trees = self.estimator.n_estimators
        scores = np.zeros((n_trees, n_features))
        n_stumps = np.zeros((n_trees, n_features))
        n_stumps_chosen = np.zeros((n_trees, n_features))
        self.estimator.fit(X, y, sample_weight)

        for idx, estimator in enumerate(self.estimator.estimators_):
            tree_transformer = TreeTransformer(estimator=estimator, pca=False, add_raw=self.add_raw,
                                               normalize_raw=self.normalize_raw)
            oob_indices = _generate_unsampled_indices(estimator.random_state, n_samples, n_samples)
            X_oob = X[oob_indices, :]
            y_oob = y[oob_indices]
            if sample_weight is not None:
                sample_weight_oob = sample_weight[oob_indices]
            else:
                sample_weight_oob = None
            for k in range(n_features):
                X_transformed_oob = tree_transformer.transform_one_feature(X_oob, k)
                if X_transformed_oob is None:
                    n_stumps[idx, k] = 0
                    n_stumps_chosen[idx, k] = 0
                    scores[idx, k] = 0
                else:
                    n_stumps[idx, k] = X_transformed_oob.shape[1]
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore")
                        self.scorer.fit(X_transformed_oob, y_oob, sample_weight_oob)
                    n_stumps_chosen[idx, k] = self.scorer.get_model_size()
                    scores[idx, k] = self.scorer.get_score()
        imp_values = scores.mean(axis=0)

        if diagnostics:
            return imp_values, scores, n_stumps, n_stumps_chosen
        else:
            return imp_values


class ScorerBase(ABC):
    """
    ABC for scoring an original feature based on a transformed representation
    """

    def __init__(self, metric):
        self.selected_features = None
        self.score = 0
        if metric is None:
            self.metric = metrics.r2_score
        else:
            self.metric = metric

    @abstractmethod
    def fit(self, X, y):
        """
        Method that sets self.score and self.selected_features
        """
        pass

    def get_selected_features(self):
        return self.selected_features

    def get_model_size(self):
        return len(self.selected_features)

    def get_score(self):
        return self.score


class LassoScorer(ScorerBase, ABC):

    def __init__(self, metric=None, criterion="bic", refit=True):
        super().__init__(metric)
        if criterion == "cv" or criterion == "cv_1se":
            self.lasso_model = LassoCV(normalize=False, fit_intercept=True)
        else:
            self.lasso_model = LassoLarsICc(criterion=criterion, normalize=False, fit_intercept=True)
        self.criterion = criterion
        self.refit = refit

    def fit(self, X, y):
        if self.criterion == "cv_1se":
            # get grid of alphas
            lasso_tmp = LassoCV(normalize=False, fit_intercept=True)
            lasso_tmp.fit(X, y)
            alphas = lasso_tmp.alphas_
            # fit lasso with cv 1se rule
            lasso = Lasso(fit_intercept=True)
            self.lasso_model = GridSearchCV(lasso, [{"alpha": alphas}], refit=cv_one_se_rule)
            self.lasso_model.fit(X, y)
            self.selected_features = np.nonzero(self.lasso_model.best_estimator_.coef_)[0]
        else:
            self.lasso_model.fit(X, y)
            self.selected_features = np.nonzero(self.lasso_model.coef_)[0]
        if self.refit and self.get_model_size() > 0:
            X_sel = X[:, self.selected_features]
            lr = LinearRegression().fit(X_sel, y)
            y_pred = lr.predict(X_sel)
        else:
            y_pred = self.lasso_model.predict(X)

        self.score = self.metric(y, y_pred)


class RidgeScorer(ScorerBase, ABC):

    def __init__(self, metric=None, criterion="gcv", alphas=np.logspace(-4, 3, 100)):
        super().__init__(metric)
        assert criterion in ["gcv", "gcv_1se", "cv_1se"]
        self.criterion = criterion
        self.alphas = alphas

    def fit(self, X, y, sample_weight=None):
        if self.criterion == "cv_1se":
            alphas = self.alphas
            ridge = Ridge(normalize=False, fit_intercept=True)
            ridge_model = GridSearchCV(ridge, [{"alpha": alphas}], refit=cv_one_se_rule)
            ridge_model.fit(X, y, sample_weight=sample_weight)
            self.selected_features = np.nonzero(ridge_model.best_estimator_.coef_)[0]
        elif self.criterion == "gcv_1se":
            alphas = self.alphas
            ridge_model = RidgeCV(alphas=alphas, normalize=False, fit_intercept=True, store_cv_values=True)
            ridge_model.fit(X, y, sample_weight=sample_weight)
            cv_mean = np.mean(ridge_model.cv_values_, axis=0)
            cv_std = np.std(ridge_model.cv_values_, axis=0)
            best_alpha_index = one_se_rule(alphas, cv_mean, cv_std, X.shape[0], "min")
            best_alpha = alphas[best_alpha_index]
            ridge_model = Ridge(alpha=best_alpha, fit_intercept=True)
            ridge_model.fit(X, y, sample_weight=sample_weight)
            self.selected_features = np.nonzero(ridge_model.coef_)[0]
        elif self.criterion == "gcv":
            alphas = self.alphas
            ridge_model = RidgeCV(alphas=alphas, normalize=False, fit_intercept=True,store_cv_values = True).fit(X, y,
                                                                                          sample_weight=sample_weight)
            self.selected_features = np.nonzero(ridge_model.coef_)[0]
        y_pred = ridge_model.predict(X)
        if self.metric == "gcv":
            best_alpha_index = np.where(ridge_model.alphas == ridge_model.alpha_)[0][0]
            LOO_error = np.sum(ridge_model.cv_values_[:,best_alpha_index])/len(y)
            R2 = 1.0 - (LOO_error/np.var(y))
            self.score = R2
        else:
            self.score = self.metric(y, y_pred)


class RobustScorer(ScorerBase, ABC):

    def __init__(self, metric=None, strategy="huber"):
        super().__init__(metric)
        if strategy == "huber":
            self.robust_model = HuberRegressor(epsilon = 1.0)
        elif strategy == "theilsen":
            self.robust_model = TheilSenRegressor()
        elif strategy == "median":
            self.robust_model = QuantileRegressor()
        else:
            raise ValueError("Not a valid robust regression strategy")

    def fit(self, X, y,sample_weight = None):
        self.robust_model.fit(X, y)
        self.selected_features = np.nonzero(self.robust_model.coef_)[0]
        y_pred = self.robust_model.predict(X)
        self.score = self.metric(y, y_pred)


class ElasticNetScorer(ScorerBase, ABC):

    def __init__(self, metric=None, refit=True):
        super().__init__(metric)
        self.elasticnet_model = ElasticNetCV(normalize=False, fit_intercept=True)
        self.refit = refit

    def fit(self, X, y):
        self.elasticnet_model.fit(X, y)
        self.selected_features = np.nonzero(self.elasticnet_model.coef_)[0]
        if self.refit and self.get_model_size() > 0:
            X_sel = X[:, self.selected_features]
            lr = LinearRegression().fit(X_sel, y)
            y_pred = lr.predict(X_sel)
        else:
            y_pred = self.elasticnet_model.predict(X)

        self.score = self.metric(y, y_pred)


class LogisticScorer(ScorerBase, ABC):

    def __init__(self, metric=None, penalty="l2"):
        self.penalty = penalty
        super().__init__(metric)

    def fit(self, X, y, sample_weight=None):
        clf = LogisticRegressionCV(fit_intercept=True).fit(X, y, sample_weight)
        self.selected_features = np.nonzero(clf.coef_)[0]
        y_pred = clf.predict_proba(X)[:, 1]
        self.score = self.metric(y, y_pred, sample_weight=sample_weight)


class JointScorerBase(ABC):

    def __init__(self, metric):
        self.scores = defaultdict(lambda: None)
        self.n_stumps = defaultdict(lambda: None)
        self.model_sizes = defaultdict(lambda: None)
        if metric is None:
            self.metric = metrics.r2_score
        else:
            self.metric = metric

    @abstractmethod
    def fit(self, X, y):
        """
        Method that sets self.score and self.selected_features
        """
        pass

    def get_score(self, k):
        return self.scores[k]

    def get_model_size(self, k):
        return self.model_sizes[k]

    def get_n_stumps(self, k):
        return self.n_stumps[k]


class JointRidgeScorer(JointScorerBase, ABC):

    def __init__(self, metric=None, criterion="gcv", alphas=np.logspace(-4, 3, 100), split_sample=False):
        super().__init__(metric)
        assert criterion in ["gcv", "gcv_1se", "cv_1se"]
        self.criterion = criterion
        self.alphas = alphas
        self.split_sample = split_sample

    def fit(self, X, y, start_indices, sample_weight):

        if self.criterion == "cv_1se":
            ridge = Ridge(normalize=False, fit_intercept=True)
            ridge_model = GridSearchCV(ridge, [{"alpha": self.alphas}], refit=cv_one_se_rule)
        elif self.criterion == "gcv_1se":
            ridge_model = RidgeCV(alphas=self.alphas, normalize=False, fit_intercept=True, store_cv_values=True)
            ridge_model.fit(X, y, sample_weight=sample_weight)
            cv_mean = np.mean(ridge_model.cv_values_, axis=0)
            cv_std = np.std(ridge_model.cv_values_, axis=0)
            best_alpha_index = one_se_rule(self.alphas, cv_mean, cv_std, X.shape[0], "min")
            best_alpha = self.alphas[best_alpha_index]
            ridge_model = Ridge(alpha=best_alpha, fit_intercept=True)
        elif self.criterion == "gcv":
            ridge_model = RidgeCV(alphas=self.alphas, normalize=False, fit_intercept=True,store_cv_values = True)
        else:
            raise ValueError("Invalid criterion type")
        if self.split_sample:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
        else:
            X_train = X
            X_test = X
            y_train = y
            y_test = y
        ridge_model.fit(X_train, y_train, sample_weight=sample_weight)
        if self.metric == "loocv":

            def _get_partial_model_looe(X, y, start_indices, alpha, beta):
                B = np.linalg.inv(X.T @ X + alpha * np.eye(X.shape[1])) @ X.T
                h_vals = np.diag(X @ B)
                y_preds = X @ beta
                n_feats = len(start_indices) - 1
                n_samples = X.shape[0]
                looe_vals = np.zeros((n_samples, n_feats))
                for k in range(len(start_indices) - 1):
                    X_partial = X[:, start_indices[k]:start_indices[k + 1]]
                    beta_partial = beta[start_indices[k]:start_indices[k + 1]]
                    B_partial = B[start_indices[k]:start_indices[k + 1], :]
                    if X_partial.shape[1] > 0:
                        y_preds_partial = X_partial @ beta_partial
                        h_vals_partial = np.diag(X_partial @ B_partial)
                        looe_vals[:, k] = ((1 - h_vals + h_vals_partial) * (y_preds_partial - y) + h_vals_partial *
                                           (y_preds - y_preds_partial)) / (1 - h_vals)
                    else:
                        looe_vals[:, k] = y
                return looe_vals

            looe = _get_partial_model_looe(X_test, y_test, start_indices, ridge_model.alpha_, ridge_model.coef_)
            y_norm_sq = np.linalg.norm(y) ** 2
            for k in range(len(start_indices) - 1):
                self.scores[k] = 1 - np.sum(looe[:, k] ** 2) / y_norm_sq
            else:
                self.scores[k] = 0
        else:
            for k in range(len(start_indices) - 1):
                restricted_feats = X_test[:, start_indices[k]:start_indices[k + 1]]
                restricted_coefs = ridge_model.coef_[start_indices[k]:start_indices[k + 1]]
                self.n_stumps[k] = start_indices[k + 1] - start_indices[k]
                self.model_sizes[k] = int(np.sum(restricted_coefs != 0))
                if len(restricted_coefs) > 0:
                    restricted_preds = restricted_feats @ restricted_coefs + ridge_model.intercept_
                    self.scores[k] = self.metric(y_test, restricted_preds)
                else:
                    self.scores[k] = 0


class JointLogisticScorer(JointScorerBase, ABC):

    def __init__(self, metric=None, penalty="l2"):
        self.penalty = penalty
        super().__init__(metric)

    def fit(self, X, y, start_indices, sample_weight=None):
        clf = LogisticRegressionCV(fit_intercept=True).fit(X, y, sample_weight)
        for k in range(len(start_indices) - 1):
            restricted_feats = X[:, start_indices[k]:start_indices[k + 1]]
            restricted_coefs = clf.coef_[0,start_indices[k]:start_indices[k + 1]]
            self.n_stumps[k] = start_indices[k + 1] - start_indices[k]
            self.model_sizes[k] = int(np.sum(restricted_coefs != 0))
            if len(restricted_coefs) > 0:
                restricted_preds = expit(restricted_feats @ restricted_coefs + clf.intercept_)
                self.scores[k] = self.metric(y, restricted_preds, sample_weight=sample_weight)
            else:
                self.scores[k] = 0

class JointLassoScorer(JointScorerBase,ABC):
    
    def __init__(self, metric=None):
        super().__init__(metric)
    
    def fit(self, X, y, start_indices, sample_weight):
        lasso_model = LassoCV(fit_intercept = True).fit(X,y,sample_weight)
        for k in range (len(start_indices) - 1):
            restricted_feats = X[:, start_indices[k]:start_indices[k + 1]]
            restricted_coefs = lasso_model.coef_[start_indices[k]:start_indices[k + 1]]
            self.n_stumps[k] = start_indices[k + 1] - start_indices[k]
            self.model_sizes[k] = int(np.sum(restricted_coefs != 0))
            if len(restricted_coefs) > 0:
                restricted_preds = restricted_feats @ restricted_coefs + lasso_model.intercept_
                self.scores[k] = self.metric(y, restricted_preds, sample_weight=sample_weight)
            else:
                self.scores[k] = 0


class JointRobustScorer(JointScorerBase, ABC):

    def __init__(self, metric=None, strategy="huber"):
        super().__init__(metric)
        if strategy == "huber":
            self.robust_model = HuberRegressor(epsilon=1.0)
        elif strategy == "theilsen":
            self.robust_model = TheilSenRegressor()
        elif strategy == "median":
            self.robust_model = QuantileRegressor()
        else:
            raise ValueError("Not a valid robust regression strategy")

    def fit(self, X, y, start_indices,sample_weight = None):
        self.robust_model.fit(X, y)
        for k in range(len(start_indices) - 1):
            restricted_feats = X[:, start_indices[k]:start_indices[k + 1]]
            restricted_coefs = self.robust_model.coef_[start_indices[k]:start_indices[k + 1]]
            self.n_stumps[k] = start_indices[k + 1] - start_indices[k]
            self.model_sizes[k] = int(np.sum(restricted_coefs != 0))
            if len(restricted_coefs) > 0:
                restricted_preds = restricted_feats @ restricted_coefs + self.robust_model.intercept_
                self.scores[k] = self.metric(y, restricted_preds)
            else:
                self.scores[k] = 0


class GeneralizedMDIJoint:

    def __init__(self, estimator=None, scorer=None, normalize=False, add_raw=True, random_state=None,
                 normalize_raw=False):

        if estimator is None:
            self.estimator = RandomForestRegressor(n_estimators=100, min_samples_leaf=5, max_features=0.33,
                                                   random_state=random_state)
        else:
            self.estimator = copy.deepcopy(estimator)
        self.normalize = normalize
        self.add_raw = add_raw
        self.normalize_raw = normalize_raw
        if scorer is None:
            self.scorer = JointRidgeScorer()
        else:
            self.scorer = copy.deepcopy(scorer)

    def get_importance_scores(self, X, y, sample_weight=None, diagnostics=False):

        n_samples, n_features = X.shape
        n_trees = self.estimator.n_estimators
        scores = np.zeros((n_trees, n_features))
        n_stumps = np.zeros((n_trees, n_features))
        n_stumps_chosen = np.zeros((n_trees, n_features))
        self.estimator.fit(X, y, sample_weight)

        for idx, estimator in enumerate(self.estimator.estimators_):
            tree_transformer = TreeTransformer(estimator=estimator, pca=False, add_raw=self.add_raw,
                                               normalize_raw=self.normalize_raw)
            oob_indices = _generate_unsampled_indices(estimator.random_state, n_samples, n_samples)
            X_oob = X[oob_indices, :]
            y_oob = y[oob_indices] - np.mean(y[oob_indices])
            if sample_weight is not None:
                sample_weight_oob = sample_weight[oob_indices]
            else:
                sample_weight_oob = None
            X_transformed_oob, start_indices = tree_transformer.transform(X_oob, center=True, return_indices=True)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                self.scorer.fit(X_transformed_oob, y_oob, start_indices, sample_weight_oob)
            for k in range(n_features):
                n_stumps[idx, k] = self.scorer.get_n_stumps(k)
                n_stumps_chosen[idx, k] = self.scorer.get_model_size(k)
                scores[idx, k] = self.scorer.get_score(k)
        imp_values = scores.mean(axis=0)

        if diagnostics:
            return imp_values, scores, n_stumps, n_stumps_chosen
        else:
            return imp_values

            # y_val_centered = y_val - np.mean(y_val)
            # if self.criterion == "cv":
            #    if self.linear_method == "lasso":
            #        lm = LassoCV(fit_intercept=False, normalize=False)
            #        lm.fit(X_transformed,y_val_centered)
            #        n_components_chosen[k] = np.count_nonzero(lm.coef_)
            #    elif self.linear_method == "ridge":
            #        lm = RidgeCV(fit_intercept=False, normalize=False, cv = None)
            #        lm.fit(X_transformed,y_val_centered)
            #        n_components_chosen[k] = np.count_nonzero(lm.coef_)
            #    else:
            #        lm = ElasticNetCV(fit_intercept=False,normalize=False)
            #        lm.fit(X_transformed,y_val_centered)
            #        n_components_chosen[k] = np.count_nonzero(lm.coef_)
            # elif self.criterion == "f_regression":
            #    f_stat, p_vals = f_regression(X_transformed, y_val_centered)
            #    chosen_components = []
            #    for i in range(len(p_vals)):
            #        if p_vals[i] <= 0.05:
            #            chosen_components.append(i)
            #    n_components_chosen[k] = len(chosen_components)
            # else:
            #    lm = LassoLarsICc(criterion=self.criterion, normalize=False, fit_intercept=False,use_noise_variance = self.use_noise_variance) #LassoLarsIC
            #    lm.fit(X_transformed, y_val_centered)
            #    n_components_chosen[k] = np.count_nonzero(lm.coef_)
            # if self.refit:
            #    if self.criterion == "f_regression":
            #        support = chosen_components
            #    else:
            #        support = np.nonzero(lm.coef_)[0]
            #    if len(support) == 0:
            #        r_squared[k] = 0.0
            #    else:
            #        lr = LinearRegression().fit(X_transformed[:, support], y_val_centered)
            #        r_squared[k] = lr.score(X_transformed[:, support], y_val_centered)
            # else:
            #    r_squared[k] = lm.score(X_transformed, y_val_centered)


def one_se_rule(alphas, mean_cve, std_cve, K, optimum="max"):
    """
    Select penalty parameter according to 1 SE rule.
    :param alphas: List of penalty parameters.
    :param mean_cve: Mean CV error for each alpha.
    :param std_cve: Standard deviation of CV error for each alpha.
    :param K: Number of folds in CV.
    :param optimum: Either "min" or "max", indicating the type of optimum.
    :return: Index of penalty parameter selected according to 1 SE rule.
    """
    assert optimum in ["min", "max"]
    if optimum == "min":
        mean_cve = -mean_cve
    mean_per_alpha = pd.Series(mean_cve, index=alphas)
    std_per_alpha = pd.Series(std_cve, index=alphas)
    sem_per_alpha = std_per_alpha / np.sqrt(K)

    max_score = mean_per_alpha.max()
    max_score_idx = mean_per_alpha.idxmax()
    sem = sem_per_alpha[max_score_idx]

    best_alpha = mean_per_alpha[mean_per_alpha >= max_score - sem].index.max()
    best_alpha_index = int(np.argwhere(alphas == best_alpha)[0])

    return best_alpha_index


def cv_one_se_rule(results):
    """
    CV wrapper to select penalty parameter according to 1 SE rule. Callable to be used in refit.
    :param results: results from GridSearchCV()
    :return: Index of penalty parameter selected according to 1 SE rule.
    """

    K = len([x for x in list(results.keys()) if x.startswith('split') and x.endswith('test_score')])
    alpha_range = results['param_alpha'].data
    return one_se_rule(alpha_range, results["mean_test_score"], results["std_test_score"], K)


def kendall_tau_metric(y_true, y_pred):
    y_true_ranks = rankdata(y_true)
    y_pred_ranks = rankdata(y_pred)
    kendall_tau_corr = kendalltau(y_true_ranks, y_pred_ranks)[0]
    if np.isnan(kendall_tau_corr):
        return 0
    else:
        return kendall_tau_corr


