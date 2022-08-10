import copy
import warnings
import scipy
from abc import ABC, abstractmethod
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import rankdata, kendalltau
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV, LinearRegression, ElasticNet,LassoLarsIC, LogisticRegressionCV, \
    TheilSenRegressor, QuantileRegressor, Lasso, Ridge,HuberRegressor, RidgeClassifier, RidgeClassifierCV,LogisticRegression,enet_path
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble._forest import _generate_unsampled_indices
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import f_regression
import sklearn.metrics as metrics
import statsmodels.api as sm
from tqdm import tqdm

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
        self.classes = None
        self.class_scores = defaultdict(lambda x: None)

    def fit(self, X, y, start_indices, sample_weight):
        if isinstance(y[0], str):
            mod = RidgeClassifier
            modCV = RidgeClassifierCV
            multi_class = len(np.unique(y)) > 2
        else:
            mod = Ridge
            modCV = RidgeCV
            multi_class = False

        if self.criterion == "cv_1se":
            ridge = mod(normalize=False, fit_intercept=True)
            ridge_model = GridSearchCV(ridge, [{"alpha": self.alphas}], refit=cv_one_se_rule)
        elif self.criterion == "gcv_1se":
            ridge_model = modCV(alphas=self.alphas, normalize=False, fit_intercept=True, store_cv_values=True)
            ridge_model.fit(X, y, sample_weight=sample_weight)
            cv_mean = np.mean(ridge_model.cv_values_, axis=0)
            cv_std = np.std(ridge_model.cv_values_, axis=0)
            best_alpha_index = one_se_rule(self.alphas, cv_mean, cv_std, X.shape[0], "min")
            best_alpha = self.alphas[best_alpha_index]
            ridge_model = mod(alpha=best_alpha, fit_intercept=True)
        elif self.criterion == "gcv":
            ridge_model = modCV(alphas=self.alphas, normalize=False, fit_intercept=True,store_cv_values = True)
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
        if multi_class:
            self.classes = ridge_model.classes_
            y_test_onehot = np.ones((len(y_test), len(self.classes)))
            y_onehot = np.ones((len(y), len(self.classes)))
            for class_idx, class_label in enumerate(self.classes):
                y_test_onehot[y_test != class_label, class_idx] = -1
                y_onehot[y != class_label, class_idx] = -1

        if self.metric == "loocv":

            def _get_partial_model_looe(X, y, start_indices, alpha, beta, intercept):
                X1 = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
                B = np.linalg.inv(X1.T @ X1 + alpha * np.diag([0] + [1] * X.shape[1])) @ X1.T
                h_vals = np.diag(X1 @ B)
                y_preds = X @ beta + intercept
                n_feats = len(start_indices) - 1
                n_samples = X.shape[0]
                looe_vals = np.zeros((n_samples, n_feats))
                for k in range(len(start_indices) - 1):
                    X_partial = X[:, start_indices[k]:start_indices[k + 1]]
                    X_partial1 = np.concatenate((np.ones((X_partial.shape[0], 1)), X_partial), axis=1)
                    beta_partial = beta[start_indices[k]:start_indices[k + 1]]
                    keep_idxs = [0] + [idx + 1 for idx in range(start_indices[k], start_indices[k + 1])]
                    B_partial = B[keep_idxs, :]
                    # B_partial = B[start_indices[k]:start_indices[k + 1], :]
                    if X_partial.shape[1] > 0:
                        y_preds_partial = X_partial @ beta_partial + intercept
                        h_vals_partial = np.diag(X_partial1 @ B_partial)
                        looe_vals[:, k] = ((1 - h_vals + h_vals_partial) * (y_preds_partial - y) + h_vals_partial *
                                           (y_preds - y_preds_partial)) / (1 - h_vals)
                    else:
                        looe_vals[:, k] = y - intercept
                return looe_vals

            def _get_partial_model_looe_multiclass(X, y_onehot, start_indices, alpha, beta, intercept):
                X1 = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
                B = np.linalg.inv(X1.T @ X1 + alpha * np.diag([0] + [1] * X.shape[1])) @ X1.T
                h_vals = np.diag(X1 @ B)
                y_preds = X @ np.transpose(beta) + intercept
                n_feats = len(start_indices) - 1
                n_samples = X.shape[0]
                looe_vals = np.zeros((n_samples, n_feats, y_onehot.shape[1]))
                for k in range(len(start_indices) - 1):
                    X_partial = X[:, start_indices[k]:start_indices[k + 1]]
                    X_partial1 = np.concatenate((np.ones((X_partial.shape[0], 1)), X_partial), axis=1)
                    beta_partial = beta[:, start_indices[k]:start_indices[k + 1]]
                    keep_idxs = [0] + [idx + 1 for idx in range(start_indices[k], start_indices[k + 1])]
                    B_partial = B[keep_idxs, :]
                    # B_partial = B[start_indices[k]:start_indices[k + 1], :]
                    if X_partial.shape[1] > 0:
                        y_preds_partial = X_partial @ np.transpose(beta_partial) + intercept
                        h_vals_partial = np.diag(X_partial1 @ B_partial)
                        for class_idx in range(y_onehot.shape[1]):
                            looe_vals[:, k, class_idx] = ((1 - h_vals + h_vals_partial) * (y_preds_partial[:, class_idx] - y_onehot[:, class_idx]) + h_vals_partial *
                                                          (y_preds[:, class_idx] - y_preds_partial[:, class_idx])) / (1 - h_vals)
                    else:
                        looe_vals[:, k, :] = y_onehot - intercept
                return looe_vals

            if multi_class:
                looe = _get_partial_model_looe_multiclass(X_test, y_test_onehot, start_indices, ridge_model.alpha_, ridge_model.coef_, ridge_model.intercept_)
                for k in range(len(start_indices) - 1):
                    R2 = 1 - np.sum(looe[:, k, :] ** 2, axis=0) / (np.var(y_test_onehot, axis=0) * y_test_onehot.shape[0])
                    self.scores[k] = np.sum(R2 * (y_test_onehot == 1).mean(axis=0))
                    self.class_scores[k] = dict(zip(self.classes, R2))
            else:
                looe = _get_partial_model_looe(X_test, y_test, start_indices, ridge_model.alpha_, ridge_model.coef_, ridge_model.intercept_)
                for k in range(len(start_indices) - 1):
                    self.scores[k] = 1 - np.sum(looe[:, k] ** 2) / (np.var(y_test) * len(y_test))
        else:
            for k in range(len(start_indices) - 1):
                if multi_class:
                    restricted_feats = X_test[:, start_indices[k]:start_indices[k + 1]]
                    restricted_coefs = ridge_model.coef_[:, start_indices[k]:start_indices[k + 1]]
                    self.n_stumps[k] = start_indices[k + 1] - start_indices[k]
                    self.model_sizes[k] = int(np.sum(np.sum(restricted_coefs != 0, axis=0) > 0))
                    if restricted_coefs.shape[1] > 0:
                        restricted_preds = restricted_feats @ np.transpose(restricted_coefs) + ridge_model.intercept_
                        metric_output = self.metric(y_test_onehot, restricted_preds)
                        self.scores[k] = np.sum(metric_output * (y_test_onehot == 1).mean(axis=0))
                        self.class_scores[k] = dict(zip(self.classes, metric_output))
                    else:
                        self.scores[k] = np.NaN
                        class_scores = np.zeros(len(self.classes))
                        class_scores[:] = np.NaN
                        self.class_scores[k] = dict(zip(self.classes, copy.deepcopy(class_scores)))
                else:
                    restricted_feats = X_test[:, start_indices[k]:start_indices[k + 1]]
                    restricted_coefs = ridge_model.coef_[start_indices[k]:start_indices[k + 1]]
                    self.n_stumps[k] = start_indices[k + 1] - start_indices[k]
                    self.model_sizes[k] = int(np.sum(restricted_coefs != 0))
                    if len(restricted_coefs) > 0:
                        restricted_preds = restricted_feats @ restricted_coefs + ridge_model.intercept_
                        self.scores[k] = self.metric(y_test, restricted_preds)
                    else:
                        self.scores[k] = np.NaN


class JointLogisticScorer(JointScorerBase, ABC):

    def __init__(self, metric=None, penalty="l2"):
        self.penalty = penalty
        self.classes = None
        self.class_scores = defaultdict(lambda x: None)
        super().__init__(metric)

    def fit(self, X, y, start_indices, sample_weight=None):
        clf = LogisticRegressionCV(fit_intercept=True,max_iter = 1000).fit(X, y, sample_weight)
        self.classes = clf.classes_
        for k in range(len(start_indices) - 1):
            restricted_feats = X[:, start_indices[k]:start_indices[k + 1]]
            restricted_coefs = clf.coef_[:, start_indices[k]:start_indices[k + 1]]
            self.n_stumps[k] = start_indices[k + 1] - start_indices[k]
            self.model_sizes[k] = int(np.sum(np.sum(restricted_coefs != 0, axis=0) > 0))
            if restricted_coefs.shape[1] > 0:
                restricted_preds = expit(restricted_feats @ np.transpose(restricted_coefs) + clf.intercept_)
                if isinstance(y[0], str):
                    y_onehot = np.ones((len(y), len(self.classes)))
                    for class_idx, class_label in enumerate(self.classes):
                        y_onehot[y != class_label, class_idx] = -1
                    metric_output = self.metric(y_onehot, restricted_preds, sample_weight=sample_weight)
                    self.scores[k] = np.sum(metric_output * (y_onehot == 1).mean(axis=0))
                    self.class_scores[k] = dict(zip(self.classes, metric_output))
                else:
                    self.scores[k] = self.metric(y, restricted_preds, sample_weight=sample_weight)
            else:
                self.scores[k] = np.NaN
                if isinstance(y[0], str):
                    class_scores = np.zeros(len(self.classes))
                    class_scores[:] = np.NaN
                    self.class_scores[k] = copy.deepcopy(class_scores)
                    
                    
        #for i,alpha in enumerate(self.alphas):
        #    for j,l1_ratio in enumerate(self.l1_ratios):
        #        alphas_enet, coefs_enet, _ = enet_path(X, y,l1_ratio=l1_ratio,fit_intercept = True)
                
                #lr_model = ElasticNet(fit_intercept=True,alpha = alpha,l1_ratio = l1_ratio).fit(X,y)
                #beta = np.append(lr_model.intercept_,lr_model.coef_)
                #ip = np.dot(X1,beta)
        #        support = np.nonzero(lr_model.coef_)[0]
        #        b = alpha - alpha*l1_ratio
        #        h_val = compute_leverage_scores(X,support,b)
        #        if h_val is np.NaN:
        #            lr_dict[lr_model] = np.inf
        #        else:
        #            lr_dict[lr_model] = compute_alo(y,ip,h_val)
        #print(lr_dict)s
                    
                    
class JointALOElasticNetScorer(JointScorerBase,ABC):
    def __init__(self,metric = None,alphas = np.logspace(-4,3,100),l1_ratios = [10**-10,0.5,1.0]):
        self.alphas = alphas
        self.l1_ratios = l1_ratios 
        super().__init__(metric)
        
    def fit(self,X,y,start_indices,sample_weight = None):
        def compute_loss_derivative(ip,y):
            return ip - y
        def compute_loss_second_derivative(ip):
            return 1
        def compute_leverage_scores(X,support,b):
            #support_complement = np.array(list(set(np.arange(X.shape[1])).difference(support))).astype(int)
            X_S1 = X[:,support]
            #X_S[:,support_complement] = 0.0
            #X_S1 = np.concatenate((np.ones((X_S.shape[0], 1)), X_S), axis=1)
            J = X_S1.T@X_S1 + b*np.diag([0] + [1] * (X_S1.shape[1]-1))
            try:
                J_inverse = np.linalg.inv(J)
                H = X_S1@J_inverse@X_S1.T
                return np.diag(H)
            except:
                return np.NaN
        def compute_alo(y,ip,h_val):
            leverage_score_ratio = h_val/(1.0-h_val)
            loo_linear_preds = ip + (ip-y)*leverage_score_ratio
            residuals = (y - loo_linear_preds)**2
            return np.sum(residuals)
        def compute_partial_preds(opt_support,opt_b,opt_loss_derivative,opt_h_val,opt_lr_model):
            #opt_support_complement = np.array(list(set(np.arange(X.shape[1])).difference(opt_support))).astype(int)
            X_S = X[:,opt_support]
            #X_S[:,opt_support_complement] = 0.0
            X1_S = np.concatenate((np.ones((X.shape[0], 1)), X_S), axis=1)
            J_opt =  X1_S.T@X1_S + opt_b*np.diag([0] + [1] * X_S.shape[1])
            J_opt_inverse = np.linalg.inv(J_opt)
            n_samples = X.shape[0]
            n_feats = len(start_indices) - 1
            looe_preds = np.zeros((n_samples, n_feats))
            for k in range(len(start_indices) - 1):
                partial_model_support = np.array(list(set(np.arange(start_indices[k],start_indices[k+1])).intersection(opt_support))).astype(int)
                #print(k,start_indices[k],start_indices[k+1],opt_support,np.arange(start_indices[k],start_indices[k+1]),partial_model_support)
                if len(partial_model_support) > 0: 
                    X_partial =  X[:, partial_model_support]
                    X_partial1 = np.concatenate((np.ones((X_partial.shape[0], 1)), X_partial), axis=1)
                    beta_partial = np.append(opt_lr_model.intercept_,opt_lr_model.coef_[partial_model_support])
                    partial_ips = np.dot(X_partial1,beta_partial)
                    partial_model_support_idxs = [0] + [opt_support.tolist().index(idx) + 1 for idx in partial_model_support]
                    J_opt_inverse_partial = J_opt_inverse[partial_model_support_idxs,:]
                #X1S_partial =  np.concatenate((np.ones((XS_partial.shape[0], 1)), XS_partial), axis=1)
                    h_vals_partial = (X_partial1.dot(J_opt_inverse_partial) * X1_S).sum(-1)#np.diag(X_partial1@J_opt_inverse_partial@X1_S.T)#(
                #looe_vals[:, k] = ((1 - opt_h_val + h_vals_partial) * (y_preds_partial - y) + h_vals_partial *
                #                           (y_preds - y_preds_partial)) / (1 - h_vals)
                    looe_preds[:,k] = partial_ips + ((h_vals_partial)/(1.0-opt_h_val))*opt_loss_derivative
                else:
                     looe_preds[:,k] = y - opt_lr_model.intercept_
            return looe_preds
            
        lr_dict = {}
        X1 = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        for i,l1_ratio in enumerate(self.l1_ratios):
            if l1_ratio > 0.0:
                alphas_enet, coefs_enet, _ = enet_path(X1, y,l1_ratio=l1_ratio)
            else:
                alphas_enet, coefs_enet, _ = enet_path(X1, y,l1_ratio=l1_ratio,alphas = self.alphas)
            for j,alpha in enumerate(alphas_enet):
                b = alpha - alpha*l1_ratio
                support = np.nonzero(coefs_enet[:,j])[0]
                h_val = compute_leverage_scores(X1,support,b)
                if h_val is np.NaN:
                    lr_dict[(l1_ratio,alpha)] = np.inf
                else:
                    lr_dict[(l1_ratio,alpha)] = compute_alo(y,np.dot(X1,coefs_enet[:,j]),h_val)
        #opt_lr_model = min(lr_dict, key=lr_dict.get)
        #opt_l1_ratio = opt_lr_model.l1_ratio
        #opt_alpha = opt_lr_model.alpha
        #print(opt_l1_ratio,opt_alpha,X.shape)
        #en_test = ElasticNetCV(l1_ratio = self.l1_ratios,alphas = self.alphas,fit_intercept = True).fit(X,y)
        #print(en_test.l1_ratio_,en_test.alpha_)
        opt_l1_ratio,opt_alpha = min(lr_dict, key=lr_dict.get)
        opt_lr_model = ElasticNet(fit_intercept = True,l1_ratio = opt_l1_ratio,alpha = opt_alpha).fit(X,y)
        opt_b = opt_alpha - opt_alpha*opt_l1_ratio
        opt_beta_1 = np.append(opt_lr_model.intercept_,opt_lr_model.coef_)
        opt_loss_derivative = np.dot(X1,opt_beta_1) - y
        opt_support = np.nonzero(opt_lr_model.coef_)[0]
        opt_support_complement_test = np.array(list(set(np.arange(X.shape[1])).difference(opt_support))).astype(int)
        #print(opt_support_complement_test)
        opt_h_val = compute_leverage_scores(X,opt_support,opt_b)
        looe_preds = compute_partial_preds(opt_support = opt_support,opt_b = opt_b,opt_loss_derivative = opt_loss_derivative,opt_h_val = opt_h_val,opt_lr_model = opt_lr_model)
        for k in range(len(start_indices)-1):
            partial_model_support = np.array(list(set(np.arange(start_indices[k],start_indices[k+1])).intersection(opt_support))).astype(int)
            if len(partial_model_support) == 0:
                self.scores[k] = np.NaN
            else:
                try:
                    self.scores[k] = self.metric(y, looe_preds[:,k])#np.sum(looe[:,k])*-1 #log-likelihood     
                except:
                    print(looe_preds[:,k])



class JointALOLogisticScorer(JointScorerBase,ABC):
    def __init__(self, metric=None, penalty="l2",Cs =  np.logspace(-4,4,10)):
        self.penalty = penalty
        self.classes = None
        self.Cs = Cs
        super().__init__(metric)
    def fit(self, X, y, start_indices, sample_weight=None):
        def compute_loss_derivative(ip,y):
            return -y + (np.exp(ip)/(1 + np.exp(ip)))
        def compute_loss_second_derivative(ip):
            return (np.exp(ip))/(1 + np.exp(ip))**2
        def compute_leverage_scores(log_loss_second_derivative,X1,C):
            alpha = 1.0/C
            J = X1.T@np.diag(log_loss_second_derivative)@X1 + alpha*np.diag([0] + [1] * X.shape[1])
            J_inverse = np.linalg.inv(J)
            H = X1@J_inverse@X1.T@np.diag(log_loss_second_derivative)
            return np.diag(H)
        def compute_alo(y,ip,h_val,log_loss_derivative,log_loss_second_derivative):
            leverage_score_ratio = h_val/(1.0-h_val)
            derivative_ratio = log_loss_derivative/log_loss_second_derivative
            log_loss = (-y*ip) + (-y*derivative_ratio*leverage_score_ratio) + np.log(1.0 + np.exp(ip + leverage_score_ratio*derivative_ratio))
            return np.sum(log_loss)
        def compute_partial_preds(opt_h_val,opt_derivative,opt_second_derivative,opt_alpha,opt_lr):
            J_opt =  X1.T@np.diag(opt_second_derivative)@X1 + opt_alpha*np.diag([0] + [1] * X.shape[1])
            J_opt_inverse = np.linalg.inv(J_opt)
            n_samples = X.shape[0]
            n_feats = len(start_indices) - 1
            looe_preds = np.zeros((n_samples, n_feats))
            for k in range(len(start_indices) - 1):
                X_partial = X[:, start_indices[k]:start_indices[k + 1]]
                X_partial1 = np.concatenate((np.ones((X_partial.shape[0], 1)), X_partial), axis=1)
                beta_partial = np.append(opt_lr.intercept_,opt_lr.coef_[0][start_indices[k]:start_indices[k + 1]])
                partial_ips = np.dot(X_partial1,beta_partial)
                keep_idxs = [0] + [idx + 1 for idx in range(start_indices[k], start_indices[k + 1])]
                J_opt_inverse_partial = J_opt_inverse[keep_idxs, :]
                h_vals_partial = (X_partial1.dot(J_opt_inverse_partial) * X1).sum(-1)
                linear_partial_preds = partial_ips + ((h_vals_partial)/(1.0-opt_h_val))*opt_derivative
                looe_preds[:,k] = 1.0 / (1.0 + np.exp(-linear_partial_preds))
            return looe_preds

        lr_models = [LogisticRegression(fit_intercept=True,C = C,max_iter = 1000).fit(X,y) for C in self.Cs]
        betas = [np.append(lr.intercept_,lr.coef_[0]) for lr in lr_models]
        X1 = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        inner_products = [np.dot(X1,beta) for beta in betas]
        log_loss_derivative = [compute_loss_derivative(ip,y) for ip in inner_products]
        log_loss_second_derivative = [compute_loss_second_derivative(ip) for ip in inner_products]
        h_vals = [compute_leverage_scores(log_loss_second_derivative[i],X1,self.Cs[i]) for i in range(len(log_loss_second_derivative))]
        alos = [compute_alo(y,inner_products[i],h_vals[i],log_loss_derivative[i],log_loss_second_derivative[i]) for i in range(len(log_loss_second_derivative))]
        looe_preds = compute_partial_preds(opt_h_val = h_vals[np.argmin(alos)],opt_derivative = log_loss_derivative[np.argmin(alos)],
                                           opt_second_derivative = log_loss_second_derivative[np.argmin(alos)],
                                           opt_alpha = 1.0/self.Cs[np.argmin(alos)],opt_lr = lr_models[np.argmin(alos)])
        for k in range(len(start_indices)-1):
            if len(lr_models[np.argmin(alos)].coef_[0][start_indices[k]:start_indices[k + 1]]) == 0:
                self.scores[k] = np.NaN
            else:
                if self.metric == "log_loss":
                    self.scores[k] = metrics.log_loss(y,looe_preds[:,k])*-1.0
                else:
                    print(looe_preds[:,k])
                    self.scores[k] = self.metric(y, looe_preds[:,k])#np.sum(looe[:,k])*-1 #log-likelihood                                                                                                      


class JointLassoScorer(JointScorerBase,ABC):
    
    def __init__(self, metric=None,sample_split = False):
        super().__init__(metric)
        self.sample_split = sample_split
    
    def fit(self, X, y, start_indices, sample_weight):
        if self.sample_split:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
        else:
            X_train = X
            X_test = X
            y_train = y
            y_test = y
            
        lasso_model = LassoCV(fit_intercept = True).fit(X_train,y_train,sample_weight)
        for k in range (len(start_indices) - 1):
            restricted_feats = X_test[:, start_indices[k]:start_indices[k + 1]]
            restricted_coefs = lasso_model.coef_[start_indices[k]:start_indices[k + 1]]
            self.n_stumps[k] = start_indices[k + 1] - start_indices[k]
            self.model_sizes[k] = int(np.sum(restricted_coefs != 0))
            if len(restricted_coefs) > 0:
                restricted_preds = restricted_feats @ restricted_coefs + lasso_model.intercept_
                self.scores[k] = self.metric(y_test, restricted_preds, sample_weight=sample_weight)
            else:
                self.scores[k] = np.NaN


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
                self.scores[k] = np.NaN


class GeneralizedMDIJoint:

    def __init__(self, estimator=None, scorer=None, normalize=False, add_raw=True, random_state=None,
                 normalize_raw=False,oob = True):

        if estimator is None:
            self.estimator = RandomForestRegressor(n_estimators=100, min_samples_leaf=5, max_features=0.33,
                                                   random_state=random_state)
        else:
            self.estimator = copy.deepcopy(estimator)
        self.normalize = normalize
        self.add_raw = add_raw
        self.normalize_raw = normalize_raw
        self.oob = oob
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
        multi_class = isinstance(y[0], str)
        self.estimator.fit(X, y, sample_weight)

        if multi_class:
            class_scores_dict = defaultdict(lambda x: None)
            for tree_id in range(n_trees):
                class_scores_dict[tree_id] = defaultdict(lambda x: None)
                for feat_id in range(n_features):
                    class_scores_dict[tree_id][feat_id] = defaultdict(lambda x: None)

        for idx, estimator in tqdm(enumerate(self.estimator.estimators_)):
            tree_transformer = TreeTransformer(estimator=estimator, pca=False, add_raw=self.add_raw,
                                               normalize_raw=self.normalize_raw)
            oob_indices = _generate_unsampled_indices(estimator.random_state, n_samples, n_samples)
            if self.oob: 
                X_oob = X[oob_indices, :]
                y_oob = y[oob_indices] #- np.mean(y[oob_indices])
            else:
                X_oob = X
                y_oob = y
            if sample_weight is not None:
                sample_weight_oob = sample_weight[oob_indices]
            else:
                sample_weight_oob = None
            X_transformed_oob, start_indices = tree_transformer.transform(X_oob, center=True, return_indices=True)
            if len(X_transformed_oob) == 0: 
                for k in range(n_features):
                    n_stumps[idx, k] = np.NaN
                    n_stumps_chosen[idx, k] = np.NaN
                    scores[idx, k] = np.NaN
            else:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    self.scorer.fit(X_transformed_oob, y_oob, start_indices, sample_weight_oob)
                for k in range(n_features):
                    n_stumps[idx, k] = self.scorer.get_n_stumps(k)
                    n_stumps_chosen[idx, k] = self.scorer.get_model_size(k)
                    scores[idx, k] = self.scorer.get_score(k)
                    if multi_class:
                        class_scores_dict[idx][k] = self.scorer.class_scores[k]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            imp_values = np.nanmean(scores, axis=0)
        imp_values[np.isnan(imp_values)] = -np.inf

        if multi_class:
            if diagnostics:
                class_scores = pd.DataFrame.from_dict({(i,j): class_scores_dict[i][j]
                                                       for i in class_scores_dict.keys()
                                                       for j in class_scores_dict[i].keys()},
                                                      orient='index').reset_index().\
                    rename(columns={"level_0": "tree", "level_1": "feature"})
                return imp_values, scores, class_scores, n_stumps, n_stumps_chosen
            else:
                return imp_values
        else:
            if diagnostics:
                return imp_values, scores, n_stumps, n_stumps_chosen
            else:
                return imp_values


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
