import copy
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict

import numpy as np
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV,LinearRegression, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble._forest import _generate_unsampled_indices
from sklearn.model_selection import train_test_split
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


    def __init__(self, estimator=None, max_components_type="auto", alpha=0.5, normalize=False, random_state=None,use_noise_variance = True,scorer = None,treelet = False,
                 criterion="bic", refit=True, add_raw=True, split_data=True, val_size=0.5, n_splits=10,rank_by_p_val = False,pca=True,normalize_raw = False):

        if estimator is None:
            self.estimator = RandomForestRegressor(n_estimators=100, min_samples_leaf=5, max_features=0.33)
        else:
            self.estimator = estimator
        if scorer is None:
            self.scorer = LassoScorer(criterion = self.criterion,refit = self.refit)
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

    def _feature_learning_one_split(self, X_train, y_train,sample_weight=None):
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
        tree_transformer = TreeTransformer(estimator=estimator, max_components_type=max_components_type,add_raw = self.add_raw,treelet = self.treelet,
                                           alpha=self.alpha, normalize=self.normalize, pca=self.pca,normalize_raw = self.normalize_raw)
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
                    self.scorer.fit(X_transformed,y_val)
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

    def __init__(self, estimator=None, scorer=None, normalize=False, add_raw=True, refit = True,
                 criterion = "aic_c",random_state=None,normalize_raw = False):

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
            self.scorer = LassoScorer(criterion = self.criterion,refit = self.refit)
        else:
            self.scorer = copy.deepcopy(scorer)

    def get_importance_scores(self, X, y, sample_weight = None,diagnostics=False):
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
        self.estimator.fit(X, y)

        for idx, estimator in enumerate(self.estimator.estimators_):
            tree_transformer = TreeTransformer(estimator=estimator, pca=False, add_raw=self.add_raw,normalize_raw = self.normalize_raw)
            oob_indices = _generate_unsampled_indices(estimator.random_state, n_samples, n_samples)
            X_oob = X[oob_indices, :]
            y_oob = y[oob_indices]
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
                        self.scorer.fit(X_transformed_oob, y_oob)
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
        if criterion == "cv":
            self.lasso_model = LassoCV(normalize=False, fit_intercept=True)
        else:
            self.lasso_model = LassoLarsICc(criterion=criterion, normalize=False, fit_intercept=True)
        self.refit = refit

    def fit(self, X, y):
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

    def __init__(self, metric=None):
        super().__init__(metric)

    def fit(self, X, y):
        ridge_model = RidgeCV(normalize = False,fit_intercept = True).fit(X, y)
        self.selected_features = np.nonzero(ridge_model.coef_)[0]
        y_pred = ridge_model.predict(X)
        self.score = self.metric(y, y_pred)
        
class ElasticNetScorer(ScorerBase, ABC):
    
    def __init__(self, metric=None,refit = True):
        super().__init__(metric)
        self.elasticnet_model = ElasticNetCV(normalize = False, fit_intercept = True)
        self.refit = refit

    def fit(self, X, y):
        self.elasticnet_model.fit(X,y)
        self.selected_features = np.nonzero(self.elasticnet_model.coef_)[0]
        if self.refit and self.get_model_size() > 0:
            X_sel = X[:, self.selected_features]
            lr = LinearRegression().fit(X_sel, y)
            y_pred = lr.predict(X_sel)
        else:
            y_pred = self.elasticnet_model.predict(X)

        self.score = self.metric(y, y_pred)

        
class JointScorerBase(ABC):

    def __init__(self, metric):
        self.scores = defaultdict(lambda x: None)
        self.n_stumps = defaultdict(lambda x: None)
        self.model_sizes = defaultdict(lambda x: None)
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

    def __init__(self, metric=None):
        super().__init__(metric)

    def fit(self, X, y, start_indices):
        ridge_model = RidgeCV(fit_intercept=True).fit(X, y)
        for k in range(len(start_indices) - 1):
            restricted_feats = X[:, start_indices[k]:start_indices[k+1]]
            restricted_coefs = ridge_model.coef_[start_indices[k]:start_indices[k+1]]
            self.n_stumps[k] = start_indices[k+1] - start_indices[k]
            self.model_sizes[k] = int(np.sum(restricted_coefs != 0))
            if len(restricted_coefs) > 0:
                restricted_preds = restricted_feats @ restricted_coefs + ridge_model.intercept_
                self.scores[k] = self.metric(y, restricted_preds)
            else:
                self.scores[k] = 0


class GeneralizedMDIJoint:

    def __init__(self, estimator=None, scorer=None, normalize=False, add_raw=True, random_state=None, normalize_raw=False):

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

    def get_importance_scores(self, X, y, diagnostics=False):

        n_samples, n_features = X.shape
        n_trees = self.estimator.n_estimators
        scores = np.zeros((n_trees, n_features))
        n_stumps = np.zeros((n_trees, n_features))
        n_stumps_chosen = np.zeros((n_trees, n_features))
        self.estimator.fit(X, y)

        for idx, estimator in enumerate(self.estimator.estimators_):
            tree_transformer = TreeTransformer(estimator=estimator, pca=False, add_raw=self.add_raw, normalize_raw=self.normalize_raw)
            oob_indices = _generate_unsampled_indices(estimator.random_state, n_samples, n_samples)
            X_oob = X[oob_indices, :]
            y_oob = y[oob_indices]
            X_transformed_oob, start_indices = tree_transformer.transform(X_oob, return_indices=True)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                self.scorer.fit(X_transformed_oob, y_oob, start_indices)
            for k in range(n_features):
                n_stumps[idx, k] = self.scorer.get_n_stumps(k)
                n_stumps_chosen[idx, k] = self.scorer.get_model_size(k)
                scores[idx, k] = self.scorer.get_score(k)
        imp_values = scores.mean(axis=0)

        if diagnostics:
            return imp_values, scores, n_stumps, n_stumps_chosen
        else:
            return imp_values
                    
                    #y_val_centered = y_val - np.mean(y_val)
                    #if self.criterion == "cv":
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
                    #elif self.criterion == "f_regression":
                    #    f_stat, p_vals = f_regression(X_transformed, y_val_centered)
                    #    chosen_components = []
                    #    for i in range(len(p_vals)):
                    #        if p_vals[i] <= 0.05:
                    #            chosen_components.append(i)
                    #    n_components_chosen[k] = len(chosen_components)
                    #else:
                    #    lm = LassoLarsICc(criterion=self.criterion, normalize=False, fit_intercept=False,use_noise_variance = self.use_noise_variance) #LassoLarsIC
                    #    lm.fit(X_transformed, y_val_centered)
                    #    n_components_chosen[k] = np.count_nonzero(lm.coef_)
                    #if self.refit:
                    #    if self.criterion == "f_regression":
                    #        support = chosen_components
                    #    else:
                    #        support = np.nonzero(lm.coef_)[0]
                    #    if len(support) == 0:
                    #        r_squared[k] = 0.0
                    #    else:
                    #        lr = LinearRegression().fit(X_transformed[:, support], y_val_centered)
                    #        r_squared[k] = lr.score(X_transformed[:, support], y_val_centered)
                    #else:
                    #    r_squared[k] = lm.score(X_transformed, y_val_centered)