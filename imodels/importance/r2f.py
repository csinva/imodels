import copy
import warnings

import numpy as np
from sklearn.linear_model import RidgeCV, LassoCV, LinearRegression, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from imodels.importance.representation import TreeTransformer


class R2F:
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
                 criterion="bic", refit=True, add_raw=True, n_splits=10):

        if estimator is None:
            self.estimator = RandomForestRegressor(n_estimators=100, min_samples_leaf=5, max_features=0.33)
        else:
            self.estimator = estimator
        self.max_components_type = max_components_type
        self.alpha = alpha
        self.normalize = normalize
        self.random_state = random_state
        self.criterion = criterion
        self.refit = refit
        self.add_raw = add_raw
        self.n_splits = n_splits

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
        if self.n_splits % 2 != 0:
            raise ValueError("n_splits has to be an even integer")
        if sample_weight is None:
            sample_weight = np.ones_like(y)
        for i in range(self.n_splits // 2):
            if self.random_state is not None:
                random_state = self.random_state + i
            else:
                random_state = None
            X_a, X_b, y_a, y_b, sample_weight_a, sample_weight_b = \
                train_test_split(X, y, sample_weight, test_size=0.5, random_state=random_state)
            tree_transformer_a = self._feature_learning_one_split(X_a, y_a, sample_weight_a)
            r_squared[2*i, :], n_stumps[2*i, :], n_components_chosen[2*i, :] = \
                self._model_selection_r2_one_split(tree_transformer_a, X_b, y_b)
            tree_transformer_b = self._feature_learning_one_split(X_b, y_b, sample_weight_b)
            r_squared[2*i+1, :], n_stumps[2*i+1, :], n_components_chosen[2*i+1, :] = \
                self._model_selection_r2_one_split(tree_transformer_b, X_a, y_a)
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
                                           alpha=self.alpha, normalize=self.normalize)
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
                y_val_centered = y_val - np.mean(y_val)
                if self.add_raw:
                    X_transformed = np.hstack([X_val[:, [k]] - np.mean(X_val[:, k]), X_transformed])
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    if self.criterion == "cv":
                        lasso = LassoCV(fit_intercept=False, normalize=False)
                    else:
                        lasso = LassoLarsIC(criterion=self.criterion, normalize=False, fit_intercept=False)
                    lasso.fit(X_transformed, y_val_centered)
                    n_components_chosen[k] = np.count_nonzero(lasso.coef_)
                    if self.refit:
                        support = np.nonzero(lasso.coef_)[0]
                        if len(support) == 0:
                            r_squared[k] = 0.0
                        else:
                            lr = LinearRegression().fit(X_transformed[:, support], y_val_centered)
                            r_squared[k] = lr.score(X_transformed[:, support], y_val_centered)
                    else:
                        r_squared[k] = lasso.score(X_transformed, y_val_centered)
        return r_squared, n_stumps, n_components_chosen