import copy
import warnings

import numpy as np
import statsmodels.api as sm
import statsmodels.stats.multitest as smt
from numpy import linalg as LA
from scipy import stats
from sklearn.linear_model import RidgeCV, LassoCV, LinearRegression, LassoLarsIC
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from representation import TreeTransformer


# sys.path.append("../../nonlinear_significance/scripts/")
# os.chdir("../../nonlinear_significance/scripts/")


class R2F:
    """
    Class to compute
    """

    def __init__(self, estimator, max_components_type="min_fracnsamples_nstumps", alpha=0.5, normalize=False,
                 random_state=None):
        self.estimator = estimator
        self.max_components_type = max_components_type
        self.alpha = alpha
        self.normalize = normalize
        self.random_state = random_state

    def get_importance_scores(self, X, y, criterion="bic", refit=True, add_raw=True, n_splits=10, sample_weight=None,
                              diagnostics=False):
        n_features = X.shape[1]
        r_squared = np.zeros((n_splits, n_features))
        n_components_chosen = np.zeros((n_splits, n_features))
        n_stumps = np.zeros((n_splits, n_features))
        if n_splits % 2 != 0:
            raise ValueError("n_splits has to be an even integer")
        for i in range(n_splits // 2):
            X_a, X_b, y_a, y_b, sample_weight_a, sample_weight_b = \
                train_test_split(X, y, sample_weight, test_size=0.5, random_state=self.random_state+i)
            tree_transformer_a = self._feature_learning_one_split(X_a, y_a, sample_weight_a,
                                                                  random_state=self.random_state+i)
            r_squared[2*i, :], n_stumps[2*i, :], n_components_chosen[2*i, :] = \
                self._model_selection_r2_one_split(tree_transformer_a, X_b, y_b, criterion, refit, add_raw,
                                                   sample_weight_b)
            tree_transformer_b = self._feature_learning_one_split(X_b, y_b, sample_weight_b,
                                                                  random_state=self.random_state+i)
            r_squared[2*i+1, :], n_stumps[2*i+1, :], n_components_chosen[2*i+1, :] = \
                self._model_selection_r2_one_split(tree_transformer_b, X_a, y_a, criterion, refit, add_raw,
                                                   sample_weight_a)
        r_squared_mean = np.mean(r_squared, axis=0)
        if diagnostics:
            return r_squared_mean, r_squared, n_stumps, n_components_chosen
        else:
            return r_squared_mean

    def _feature_learning_one_split(self, X_train, y_train, sample_weight=None, random_state=None):
        estimator = copy.deepcopy(self.estimator)
        if sample_weight is None:
            sample_weight = np.ones_like(y_train)
        estimator.fit(X_train, y_train, sample_weight=sample_weight, random_state=random_state)
        tree_transformer = TreeTransformer(estimator=estimator, max_components_type=self.max_components_type,
                                           alpha=self.alpha, normalize=self.normalize)
        tree_transformer.fit(X_train)
        return tree_transformer

    def _model_selection_r2_one_split(self, tree_transformer, X_val, y_val, criterion="bic", refit=True,
                                      add_raw=True, sample_weight=None):
        if sample_weight is None:
            sample_weight = np.ones_like(y_val)
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
                if add_raw:
                    X_transformed = np.hstack([X_val[:, [k]] - np.mean(X_val[:, k]), X_transformed])
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    if criterion == "cv":
                        lasso = LassoCV(fit_intercept=False, normalize=False)
                    else:
                        lasso = LassoLarsIC(criterion=criterion, normalize=False, fit_intercept=False)
                    lasso.fit(X_transformed, y_val_centered, sample_weight=sample_weight)
                    n_components_chosen[k] = np.count_nonzero(lasso.coef_)
                    if refit:
                        support = np.nonzero(lasso.coef_)[0]
                        if len(support) == 0:
                            r_squared[k] = 0.0
                        else:
                            lr = LinearRegression().fit(X_transformed[:, support], y_val_centered)
                            r_squared[k] = lr.score(X_transformed[:, support], y_val_centered)
                    else:
                        r_squared[k] = lasso.score(X_transformed, y_val_centered)
        return r_squared, n_stumps, n_components_chosen