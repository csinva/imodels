import numpy as np
import pandas as pd
from collections import defaultdict

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.decomposition import PCA
from sklearn.ensemble import BaseEnsemble
import statistics
import statsmodels.api as sm
from scipy import stats
from sklearn.model_selection import train_test_split
import statsmodels.stats.multitest as smt
from tqdm import tqdm
# import torch
import sys, os
# from torch import nn
from numpy import linalg as LA
# from torch.functional import F
import copy
from sklearn.model_selection import GridSearchCV
# from torch.autograd import Variable
import numpy as np
from collections import defaultdict
from joblib import delayed, Parallel
from sklearn.feature_selection import RFECV
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.decomposition import PCA
from sklearn.ensemble import BaseEnsemble
from sklearn.metrics import r2_score
import statistics, warnings
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from scipy import sparse
from sklearn.linear_model import RidgeCV, LassoCV, LinearRegression,LassoLarsIC

# from nonlinear_significance.scripts.util import *
from representation import TreeTransformer
# sys.path.append("../../nonlinear_significance/scripts/")
# from util import TreeTransformer
# os.chdir("../../nonlinear_significance/scripts/")


def get_r_squared(OLS_results, tree_transformer, transformed_feats, y_test, origin_feat):
    feat_pcs = tree_transformer.original_feat_to_transformed_mapping[origin_feat]
    restricted_model_coeffs = OLS_results.params[feat_pcs]
    a = np.transpose(y_test - transformed_feats[:, feat_pcs] @ restricted_model_coeffs) @ (
            y_test - transformed_feats[:, feat_pcs] @ restricted_model_coeffs)
    return 1.0 - (a / (np.transpose(y_test) @ y_test))


class TreeTester:

    def __init__(self, estimator, max_components_type='median', fraction_chosen=1.0, normalize=False):
        """

        :param estimator:
        :param max_components: Method for choosing the number of components for PCA. Can be either "median", "max",
            or a fraction in [0, 1]. If "median" (respectively "max") then this is set as the median (respectively max
            number of splits on that feature in the RF. If a fraction, then this is set to be the fraction * n
        :param normalize:
        """
        self.estimator = estimator
        self.max_components_type = max_components_type
        self.fraction_chosen = fraction_chosen
        self.normalize = normalize

    def get_feature_significance_and_ranking(self, X, y, num_splits=10, add_linear=True, joint=False,
                                             diagnostics=False, adjusted_r2=False):
        p_vals = np.zeros((num_splits, X.shape[1]))
        r_squared = np.zeros((num_splits, X.shape[1]))
        n_stumps = np.zeros((num_splits, X.shape[1]))
        num_components_chosen = np.zeros((num_splits, X.shape[1]))
        for i in tqdm(range(num_splits)):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5,
                                                                random_state=i)  # perform sample splitting
            self.estimator.fit(X_train, y_train)  # fit on half of sample to learn tree structure and features
            tree_transformer = TreeTransformer(estimator=self.estimator, max_components_type=self.max_components_type,
                                               fraction_chosen=self.fraction_chosen, normalize=self.normalize)
            tree_transformer.fit(X_train)  # Apply PCA on X_train
            # transformed_feats = tree_transformer.transform(X_test)  # apply tree mapping on X_test
            if joint:  # Fit joint linear model
                raise NotImplementedError()
            else:
                for j in range(X.shape[1]):  # Iterate over original features
                    n_stumps[i, j] = len(tree_transformer.original_feat_to_stump_mapping[j])
                    transformed_feats_for_j = tree_transformer.transform_one_feature(X_test, j)
                    if add_linear and transformed_feats_for_j is not None:
                        transformed_feats_for_j = np.hstack(
                            [X_test[:, [j]] - np.mean(X_test[:, j]), transformed_feats_for_j])
                    if transformed_feats_for_j is None:  # if transformed_feats_for_j.shape[1] == 0:
                        p_vals[i, j] = 1.0
                        r_squared[i, j] = 0.0
                        num_components_chosen[i, j] = 0
                    else:
                        with warnings.catch_warnings():
                            warnings.filterwarnings("ignore")
                            OLS_for_j = sm.OLS(y_test - np.mean(y_test), transformed_feats_for_j).fit(cov_type="HC0")
                            if adjusted_r2:
                                r_squared[i, j] = OLS_for_j.rsquared_adj
                            else:
                                r_squared[i, j] = OLS_for_j.rsquared
                            p_vals[i, j] = OLS_for_j.f_pvalue
                            num_components_chosen[i, j] = transformed_feats_for_j.shape[1]
        p_vals[np.isnan(p_vals)] = 1.0
        median_p_vals = 2 * np.median(p_vals, axis=0)
        r_squared = np.mean(r_squared, axis=0)
        median_p_vals[median_p_vals > 1.0] = 1.0

        if diagnostics:
            return median_p_vals, r_squared, num_components_chosen, n_stumps
        else:
            return median_p_vals, r_squared

    def multiple_testing_correction(self, p_vals, method='bonferroni', alpha=0.05):
        return smt.multipletests(p_vals, method=method)[1]

    def get_r_squared_sig_threshold(self, X, y, num_splits=10, add_linear=True, threshold=0.05, first_ns=True,
                                    diagnostics=False):
        """
        Get r squared values, but only with respect to a subset of the engineered features, depending on a thresholding
        criterion.

        :param X:
        :param y:
        :param num_splits:
        :param add_linear:
        :param threshold:
        :param first_ns: Flag, if True, then use only engineered features with indices less than the first one
            that has nonsignificant p-value
        :return:
        """
        r_squared = np.zeros((num_splits, X.shape[1]))
        num_components_chosen = np.zeros((num_splits, X.shape[1]))
        n_stumps = np.zeros((num_splits, X.shape[1]))
        for i in tqdm(range(num_splits)):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5,
                                                                random_state=i)  # perform sample splitting
            self.estimator.fit(X_train, y_train)  # fit on half of sample to learn tree structure and features
            # if self.max_components == 'median':
            tree_transformer = TreeTransformer(estimator=self.estimator, max_components_type=self.max_components_type,
                                               fraction_chosen=self.fraction_chosen, normalize=self.normalize)
            tree_transformer.fit(X_train)  # Apply PCA on X_train
            # transformed_feats = tree_transformer.transform(X_test)
            # transformed_feats = tree_transformer.transform(X_test)  # apply tree mapping on X_test
            for j in range(X.shape[1]):  # Iterate over original features
                transformed_feats_for_j = tree_transformer.transform_one_feature(X_test, j)
                n_stumps[i, j] = len(tree_transformer.original_feat_to_stump_mapping[j])
                if transformed_feats_for_j is None:
                    r_squared[i, j] = 0.0
                    num_components_chosen[i, j] = 0
                else:
                    #
                    # if add_linear:
                    #     transformed_feats_for_j = np.hstack(
                    #         [X_test[:, [j]] - np.mean(X_test[:, j]), transformed_feats_for_j])
                    # if transformed_feats_for_j.shape[1] == 0:
                    #     r_squared[i, j] = 0.0
                    #     num_components_chosen[i, j] = 0
                    # else:
                    # print(transformed_feats_for_j.shape)
                    f_p_values = sequential_F_test(transformed_feats_for_j, y_test - np.mean(y_test))
                    f_p_values = np.nan_to_num(f_p_values, copy=True, nan=1.0, posinf=None, neginf=None)
                    if first_ns:
                        if np.all(f_p_values <= threshold):
                            stopping_index = transformed_feats_for_j.shape[1]
                        else:
                            stopping_index = np.nonzero(f_p_values > threshold)[0][0]
                        filtered_transformed_feats_for_j = transformed_feats_for_j[:, np.arange(stopping_index)]
                    else:
                        filtered_transformed_feats_for_j = transformed_feats_for_j[:, f_p_values <= threshold]
                    if add_linear:
                        filtered_transformed_feats_for_j = np.hstack([X_test[:, [j]] - np.mean(X_test[:, j]),
                                                                      filtered_transformed_feats_for_j])
                    num_components_chosen[i, j] = filtered_transformed_feats_for_j.shape[1]
                    if filtered_transformed_feats_for_j.shape[1] == 0:
                        r_squared[i, j] = 0.0
                    else:
                        OLS_for_j = sm.OLS(y_test - np.mean(y_test), filtered_transformed_feats_for_j).fit(
                            cov_type="HC0")
                        r_squared[i, j] = OLS_for_j.rsquared
        r_squared = np.mean(r_squared, axis=0)
        if diagnostics:
            return r_squared, num_components_chosen, n_stumps
        else:
            return r_squared

    def get_r_squared_stepwise_regression(self, X, y, num_splits=10, add_linear=True, threshold=0.05,
                                          diagnostics=False):
        r_squared = np.zeros((num_splits, X.shape[1]))
        num_components_chosen = np.zeros((num_splits, X.shape[1]))
        n_stumps = np.zeros((num_splits, X.shape[1]))
        for i in tqdm(range(num_splits)):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5,
                                                                random_state=i)  # perform sample splitting
            self.estimator.fit(X_train, y_train)  # fit on half of sample to learn tree structure and features
            tree_transformer = TreeTransformer(estimator=self.estimator, max_components_type=self.max_components_type,
                                               fraction_chosen=self.fraction_chosen, normalize=self.normalize)
            tree_transformer.fit(X_train)  # Apply PCA on X_train
            for j in range(X.shape[1]):  # Iterate over original features
                transformed_feats_for_j = tree_transformer.transform_one_feature(X_test, j)
                n_stumps[i, j] = len(tree_transformer.original_feat_to_stump_mapping[j])
                if transformed_feats_for_j is None:
                    r_squared[i, j] = 0.0
                    num_components_chosen[i, j] = 0
                else:
                    if add_linear:
                        transformed_feats_for_j = np.hstack(
                            [X_test[:, [j]] - np.mean(X_test[:, j]), transformed_feats_for_j])
                    active_set = stepwise_regression_test(transformed_feats_for_j, y_test - np.mean(y_test), threshold)
                    num_components_chosen[i, j] = len(active_set)
                    if len(active_set) == 0:
                        r_squared[i, j] = 0.0
                    else:
                        OLS_for_j = sm.OLS(y_test - np.mean(y_test), transformed_feats_for_j[:, active_set]).fit(
                            cov_type="HC0")
                        r_squared[i, j] = OLS_for_j.rsquared
        r_squared = np.mean(r_squared, axis=0)
        if diagnostics:
            return r_squared, num_components_chosen, n_stumps
        else:
            return r_squared

    def get_r_squared_nonsequential_bic(self, X, y, num_splits=10, add_linear=True, direction="forward",
                                        diagnostics=False, adjusted_r2=False):
        r_squared = np.zeros((num_splits, X.shape[1]))
        num_components_chosen = np.zeros((num_splits, X.shape[1]))
        n_stumps = np.zeros((num_splits, X.shape[1]))
        for i in tqdm(range(num_splits)):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5,
                                                                random_state=i)  # perform sample splitting
            self.estimator.fit(X_train, y_train)  # fit on half of sample to learn tree structure and features
            tree_transformer = TreeTransformer(estimator=self.estimator, max_components_type=self.max_components_type,
                                               fraction_chosen=self.fraction_chosen, normalize=self.normalize)
            tree_transformer.fit(X_train)  # Apply PCA on X_train
            for j in range(X.shape[1]):  # Iterate over original features
                transformed_feats_for_j = tree_transformer.transform_one_feature(X_test, j)
                n_stumps[i, j] = len(tree_transformer.original_feat_to_stump_mapping[j])
                if transformed_feats_for_j is None:
                    r_squared[i, j] = 0.0
                    num_components_chosen[i, j] = 0
                else:
                    if add_linear:
                        transformed_feats_for_j = np.hstack(
                            [X_test[:, [j]] - np.mean(X_test[:, j]), transformed_feats_for_j])
                    active_set = nonsequential_bic(transformed_feats_for_j, y_test - np.mean(y_test), direction)
                    if len(active_set) == 0:
                        r_squared[i, j] = 0.0
                        num_components_chosen[i, j] = 0
                    else:
                        OLS_for_j = sm.OLS(y_test - np.mean(y_test), transformed_feats_for_j[:, active_set]).fit(
                            cov_type="HC0")
                        if adjusted_r2:
                            r_squared[i, j] = OLS_for_j.rsquared_adj
                        else:
                            r_squared[i, j] = OLS_for_j.rsquared
                        num_components_chosen[i, j] = len(active_set)
        r_squared = np.mean(r_squared, axis=0)
        if diagnostics:
            return r_squared, num_components_chosen, n_stumps
        else:
            return r_squared

    def get_r_squared_sequential_bic(self, X, y, num_splits=10, add_linear=True, diagnostics=False, adjusted_r2=False):
        r_squared = np.zeros((num_splits, X.shape[1]))
        num_components_chosen = np.zeros((num_splits, X.shape[1]))
        n_stumps = np.zeros((num_splits, X.shape[1]))
        for i in tqdm(range(num_splits)):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5,
                                                                random_state=i)  # perform sample splitting
            self.estimator.fit(X_train, y_train)  # fit on half of sample to learn tree structure and features
            tree_transformer = TreeTransformer(estimator=self.estimator, max_components_type=self.max_components_type,
                                               fraction_chosen=self.fraction_chosen, normalize=self.normalize)
            tree_transformer.fit(X_train)  # Apply PCA on X_train
            for j in range(X.shape[1]):  # Iterate over original features
                transformed_feats_for_j = tree_transformer.transform_one_feature(X_test, j)
                n_stumps[i, j] = len(tree_transformer.original_feat_to_stump_mapping[j])
                if transformed_feats_for_j is None:
                    r_squared[i, j] = 0.0
                    num_components_chosen[i, j] = 0
                else:
                    if add_linear:
                        transformed_feats_for_j = np.hstack(
                            [X_test[:, [j]] - np.mean(X_test[:, j]), transformed_feats_for_j])
                    active_set = sequential_bic(transformed_feats_for_j, y_test - np.mean(y_test))
                    if len(active_set) == 0:
                        r_squared[i, j] = 0.0
                        num_components_chosen[i, j] = 0
                    else:
                        OLS_for_j = sm.OLS(y_test - np.mean(y_test), transformed_feats_for_j[:, active_set]).fit(
                            cov_type="HC0")
                        if adjusted_r2:
                            r_squared[i, j] = OLS_for_j.rsquared_adj
                        else:
                            r_squared[i, j] = OLS_for_j.rsquared
                        num_components_chosen[i, j] = len(active_set)
        r_squared = np.mean(r_squared, axis=0)
        if diagnostics:
            return r_squared, num_components_chosen, n_stumps
        else:
            return r_squared

    def get_r_squared_stepwise_adjusr2(self, X, y, num_splits=10, add_linear=True, threshold=0.05, diagnostics=False):
        raise NotImplementedError("Not implemented")
        # r_squared = np.zeros((num_splits, X.shape[1]))
        # num_components_chosen = np.zeros((num_splits, X.shape[1]))
        # n_stumps = np.zeros((num_splits, X.shape[1]))
        # for i in tqdm(range(num_splits)):
        #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5,
        #                                                         random_state=i)  # perform sample splitting
        #     self.estimator.fit(X_train, y_train)  # fit on half of sample to learn tree structure and features
        #     tree_transformer = TreeTransformer(estimator=self.estimator, max_components_type=self.max_components_type,
        #                                        fraction_chosen=self.fraction_chosen, normalize=self.normalize)
        #     tree_transformer.fit(X_train)  # Apply PCA on X_train
        #     for j in range(X.shape[1]):  # Iterate over original features
        #         transformed_feats_for_j = tree_transformer.transform_one_feature(X_test, j)
        #         n_stumps[i, j] = len(tree_transformer.original_feat_to_stump_mapping[j])
        #         if transformed_feats_for_j is None:
        #             r_squared[i, j] = 0.0
        #             num_components_chosen[i, j] = 0
        #         else:
        #             if add_linear:
        #                 transformed_feats_for_j = np.hstack(
        #                     [X_test[:, [j]] - np.mean(X_test[:, j]), transformed_feats_for_j])
        #             transformed_feats_for_j = pd.DataFrame(transformed_feats_for_j)
        #             transformed_feats_for_j.columns = ['orig_feat'] + ['PC' + str(i) for i in
        #                                                                range(1, transformed_feats_for_j.shape[1])]
        #             transformed_feats_for_j['response'] = y_test - np.mean(y_test)
        #             ols_r2_model, ols_r2 = forward_selected(transformed_feats_for_j, 'response')
        #             r_squared[i, j] = ols_r2
        #     r_squared = np.mean(r_squared, axis=0)
        #     if diagnostics:
        #         return r_squared, num_components_chosen
        #     else:
        #         return r_squared

    def get_r_squared_ridge(self, X, y, num_splits=10, add_linear=True, diagnostics=False):
        r_squared = np.zeros((num_splits, X.shape[1]))
        num_components_chosen = np.zeros((num_splits, X.shape[1]))
        n_stumps = np.zeros((num_splits, X.shape[1]))
        for i in tqdm(range(num_splits)):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5,
                                                                random_state=i)  # perform sample splitting
            self.estimator.fit(X_train,
                               y_train)  # fit on half of sample to learn tree structure and features # if self.max_components == 'median':
            tree_transformer = TreeTransformer(estimator=self.estimator, max_components_type=self.max_components_type,
                                               fraction_chosen=self.fraction_chosen, normalize=self.normalize)
            tree_transformer.fit(X_train)  # Apply PCA on X_train
            # tree_transformed_test = tree_transformer.transform(X_test)  # transformed_feats = tree_transformer.transform(X_test)  # apply tree mapping on X_test
            for j in range(X.shape[1]):  # Iterate over original features
                transformed_feats_for_j = tree_transformer.transform_one_feature(X_test, j)
                n_stumps[i, j] = len(tree_transformer.original_feat_to_stump_mapping[j])
                if transformed_feats_for_j is None:
                    r_squared[i, j] = 0.0
                    num_components_chosen[i, j] = 0
                else:
                    if add_linear:
                        transformed_feats_for_j = np.hstack([X_test[:, [j]] - np.mean(X_test[:, j]),
                                                             transformed_feats_for_j])
                    num_components_chosen[i, j] = transformed_feats_for_j.shape[1]
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore")
                        clf = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1, 10.0, 100.0, 500.0]).fit(transformed_feats_for_j,
                                                                                            y_test - np.mean(y_test))
                        r_squared[i, j] = clf.score(transformed_feats_for_j, y_test - np.mean(y_test))
                    # r2_score(y_test,clf.predict(#clf.score(transformed_feats_for_j,y_test)
        r_squared = np.mean(r_squared, axis=0)
        if diagnostics:
            return r_squared, num_components_chosen, n_stumps
        else:
            return r_squared

    def get_r_squared_lasso(self, X, y, num_splits=10, add_linear=True, criteria = "bic",diagnostics=False,refit = True):
        r_squared = np.zeros((num_splits, X.shape[1]))
        num_components_chosen = np.zeros((num_splits, X.shape[1]))
        n_stumps = np.zeros((num_splits, X.shape[1]))
        for i in tqdm(range(num_splits)):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5,
                                                                random_state=i)  # perform sample splitting
            self.estimator.fit(X_train,y_train)
            tree_transformer = TreeTransformer(estimator=self.estimator, max_components_type=self.max_components_type,
                                               fraction_chosen=self.fraction_chosen, normalize=self.normalize)
            tree_transformer.fit(X_train)  # Apply PCA on X_train
            for j in range(X.shape[1]):  # Iterate over original features
                transformed_feats_for_j = tree_transformer.transform_one_feature(X_test, j)
                n_stumps[i, j] = len(tree_transformer.original_feat_to_stump_mapping[j])
                if transformed_feats_for_j is None:
                    r_squared[i, j] = 0.0
                    num_components_chosen[i, j] = 0
                else:
                    if add_linear:
                        transformed_feats_for_j = np.hstack([X_test[:, [j]] - np.mean(X_test[:, j]),
                                                             transformed_feats_for_j])
                    #num_components_chosen[i, j] = transformed_feats_for_j.shape[1]
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore")
                        if criteria == "cv":
                            clf = LassoCV(fit_intercept = False)
                            clf.fit(transformed_feats_for_j,y_test - np.mean(y_test))
                        elif criteria == "aic":
                            clf = LassoLarsIC(criterion="aic", normalize=False,fit_intercept = False)
                            clf.fit(transformed_feats_for_j,y_test - np.mean(y_test))
                        else:
                            clf = LassoLarsIC(criterion="bic", normalize=False,fit_intercept = False)
                            clf.fit(transformed_feats_for_j,y_test - np.mean(y_test))
                        num_components_chosen[i, j] = np.count_nonzero(clf.coef_)
                        if refit == True:
                            support = np.nonzero(clf.coef_)[0]
                            if len(support) == 0:
                                r_squared[i, j] = 0.0
                            else:
                                lr = LinearRegression().fit(transformed_feats_for_j[:,support],y_test - np.mean(y_test))
                                r_squared[i, j] = lr.score(transformed_feats_for_j[:,support], y_test - np.mean(y_test))
                        else:
                            r_squared[i, j] = clf.score(transformed_feats_for_j, y_test - np.mean(y_test))



                    # r2_score(y_test,clf.predict(#clf.score(transformed_feats_for_j,y_test)
        r_squared = np.mean(r_squared, axis=0)
        if diagnostics:
            return r_squared, num_components_chosen, n_stumps
        else:
            return r_squared


    def get_r_squared_pca_var_explained(self, X, y, num_splits=10, add_linear=True, threshold=0.5, diagnostics=False):
        r_squared = np.zeros((num_splits, X.shape[1]))
        num_components_chosen = np.zeros((num_splits, X.shape[1]))
        n_stumps = np.zeros((num_splits, X.shape[1]))
        for i in tqdm(range(num_splits)):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5,
                                                                random_state=i)  # perform sample splitting
            self.estimator.fit(X_train, y_train)  # fit on half of sample to learn tree structure and features
            # if self.max_components == 'median':
            tree_transformer = TreeTransformer(estimator=self.estimator, max_components_type=self.max_components_type,
                                               fraction_chosen=self.fraction_chosen, normalize=self.normalize)
            tree_transformer.fit(X_train)  # Apply PCA on X_train
            # transformed_feats = tree_transformer.transform(X_test)  # apply tree mapping on X_test
            for j in range(X.shape[1]):  # Iterate over original features
                transformed_feats_for_j = tree_transformer.transform_one_feature(X_test, j)
                n_stumps[i, j] = len(tree_transformer.original_feat_to_stump_mapping[j])
                if transformed_feats_for_j is None:
                    r_squared[i, j] = 0.0
                    num_components_chosen[i, j] = 0
                else:
                    pca_var_explained = tree_transformer.pca_transformers[j].explained_variance_ratio_
                    cum_var_explained = np.cumsum(pca_var_explained)
                    if cum_var_explained[-1] < threshold:
                        stopping_index = len(pca_var_explained)
                    else:
                        stopping_index = np.where(cum_var_explained >= threshold)[0][0]
                    transformed_feats_for_j = tree_transformer.transform_one_feature(X_test, j)
                    filtered_transformed_feats_for_j = transformed_feats_for_j[:, np.arange(stopping_index)]
                    if add_linear:
                        filtered_transformed_feats_for_j = np.hstack([X_test[:, [j]] - np.mean(X_test[:, j]),
                                                                      filtered_transformed_feats_for_j])
                    num_components_chosen[i, j] = filtered_transformed_feats_for_j.shape[1]
                    OLS_for_j = sm.OLS(y_test - np.mean(y_test), filtered_transformed_feats_for_j).fit(cov_type="HC0")
                    r_squared[i, j] = OLS_for_j.rsquared
        r_squared = np.mean(r_squared, axis=0)
        if diagnostics:
            return r_squared, num_components_chosen, n_stumps
        else:
            return r_squared

    def get_r_squared_pca_cv(self, X, y, num_splits=10, cv=5, geom_grid_spacing=8, add_linear=True, diagnostics=False):
        r_squared = np.zeros((num_splits, X.shape[1]))
        num_components_chosen = np.zeros((num_splits, X.shape[1]))
        n_stumps = np.zeros((num_splits, X.shape[1]))
        for i in tqdm(range(num_splits)):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=i)
            tree_transformer = TreeTransformer(estimator=self.estimator, max_components_type=self.max_components_type,
                                               fraction_chosen=self.fraction_chosen, normalize=self.normalize)
            tree_transformer.fit(X_train)
            for j in range(X.shape[1]):
                transformed_feats_for_j = tree_transformer.transform_one_feature(X_test, j)
                n_stumps[i, j] = len(tree_transformer.original_feat_to_stump_mapping[j])
                if transformed_feats_for_j is None:
                    r_squared[i, j] = 0.0
                    num_components_chosen[i, j] = 0
                else:
                    pc_grid = np.geomspace(1, transformed_feats_for_j.shape[1], num=geom_grid_spacing, dtype=int)
                    all_scores = []
                    for fold in range(cv):
                        fold_scores = []
                        train_size = 1.0 - (1.0 / cv)
                        transformed_feats_for_j_train, transformed_feats_for_j_test, y_test_train, y_test_val = train_test_split(
                            transformed_feats_for_j, y_test, train_size=train_size)
                        for num_pcs in pc_grid:
                            transformed_feats_for_j_train_limited_pcs = transformed_feats_for_j_train[:, :num_pcs]
                            transformed_feats_for_j_test_limited_pcs = transformed_feats_for_j_test[:, :num_pcs]
                            reg = LinearRegression().fit(transformed_feats_for_j_train_limited_pcs, y_test_train)
                            fold_scores.append(
                                r2_score(y_test_val, reg.predict(transformed_feats_for_j_test_limited_pcs)))
                        all_scores.append(fold_scores)
                    pc_scores_for_j = np.mean(all_scores, axis=0)
                    optimal_pcs_for_j = pc_grid[np.argmax(pc_scores_for_j)]
                    transformed_feats_for_j = transformed_feats_for_j[:, :optimal_pcs_for_j]
                    if add_linear:
                        transformed_feats_for_j = np.hstack(
                            [X_test[:, [j]] - np.mean(X_test[:, j]), transformed_feats_for_j])
                    OLS_for_j = sm.OLS(y_test - np.mean(y_test), transformed_feats_for_j).fit(cov_type="HC0")
                    r_squared[i, j] = OLS_for_j.rsquared
                    num_components_chosen[i, j] = transformed_feats_for_j.shape[1]
        r_squared = np.mean(r_squared, axis=0)
        if diagnostics:
            return r_squared, num_components_chosen, n_stumps
        else:
            return r_squared


class optimalTreeTester:  # This class is trying to improve the power of TreeTester by implementing an optimal weighting scheme that favors big nodes...

    def __init__(self, estimator, normalize=True):
        self.estimator = estimator
        self.normalize = normalize

    def get_feature_significance(self, X, y, num_splits=10, eta=None, lr=.1, n_steps=3000, num_reps=20000,
                                 max_components='median', params={}):
        p_vals = np.ones((num_splits, X.shape[1]))
        r_squared = np.zeros((num_splits, X.shape[1]))
        for i in tqdm(range(num_splits)):
            X_sel, X_inf, y_sel, y_inf = train_test_split(X, y, test_size=0.5)

            if len(params) != 0:
                gs_estimator = GridSearchCV(self.estimator, param_grid=params, scoring='r2', cv=5)
                gs_estimator.fit(X_sel, y_sel)
                self.estimator = gs_estimator.best_estimator_
                self.estimator.fit(X_sel, y_sel)  # fit on half of sample to learn tree structure and features
            else:
                self.estimator.fit(X_sel, y_sel)

            if max_components == 'median':
                tree_transformer_sel = TreeTransformer(estimator=copy.deepcopy(self.estimator),
                                                       max_components_type='median')
                tree_transformer_inf = TreeTransformer(estimator=copy.deepcopy(self.estimator),
                                                       max_components_type='median')

            else:
                tree_transformer_sel = TreeTransformer(estimator=self.estimator,
                                                       max_components_type=int(self.max_components * X_train.shape[0]))
                tree_transformer_inf = TreeTransformer(estimator=self.estimator,
                                                       max_components_type=int(self.max_components * X_train.shape[0]))

            # tree_transformer_sel = TreeTransformer(estimator = copy.deepcopy(self.estimator), max_components= int(X_sel.shape[0]*max_components) )
            tree_transformer_sel.fit(X_sel)
            transformed_feats_sel = tree_transformer_sel.transform(X_sel)
            transformed_feats_inf = tree_transformer_sel.transform(X_inf)

            # tree_transformer_inf = TreeTransformer(estimator = copy.deepcopy(self.estimator), max_components= int(X_sel.shape[0]*max_components) )
            # tree_transformer_inf.fit(X_sel)
            # transformed_feats_inf = tree_transformer_inf.transform(X_inf)#tree_transformer_sel.transform(X_inf)#tree_transformer_inf.transform(X_inf)

            n_sel = len(y_sel)
            n_inf = len(y_inf)
            p_sel = transformed_feats_sel.shape[1]
            p_inf = transformed_feats_inf.shape[1]
            for j in range(X.shape[1]):
                stumps_sel_for_feat = tree_transformer_sel.original_feat_to_transformed_mapping[j]
                num_splits_for_feat = len(stumps_sel_for_feat)
                if num_splits_for_feat == 0:
                    p_vals[i, j] = 1.0
                else:
                    stumps_inf_for_feat = tree_transformer_sel.original_feat_to_transformed_mapping[j]
                    # tree_transformer_inf.original_feat_to_transformed_mapping[j]
                    X_sel_for_feat = transformed_feats_sel[:, stumps_sel_for_feat]
                    p_sel_feat = X_sel_for_feat.shape[1]
                    sigma_sel = (np.sum((y_sel - np.mean(y_sel)) ** 2)) / (n_sel - p_sel_feat - 1)
                    if eta is None:
                        eta = sigma_sel
                    X_inf_for_feat = transformed_feats_inf[:, stumps_inf_for_feat]
                    p_inf_feat = X_inf_for_feat.shape[1]
                    sigma_inf = (np.sum((y_inf - np.mean(y_inf)) ** 2)) / (n_inf - p_inf_feat - 1)
                    optimal_lambda_for_feat = self.get_optimal_lambda(X_sel_for_feat, y_sel, eta, sigma_sel, lr,
                                                                      n_steps)
                    p_vals[i, j] = self.compute_p_val(optimal_lambda_for_feat, X_inf_for_feat, y_inf, sigma_inf,
                                                      num_reps, n_sel, n_inf, p_sel_feat, p_inf_feat)
                    OLS_results = sm.OLS(y_inf, transformed_feats_inf).fit()
                    r_squared[i, j] = get_r_squared(OLS_results, tree_transformer_sel, transformed_feats_inf, y_inf, j)

        p_vals[np.isnan(p_vals)] = 1.0
        median_p_vals = 2 * np.median(p_vals, axis=0)
        median_p_vals[median_p_vals > 1.0] = 1.0
        r_squared = np.mean(r_squared, axis=0)

        # return median_p_vals,p_vals,self.multiple_testing_correction(median_p_vals)
        return median_p_vals, r_squared

    def multiple_testing_correction(self, p_vals, method='bonferroni', alpha=0.05):
        return smt.multipletests(p_vals, method=method)[1]

    def compute_p_val(self, optimal_lambda, X_inf, y_inf, sigma_inf, num_reps, n_sel, n_inf, p_sel_feat, p_inf_feat):
        u_inf, s_inf, vh_inf = np.linalg.svd(X_inf, full_matrices=False)
        optimal_weights = optimal_lambda  # optimal_lambda.cpu().detach().numpy()
        weighted_chi_squared_samples = np.sort(
            np.array(self.get_weighted_chi_squared(optimal_weights, n_sel, n_inf, p_sel_feat, p_inf_feat, num_reps)))
        test_statistic = (np.sum((optimal_weights * (np.transpose(u_inf) @ y_inf)) ** 2)) / sigma_inf
        quantile = stats.percentileofscore(weighted_chi_squared_samples, test_statistic, 'weak') / 100.0
        return 1.0 - quantile

    def get_optimal_lambda(self, X, y, eta, sigma_sel, lr, n_steps):
        u_sel, s_sel, vh_sel = np.linalg.svd(X, full_matrices=False)
        betas = np.transpose(u_sel) @ y
        weights = []
        for i in range(u_sel.shape[1]):
            weights.append(np.random.uniform())
        weights = np.array(weights)
        difference_in_weights = np.array([1.0])
        gradient = np.array([1.0])
        num_steps = 0
        for i in range(
                n_steps):  # while any(i > 0.00001 for i in difference_in_weights):## #or i < n_steps:#for i in range(n_steps):
            gradient = self.compute_gradient(weights, betas, u_sel, y, eta, sigma_sel)
            new_weights = np.add(weights, lr * gradient)
            difference_in_weights = new_weights - weights
            weights = new_weights
            num_steps += 1
        return weights

    def compute_gradient(self, weights, betas, u_sel, y_sel, eta, sigma_sel):
        gradients = []
        g = np.dot(weights ** 2, betas ** 2) - eta * ((LA.norm(weights)) ** 2)
        h = sigma_sel * np.sqrt(np.sum(weights ** 4))
        for i in range(len(weights)):
            g_prime = 2.0 * weights[i] * (betas[i] ** 2) - (2.0 * eta * weights[i])
            h_prime = ((2.0 * (weights[i] ** 3)) * (sigma_sel)) / (np.sqrt(np.sum(weights ** 4)))
            grad_weight = (g_prime * h - h_prime * g) / (h ** 2)
            gradients.append(grad_weight)
        return np.array(gradients)

    def get_weighted_chi_squared(self, weights, n_sel, n_inf, p_sel_feat, p_inf_feat, num_reps=10000000):
        k = len(weights)
        samples = []
        for n in range(num_reps):
            numerator_sample = 0.0
            denominator_sample = np.random.chisquare(n_sel - p_sel_feat)  # /(n_sel-p_sel_feat-1)
            for i in range(k):
                numerator_sample += ((weights[i] ** 2) * np.random.chisquare(1, size=None))
            numerator_sample = numerator_sample * (n_sel - p_sel_feat - 1)
            samples.append(numerator_sample / (denominator_sample))
        return samples

        # def forward(self,weights,u_sel,y_sel,eta,sigma_sel):
    #    T1 = torch.from_numpy(np.transpose(u_sel) @ y_sel)
    #    T1 = T1.type(torch.FloatTensor)
    #    T1 = weights * T1
    #    T1 = torch.linalg.vector_norm(T1)**2
    #    T2 = eta*torch.sum(weights**2)
    #   T3 = sigma_sel*torch.sqrt(torch.sum(weights**4))
    #   return torch.divide(torch.subtract(T2,T1),T3)
    # print("g_prime" + str(g_prime))
    # print("g" + str(g))
    # print("h_prime:" + str(h_prime))
    ##print("h" + str(h))
    # print("grad weight numerator:" + str(g_prime*h - h_prime*g))
    # print("grad_weight:" + str(grad_weight))
    # print(g_prime*h - h_prime*g[0])

    # print(weights)
    # if all(i <= 0.000001 for i in difference_in_weights):
    #    #print("im'here")
    #    break
    # else:
    #    new_weights
    # opt.zero_grad()
    # z = self.forward(weights,u_sel,y,eta,sigma_sel)#torch.linalg.vector_norm(x)#x*x
    # z.sum().backward()
    # z.sum().backward()
    # print(weights.grad.data)


#            z.sum().backward() # Calculate gradients
# opt.step()
# while all(gradient)
# tns = torch.distributions.Uniform(0,1.0).sample((u_sel.shape[1],))
# weights = Variable(tns, requires_grad=True)
# opt = torch.optim.SGD([weights], lr=lr)


def sequential_F_test(X, y, cov_type="HC0"):
    """
    Takes results from a statsmodel OLS model fit, and obtain F-statistic p-values for adding each feature into
    the model.

    :param X: covariate matrix
    :param y: response vector
    :return:
    """

    d = X.shape[1]
    p_values = np.zeros(d)
    ols_full = sm.OLS(y, X[:, 0]).fit(cov_type=cov_type)
    p_values[0] = ols_full.f_pvalue
    for i in range(1, d):
        ols_restricted = ols_full
        ols_full = sm.OLS(y, X[:, np.arange(i + 1)]).fit(cov_type=cov_type)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            p_values[i] = ols_full.compare_f_test(ols_restricted)[1]

    return p_values


def stepwise_regression_test(X, y, threshold, cov_type="HC0"):
    d = X.shape[1]
    active_set = set()  # indices of features to be included in model
    non_active_set = set(range(0, d))
    p_vals_non_active_set = {i: 0 for i in range(0, d)}
    while (len(p_vals_non_active_set) != 0):  # any(val < threshold for val in p_vals_non_active_set.values()) and
        if len(active_set) == 0:
            for feat_considered in copy.deepcopy(non_active_set):
                X_feat = X[:, feat_considered]
                ols = sm.OLS(y, X_feat).fit(cov_type=cov_type)
                p_vals_non_active_set[feat_considered] = ols.f_pvalue
        else:
            X_active_set = X[:, list(active_set)]
            ols_active_set = sm.OLS(y, X_active_set).fit(cov_type=cov_type)
            for feat_considered in copy.deepcopy(non_active_set):
                active_set_under_consideration = copy.deepcopy(active_set)
                active_set_under_consideration.add(feat_considered)
                X_active_union_feat = X[:, list(active_set_under_consideration)]
                ols_active_union_feat = sm.OLS(y, X_active_union_feat).fit(cov_type=cov_type)
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    p_vals_non_active_set[feat_considered] = ols_active_union_feat.compare_f_test(ols_active_set)[1]
        smallest_p_val_feat = min(p_vals_non_active_set, key=p_vals_non_active_set.get)  # np.argmin(p_vals) + 1
        smallest_p_val = p_vals_non_active_set[smallest_p_val_feat]
        if smallest_p_val < threshold:
            active_set.add(smallest_p_val_feat)
            non_active_set.remove(smallest_p_val_feat)
            del p_vals_non_active_set[smallest_p_val_feat]
        else:
            break
    return [active_feat for active_feat in active_set]


def nonsequential_bic(X, y, direction="forward", cov_type="HC0"):
    d = X.shape[1]
    active_set = set() #indices of features to be included in model
    non_active_set = set(range(0,d))
    bic_vals_non_active_set = {i:0 for i in range(0,d)}

    # intercept only model for comparison
    ols = sm.OLS(y,sm.add_constant(X[:,list(active_set)])).fit(cov_type=cov_type)
    current_bic = ols.bic

    while (len(bic_vals_non_active_set) != 0):
        # collect bic values when including each non_active feature
        if len(active_set) == 0:
            for feat_considered in copy.deepcopy(non_active_set):
                X_feat = X[:,feat_considered]
                ols = sm.OLS(y,sm.add_constant(X_feat)).fit(cov_type=cov_type)
                bic_vals_non_active_set[feat_considered] = ols.bic
        else:
            for feat_considered in copy.deepcopy(non_active_set):
                active_set_under_consideration = copy.deepcopy(active_set)
                active_set_under_consideration.add(feat_considered)
                X_active_union_feat = X[:,list(active_set_under_consideration)]
                ols_active_union_feat = sm.OLS(y,sm.add_constant(X_active_union_feat)).fit(cov_type=cov_type)
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    bic_vals_non_active_set[feat_considered] = ols_active_union_feat.bic

        # choose feature with lowest bic
        smallest_bic_val_feat = min(bic_vals_non_active_set, key=bic_vals_non_active_set.get)
        smallest_bic_val = bic_vals_non_active_set[smallest_bic_val_feat]

        if smallest_bic_val < current_bic:
            # update current model bic, add feature to active set, remove from non_active
            current_bic = smallest_bic_val
            active_set.add(smallest_bic_val_feat)
            non_active_set.remove(smallest_bic_val_feat)
            del bic_vals_non_active_set[smallest_bic_val_feat]
        else:
            # stop, no improvements
            break

        if direction == "both" and len(active_set) != 0:
            # collect bic when excluding each active feature
            bic_vals_tmp = {i:0 for i in copy.deepcopy(active_set)}
            for feat_considered in copy.deepcopy(active_set):
                active_set_under_consideration = copy.deepcopy(active_set)
                active_set_under_consideration.remove(feat_considered)
                X_active_union_feat = X[:,list(active_set_under_consideration)]
                ols_active_union_feat = sm.OLS(y,sm.add_constant(X_active_union_feat)).fit(cov_type=cov_type)
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    bic_vals_tmp[feat_considered] = ols_active_union_feat.bic

            smallest_bic_val_feat = min(bic_vals_tmp, key=bic_vals_tmp.get)
            smallest_bic_val = bic_vals_tmp[smallest_bic_val_feat]

            if smallest_bic_val < current_bic:
                # update current model bic, remove feature from active, add to non active
                current_bic = smallest_bic_val
                active_set.remove(smallest_bic_val_feat)
                non_active_set.add(smallest_bic_val_feat)
                bic_vals_non_active_set[smallest_bic_val_feat] = smallest_bic_val

    return [active_feat for active_feat in active_set]


def sequential_bic(X, y, cov_type="HC0"):
    d = X.shape[1]
    active_set = set()
    bic_values = np.zeros(d)

    # intercept only model for comparison
    ols = sm.OLS(y,sm.add_constant(X[:,list(active_set)])).fit(cov_type=cov_type)
    current_bic = ols.bic
    for i in range(0, d):
        ols = sm.OLS(y, sm.add_constant(X[:, np.arange(i+1)])).fit(cov_type=cov_type)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            bic_values[i] = ols.bic
        if bic_values[i] < current_bic:
            current_bic = bic_values[i]
            active_set.add(i)
        else:
            break
    return [active_feat for active_feat in active_set]


import statsmodels.formula.api as smf


def forward_selected(data, response):
    """Linear model designed by forward selection.

    Parameters:
    -----------
    data : pandas DataFrame with all possible predictors and response

    response: string, name of response column in data

    Returns:
    --------
    model: an "optimal" fitted statsmodels linear model
           with an intercept
           selected by forward selection
           evaluated by adjusted R-squared
    """
    remaining = set(data.columns)
    remaining.remove(response)
    selected = []
    current_score, best_new_score = 0.0, 0.0
    while remaining and current_score == best_new_score:
        scores_with_candidates = []
        for candidate in remaining:
            formula = "{} ~ {} + 1".format(response,
                                           ' + '.join(selected + [candidate]))
            score = smf.ols(formula, data).fit().rsquared_adj
            scores_with_candidates.append((score, candidate))
        scores_with_candidates.sort()
        best_new_score, best_candidate = scores_with_candidates.pop()
        pre_remaining = copy.deepcopy(remaining)
        if current_score < best_new_score:
            remaining.remove(best_candidate)
            selected.append(best_candidate)
            current_score = best_new_score
        if pre_remaining == remaining:
            break
    formula = "{} ~ {} + 1".format(response, ' + '.join(selected))
    model = smf.ols(formula, data).fit()
    return model, model.rsquared

# print("active_set")  p_vals = np.nan_to_num(p_vals, copy=True, nan=1.0, posinf=None, neginf=None)
#        print(active_set)
#        print("non_active_set")
#        print(non_active_set)

#  def get_r_squared_lasso(self, X, y, num_splits=10, add_linear=True):
#            r_squared = np.zeros((num_splits, X.shape[1]))
#            for i in tqdm(range(num_splits)):
#                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5,
#                                                                random_state=i)  # perform sample splitting
##                self.estimator.fit(X_train, y_train)  # fit on half of sample to learn tree structure and features
# if self.max_components == 'median':
#                tree_transformer = TreeTransformer(estimator=self.estimator, max_components=self.max_components)
#                tree_transformer.fit(X_train,always_pca = False)  # Apply PCA on X_train
#                tree_transformed_test = tree_transformer.transform(X_test)
# transformed_feats = tree_transformer.transform(X_test)  # apply tree mapping on X_test
#                for j in range(X.shape[1]):  # Iterate over original features
#                    transformed_feats_for_j = tree_transformer.get_transformed_X_for_feat(tree_transformed_test,j,self.max_components)
#                    if add_linear:
#                        transformed_feats_for_j = np.hstack([X_test[:, [j]] - np.mean(X_test[:, j]),
#                                                                      transformed_feats_for_j])
#                    if transformed_feats_for_j is None:
#                        r_squared[i, j] = 0.0
#                        num_components_chosen[i, j] = 0
#                    else:
#                        clf = LassoCV().fit(transformed_feats_for_j,y_test - np.mean(y_test))
#                        r_squared[i, j] = clf.score(transformed_feats_for_j,y_test - np.mean(y_test))
#                        #r2_score(y_test,clf.predict(#clf.score(transformed_feats_for_j,y_test)
#            r_squared = np.mean(r_squared, axis=0)
#            return r_squared

# def get_r_squared_ridge_RFE(self, X, y, num_splits=10, add_linear=True):
#            r_squared = np.zeros((num_splits, X.shape[1]))
#            for i in tqdm(range(num_splits)):
#                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5,random_state=i)  # perform sample splitting
#                self.estimator.fit(X_train, y_train)  # fit on half of sample to learn tree structure and features
#            # if self.max_components == 'median':
#                tree_transformer = TreeTransformer(estimator=self.estimator, max_components=self.max_components)
#                tree_transformer.fit(X_train,always_pca = False)  # Apply PCA on X_train
#                tree_transformed_test = tree_transformer.transform(X_test)
# transformed_feats = tree_transformer.transform(X_test)  # apply tree mapping on X_test
#                for j in range(X.shape[1]):  # Iterate over original features
#                    transformed_feats_for_j = tree_transformer.get_transformed_X_for_feat(tree_transformed_test,j,self.max_components)
#                    if add_linear:
#                        transformed_feats_for_j = np.hstack([X_test[:, [j]] - np.mean(X_test[:, j]),
#                                                                      transformed_feats_for_j])
#                    if transformed_feats_for_j is None:
#                        r_squared[i, j] = 0.0
#                        num_components_chosen[i, j] = 0
#                    else:
#                        clf = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1,10.0,100.0,500.0]).fit(transformed_feats_for_j,y_test - np.mean(y_test))
#                        if transformed_feats_for_j.shape[1] == 1:
#                            with warnings.catch_warnings():
#                                warnings.filterwarnings("ignore")
#                                r_squared[i, j] = clf.score(transformed_feats_for_j,y_test - np.mean(y_test))
#                        else:
#                            with warnings.catch_warnings():
#                                warnings.filterwarnings("ignore")
##                                selector = RFECV(clf, step=max(0.1*transformed_feats_for_j.shape[1],1), cv=3)
#                               selector = selector.fit(transformed_feats_for_j,y_test - np.mean(y_test) )
#                               r_squared[i, j] = selector.score(transformed_feats_for_j,y_test - np.mean(y_test))
#                       #r2_score(y_test,clf.predict(#clf.score(transformed_feats_for_j,y_test)
#           r_squared = np.mean(r_squared, axis=0)
#           return r_squared

# def get_r_squared_r2_forward(self, X, y, num_splits=10, add_linear=True):
#            r_squared = np.zeros((num_splits, X.shape[1]))
#            for i in tqdm(range(num_splits)):
#                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5,
#                                                                random_state=i)  # perform sample splitting
#                self.estimator.fit(X_train, y_train)  # fit on half of sample to learn tree structure and features
#            # if self.max_components == 'median':
#                tree_transformer = TreeTransformer(estimator=self.estimator, max_components=self.max_components)
##                tree_transformer.fit(X_train,always_pca = False)  # Apply PCA on X_train
#               tree_transformed_test = tree_transformer.transform(X_test)
#           # transformed_feats = tree_transformer.transform(X_test)  # apply tree mapping on X_test
#               for j in range(X.shape[1]):  # Iterate over original features
#                   transformed_feats_for_j = tree_transformer.get_transformed_X_for_feat(tree_transformed_test,j,self.max_components)
#                   if add_linear:
##                       transformed_feats_for_j = np.hstack([X_test[:, [j]] - np.mean(X_test[:, j]),
#                                                                    transformed_feats_for_j])
#                  if transformed_feats_for_j is None:
#                      r_squared[i, j] = 0.0
#                      num_components_chosen[i, j] = 0
#                  else:
#                      with warnings.catch_warnings():
#                          warnings.filterwarnings("ignore")
##                          clf = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1,10.0,100.0,500.0]).fit(transformed_feats_for_j,y_test - np.mean(y_test))
##                         r_squared[i, j] = clf.score(transformed_feats_for_j,y_test - np.mean(y_test))
# r2_score(y_test,clf.predict(#clf.score(transformed_feats_for_j,y_test)
#        r_squared = np.mean(r_squared, axis=0)
#        return r_squared
