import numpy as np
import random

from sklearn.linear_model import Ridge, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

import imodels.importance.representation_cleaned as rep_new
import imodels.importance.r2f_exp_cleaned as gmdi_new
import imodels.importance.representation as rep_old
import imodels.importance.r2f_experimental as gmdi_old

class TestRepresentation:

    def setup(self):
        np.random.seed(42)
        random.seed(42)
        self.p = 10
        self.n = 100
        self.beta = np.array([1] + [0] * (self.p - 1))
        self.sigma = 1
        self.X = np.random.randn(self.n, self.p)
        self.y = self.X @ self.beta + self.sigma * np.random.randn(self.n)

        self.tree_model = DecisionTreeRegressor(max_leaf_nodes=5)
        self.tree_model.fit(self.X, self.y)

        self.rf_model = RandomForestRegressor(max_features=0.33, min_samples_leaf=5, n_estimators=5)
        self.rf_model.fit(self.X, self.y)

    def test_single_tree_representation_no_normalization(self):
        # Note that centering is enabled for both old and new code (default for new code does centering,
        # default for old code does not.)

        old_transformer = rep_old.TreeTransformer(estimator=self.tree_model, pca=False, add_raw=True,
                                                  normalize_raw=False)
        new_transformer = rep_new.CompositeTransformer([rep_new.IdentityTransformer(self.p),
                                                        rep_new.TreeTransformer(self.p, self.tree_model)], adj_std=None)
        transformed_x1_old = old_transformer.transform_one_feature(self.X, 0, center=True)
        transformed_x1_new = new_transformer.transform_one_feature(self.X, 0)
        assert np.all(np.isclose(transformed_x1_old, transformed_x1_new))
        transformed_data_old = old_transformer.transform(self.X, center=True)
        transformed_data_new = new_transformer.transform(self.X).get_all_data()
        assert np.all(np.isclose(transformed_data_old, transformed_data_new))

    def test_single_tree_representation_normalize_raw(self):
        old_transformer = rep_old.TreeTransformer(estimator=self.tree_model, pca=False, add_raw=True,
                                                  normalize_raw=True)
        new_transformer = rep_new.CompositeTransformer([rep_new.IdentityTransformer(self.p),
                                                        rep_new.TreeTransformer(self.p, self.tree_model)], adj_std="max")
        transformed_x1_old = old_transformer.transform_one_feature(self.X, 0, center=True)
        transformed_x1_new = new_transformer.transform_one_feature(self.X, 0)
        assert np.all(np.isclose(transformed_x1_old, transformed_x1_new))

        transformed_data_old = old_transformer.transform(self.X)
        transformed_data_new = new_transformer.transform(self.X).get_all_data()
        assert np.all(np.isclose(transformed_data_old, transformed_data_new))

    def test_rf_representation_no_normalization(self):
        estimator = self.rf_model.estimators_[0]
        old_transformer = rep_old.TreeTransformer(estimator=estimator, pca=False, add_raw=True,
                                                  normalize_raw=False)
        new_transformer = rep_new.CompositeTransformer([rep_new.IdentityTransformer(self.p),
                                                        rep_new.TreeTransformer(self.p, estimator)], adj_std=None)
        transformed_x1_old = old_transformer.transform_one_feature(self.X, 0, center=True)
        transformed_x1_new = new_transformer.transform_one_feature(self.X, 0)
        assert np.all(np.isclose(transformed_x1_old, transformed_x1_new))
        transformed_data_old = old_transformer.transform(self.X, center=True)
        transformed_data_new = new_transformer.transform(self.X).get_all_data()
        assert np.all(np.isclose(transformed_data_old, transformed_data_new))

    def test_rf_representation_normalize_raw(self):
        estimator = self.rf_model.estimators_[0]
        old_transformer = rep_old.TreeTransformer(estimator=estimator, pca=False, add_raw=True,
                                                  normalize_raw=True)
        new_transformer = rep_new.CompositeTransformer([rep_new.IdentityTransformer(self.p),
                                                        rep_new.TreeTransformer(self.p, estimator)], adj_std="max")
        transformed_x1_old = old_transformer.transform_one_feature(self.X, 0, center=True)
        transformed_x1_new = new_transformer.transform_one_feature(self.X, 0)
        assert np.all(np.isclose(transformed_x1_old, transformed_x1_new))

        transformed_data_old = old_transformer.transform(self.X, center=True)
        transformed_data_new = new_transformer.transform(self.X).get_all_data()
        assert np.all(np.isclose(transformed_data_old, transformed_data_new))


class TestLOOParams:

    def setup(self):
        np.random.seed(42)
        random.seed(42)
        self.p = 10
        self.n = 100
        self.beta = np.array([1] + [0] * (self.p - 1))
        self.sigma = 1
        self.X = np.random.randn(self.n, self.p)
        self.blocked_data = rep_new.IdentityTransformer(self.p).transform(self.X)
        self.y = self.X @ self.beta + self.sigma * np.random.randn(self.n)

    def manual_LOO_coefs(self, model):
        loo_coefs = []
        for i in range(self.n):
            train_indices = [j != i for j in range(self.n)]
            X_partial = self.X[train_indices, :]
            y_partial = self.y[train_indices]
            model.fit(X_partial, y_partial)
            loo_coefs.append(model.coef_)
        return np.array(loo_coefs)

    def test_loo_params_linear_no_intercept(self):
        linear_ppm = gmdi_new.GenericLOOPPM(estimator=LinearRegression(), alpha_grid=[0])
        lr = LinearRegression(fit_intercept=False)
        manual_params = self.manual_LOO_coefs(lr)
        lr.fit(self.X, self.y)
        gmdi_params = linear_ppm._get_loo_fitted_parameters(self.X, self.y, lr.coef_).T

        assert np.all(np.isclose(manual_params, gmdi_params))

    def test_loo_params_linear_intercept(self):
        linear_ppm = gmdi_new.GenericLOOPPM(estimator=LinearRegression(), alpha_grid=[0])
        lr = LinearRegression(fit_intercept=True)
        manual_params = self.manual_LOO_coefs(lr)
        lr.fit(self.X, self.y)
        X_aug = np.hstack([self.X, np.ones((self.X.shape[0], 1))])
        coef_aug = np.array(list(lr.coef_) + [lr.intercept_])
        gmdi_params = linear_ppm._get_loo_fitted_parameters(X_aug, self.y, coef_aug).T[:, :-1]

        assert np.all(np.isclose(manual_params, gmdi_params))

    def test_loo_params_ridge_intercept(self):
        ridge_ppm = gmdi_new.GenericLOOPPM(estimator=Ridge(alpha=1), alpha_grid=[0])
        ridge = Ridge(alpha=1, fit_intercept=True)
        manual_params = self.manual_LOO_coefs(ridge)
        ridge.fit(self.X, self.y)
        X_aug = np.hstack([self.X, np.ones((self.X.shape[0], 1))])
        coef_aug = np.array(list(ridge.coef_) + [ridge.intercept_])
        gmdi_params = ridge_ppm._get_loo_fitted_parameters(X_aug, self.y, coef_aug, alpha=1).T[:, :-1]

        assert np.all(np.isclose(manual_params, gmdi_params))


class TestScorer:

    def setup(self):
        np.random.seed(42)
        random.seed(42)
        self.p = 10
        self.n = 100
        self.beta = np.array([1] + [0] * (self.p - 1))
        self.sigma = 1
        self.X = np.random.randn(self.n, self.p)
        self.blocked_data = rep_new.IdentityTransformer(self.p).transform(self.X)
        self.y = self.X @ self.beta + self.sigma * np.random.randn(self.n)

    def test_ridge_ppm(self):
        """
        Settings for old code:
        - sample_weight = None

        """
        old_scorer = gmdi_old.JointRidgeScorer(metric="loocv")
        new_scorer = gmdi_new.RidgeLOOPPM(alpha_grid=np.logspace(-4, 3, 100))
        old_scorer.fit(self.X, self.y, start_indices=range(self.p), sample_weight=None)
        new_scorer.fit(self.blocked_data, self.y)

        old_alpha_ = old_scorer.alpha
        new_alpha_ = new_scorer.alpha_
        assert np.isclose(old_alpha_, new_alpha_)

        ridge = Ridge(alpha=old_alpha_)
        ridge.fit(self.X, self.y)
        old_beta_ = ridge.coef_
        old_intercept_ = ridge.intercept_
        old_errors = _get_partial_model_looe_OLD(self.X, self.y, range(self.p), old_alpha_, old_beta_,
                                                old_intercept_)[:, 0]
        new_preds = new_scorer.get_partial_predictions(0)
        new_errors = self.y - new_preds

        assert np.all(np.isclose(old_errors, new_errors))

        old_score = old_scorer.get_score(0)
        new_score = r2_score(self.y, new_scorer.get_partial_predictions(0))

        assert np.isclose(old_score, new_score)


def _get_partial_model_looe_OLD(X, y, start_indices, alpha, beta, intercept):
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