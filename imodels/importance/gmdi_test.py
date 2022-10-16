import numpy as np
import random
import scipy as sp

from sklearn.linear_model import Ridge, LinearRegression, RidgeCV, LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, log_loss, roc_auc_score

import imodels.importance.representation_cleaned as rep_new
import imodels.importance.r2f_exp_cleaned as gmdi_new
import imodels.importance.representation as rep_old
import imodels.importance.r2f_experimental as gmdi_old

class TestRepresentation:

    def setup(self):
        np.random.seed(42)
        random.seed(42)
        self.p = 10
        self.n = 10
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
        if transformed_x1_new.shape[1] == 0:
            assert transformed_x1_old is None
        else:
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
        if transformed_x1_new.shape[1] == 0:
            assert transformed_x1_old is None
        else:
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
        if transformed_x1_new.shape[1] == 0:
            assert transformed_x1_old is None
        else:
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
        if transformed_x1_new.shape[1] == 0:
            assert transformed_x1_old is None
        else:
            assert np.all(np.isclose(transformed_x1_old, transformed_x1_new))

        transformed_data_old = old_transformer.transform(self.X, center=True)
        transformed_data_new = new_transformer.transform(self.X).get_all_data()
        assert np.all(np.isclose(transformed_data_old, transformed_data_new))


class TestLOOParams:
    """
    Check if new LOO PPM computed using closed form formulas is the same as computing the values manually.
    """

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

    def manual_LOO_coefs(self, model, return_intercepts=False, center=False):
        loo_coefs = []
        loo_intercepts = []
        for i in range(self.n):
            train_indices = [j != i for j in range(self.n)]
            if center:
                X = self.X - self.X.mean(axis=0)
            else:
                X = self.X
            X_partial = X[train_indices, :]
            y_partial = self.y[train_indices]
            model.fit(X_partial, y_partial)
            loo_coefs.append(model.coef_)
            loo_intercepts.append(model.intercept_)
        if return_intercepts:
            return np.array(loo_coefs), np.array(loo_intercepts)
        else:
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

    def test_partial_predictions_ridge(self):
        """
        Need to center original X first
        New code automatically centers it
        """
        ridge_ppm = gmdi_new.RidgeLOOPPM(alpha_grid=[1])
        ridge = Ridge(alpha=1, fit_intercept=True)
        blocked_data = rep_new.IdentityTransformer(self.p).transform(self.X)
        ridge_ppm.fit(blocked_data, self.y, blocked_data)
        gmdi_pps = ridge_ppm.get_partial_predictions(0)
        manual_params, manual_intercepts = self.manual_LOO_coefs(ridge, return_intercepts=True, center=True)
        manual_pps = (self.X[:, 0] - self.X[:, 0].mean()) * manual_params[:, 0] + manual_intercepts

        assert np.all(np.isclose(manual_pps, gmdi_pps))


class TestNewVOldLOOPPM:
    """
    Need to add a line of code to the old JointRidgeScorer code to keep track of the fitted alpha hyperparameter.

    """

    def setup(self):
        np.random.seed(42)
        random.seed(42)
        self.p = 20
        self.n = 50
        self.beta = np.array([1] + [0] * (self.p - 1))
        self.sigma = 1
        self.X = np.random.randn(self.n, self.p)
        self.blocked_data = rep_new.IdentityTransformer(self.p).transform(self.X)
        self.y = self.X @ self.beta + self.sigma * np.random.randn(self.n)

    def test_ridge_ppm_alpha(self):
        """
        Check if fitted alphas are the same between old and new code.

        Settings for old code:
        - sample_weight = None

        """
        old_scorer = gmdi_old.JointRidgeScorer(metric="loocv")
        new_scorer = gmdi_new.RidgeLOOPPM(alpha_grid=np.logspace(-4, 3, 100))
        old_scorer.fit(self.X, self.y, start_indices=range(self.p), sample_weight=None)
        new_scorer.fit(self.blocked_data, self.y, self.blocked_data)

        old_alpha_ = old_scorer.alpha
        new_alpha_ = new_scorer.alpha_
        assert np.isclose(old_alpha_, new_alpha_)

    def test_ridge_ppm_looe(self):
        """
        Check if the old method for getting LOO residuals is the same as that for the new method

        Note: Need to center X before handing it to the old scorer
        """
        old_scorer = gmdi_old.JointRidgeScorer(metric="loocv")
        new_scorer = gmdi_new.RidgeLOOPPM(alpha_grid=np.logspace(-4, 3, 100))
        old_scorer.fit(self.X, self.y, start_indices=range(self.p), sample_weight=None)
        new_scorer.fit(self.blocked_data, self.y, self.blocked_data)

        old_alpha_ = old_scorer.alpha
        new_alpha_ = new_scorer.alpha_
        assert np.isclose(old_alpha_, new_alpha_)

        ridge = Ridge(alpha=old_alpha_)
        X = self.X - self.X.mean(axis=0)
        ridge.fit(X, self.y)
        old_beta_ = ridge.coef_
        old_intercept_ = ridge.intercept_
        old_errors = self._get_partial_model_looe_OLD(X, self.y, range(self.p + 1), old_alpha_, old_beta_,
                                                      old_intercept_)
        new_preds = np.array([new_scorer.get_partial_predictions(k) for k in range(self.p)])
        new_errors = (new_preds - self.y).T

        assert np.all(np.isclose(old_errors, new_errors))

    def test_ridge_ppm_scores(self):
        """
        Check if the old method for getting LOO scores is the same as that for the new method

        Note: Need to center X before handing it to the old scorer
        """
        old_scorer = gmdi_old.JointRidgeScorer(metric="loocv")
        new_scorer = gmdi_new.RidgeLOOPPM(alpha_grid=np.logspace(-4, 3, 100))
        old_scorer.fit(self.X - self.X.mean(axis=0), self.y, start_indices=range(self.p), sample_weight=None)
        new_scorer.fit(self.blocked_data, self.y, self.blocked_data)

        old_alpha_ = old_scorer.alpha
        new_alpha_ = new_scorer.alpha_
        assert np.isclose(old_alpha_, new_alpha_)

        old_score = old_scorer.get_score(0)
        new_score = r2_score(self.y, new_scorer.get_partial_predictions(0))

        assert np.isclose(old_score, new_score)

    def _get_partial_model_looe_OLD(self, X, y, start_indices, alpha, beta, intercept):
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


class TestGMDI:

    def setup(self):
        np.random.seed(42)
        random.seed(42)
        self.p = 10
        self.n = 100
        self.beta = np.array([1] + [0] * (self.p - 1))
        self.sigma = 1
        self.X = np.random.randn(self.n, self.p)
        self.y = self.X @ self.beta + self.sigma * np.random.randn(self.n)
        # self.tree_model = DecisionTreeRegressor(max_leaf_nodes=5)
        # self.tree_model.fit(self.X, self.y)
        self.one_tree_forest = RandomForestRegressor(max_features="auto", min_samples_leaf=5, n_estimators=1,
                                                     bootstrap=False)
        self.one_tree_forest.fit(self.X, self.y)
        self.tree_model = self.one_tree_forest.estimators_[0]
        self.rf_model = RandomForestRegressor(max_features=0.33, min_samples_leaf=5, n_estimators=5)
        self.rf_model.fit(self.X, self.y)

    def test_one_tree_transformation(self):
        old_transformer = rep_old.TreeTransformer(estimator=self.tree_model, pca=False, add_raw=True,
                                                  normalize_raw=True)
        new_transformer = rep_new.CompositeTransformer([rep_new.IdentityTransformer(self.p),
                                                        rep_new.TreeTransformer(self.p, self.tree_model)], adj_std="max")
        transformed_x1_old = old_transformer.transform_one_feature(self.X, 0, center=True)
        transformed_x1_new = new_transformer.transform_one_feature(self.X, 0)
        if transformed_x1_new.shape[1] == 0:
            assert transformed_x1_old is None
        else:
            assert np.all(np.isclose(transformed_x1_old, transformed_x1_new))

        transformed_data_old = old_transformer.transform(self.X)
        transformed_data_new = new_transformer.transform(self.X).get_all_data()
        assert np.all(np.isclose(transformed_data_old, transformed_data_new))

    def test_single_tree_looe(self):
        new_transformer = rep_new.CompositeTransformer([rep_new.IdentityTransformer(self.p),
                                                        rep_new.TreeTransformer(self.p, self.tree_model)], adj_std="max")
        new_ppm = gmdi_new.RidgeLOOPPM(alpha_grid=np.logspace(-4, 3, 100), fixed_intercept=True)
        blocked_data = new_transformer.transform(self.X)
        new_ppm.fit(blocked_data, self.y, blocked_data, self.y)

        alpha = new_ppm.alpha_
        old_transformer = rep_old.TreeTransformer(estimator=self.tree_model, pca=False, add_raw=True,
                                                  normalize_raw=True)
        X_transformed_old, start_indices_old = old_transformer.transform(self.X, center=True, return_indices=True)
        ridge = Ridge(alpha=alpha)
        ridge.fit(X_transformed_old, self.y)
        old_beta_ = ridge.coef_
        old_intercept_ = ridge.intercept_
        old_looe = self._get_partial_model_looe_OLD(X_transformed_old, self.y, start_indices_old,
                                                    alpha, old_beta_, old_intercept_)
        new_looe = np.zeros((self.n, self.p))
        for k in range(self.p):
            new_preds = new_ppm.get_partial_predictions(k)
            new_looe[:, k] = new_preds - self.y

        assert np.all(np.isclose(old_looe, new_looe))

    def test_single_tree_gmdi(self):
        new_transformer = rep_new.CompositeTransformer([rep_new.IdentityTransformer(self.p),
                                                        rep_new.TreeTransformer(self.p, self.tree_model)], adj_std="max")
        new_ppm = gmdi_new.RidgeLOOPPM(alpha_grid=np.logspace(-4, 3, 100), fixed_intercept=True)
        gmdi_obj_new = gmdi_new.GMDI(new_transformer, new_ppm, r2_score)
        new_scores = gmdi_obj_new.get_scores(self.X, self.y)

        old_settings = {"normalize_raw": True,
                        "oob": False}
        old_scorer = gmdi_old.JointRidgeScorer(metric="loocv")

        gmdi_obj_old = gmdi_old.GeneralizedMDIJoint(self.one_tree_forest, scorer=old_scorer, **old_settings)
        old_scores = gmdi_obj_old.get_importance_scores(self.X, self.y)

        old_alpha = gmdi_obj_old.scorer.alpha
        new_alpha = gmdi_obj_new.partial_prediction_model.alpha_
        assert old_alpha == new_alpha

        assert np.all(np.isclose(old_scores, new_scores))

    def test_single_tree_looe_bootstrap(self):
        one_tree_forest = RandomForestRegressor(max_features="auto", min_samples_leaf=5, n_estimators=1,
                                                bootstrap=True)
        one_tree_forest.fit(self.X, self.y)
        tree_model = one_tree_forest.estimators_[0]
        new_transformer = rep_new.CompositeTransformer([rep_new.IdentityTransformer(self.p),
                                                        rep_new.TreeTransformer(self.p, tree_model)], adj_std="max")
        new_ppm = gmdi_new.RidgeLOOPPM(alpha_grid=np.logspace(-4, 3, 100), fixed_intercept=True)
        blocked_data = new_transformer.transform(self.X)
        new_ppm.fit(blocked_data, self.y, blocked_data, self.y)

        alpha = new_ppm.alpha_
        old_transformer = rep_old.TreeTransformer(estimator=tree_model, pca=False, add_raw=True,
                                                  normalize_raw=True)
        X_transformed_old, start_indices_old = old_transformer.transform(self.X, center=True, return_indices=True)
        ridge = Ridge(alpha=alpha)
        ridge.fit(X_transformed_old, self.y)
        old_beta_ = ridge.coef_
        old_intercept_ = ridge.intercept_
        old_looe = self._get_partial_model_looe_OLD(X_transformed_old, self.y, start_indices_old,
                                                    alpha, old_beta_, old_intercept_)
        new_looe = np.zeros((self.n, self.p))
        for k in range(self.p):
            new_preds = new_ppm.get_partial_predictions(k)
            new_looe[:, k] = new_preds - self.y

        assert np.all(np.isclose(old_looe, new_looe))

    def test_single_tree_gmdi_bootstrap(self):
        one_tree_forest = RandomForestRegressor(max_features="auto", min_samples_leaf=5, n_estimators=1,
                                                bootstrap=True)
        one_tree_forest.fit(self.X, self.y)
        tree_model = one_tree_forest.estimators_[0]
        new_transformer = rep_new.CompositeTransformer([rep_new.IdentityTransformer(self.p),
                                                        rep_new.TreeTransformer(self.p, tree_model)], adj_std="max")
        new_ppm = gmdi_new.RidgeLOOPPM(alpha_grid=np.logspace(-4, 3, 100), fixed_intercept=True)
        blocked_data = new_transformer.transform(self.X)
        new_ppm.fit(blocked_data, self.y, blocked_data, self.y)
        new_looe = np.zeros((self.n, self.p))
        for k in range(self.p):
            new_preds = new_ppm.get_partial_predictions(k)
            new_looe[:, k] = new_preds - self.y
        gmdi_obj_new = gmdi_new.GMDI(new_transformer, new_ppm, r2_score)
        new_scores = gmdi_obj_new.get_scores(self.X, self.y)

        manual_scores = 1 - np.sum(new_looe ** 2, axis=0) / (np.var(self.y) * len(self.y))

        assert np.all(np.isclose(manual_scores, new_scores))

        old_settings = {"normalize_raw": True,
                        "oob": False}
        old_scorer = gmdi_old.JointRidgeScorer(metric="loocv")

        gmdi_obj_old = gmdi_old.GeneralizedMDIJoint(one_tree_forest, scorer=old_scorer, **old_settings)
        old_scores = gmdi_obj_old.get_importance_scores(self.X, self.y)

        # old_alpha = gmdi_obj_old.scorer.alpha
        # new_alpha = gmdi_obj_new.partial_prediction_model.alpha_
        # assert old_alpha == new_alpha

        assert np.all(np.isclose(old_scores, new_scores))

    def test_gmdi_ensemble(self):
        new_transformer_list = [rep_new.CompositeTransformer([
            rep_new.IdentityTransformer(self.p), rep_new.TreeTransformer(self.p, estimator)], adj_std="max")
            for estimator in self.rf_model.estimators_]
        new_ppm = gmdi_new.RidgeLOOPPM(alpha_grid=np.logspace(-4, 3, 100), fixed_intercept=True)
        gmdi_obj_new = gmdi_new.GMDIEnsemble(new_transformer_list, new_ppm, r2_score)
        new_scores = gmdi_obj_new.get_scores(self.X, self.y)

        old_settings = {"normalize_raw": True,
                        "oob": False}
        old_scorer = gmdi_old.JointRidgeScorer(metric="loocv")
        gmdi_obj_old = gmdi_old.GeneralizedMDIJoint(self.rf_model, scorer=old_scorer, **old_settings)
        old_scores = gmdi_obj_old.get_importance_scores(self.X, self.y)

        assert np.all(np.isclose(old_scores, new_scores))

    def test_gmdi_pipeline(self):
        """
        Need to change the initialization of RidgeLOOPPM in GMDI_pipeline to:
        partial_prediction_model = RidgeLOOPPM(alpha_grid=np.logspace(-4, 3, 100), fixed_intercept=True)
        """
        new_scores = gmdi_new.GMDI_pipeline(self.X, self.y, self.rf_model)["importance"]
        old_scorer = gmdi_old.JointRidgeScorer(metric="loocv")
        old_settings = {"normalize_raw": True,
                        "oob": False}
        gmdi_obj_old = gmdi_old.GeneralizedMDIJoint(self.rf_model, scorer=old_scorer, **old_settings)
        old_scores = gmdi_obj_old.get_importance_scores(self.X, self.y)
        # [ 4.09734644e-01 -1.00398643e-02 -1.26126245e-02 -1.42087148e-02, -2.35163131e-03 -1.02161585e-03
        # -2.52998766e-03 -8.33837613e-03, -5.28337568e-03  4.36833381e-05]

        assert np.all(np.isclose(old_scores, new_scores))

    def _get_partial_model_looe_OLD(self, X, y, start_indices, alpha, beta, intercept):
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
                looe_vals[:, k] = intercept - y
        return looe_vals

# X_new = new_transformer.transform_one_feature(self.X, 8)
#
# old_transformer = rep_old.TreeTransformer(estimator=self.tree_model, pca=False, add_raw=True,
# normalize_raw=True)
# X_old = old_transformer.transform_one_feature(self.X, 8)
# assert np.all(np.isclose(X_new, X_old)

class TestLOOvsNonLOOGMDI:

    def setup(self):
        np.random.seed(42)
        random.seed(42)
        self.p = 10
        self.n = 100
        self.beta = np.array([1] + [0] * (self.p - 1))
        self.sigma = 1
        self.X = np.random.randn(self.n, self.p)
        self.y = self.X @ self.beta + self.sigma * np.random.randn(self.n)
        # self.tree_model = DecisionTreeRegressor(max_leaf_nodes=5)
        # self.tree_model.fit(self.X, self.y)
        self.one_tree_forest = RandomForestRegressor(max_features="auto", min_samples_leaf=5, n_estimators=1,
                                                     bootstrap=False)
        self.one_tree_forest.fit(self.X, self.y)
        self.tree_model = self.one_tree_forest.estimators_[0]
        self.rf_model = RandomForestRegressor(max_features=0.33, min_samples_leaf=5, n_estimators=5)
        self.rf_model.fit(self.X, self.y)

    def test_single_tree_loo_v_nloo_ridge(self):
        new_transformer = rep_new.CompositeTransformer([rep_new.IdentityTransformer(self.p),
                                                        rep_new.TreeTransformer(self.p, self.tree_model)], adj_std="max")
        loo_ppm = gmdi_new.RidgeLOOPPM(alpha_grid=np.logspace(-4, 3, 100), fixed_intercept=True)
        nonloo_ppm = gmdi_new.RidgePPM(alphas=np.logspace(-4, 3, 100))
        gmdi_loo = gmdi_new.GMDI(new_transformer, loo_ppm, r2_score)
        gmdi_nonloo = gmdi_new.GMDI(new_transformer, nonloo_ppm, r2_score)
        gmdi_loo.get_scores(self.X, self.y)
        gmdi_nonloo.get_scores(self.X, self.y)

        loo_alpha = gmdi_loo.partial_prediction_model.alpha_
        nloo_alpha = gmdi_nonloo.partial_prediction_model.estimator.alpha_
        assert loo_alpha == nloo_alpha

class TestLOOIntercept:

    def setup(self):
        np.random.seed(42)
        random.seed(42)
        self.p = 10
        self.n = 100
        self.beta = np.array([1] + [0] * (self.p - 1))
        self.sigma = 1
        self.X = np.random.randn(self.n, self.p)
        self.y = self.X @ self.beta + self.sigma * np.random.randn(self.n)
        # self.tree_model = DecisionTreeRegressor(max_leaf_nodes=5)
        # self.tree_model.fit(self.X, self.y)
        self.one_tree_forest = RandomForestRegressor(max_features="auto", min_samples_leaf=5, n_estimators=1,
                                                     bootstrap=False)
        self.one_tree_forest.fit(self.X, self.y)
        self.tree_model = self.one_tree_forest.estimators_[0]
        self.rf_model = RandomForestRegressor(max_features=0.33, min_samples_leaf=5, n_estimators=5)
        self.rf_model.fit(self.X, self.y)

    def test_nonfixed_intercept(self):
        new_transformer = rep_new.CompositeTransformer([rep_new.IdentityTransformer(self.p),
                                                        rep_new.TreeTransformer(self.p, self.tree_model)], adj_std="max")
        loo_ppm = gmdi_new.RidgeLOOPPM(alpha_grid=np.logspace(-4, 3, 100), fixed_intercept=False)
        gmdi_loo = gmdi_new.GMDI(new_transformer, loo_ppm, r2_score)
        scores = gmdi_loo.get_scores(self.X, self.y)

        assert (scores[1] == scores[6]) and (scores[1] == scores[9])


class TestALooCalculator:

    def setup(self):
        np.random.seed(42)
        random.seed(42)
        self.p = 10
        self.n = 100
        self.beta = np.array([1] + [0] * (self.p - 1))
        self.sigma = 1
        self.X = np.random.randn(self.n, self.p)
        self.y = self.X @ self.beta + self.sigma * np.random.randn(self.n)
        self.y_log = np.random.binomial(1, sp.special.expit(self.y))

    def manual_LOO_coefs(self, model, log=False, return_intercepts=False, center=False):
        loo_coefs = []
        loo_intercepts = []
        for i in range(self.n):
            train_indices = [j != i for j in range(self.n)]
            if center:
                X = self.X - self.X.mean(axis=0)
            else:
                X = self.X
            X_partial = X[train_indices, :]
            if log:
                y_partial = self.y_log[train_indices]
            else:
                y_partial = self.y[train_indices]
            model.fit(X_partial, y_partial)
            coef_, intercept_ = gmdi_new.extract_coef_and_intercept(model)
            loo_coefs.append(coef_)
            loo_intercepts.append(intercept_)
        if return_intercepts:
            return np.array(loo_coefs), np.array(loo_intercepts)
        else:
            return np.array(loo_coefs)

    def test_ridge_LOO_coefs(self):
        linear_loo = gmdi_new.GlmAlooCalculator(LinearRegression())
        computed = linear_loo.get_aloo_fitted_parameters(self.X, self.y)
        true_coefs, true_intercepts = self.manual_LOO_coefs(LinearRegression(), return_intercepts=True)
        true = np.hstack([true_coefs, true_intercepts[:, np.newaxis]])
        assert np.all(np.isclose(computed.T, true))

    def test_ridge_hyperparameter_opt(self):
        ridge_cv = RidgeCV(alphas=np.logspace(-4, 4, 50))
        ridge_loo = gmdi_new.GlmAlooCalculator(Ridge(), alpha_grid=np.logspace(-4, 4, 50))
        computed = ridge_loo.get_aloocv_alpha(self.X, self.y)
        ridge_cv.fit(self.X, self.y)
        true = ridge_cv.alpha_
        assert true == computed

    def test_logistic_LOO_coefs(self):
        log_loo = gmdi_new.GlmAlooCalculator(LogisticRegression(C=1))
        computed = log_loo.get_aloo_fitted_parameters(self.X, self.y_log, alpha=1)
        true_coefs, true_intercepts = self.manual_LOO_coefs(LogisticRegression(C=1), log=True, return_intercepts=True)
        true = np.hstack([true_coefs, true_intercepts[:, np.newaxis]])
        # assert np.all(np.isclose(computed.T, true))


class TestRidgeGMDI:

    def setup(self):
        np.random.seed(42)
        random.seed(42)
        self.p = 10
        self.n = 100
        self.beta = np.array([1] + [0] * (self.p - 1))
        self.sigma = 1
        self.X = np.random.randn(self.n, self.p)
        self.y = self.X @ self.beta + self.sigma * np.random.randn(self.n)
        # self.tree_model = DecisionTreeRegressor(max_leaf_nodes=5)
        # self.tree_model.fit(self.X, self.y)
        self.rf_model = RandomForestRegressor(max_features=0.33, min_samples_leaf=5, n_estimators=5)
        self.rf_model.fit(self.X, self.y)

    def test_oob_nonloo(self):
        new_transformer_list = [rep_new.CompositeTransformer([
            rep_new.IdentityTransformer(self.p), rep_new.TreeTransformer(self.p, estimator)], adj_std="max")
            for estimator in self.rf_model.estimators_]
        new_ppm = gmdi_new.RidgePPM(alphas=np.logspace(-4, 3, 100))
        gmdi_obj_new = gmdi_new.GMDIEnsemble(new_transformer_list, new_ppm, r2_score, oob=True)
        new_scores_oob = gmdi_obj_new.get_scores(self.X, self.y)
        print(new_scores_oob)
        gmdi_obj_new = gmdi_new.GMDIEnsemble(new_transformer_list, new_ppm, r2_score, oob=False)
        new_scores = gmdi_obj_new.get_scores(self.X, self.y)
        print(new_scores)
        return

    def test_oob_loo(self):
        new_transformer_list = [rep_new.CompositeTransformer([
            rep_new.IdentityTransformer(self.p), rep_new.TreeTransformer(self.p, estimator)], adj_std="max")
            for estimator in self.rf_model.estimators_]
        new_ppm = gmdi_new.RidgeLOOPPM()
        gmdi_obj_new = gmdi_new.GMDIEnsemble(new_transformer_list, new_ppm, r2_score, oob=False)
        new_scores = gmdi_obj_new.get_scores(self.X, self.y)
        print(new_scores)
        gmdi_obj_new = gmdi_new.GMDIEnsemble(new_transformer_list, new_ppm, r2_score, oob=True)
        new_scores_oob = gmdi_obj_new.get_scores(self.X, self.y)
        print(new_scores_oob)
        return


class TestLogisticGMDI:

    def setup(self):
        np.random.seed(42)
        random.seed(42)
        self.p = 10
        self.n = 100
        self.beta = np.array([1] + [0] * (self.p - 1))
        self.sigma = 1
        self.X = np.random.randn(self.n, self.p)
        score = self.X @ self.beta + self.sigma * np.random.randn(self.n)
        self.y = np.random.binomial(1, sp.special.expit(score))
        self.one_tree_forest = RandomForestClassifier(max_features="auto", min_samples_leaf=5, n_estimators=1,
                                                      bootstrap=False)
        self.one_tree_forest.fit(self.X, self.y)
        self.tree_model = self.one_tree_forest.estimators_[0]
        self.rf_model = RandomForestClassifier(max_features=0.33, min_samples_leaf=5, n_estimators=5)
        self.rf_model.fit(self.X, self.y)

    def test_setup(self):
        log_ppm = gmdi_new.LogisticPPM()
        log_ppm.fit(self.X, self.y)

    def test_ppm(self):
        blocked_data = rep_new.IdentityTransformer(self.p).transform(self.X)
        train_indices = np.arange(70)
        test_indices = np.arange(70, 100)
        train_blocked_data, test_blocked_data = blocked_data.train_test_split(train_indices, test_indices)
        y_train = self.y[train_indices]
        y_test = self.y[test_indices]
        ppm = gmdi_new.LogisticPPM(alphas=np.logspace(-4, 3, 100))
        ppm.fit(train_blocked_data, y_train, test_blocked_data, y_test)
        scores = np.zeros(self.p)
        for k in range(self.p):
            partial_preds = ppm.get_partial_predictions(k)
            scores[k] = log_loss(y_test, partial_preds)
        return scores


    def test_oob_nonloo(self):
        new_transformer_list = [rep_new.CompositeTransformer([
            rep_new.IdentityTransformer(self.p), rep_new.TreeTransformer(self.p, estimator)], adj_std="max")
            for estimator in self.rf_model.estimators_]
        new_ppm = gmdi_new.LogisticPPM(alphas=np.logspace(-4, 3, 100))
        gmdi_obj_new = gmdi_new.GMDIEnsemble(new_transformer_list, new_ppm, roc_auc_score, oob=True)
        new_scores_oob = gmdi_obj_new.get_scores(self.X, self.y)
        print(new_scores_oob)
        gmdi_obj_new = gmdi_new.GMDIEnsemble(new_transformer_list, new_ppm, roc_auc_score, oob=False)
        new_scores = gmdi_obj_new.get_scores(self.X, self.y)
        print(new_scores)
        return