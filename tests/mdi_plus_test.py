import copy

import numpy as np
import random
import scipy as sp

from sklearn.linear_model import Ridge, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, log_loss, roc_auc_score, \
    mean_squared_error

from imodels.importance.block_transformers import IdentityTransformer, \
    TreeTransformer, CompositeTransformer, MDIPlusDefaultTransformer
from imodels.importance.ppms import RidgeRegressorPPM, \
    LogisticClassifierPPM, RobustRegressorPPM
from imodels.importance.mdi_plus import TreeMDIPlus, ForestMDIPlus
from imodels.importance.rf_plus import RandomForestPlusClassifier


class TestTransformers:

    def setup_method(self):
        np.random.seed(42)
        random.seed(42)
        self.p = 10
        self.n = 50
        self.beta = np.array([1] + [0] * (self.p - 1))
        self.sigma = 1
        self.X = np.random.randn(self.n, self.p)
        self.y = self.X @ self.beta + self.sigma * np.random.randn(self.n)

        self.tree_model = DecisionTreeRegressor(max_leaf_nodes=5)
        self.tree_model.fit(self.X, self.y)

        self.rf_model = RandomForestRegressor(max_features=0.33,
                                              min_samples_leaf=5,
                                              n_estimators=5)
        self.rf_model.fit(self.X, self.y)
        self.n_internal_nodes = (self.tree_model.tree_.node_count - 1) // 2

    def test_identity(self):
        id_transformer = IdentityTransformer()
        X0 = id_transformer.fit_transform_one_feature(self.X, 0, center=False).\
            ravel()
        assert_array_equal(X0, self.X[:, 0])
        X_transformed = id_transformer.fit_transform(self.X, center=False)
        assert_array_equal(X_transformed.get_all_data(), self.X)

    def test_tree_transformer(self):
        tree_transformer = TreeTransformer(self.tree_model)
        assert sum(tree_transformer.n_splits.values()) == self.n_internal_nodes
        lin_reg = LinearRegression()
        tree_rep = tree_transformer.fit_transform(self.X).get_all_data()
        lin_reg.fit(tree_rep, self.y)
        assert_array_equal(lin_reg.predict(tree_rep),
                           self.tree_model.predict(self.X))

    def test_composite_transformer(self):
        composite_transformer = CompositeTransformer([IdentityTransformer(),
                                                      IdentityTransformer()])
        X0_doubled = composite_transformer.fit_transform_one_feature(
            self.X, 0, center=False)
        assert X0_doubled.shape[1] == 2

    def test_gmdi_default(self):
        # Test number of engineered features without drop_features
        gmdi_transformer = MDIPlusDefaultTransformer(tree_model=self.tree_model,
                                                     drop_features=False)
        assert gmdi_transformer.fit_transform(self.X).get_all_data().shape[1] == \
            self.p + self.n_internal_nodes
        # Test number of engineered features with drop_features
        gmdi_transformer = MDIPlusDefaultTransformer(tree_model=self.tree_model,
                                                     drop_features=True)

        assert gmdi_transformer.fit_transform(self.X).get_all_data().shape[1] == \
            self.n_internal_nodes + \
            len(gmdi_transformer.block_transformer_list[0].n_splits)
        # Test scaling
        tree_transformer = TreeTransformer(self.tree_model)
        tree_transformer_max = max(
            tree_transformer.fit_transform_one_feature(self.X, 0).std(axis=0))
        composite_transformer_rescaling = gmdi_transformer. \
            fit_transform_one_feature(self.X, 0).std(axis=0)[3]
        assert np.isclose(tree_transformer_max,
                          composite_transformer_rescaling)
        gmdi_transformer = MDIPlusDefaultTransformer(tree_model=self.tree_model,
                                                     rescale_mode="mean",
                                                     drop_features=True)
        tree_transformer_mean = np.mean(
            tree_transformer.fit_transform_one_feature(self.X, 0).std(axis=0))
        composite_transformer_rescaling = gmdi_transformer. \
            fit_transform_one_feature(self.X, 0).std(axis=0)[3]
        assert np.isclose(tree_transformer_mean,
                          composite_transformer_rescaling)


class TestLOOParams:
    """
    Check if new LOO PPM computed using closed form formulas is the same as
    computing the values manually.
    """

    def setup_method(self):
        np.random.seed(42)
        random.seed(42)
        self.p = 10
        self.n = 100
        self.beta = np.array([1] + [0] * (self.p - 1))
        self.sigma = 1
        self.X = np.random.randn(self.n, self.p)
        self.blocked_data = IdentityTransformer().fit_transform(self.X)
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

    def test_loo_params_linear(self):
        linear_ppm = RidgeRegressorPPM(loo=True, alpha_grid=[0])
        lr = LinearRegression(fit_intercept=True)
        manual_params, manual_intercepts = \
            self.manual_LOO_coefs(lr, return_intercepts=True)
        augmented_params = np.hstack([manual_params,
                                      manual_intercepts[:, np.newaxis]])
        gmdi_params = linear_ppm._fit_loo_coefficients(self.X, self.y, 0)
        assert_array_equal(augmented_params, gmdi_params)

    def test_loo_params_ridge(self):
        ridge_ppm = RidgeRegressorPPM(loo=True, alpha_grid=[1])
        ridge = Ridge(alpha=1, fit_intercept=True)
        manual_params, manual_intercepts = \
            self.manual_LOO_coefs(ridge, return_intercepts=True)
        augmented_params = np.hstack([manual_params,
                                      manual_intercepts[:, np.newaxis]])
        gmdi_params = ridge_ppm._fit_loo_coefficients(self.X, self.y, 1)
        assert_array_equal(augmented_params, gmdi_params)

    def test_partial_predictions_ridge(self):
        """
        Check if partial predictions for the identity representation are
        correct. Note that we need to center original X first
        """
        ridge_ppm = RidgeRegressorPPM(loo=True, alpha_grid=[1])
        ridge = Ridge(alpha=1, fit_intercept=True)
        blocked_data = IdentityTransformer().fit_transform(self.X)
        ridge_ppm.fit(blocked_data.get_all_data(), self.y)
        for k in range(self.p):
            gmdi_pps = ridge_ppm.predict_partial_k(
                blocked_data, k, mode="keep_k")
            manual_params, manual_intercepts = \
                self.manual_LOO_coefs(
                    ridge, return_intercepts=True, center=True)
            manual_pps = (self.X[:, k] - self.X[:, k].mean()) * \
                manual_params[:, k] + manual_intercepts
            assert_array_equal(manual_pps, gmdi_pps)


class TestPPM:

    def setup_method(self):
        np.random.seed(42)
        random.seed(42)
        self.p = 10
        self.n = 100
        self.beta = np.array([1] + [0] * (self.p - 1))
        self.sigma = 1
        self.X = np.random.randn(self.n, self.p)
        self.blocked_data = IdentityTransformer().fit_transform(self.X)
        self.y = self.X @ self.beta + self.sigma * np.random.randn(self.n)
        self.y_bin = np.random.binomial(
            1, sp.special.expit(self.X @ self.beta), self.n)
        self.tree_model = DecisionTreeRegressor(max_leaf_nodes=5)
        self.tree_model.fit(self.X, self.y)

    def test_alpha_selection(self):
        ridge_ppm = RidgeRegressorPPM(
            loo=True, alpha_grid=np.logspace(-4, 3, 100))
        ridge_ppm.fit(self.blocked_data.get_all_data(), self.y)
        assert np.isclose(ridge_ppm.alpha_[0], 10.47615752789664)
        logistic_ppm = LogisticClassifierPPM(
            loo=True, alpha_grid=np.logspace(-4, 3, 100))
        logistic_ppm.fit(self.blocked_data.get_all_data(), self.y_bin)
        assert np.isclose(logistic_ppm.alpha_[0], 8.902150854450374)

    def test_ridge_predictions(self):
        gmdi_transformer = MDIPlusDefaultTransformer(
            tree_model=self.tree_model)
        blocked_data = gmdi_transformer.fit_transform(self.X)
        ridge_ppm = RidgeRegressorPPM(
            loo=False, alpha_grid=np.logspace(-4, 3, 100))
        ridge_ppm.fit(blocked_data.get_all_data(), self.y)
        # Test full prediction
        assert np.isclose(ridge_ppm.predict_full(blocked_data)[0],
                          0.6686467658857475)
        # Test partial prediction
        assert np.isclose(ridge_ppm.predict_partial_k(blocked_data, 0, mode="keep_k")[0],
                          0.5306302415575942)
        # Test intercept model
        assert np.isclose(ridge_ppm.predict_partial_k(blocked_data, 1, mode="keep_k")[1],
                          0.1637922129748298)

    def test_ridge_loo_predictions(self):
        gmdi_transformer = MDIPlusDefaultTransformer(
            tree_model=self.tree_model)
        blocked_data = gmdi_transformer.fit_transform(self.X)
        ridge_ppm = RidgeRegressorPPM(
            loo=True, alpha_grid=np.logspace(-4, 3, 100))
        ridge_ppm.fit(blocked_data.get_all_data(), self.y)
        # Test full prediction
        assert np.isclose(ridge_ppm.predict_full(blocked_data)[0],
                          0.6286095042288156)
        # Test partial prediction
        assert np.isclose(ridge_ppm.predict_partial_k(blocked_data, 0, mode="keep_k")[0],
                          0.49988326053782545)
        # Test intercept model
        assert np.isclose(ridge_ppm.predict_partial_k(blocked_data, 1, mode="keep_k")[1],
                          0.1637922129748298)

    def test_logistic_loo_predictions(self):
        gmdi_transformer = MDIPlusDefaultTransformer(
            tree_model=self.tree_model)
        blocked_data = gmdi_transformer.fit_transform(self.X)
        logistic_ppm = LogisticClassifierPPM(
            loo=True, alpha_grid=np.logspace(-4, 3, 100))
        logistic_ppm.fit(blocked_data.get_all_data(), self.y_bin)
        # Test full prediction
        # assert np.isclose(logistic_ppm.predict_full(blocked_data)[0],
        #   0.7065047799408872)
        # Test partial prediction
        # assert np.isclose(logistic_ppm.predict_partial_k(blocked_data, 0, mode="keep_k")[0],
        #                   0.7693235069016788)
        # # Test intercept model
        # assert np.isclose(logistic_ppm.predict_partial_k(blocked_data, 1, mode="keep_k")[1],
        #                   0.609994765464111)

    def test_robust_loo_predictions(self):
        gmdi_transformer = MDIPlusDefaultTransformer(
            tree_model=self.tree_model)
        blocked_data = gmdi_transformer.fit_transform(self.X)
        robust_ppm = RobustRegressorPPM(
            loo=True, alpha_grid=np.logspace(-4, 3, 100))
        robust_ppm.fit(blocked_data.get_all_data(), self.y)
        # Test full prediction
        assert np.isclose(robust_ppm.predict_full(blocked_data)[0],
                          0.6575704560264011)
        # Test partial prediction
        assert np.isclose(robust_ppm.predict_partial_k(blocked_data, 0, mode="keep_k")[0],
                          0.4813493202027731)
        # Test intercept model
        assert np.isclose(robust_ppm.predict_partial_k(blocked_data, 1, mode="keep_k")[1],
                          0.1531074473707865)


class TestMDIPlus:

    def setup_method(self):
        np.random.seed(42)
        random.seed(42)
        self.p = 10
        self.n = 100
        self.beta = np.array([1] + [0] * (self.p - 1))
        self.sigma = 1
        self.X = np.random.randn(self.n, self.p)
        self.y = self.X @ self.beta + self.sigma * np.random.randn(self.n)
        self.y_bin = np.random.binomial(
            1, sp.special.expit(self.X @ self.beta), self.n)
        self.tree_model = DecisionTreeRegressor(max_leaf_nodes=5)
        self.tree_model.fit(self.X, self.y)
        self.rf_model = RandomForestRegressor(max_features=0.33,
                                              min_samples_leaf=5,
                                              n_estimators=5)
        self.rf_model.fit(self.X, self.y)

    def test_tree_mdi_plus(self):
        tree_transformer = MDIPlusDefaultTransformer(
            tree_model=self.tree_model)
        blocked_data = tree_transformer.fit_transform(self.X)
        ridge_ppm = RidgeRegressorPPM(
            loo=True, alpha_grid=np.logspace(-4, 3, 100))
        ridge_ppm.fit(blocked_data.get_all_data(), self.y)
        scoring_fns = r2_score
        tree_mdi = TreeMDIPlus(ridge_ppm, tree_transformer, scoring_fns,
                               tree_random_state=self.tree_model.random_state)
        scores = tree_mdi.get_scores(self.X, self.y).values.ravel()
        true_scores = np.array([0.43619799667263814,
                                0, 0, 0,
                                0.041935066728947756,
                                0, 0,
                                -0.0073188385516917975,
                                0, 0])
        assert_array_equal(scores, true_scores)

    def test_gmdi_default(self):
        ridge_ppm = RidgeRegressorPPM()
        rf_transformers = []
        rf_ppms = []
        tree_random_states = []
        for tree_model in self.rf_model.estimators_:
            transformer = MDIPlusDefaultTransformer(tree_model)
            blocked_data = transformer.fit_transform(self.X)
            rf_transformers.append(transformer)
            ridge_ppm.fit(blocked_data.get_all_data(), self.y)
            rf_ppms.append(copy.deepcopy(ridge_ppm))
            tree_random_states.append(tree_model.random_state)
        scoring_fns = {"importance": r2_score}
        gmdi = ForestMDIPlus(rf_ppms, rf_transformers, scoring_fns,
                             tree_random_states=tree_random_states)
        scores = gmdi.get_scores(self.X, self.y).importance.values
        true_scores = np.array([0.22712585381651848,
                                -0.021441281161664084,
                                -0.008501908090243582,
                                -0.008645314550603267,
                                -0.004325418217144428,
                                -0.0037645517797257667,
                                -0.0038558468903281628,
                                -0.0034596658742244825,
                                -0.014174201624713011,
                                -0.006400747217417635])
        assert_array_equal(scores, true_scores)

    def test_gmdi_oob(self):
        ridge_ppm = RidgeRegressorPPM(loo=False)
        rf_transformers = []
        rf_ppms = []
        tree_random_states = []
        for tree_model in self.rf_model.estimators_:
            transformer = MDIPlusDefaultTransformer(tree_model)
            blocked_data = transformer.fit_transform(self.X)
            rf_transformers.append(transformer)
            ridge_ppm.fit(blocked_data.get_all_data(), self.y)
            rf_ppms.append(copy.deepcopy(ridge_ppm))
            tree_random_states.append(tree_model.random_state)
        scoring_fns = {"importance": r2_score}
        gmdi = ForestMDIPlus(rf_ppms, rf_transformers, scoring_fns,
                             tree_random_states=tree_random_states,
                             sample_split="oob")
        scores = gmdi.get_scores(self.X, self.y).importance.values
        true_scores = np.array([0.24973548,
                                -0.02194494,
                                -0.01844932,
                                -0.01626793,
                                -0.022296,
                                0.01004052,
                                0.00181714,
                                -0.01403385,
                                -0.01361916,
                                -0.00903695])
        assert_array_equal(scores, true_scores)

    def test_multi_scoring(self):
        ridge_ppm = RidgeRegressorPPM()
        rf_transformers = []
        rf_ppms = []
        tree_random_states = []
        for tree_model in self.rf_model.estimators_:
            transformer = MDIPlusDefaultTransformer(tree_model)
            blocked_data = transformer.fit_transform(self.X)
            rf_transformers.append(transformer)
            ridge_ppm.fit(blocked_data.get_all_data(), self.y)
            rf_ppms.append(copy.deepcopy(ridge_ppm))
            tree_random_states.append(tree_model.random_state)
        scoring_fns = {"log_loss": log_loss, "roc_auc": roc_auc_score}
        gmdi = ForestMDIPlus(rf_ppms, rf_transformers, scoring_fns,
                             tree_random_states=tree_random_states)
        scores = gmdi.get_scores(self.X, self.y_bin)
        assert scores.shape[1] == 3

    def test_multi_target(self):
        y_multi = np.random.multinomial(1, (0.3, 0.3, 0.4), self.n)
        rf_model = RandomForestClassifier(
            max_features=0.33, min_samples_leaf=5, n_estimators=5)
        rf_plus_model = RandomForestPlusClassifier(rf_model)
        rf_plus_model.fit(self.X, y_multi)
        scores = rf_plus_model.get_mdi_plus_scores(
            self.X, y_multi, scoring_fns=mean_squared_error)


def assert_array_equal(arr1, arr2):
    assert arr1.shape == arr2.shape, "Array shapes not equal"
    assert np.all(np.isclose(arr1, arr2)), "Entries not equal"
