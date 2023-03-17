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
from imodels.importance.ppms import RidgePPM, LogisticPPM, RobustPPM
from imodels.importance.mdi_plus import GmdiHelper, GMDI


class TestTransformers:

    def setup(self):
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
        X0 = id_transformer.transform_one_feature(self.X, 0, center=False).\
            ravel()
        assert_array_equal(X0, self.X[:, 0])
        X_transformed = id_transformer.transform(self.X, center=False)
        assert_array_equal(X_transformed.get_all_data(), self.X)

    def test_tree_transformer(self):
        tree_transformer = TreeTransformer(self.tree_model)
        assert sum(tree_transformer.n_splits.values()) == self.n_internal_nodes
        lin_reg = LinearRegression()
        tree_rep = tree_transformer.transform(self.X).get_all_data()
        lin_reg.fit(tree_rep, self.y)
        assert_array_equal(lin_reg.predict(tree_rep),
                           self.tree_model.predict(self.X))

    def test_composite_transformer(self):
        composite_transformer = CompositeTransformer([IdentityTransformer(),
                                                      IdentityTransformer()])
        X0_doubled = composite_transformer.transform_one_feature(self.X, 0,
                                                                 center=False)
        assert X0_doubled.shape[1] == 2

    def test_gmdi_default(self):
        # Test number of engineered features without drop_features
        gmdi_transformer = MDIPlusDefaultTransformer(tree_model=self.tree_model,
                                                     drop_features=False)
        assert gmdi_transformer.transform(self.X).get_all_data().shape[1] == \
               self.p + self.n_internal_nodes
        # Test number of engineered features with drop_features
        gmdi_transformer = MDIPlusDefaultTransformer(tree_model=self.tree_model,
                                                     drop_features=True)

        assert gmdi_transformer.transform(self.X).get_all_data().shape[1] == \
               self.n_internal_nodes + \
               len(gmdi_transformer.block_transformer_list[0].n_splits)
        # Test scaling
        tree_transformer = TreeTransformer(self.tree_model)
        tree_transformer_max = max(
            tree_transformer.transform_one_feature(self.X, 0).std(axis=0))
        composite_transformer_rescaling = gmdi_transformer.\
            transform_one_feature(self.X, 0).std(axis=0)[3]
        assert np.isclose(tree_transformer_max, composite_transformer_rescaling)
        gmdi_transformer = MDIPlusDefaultTransformer(tree_model=self.tree_model,
                                                     rescale_mode="mean",
                                                     drop_features=True)
        tree_transformer_mean = np.mean(
            tree_transformer.transform_one_feature(self.X, 0).std(axis=0))
        composite_transformer_rescaling = gmdi_transformer. \
            transform_one_feature(self.X, 0).std(axis=0)[3]
        assert np.isclose(tree_transformer_mean, composite_transformer_rescaling)


class TestLOOParams:
    """
    Check if new LOO PPM computed using closed form formulas is the same as
    computing the values manually.
    """
    def setup(self):
        np.random.seed(42)
        random.seed(42)
        self.p = 10
        self.n = 100
        self.beta = np.array([1] + [0] * (self.p - 1))
        self.sigma = 1
        self.X = np.random.randn(self.n, self.p)
        self.blocked_data = IdentityTransformer().transform(self.X)
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
        linear_ppm = RidgePPM(loo=True, alpha_grid=[0])
        lr = LinearRegression(fit_intercept=True)
        manual_params, manual_intercepts = \
            self.manual_LOO_coefs(lr, return_intercepts=True)
        augmented_params = np.hstack([manual_params,
                                      manual_intercepts[:, np.newaxis]])
        gmdi_params = linear_ppm._fit_loo_coefficients(self.X, self.y, 0)
        assert_array_equal(augmented_params, gmdi_params)

    def test_loo_params_ridge(self):
        ridge_ppm = RidgePPM(loo=True, alpha_grid=[1])
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
        ridge_ppm = RidgePPM(loo=True, alpha_grid=[1])
        ridge = Ridge(alpha=1, fit_intercept=True)
        blocked_data = IdentityTransformer().transform(self.X)
        ridge_ppm.fit(blocked_data, self.y, blocked_data)
        for k in range(self.p):
            gmdi_pps = ridge_ppm.get_partial_predictions(k)
            manual_params, manual_intercepts = \
                self.manual_LOO_coefs(ridge, return_intercepts=True, center=True)
            manual_pps = (self.X[:, k] - self.X[:, k].mean()) * \
                         manual_params[:, k] + manual_intercepts
            assert_array_equal(manual_pps, gmdi_pps)


class TestPPM:

    def setup(self):
        np.random.seed(42)
        random.seed(42)
        self.p = 10
        self.n = 100
        self.beta = np.array([1] + [0] * (self.p - 1))
        self.sigma = 1
        self.X = np.random.randn(self.n, self.p)
        self.blocked_data = IdentityTransformer().transform(self.X)
        self.y = self.X @ self.beta + self.sigma * np.random.randn(self.n)
        self.y_bin = np.random.binomial(
            1, sp.special.expit(self.X @ self.beta), self.n)
        self.tree_model = DecisionTreeRegressor(max_leaf_nodes=5)
        self.tree_model.fit(self.X, self.y)

    def test_alpha_selection(self):
        ridge_ppm = RidgePPM(loo=True, alpha_grid=np.logspace(-4, 3, 100))
        ridge_ppm.fit(self.blocked_data, self.y, self.blocked_data)
        assert np.isclose(ridge_ppm.alpha_, 10.47615752789664)
        logistic_ppm = LogisticPPM(loo=True, alpha_grid=np.logspace(-4, 3, 100))
        logistic_ppm.fit(self.blocked_data, self.y_bin, self.blocked_data)
        assert np.isclose(logistic_ppm.alpha_, 8.902150854450374)

    def test_ridge_predictions(self):
        gmdi_transformer = MDIPlusDefaultTransformer(tree_model=self.tree_model)
        blocked_data = gmdi_transformer.transform(self.X)
        ridge_ppm = RidgePPM(loo=False, alpha_grid=np.logspace(-4, 3, 100))
        ridge_ppm.fit(blocked_data, self.y, blocked_data)
        # Test full prediction
        assert np.isclose(ridge_ppm.get_full_predictions()[0],
                          0.6686467658857475)
        # Test partial prediction
        assert np.isclose(ridge_ppm.get_partial_predictions(0)[0],
                          0.5306302415575942)
        # Test intercept model
        assert np.isclose(ridge_ppm.get_partial_predictions(1),
                          0.1637922129748298)

    def test_ridge_loo_predictions(self):
        gmdi_transformer = MDIPlusDefaultTransformer(tree_model=self.tree_model)
        blocked_data = gmdi_transformer.transform(self.X)
        ridge_ppm = RidgePPM(loo=True, alpha_grid=np.logspace(-4, 3, 100))
        ridge_ppm.fit(blocked_data, self.y, blocked_data)
        # Test full prediction
        assert np.isclose(ridge_ppm.get_full_predictions()[0],
                          0.6286095042288156)
        # Test partial prediction
        assert np.isclose(ridge_ppm.get_partial_predictions(0)[0],
                          0.49988326053782545)
        # Test intercept model
        assert np.isclose(ridge_ppm.get_partial_predictions(1),
                          0.1637922129748298)

    def test_logistic_loo_predictions(self):
        gmdi_transformer = MDIPlusDefaultTransformer(tree_model=self.tree_model)
        blocked_data = gmdi_transformer.transform(self.X)
        logistic_ppm = LogisticPPM(loo=True, alpha_grid=np.logspace(-4, 3, 100))
        logistic_ppm.fit(blocked_data, self.y_bin, blocked_data)
        # Test full prediction
        assert np.isclose(logistic_ppm.get_full_predictions()[0],
                          0.7065047799408872)
        # Test partial prediction
        assert np.isclose(logistic_ppm.get_partial_predictions(0)[0],
                          0.7693235069016788)
        # Test intercept model
        assert np.isclose(logistic_ppm.get_partial_predictions(1),
                          0.609994765464111)

    def test_robust_loo_predictions(self):
        gmdi_transformer = MDIPlusDefaultTransformer(tree_model=self.tree_model)
        blocked_data = gmdi_transformer.transform(self.X)
        robust_ppm = RobustPPM(loo=True, alpha_grid=np.logspace(-4, 3, 100))
        robust_ppm.fit(blocked_data, self.y, blocked_data)
        # Test full prediction
        assert np.isclose(robust_ppm.get_full_predictions()[0],
                          0.6575704560264011)
        # Test partial prediction
        assert np.isclose(robust_ppm.get_partial_predictions(0)[0],
                          0.4813493202027731)
        # Test intercept model
        assert np.isclose(robust_ppm.get_partial_predictions(1),
                          0.1531074473707865)


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
        self.y_bin = np.random.binomial(
            1, sp.special.expit(self.X @ self.beta), self.n)
        self.tree_model = DecisionTreeRegressor(max_leaf_nodes=5)
        self.tree_model.fit(self.X, self.y)
        self.rf_model = RandomForestRegressor(max_features=0.33,
                                              min_samples_leaf=5,
                                              n_estimators=5)
        self.rf_model.fit(self.X, self.y)

    def test_gmdi_helper(self):
        gmdi_transformer = MDIPlusDefaultTransformer(tree_model=self.tree_model)
        ridge_ppm = RidgePPM(loo=True, alpha_grid=np.logspace(-4, 3, 100))
        scoring_fns = r2_score
        gmdi_helper = GmdiHelper(gmdi_transformer, ridge_ppm, scoring_fns)
        scores = gmdi_helper.get_scores(self.X, self.y).values.ravel()
        true_scores = np.array([0.43619799667263814,
                                0, 0, 0,
                                0.041935066728947756,
                                0, 0,
                                -0.0073188385516917975,
                                0, 0])
        assert_array_equal(scores, true_scores)

    def test_gmdi_default(self):
        gmdi = GMDI(self.rf_model, refit_rf=False)
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
        gmdi = GMDI(self.rf_model, refit_rf=False, sample_split="oob")
        scores = gmdi.get_scores(self.X, self.y).importance.values
        true_scores = np.array([0.19717146364990595,
                                -0.11398750665049504,
                                -0.03840278088642357,
                                -0.03707154696242045,
                                -0.07515725468626222,
                                -0.055321005095176214,
                                -0.04356595360440514,
                                -0.04971877469991455,
                                -0.05301153482428722,
                                -0.07506932993987411])
        assert_array_equal(scores, true_scores)

    def test_multi_scoring(self):
        scoring_fns = {"log_loss": log_loss, "roc_auc": roc_auc_score}
        gmdi = GMDI(self.rf_model, refit_rf=False, task="classification",
                    scoring_fns=scoring_fns)
        scores = gmdi.get_scores(self.X, self.y_bin)
        assert scores.shape[1] == 3

    def test_multi_target(self):
        y_multi = np.random.multinomial(1, (0.3, 0.3, 0.4), self.n)
        rf_model = RandomForestClassifier(max_features=0.33,
                                          min_samples_leaf=5, n_estimators=5)
        rf_model.fit(self.X, y_multi)
        gmdi = GMDI(rf_model, refit_rf=False, task="classification",
                    scoring_fns=mean_squared_error)
        scores = gmdi.get_scores(self.X, y_multi)


def assert_array_equal(arr1, arr2):
    assert arr1.shape == arr2.shape, "Array shapes not equal"
    assert np.all(np.isclose(arr1, arr2)), "Entries not equal"