import numpy as np
import random

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

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

        self.rf_model = RandomForestRegressor(max_features=0.33, n_estimators=5)
        self.rf_model.fit(self.X, self.y)

    def test_single_tree_representation_no_normalization(self):
        # Note that centering is enabled for both old and new code (default for new code does centering,
        # default for old code does not.

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