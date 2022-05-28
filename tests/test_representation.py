import os
import random

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from imodels.importance import TreeTransformer

class TestTreeTransformer:

    def setup(self):
        '''Test on synthetic dataset
        '''
        np.random.seed(13)
        random.seed(13)
        self.n = 200
        self.p = 5
        self.X = np.random.randn(self.n, self.p)
        self.beta = np.ones(5) * 10
        self.y = self.X @ self.beta + np.random.randn(self.n)

    def test_fitting(self):
        rf_model = RandomForestRegressor(n_estimators=100, min_samples_leaf=5, max_features=0.33)
        rf_model.fit(self.X, self.y)
        tree_transformer = TreeTransformer(rf_model)
        tree_transformer.fit(self.X)

        assert len(tree_transformer.pca_transformers) == self.p