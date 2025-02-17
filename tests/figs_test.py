import os
import random
from functools import partial

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

from imodels import FIGSClassifier, FIGSRegressor, FIGSClassifierCV, FIGSRegressorCV
from imodels.experimental.figs_ensembles import FIGSExtRegressor, FIGSExtClassifier
from sklearn.ensemble import StackingRegressor, VotingRegressor, BaggingClassifier

path_to_tests = os.path.dirname(os.path.realpath(__file__))


class TestFIGS:

    def setup_method(self):
        '''Test on synthetic dataset
        '''
        np.random.seed(13)
        random.seed(13)
        self.n = 100
        self.p = 2
        self.X = (np.random.randn(self.n, self.p) > 0).astype(int)

        # y = x0 > 0 * x1 > 0
        self.y_classification_binary = (self.X[:, 0] > 0).astype(int) * (
            self.X[:, 1] > 0).astype(int)
        self.y_reg = self.X[:, 0] + self.X[:, 1]

    def test_recognized_by_sklearn(self):
        base_models = [('figs', FIGSRegressor()),
                       ('random_forest', DecisionTreeRegressor())]
        comb_model = VotingRegressor(estimators=base_models,
                                     n_jobs=10,
                                     verbose=2)
        comb_model.fit(self.X, self.y_reg)

    def test_categorical(self):
        """Test FIGS with categorical data"""
        categories = ['cat', 'dog', 'bird', 'fish']
        categories_2 = ['bear', 'chicken', 'cow']

        self.X_cat = pd.DataFrame(self.X)
        self.X_cat['pet1'] = np.random.choice(categories, size=(self.n, 1))
        self.X_cat['pet2'] = np.random.choice(categories_2, size=(self.n, 1))

        figs_reg = FIGSRegressor()
        figs_cls = FIGSClassifier()

        figs_reg.fit(self.X_cat, self.y_reg,
                     categorical_features=["pet1", 'pet2'])
        figs_reg.predict(self.X_cat, categorical_features=["pet1", 'pet2'])

        figs_cls.fit(self.X_cat, self.y_reg,
                     categorical_features=["pet1", 'pet2'])
        figs_cls.predict_proba(
            self.X_cat, categorical_features=["pet1", 'pet2'])

    def test_fitting(self):
        '''Test on a real (small) dataset
        '''
        for model_type in [
            FIGSClassifier, FIGSRegressor,
            FIGSExtClassifier, FIGSExtRegressor,
            FIGSClassifierCV, FIGSRegressorCV,
            partial(BaggingClassifier,
                    estimator=FIGSExtClassifier(max_rules=3),
                    n_estimators=2),
        ]:

            init_kwargs = {}
            m = model_type(**init_kwargs)

            X = self.X
            m.fit(X, self.y_classification_binary)

            # test predict()
            preds = m.predict(X)  # > 0.5).astype(int)
            assert preds.size == self.n, 'predict() yields right size'

            # test preds_proba()
            if model_type in [FIGSClassifier, FIGSClassifierCV, BaggingClassifier]:
                preds_proba = m.predict_proba(X)
                assert len(preds_proba.shape) == 2, 'preds_proba has 2 columns'
                assert preds_proba.shape[1] == 2, 'preds_proba has 2 columns'
                assert np.max(
                    preds_proba) < 1.1, 'preds_proba has no values over 1'
                assert (np.argmax(preds_proba, axis=1) == preds).all(), ("predict_proba and "
                                                                         "predict correspond")

            # test acc
            acc_train = np.mean(preds == self.y_classification_binary)
            assert acc_train > 0.9, 'acc greater than 0.9'
            # print(m)

            if not type(m) in [FIGSClassifierCV, FIGSRegressorCV, BaggingClassifier]:
                trees = m.trees_
                assert len(trees) == 1, 'only one tree'
                assert trees[0].feature == 1, 'split on feat 1'
                #assert np.abs(trees[0].left.value[0]) < 0.01, 'left value 0'
                assert trees[0].left.left is None and trees[0].left.right is None, 'left is leaf'
                #assert np.abs(
                #    trees[0].right.left.value[0]) < 0.01, 'right-left value 0'
                #assert np.abs(trees[0].right.right.value[0] -
                #              1) < 0.01, 'right-right value 1'


if __name__ == '__main__':
    t = TestFIGS()
    t.setup_method()
    t.test_fitting()
    t.test_categorical()
