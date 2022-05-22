import os
import random

import numpy as np
from sklearn.tree import DecisionTreeRegressor

from imodels import FIGSClassifier, FIGSRegressor, FIGSClassifierCV, FIGSRegressorCV
from imodels.experimental.figs_ensembles import FIGSExtRegressor, FIGSExtClassifier
from sklearn.ensemble import StackingRegressor,VotingRegressor
path_to_tests = os.path.dirname(os.path.realpath(__file__))


class TestFIGS:

    def setup(self):
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

    def test_fitting(self):
        '''Test on a real (small) dataset
        '''
        for model_type in [
            FIGSClassifier, FIGSRegressor,
            FIGSExtClassifier, FIGSExtRegressor,
            FIGSClassifierCV, FIGSRegressorCV,
        ]:

            init_kwargs = {}
            m = model_type(**init_kwargs)

            X = self.X
            m.fit(X, self.y_classification_binary)

            # test predict()
            preds = m.predict(X)  # > 0.5).astype(int)
            assert preds.size == self.n, 'predict() yields right size'

            # test preds_proba()
            if model_type in [FIGSClassifier]:
                preds_proba = m.predict_proba(X)
                assert len(preds_proba.shape) == 2, 'preds_proba has 2 columns'
                assert preds_proba.shape[1] == 2, 'preds_proba has 2 columns'
                assert np.max(preds_proba) < 1.1, 'preds_proba has no values over 1'
                assert (np.argmax(preds_proba, axis=1) == preds).all(), ("predict_proba and "
                                                                         "predict correspond")

            # test acc
            acc_train = np.mean(preds == self.y_classification_binary)
            assert acc_train > 0.8, 'acc greater than 0.9'
            # print(m)

            if not type(m) in [FIGSClassifierCV, FIGSRegressorCV]:
                trees = m.trees_
                assert len(trees) == 1, 'only one tree'
                assert trees[0].feature == 1, 'split on feat 1'
                assert np.abs(trees[0].left.value) < 0.01, 'left value 0'
                assert trees[0].left.left is None and trees[0].left.right is None, 'left is leaf'
                assert np.abs(trees[0].right.left.value) < 0.01, 'right-left value 0'
                assert np.abs(trees[0].right.right.value - 1) < 0.01, 'right-right value 1'


if __name__ == '__main__':
    t = TestFIGS()
    t.setup()
    t.test_fitting()
