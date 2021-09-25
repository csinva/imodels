import os
import random
import unittest
from os.path import join as oj
from random import sample

from pandas.io.parsers import read_csv

from imodels import *

test_dir = os.path.dirname(os.path.abspath(__file__))


class TestBoaClassifier(unittest.TestCase):
    def test_boa_tictactoe(self):
        '''Test classifiers are properly sklearn-compatible
        '''

        df = read_csv(oj(test_dir, 'test_data', 'tictactoe_X.txt'), header=0, sep=" ")
        Y = np.loadtxt(open(oj(test_dir, 'test_data', 'tictactoe_Y.txt'), "rb"), delimiter=" ")

        lenY = len(Y)
        idxs_train = sample(range(lenY), int(0.50 * lenY))
        idxs_test = [i for i in range(lenY) if i not in idxs_train]
        y_test = Y[idxs_test]
        model = BOAClassifier(n_rules=100,
                              supp=5,
                              maxlen=3,
                              num_iterations=100,
                              num_chains=2,
                              alpha_pos=500, beta_pos=1,
                              alpha_neg=500, beta_neg=1,
                              alpha_l=None, beta_l=None)

        # fit and check accuracy
        np.random.seed(13)
        random.seed(13)
        model.fit(df.iloc[idxs_train], Y[idxs_train])
        y_pred = model.predict(df.iloc[idxs_test])
        acc1 = np.mean(y_pred == y_test)
        assert acc1 > 0.8

        # try fitting np version
        np.random.seed(13)
        random.seed(13)
        model.fit(df.iloc[idxs_train].values, Y[idxs_train])
        y_pred = model.predict(df.iloc[idxs_test].values)
        y_test = Y[idxs_test]
        acc2 = np.mean(y_pred == y_test)
        assert acc2 > 0.8

        # assert np.abs(acc1 - acc2) < 0.05 # todo: fix seeding

