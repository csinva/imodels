import unittest

import numpy as np

from imodels.rule_list.greedy_rule_list import GreedyRuleListClassifier


class TestGRL(unittest.TestCase):

    def test_integration_stability(self):
        '''Test on synthetic dataset
        '''
        X = np.array(
            [[0, 0, 1, 1, 0],
             [1, 0, 0, 0, 0],
             [0, 0, 1, 0, 0],
             [1, 0, 0, 0, 0],
             [1, 1, 0, 1, 1],
             [1, 1, 1, 1, 1],
             [0, 1, 1, 1, 1],
             [1, 0, 1, 1, 1]])
        y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        m = GreedyRuleListClassifier()
        m.fit(X, y)
        yhat = m.predict(X)
        acc = np.mean(y == yhat) * 100
        assert acc > 99, 'acc must be 100'
