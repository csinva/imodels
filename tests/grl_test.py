import unittest
import traceback

import numpy as np
from sklearn.metrics import accuracy_score
from imodels.rule_list.greedy_rule_list import GreedyRuleListClassifier
import sklearn
from sklearn.model_selection import train_test_split

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

def test_breast_cancer():
    np.random.seed(13)
    X, Y = sklearn.datasets.load_breast_cancer(as_frame=True, return_X_y=True)
    model = GreedyRuleListClassifier(max_depth=10)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3)
    model.fit(X_train, y_train, feature_names=X_train.columns)
    y_pred = model.predict(X_test)
    # score = accuracy_score(y_test.values,y_pred)
    # print('Accuracy:', score)
    # model._print_list()

if __name__ == '__main__':
    test_breast_cancer()