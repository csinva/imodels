import unittest

import numpy as np
from imodels.rule_list.greedy_rule_list import GreedyRuleListClassifier
import sklearn
from sklearn.model_selection import train_test_split

class TestGRL(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m  = GreedyRuleListClassifier()

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
        self.m.fit(X, y)
        yhat = self.m.predict(X)
        acc = np.mean(y == yhat) * 100
        assert acc > 99 # acc must be 100

    def test_linear_separability(self):
        """Test if the model can learn a linearly separable dataset"""
        x = np.array([0.8, 0.8, 0.3, 0.3, 0.3, 0.3]).reshape(-1, 1)
        y = np.array([0, 0, 1, 1, 1, 1])
        self.m.fit(x, y, verbose=True)
        yhat = self.m.predict(x)
        acc = np.mean(y == yhat) * 100
        assert len(self.m.rules_) == 2  
        assert acc == 100 # acc must be 100

    def test_y_left_conditional_probability(self):
        """Test conditional probability of y given x in the left node"""
        x = np.array([0.8, 0.8, 0.3, 0.3, 0.3, 0.3]).reshape(-1, 1)
        y = np.array([0, 0, 1, 1, 1, 1])
        self.m.fit(x, y, verbose=True)
        assert self.m.rules_[1]["val"] == 0

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