import os
import unittest

import numpy as np
from scipy.io.arff import loadarff
from sklearn.model_selection import train_test_split

from imodels.rule_list.bayesian_rule_list.bayesian_rule_list import BayesianRuleListClassifier

path_to_tests = os.path.dirname(os.path.realpath(__file__))


class TestBRL(unittest.TestCase):

    def test_integration_stability(self):
        '''Test on synthetic dataset
        '''
        X = np.array([[0, 0, 1, 1, 0],
                      [1, 0, 0, 0, 0],
                      [0, 0, 1, 0, 0],
                      [1, 0, 0, 0, 0],
                      [1, 1, 0, 1, 1],
                      [1, 1, 1, 1, 1],
                      [0, 1, 1, 1, 1],
                      [1, 0, 1, 1, 1]])
        y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        M = BayesianRuleListClassifier(minsupport=0.02, maxcardinality=1)
        feat = ['ft1', 'ft2', 'ft3', 'ft4', 'ft5']
        M.fit(X, y, feature_names=feat)
        assert (np.array([M.predict(np.array([row]), threshold=0.5) for row in X]).flatten() == y).all()

    def test_integration_fitting(self):
        '''Test on a real (small) dataset
        '''
        np.random.seed(13)
        feature_names = ["#Pregnant", "Glucose concentration test", "Blood pressure(mmHg)",
                         "Triceps skin fold thickness(mm)",
                         "2-Hour serum insulin (mu U/ml)", "Body mass index", "Diabetes pedigree function",
                         "Age (years)"]

        data = loadarff(os.path.join(path_to_tests, "test_data/diabetes.arff"))
        data_np = np.array(list(map(lambda x: np.array(list(x)), data[0])))
        X, y_text = data_np[:, :-1].astype('float32'), data_np[:, -1].astype('str')
        y = (y_text == 'tested_positive').astype(int)  # labels 0-1

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.75)  # split

        # train classifier (allow more iterations for better accuracy; use BigDataRuleListClassifier for large datasets)
        print('training...')
        model = BayesianRuleListClassifier(max_iter=1000, minsupport=0.4, maxcardinality=1, class1label="diabetes",
                                           verbose=False)
        model.fit(X_train, y_train, feature_names=feature_names)
        preds = model.predict(X_test, threshold=0.1)
        print("RuleListClassifier Accuracy:", np.mean(y_test == preds), "Learned interpretable model:\n", model)
