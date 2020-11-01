import numpy as np

from imodels import GreedyRuleListClassifier, IRFClassifier, SkopeRulesClassifier, BayesianRuleListClassifier


class TestClassClassification:
    def setup(self):
        np.random.seed(13)
        self.n = 30
        self.p = 5
        self.X_classification_binary = np.random.randn(self.n, self.p)
        self.X_classification_binary_brl = (self.X_classification_binary > 0).astype(int)
        self.y_classification_binary = (self.X_classification_binary[:, 0] > 0).astype(int)

    def test_classification_binary(self):
        '''Test imodels on basic binary classification task
        '''
        for model_type in [GreedyRuleListClassifier, IRFClassifier, SkopeRulesClassifier]:
            if model_type == BayesianRuleListClassifier:
                X = self.X_classification_binary_brl
            else:
                X = self.X_classification_binary
#             print(model_type, X)
            m = model_type()
            m.fit(X, self.y_classification_binary)

            # if not model_type in [SkopeRulesClassifier]:
            preds_proba = m.predict_proba(X)
#             for i in range(20):
#                 print(i, self.X_classification_binary[i], preds_proba[i])
            # print('preds_proba', preds_proba)
            assert len(preds_proba.shape) == 2, 'preds_proba has columns'
            assert np.max(preds_proba) < 1.1, 'preds_proba has no values over 1'

            preds = m.predict(X) # > 0.5).astype(int)
            assert preds.size == self.n, 'predictions are right size'
            #             print(model_type, preds, self.y_classification_binary)
            for i in range(20):
                print(i, self.y_classification_binary[i], preds_proba[i], preds[i])

            acc_train = np.mean(preds == self.y_classification_binary)
            print(type(m), m)
            print('acc', acc_train)
            assert acc_train > 0.8, 'acc greater than 0.8'
