import numpy as np

from imodels import GreedyRuleListClassifier, SkopeRulesClassifier, BayesianRuleListClassifier, \
    OneRClassifier, BoostedRulesClassifier  # IRFClassifier


class TestClassClassification:
    '''Tests simple classification for different models. Note: still doesn't test BRL!
    '''

    def setup(self):
        np.random.seed(13)
        self.n = 40
        self.p = 2
        self.X_classification_binary = np.random.randn(self.n, self.p)
        self.X_classification_binary_brl = (self.X_classification_binary > 0).astype(str)
        self.y_classification_binary = (self.X_classification_binary[:, 0] > 0).astype(int)
        self.y_classification_binary[-2:] = 1 - self.y_classification_binary[-2:]  # flip labels for last few

    def test_classification_binary(self):
        '''Test imodels on basic binary classification task
        '''
        for model_type in [GreedyRuleListClassifier, SkopeRulesClassifier, BoostedRulesClassifier,
                           OneRClassifier]:  # IRFClassifier

            m = model_type()

            if model_type == BayesianRuleListClassifier:
                X = self.X_classification_binary_brl
                m.fit(X, self.y_classification_binary,
                      feature_labels=[f'X{i}' for i in range(self.p)])
            else:
                X = self.X_classification_binary
                m.fit(X, self.y_classification_binary)

            print('starting to test', type(m), '...')
            preds_proba = m.predict_proba(X)
            #             for i in range(20):
            #                 print(i, self.X_classification_binary[i], preds_proba[i])
            # print('preds_proba', preds_proba)
            assert len(preds_proba.shape) == 2, 'preds_proba has columns'
            assert np.max(preds_proba) < 1.1, 'preds_proba has no values over 1'

            preds = m.predict(X)  # > 0.5).astype(int)
            assert preds.size == self.n, 'predictions are right size'
            #             print(model_type, preds, self.y_classification_binary)
            # for i in range(20):
            #     print(i, self.y_classification_binary[i], preds_proba[i], preds[i])

            acc_train = np.mean(preds == self.y_classification_binary)
            # print(type(m), m)
            print(type(m), m, 'final acc', acc_train)
            assert acc_train > 0.8, 'acc greater than 0.8'
