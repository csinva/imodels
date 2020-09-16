import numpy as np

from imodels import GreedyRuleListClassifier, IRFClassifier, SkopeRulesClassifier


class TestClassClassification:
    def setup(self):
        np.random.seed(13)
        self.n = 20
        self.p = 5
        self.X_classification_binary = np.random.randn(self.n, self.p)
        self.y_classification_binary = (self.X_classification_binary[:, 0] > 0).astype(int)

    def test_classification_binary(self):
        '''Test imodels on basic binary classification task
        '''
        for model_type in [GreedyRuleListClassifier, IRFClassifier, SkopeRulesClassifier]:
            m = model_type()
            m.fit(self.X_classification_binary, self.y_classification_binary)

            # if not model_type in [SkopeRulesClassifier]:
            preds_proba = m.predict_proba(self.X_classification_binary)
            assert len(preds_proba.shape) == 2, 'preds_proba has columns'
            assert np.max(preds_proba) < 1.1, 'preds_proba has no values over 1'

            preds = (m.predict(self.X_classification_binary) > 0.5).astype(int)
            assert preds.size == self.n, 'predictions are right size'
            #             print(model_type, preds, self.y_classification_binary)

            acc_train = np.mean(preds == self.y_classification_binary)
            assert acc_train > 0.8, 'acc greater than 0.8'
