import numpy as np

from imodels import GreedyRuleListClassifier, SkopeRulesClassifier, BayesianRuleListClassifier, \
    OneRClassifier, BoostedRulesClassifier, RuleFitClassifier, FPLassoClassifier, FPSkopeClassifier  # IRFClassifier


class TestClassClassificationBinary:
    '''Tests simple classification for different models. Note: still doesn't test BRL!
    '''

    def setup(self):
        np.random.seed(13)
        self.n = 40
        self.p = 2
        self.X_classification_binary = np.random.randn(self.n, self.p)
        self.X_classification_binary_brl = (self.X_classification_binary > 0).astype(str)
        self.y_classification_binary = (self.X_classification_binary[:, 0] > 0).astype(int) # y = x0 > 0
        self.y_classification_binary[-2:] = 1 - self.y_classification_binary[-2:]  # flip labels for last few

    def test_classification_binary(self):
        '''Test imodels on basic binary classification task
        '''
        for model_type in [RuleFitClassifier, GreedyRuleListClassifier,
                           FPLassoClassifier, SkopeRulesClassifier,
                           FPSkopeClassifier, BoostedRulesClassifier, 
                           OneRClassifier]:  # IRFClassifier, 

            init_kwargs = {}
            if model_type == RuleFitClassifier:
                init_kwargs['max_rules'] = 5
            if model_type == SkopeRulesClassifier or model_type == FPSkopeClassifier:
                init_kwargs['random_state'] = 0
                init_kwargs['recall_min'] = 0.5
            m = model_type(**init_kwargs)

            if model_type == BayesianRuleListClassifier:
                X = self.X_classification_binary_brl
                m.fit(X, self.y_classification_binary,
                      feature_labels=[f'X{i}' for i in range(self.p)])
            else:
                X = self.X_classification_binary
                m.fit(X, self.y_classification_binary)

#             print('starting to test', type(m), '...')
            # print(m.visualize())
            preds_proba = m.predict_proba(X)
            #             for i in range(20):
            #                 print(i, self.X_classification_binary[i], preds_proba[i])
            # print('preds_proba', preds_proba)
            assert len(preds_proba.shape) == 2, 'preds_proba has 2 columns'
            assert preds_proba.shape[1] == 2, 'preds_proba has 2 columns'
            assert np.max(preds_proba) < 1.1, 'preds_proba has no values over 1'

            preds = m.predict(X)  # > 0.5).astype(int)
            assert preds.size == self.n, 'predict() yields right size'
            assert (np.argmax(preds_proba, axis=1) == preds).all(), "predict_proba and predict correspond"
            
            #             print(model_type, preds, self.y_classification_binary)
            # for i in range(20):
            #     print(i, self.y_classification_binary[i], preds_proba[i], preds[i])

            acc_train = np.mean(preds == self.y_classification_binary)
            # print(type(m), m)
            print(type(m), m, 'final acc', acc_train)
            assert acc_train > 0.8, 'acc greater than 0.8'
