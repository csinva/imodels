import random

from imodels import *


class TestClassClassificationBinary:
    '''Tests simple classification for different models. Note: still doesn't test all the models!
    '''

    def setup(self):
        np.random.seed(13)
        random.seed(13)
        self.n = 40
        self.p = 2
        self.X_classification_binary = np.random.randn(self.n, self.p)
        self.y_classification_binary = (self.X_classification_binary[:, 0] > 0).astype(int)  # y = x0 > 0
        self.y_classification_binary[-2:] = 1 - self.y_classification_binary[-2:]  # flip labels for last few

    def test_classification_binary(self):
        '''Test imodels on basic binary classification task
        '''
        for model_type in [
            RuleFitClassifier, GreedyRuleListClassifier,
            FPLassoClassifier, SkopeRulesClassifier,
            FPSkopeClassifier, BoostedRulesClassifier,
            OneRClassifier, SlipperClassifier,
        ]:  # IRFClassifier, SLIMClassifier, BOAClassifier

            init_kwargs = {}
            if model_type == RuleFitClassifier:
                init_kwargs['max_rules'] = 5
            if model_type == SkopeRulesClassifier or model_type == FPSkopeClassifier:
                init_kwargs['random_state'] = 0
                init_kwargs['max_samples_features'] = 1.
            if model_type == SlipperClassifier:
                init_kwargs['n_estimators'] = 1
            m = model_type(**init_kwargs)

            X = self.X_classification_binary
            m.fit(X, self.y_classification_binary)

            # test predict()
            preds = m.predict(X)  # > 0.5).astype(int)
            assert preds.size == self.n, 'predict() yields right size'

            # test preds_proba()
            if not model_type == CorelsRuleListClassifier:
                preds_proba = m.predict_proba(X)
                assert len(preds_proba.shape) == 2, 'preds_proba has 2 columns'
                assert preds_proba.shape[1] == 2, 'preds_proba has 2 columns'
                assert np.max(preds_proba) < 1.1, 'preds_proba has no values over 1'
                assert (np.argmax(preds_proba, axis=1) == preds).all(), "predict_proba and predict correspond"

            # test acc
            acc_train = np.mean(preds == self.y_classification_binary)
            print(type(m), m, 'final acc', acc_train)
            assert acc_train > 0.8, 'acc greater than 0.8'


if __name__ == '__main__':
    t = TestClassClassificationBinary()
    t.setup()
    t.test_classification_binary()
