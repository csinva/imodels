import random

import numpy as np

from imodels import FIGSClassifier, BoostedRulesClassifier, GreedyTreeClassifier, RuleFitClassifier, HSTreeClassifierCV


class TestAutogluon:
    '''Tests simple classification for different models. Note: still doesn't test all the models!
    '''

    def setup_method(self):
        np.random.seed(13)
        random.seed(13)
        self.n = 40
        self.p = 2
        self.X_classification_binary = (
            np.random.randn(self.n, self.p) > 0).astype(int)

        # y = x0 > 0
        self.y_classification_binary = (
            self.X_classification_binary[:, 0] > 0).astype(int)

        # flip labels for last few
        self.y_classification_binary[-2:] = 1 - \
            self.y_classification_binary[-2:]

    def test_printing_autogluon_models(self):
        '''Test imodels on basic binary classification task
        '''
        for model_type in [
            FIGSClassifier, BoostedRulesClassifier, GreedyTreeClassifier, RuleFitClassifier, HSTreeClassifierCV
        ]:

            init_kwargs = {}
            m = model_type(**init_kwargs)

            X = self.X_classification_binary
            m.fit(X, self.y_classification_binary)

            print(m)


if __name__ == '__main__':
    t = TestAutogluon()
    t.setup()
    t.test_printing_autogluon_models()
