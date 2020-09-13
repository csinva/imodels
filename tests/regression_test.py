from imodels import RuleListClassifier, RuleFit, GreedyRuleList, SkopeRules, SLIM, IRFClassifier

import numpy as np

class TestClassRegression:
    def setup(self):
        np.random.seed(13)
        self.n = 20
        self.p = 5
        self.X_regression = np.random.randn(self.n, self.p)
        self.y_regression = self.X_regression[:, 0] + np.random.randn(self.n) * 0.01    
        
            
    def test_regression(self):
        '''Test imodels on basic binary classification task
        '''
        for model_type in [RuleFit, SLIM]:
            m = model_type()
            m.fit(self.X_regression, self.y_regression)
            
            preds = m.predict(self.X_regression)
            assert preds.size == self.n, 'predictions are right size'


            mse = np.mean(np.square(preds - self.y_regression))
            assert mse < 1 , 'mse less than 1'