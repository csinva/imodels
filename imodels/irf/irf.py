from irf import irf_utils # installed from https://github.com/Yu-Group/iterative-Random-Forest
from irf.ensemble import wrf, RandomForestClassifierWithWeights # https://github.com/Yu-Group/iterative-Random-Forest
import numpy as np

class IRFClassifier():
    def __init__(self):
        self.model = wrf()
        self.predict = self.model.predict
        self.predict_proba = self.model.predict_proba
    
    
    def fit(self, X, y, lambda_reg=0.1, sample_weight=None):
        '''fit a linear model with integer coefficient and L1 regularization
        
        Params
        ------
        sample_weight: np.ndarray (n,)
            weight for each individual sample
        '''
        if 'pandas' in str(type(X)):
            X = X.values
        if 'pandas' in str(type(y)):
            y = y.values
        assert type(X) == np.ndarray, 'inputs should be ndarrays'
        assert type(y) == np.ndarray, 'inputs should be ndarrays'
        
        self.model.fit(X, y, keep_record=False)
        
        
        
        