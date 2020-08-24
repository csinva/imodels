from sklearn.linear_model import LinearRegression
import numpy as np
import cvxpy as cp # package for optimization

class SLIM():
    def __init__(self):
        self.model = LinearRegression()
        self.predict = self.model.predict
    
    
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
        
        # declare the integer-valued optimization variable
        w = cp.Variable(X.shape[1], integer=True)
        

        # set up the minimization problem
        residuals = X @ w - y
        if sample_weight is not None:
            # print('shapes', residuals.shape, sample_weight.shape)
            residuals = cp.multiply(sample_weight, residuals)
        mse = cp.sum_squares(residuals)
        l1_penalty = lambda_reg * cp.norm(w, 1)
        obj = cp.Minimize(mse + l1_penalty)
        prob = cp.Problem(obj)

        # solve the problem using an appropriate solver
        sol = prob.solve()
        self.model.coef_ = w.value.astype(np.int)
        self.model.intercept_ = 0
        
        