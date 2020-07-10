from sklearn.linear_model import LinearRegression
import numpy as np
import cvxpy as cp # package for optimization

class SLIM():
    def __init__(self):
        self.model = LinearRegression()
        self.predict = self.model.predict
    
    
    def fit(self, X, y, lambda_reg=0):
        '''fit a linear model with integer coefficient and L1 regularization
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
        mse = cp.sum_squares(X @ w - y)
        l1_penalty = lambda_reg * cp.norm(w, 1)
        obj = cp.Minimize(mse + l1_penalty)
        prob = cp.Problem(obj)

        # solve the problem using an appropriate solver
        sol = prob.solve()
        self.model.coef_ = w.value.astype(np.int)
        self.model.intercept_ = 0
        
        