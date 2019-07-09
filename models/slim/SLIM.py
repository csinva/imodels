from sklearn.linear_model import LinearRegression
import numpy as np
import cvxpy # package for optimization

class SLIM():
    def __init__(self):
        self.model = LinearRegression()
        self.predict = self.model.predict
    
    
    def fit(self, X, y, lambda_reg=0):
        '''fit a linear model with integer coefficient and L1 regularizaiton'''

        # declare the integer-valued optimization variable
        w = cvxpy.Variable(X.shape[1], integer=True)

        # set up the minimization problem
        obj = cvxpy.Minimize(cvxpy.norm(X * w - y, 2) + lambda_reg * cvxpy.norm(w, 1))
        prob = cvxpy.Problem(obj)

        # solve the problem using an appropriate solver
        sol = prob.solve(solver = 'ECOS_BB')

        # the optimal value
        self.model.coef_ = w.value.astype(np.int)
        self.model.intercept_ = 0
        self.coef_ = self.model.coef_
        
        