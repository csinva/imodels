# wrapper for sparse, integer linear models

def fit_integer_linear_model(X, y, lambda_reg=0):
    import cvxpy # package for optimization
    '''fit a linear model with integer coefficient and L1 regularizaiton'''
    
    # declare the integer-valued optimization variable
    w = cvxpy.Variable(X.shape[1], integer=True)

    # set up the minimization problem
    obj = cvxpy.Minimize(cvxpy.norm(X * w - y, 2) + lambda_reg * cvxpy.norm(w, 1))
    prob = cvxpy.Problem(obj)

    # solve the problem using an appropriate solver
    sol = prob.solve(solver = 'ECOS_BB')

    # the optimal value
    return w.value.astype(np.int)