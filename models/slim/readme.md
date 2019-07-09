# wrapper for sparse, integer linear models

minimizes norm(X * w - y, 2) + lambda_reg * norm(w, 1)

with integer coefficients in w