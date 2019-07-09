# Optimal classification trees python
Sklearn-style implementation of [Optimal Classification Trees](https://link.springer.com/article/10.1007/s10994-017-5633-9), adapted from [this repo](https://github.com/pan5431333/pyoptree) with minor modifications to make it easier to install / use

#### Then install solver (IMPORTANT!) 
The user needs to have **IBM Cplex** or **Gurobi** installed on their computer, and make sure that **the executable has been added to PATH environment variable** (i.e. command `cplex` or `gurobi` can be run on terminal). 

### Example 
```python
import pandas as pd
import numpy as np
from pyoptree.optree import OptimalTreeModel
feature_names = np.array(["x1", "x2"])

X = np.array([[1, 2, 2, 2, 3], [1, 2, 1, 0, 1]]).T
y = np.array([1, 1, 0, 0, 0]).reshape(-1, 1)
X_test = np.array([[1, 1, 2, 2, 2, 3, 3], [1, 2, 2, 1, 0, 1, 0]]).T
y_test = np.array([1, 1, 1, 0, 0, 0, 0])

X = np.random.randn(10, 5)


model = OptimalTreeModel(tree_depth=3, N_min=1, alpha=0.1) #, solver_name='baron'
model.fit(X, y) # this method is currently using the fast, but not optimal solver
preds = model.predict(X_test)
# print(np.mean())
print('acc', np.mean(preds == y_test))
```

### Todos 
1. Use the solution from the previous depth tree as a "Warm Start" to speed up the time to solve the Mixed Integer Linear Programming (MILP); （Done √）
2. Use the solution from sklearn's CART to give a good initial solution (Done √);
