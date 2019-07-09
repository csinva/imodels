# Optimal classification trees python
Sklearn-style implementation of [Optimal Classification Trees](https://link.springer.com/article/10.1007/s10994-017-5633-9), adapted from [this repo](https://github.com/pan5431333/pyoptree) with minor modifications to make it easier to install / use

#### Then install solver (IMPORTANT!) 
The user needs to have **IBM Cplex** or **Gurobi** installed on their computer, and make sure that **the executable has been added to PATH environment variable** (i.e. command `cplex` or `gurobi` can be run on terminal). 

### Example 
```python
import pandas as pd
from pyoptree.optree import OptimalHyperTreeModel, OptimalTreeModel

data = pd.DataFrame({
    "index": ['A', 'C', 'D', 'E', 'F'],
    "x1": [1, 2, 2, 2, 3],
    "x2": [1, 2, 1, 0, 1],
    "y": [1, 1, 0, 0, 0]
})
test_data = pd.DataFrame({
    "index": ['A', 'B', 'C', 'D', 'E', 'F', 'G'],
    "x1": [1, 1, 2, 2, 2, 3, 3],
    "x2": [1, 2, 2, 1, 0, 1, 0],
    "y": [1, 1, 1, 0, 0, 0, 0]
})
model = OptimalHyperTreeModel(["x1", "x2"], "y", tree_depth=2, N_min=1, alpha=0.1, solver_name="cplex")
model.train(data, train_method="mio")

print(model.predict(test_data))
```

### Todos 
1. Use the solution from the previous depth tree as a "Warm Start" to speed up the time to solve the Mixed Integer Linear Programming (MILP); （Done √）
2. Use the solution from sklearn's CART to give a good initial solution (Done √);
