# Sparse pruning and hierarchical shrinkage

Fit a CART tree, prune it with [hiCAP](https://arxiv.org/abs/0909.0411), and
optionally smooth its predictions with hierarchical shrinkage (HS).
`sp_alpha` controls pruning; `reg_param` controls shrinkage.

[Fit a model](#fit-a-model) · [Cross-validation](#cross-validation) ·
[Solution paths](#solution-paths) · [Solvers](#solver-reference) ·
[Math and implementation](optimization/README.md)

## Fit a model

### Regression

Use `SPTreeRegressor` for pruning, or `SHSTreeRegressor` for pruning + HS.
Add `CV` to let the model choose the strengths:

```python
from sklearn.datasets import load_diabetes
from imodels import SHSTreeRegressorCV

X_reg, y_reg = load_diabetes(return_X_y=True)
model = SHSTreeRegressorCV(cv=5, max_leaf_nodes=32, random_state=0).fit(X_reg, y_reg)
predictions = model.predict(X_reg)
print(model.sp_alpha_, model.reg_param_)
```

Without CV, supply `sp_alpha` yourself and, for HS, `reg_param`.
All classes import directly from `imodels`.

Without a template, `max_leaf_nodes=None` (the default) grows an uncapped tree.
Set a number to limit growth before pruning. With an `estimator_` template,
`None` preserves its settings; a number overrides its leaf cap.

### Binary and multiclass classification

The classifier versions work the same way: `SPTreeClassifier` and
`SHSTreeClassifier`, with `CV` versions for tuning.

```python
from sklearn.datasets import load_iris
from imodels import SPTreeClassifierCV

X_class, y_class = load_iris(return_X_y=True)
classifier = SPTreeClassifierCV(cv=5, max_leaf_nodes=32, random_state=0).fit(
    X_class, y_class,
)
probabilities = classifier.predict_proba(X_class)
```

The defaults, `ord=np.inf` and `solver="auto"`, use the fast structural path
for eligible regression and classification trees. Predictions come from the
retained CART nodes, optionally adjusted by HS. Penalized coefficients are
available separately if you want them.

## Tune regularization strengths

### Cross-validation

By default, CV tries every distinct structural pruning state within each
training fold, then refits the selected settings on all the data. Regression
uses shuffled KFold; classification uses shuffled StratifiedKFold.

- `cv=5` sets the number of folds (default: 3).
- `sp_alpha_list="auto"` uses structural knots for eligible trees; a list such
  as `[0, 0.01, 0.1, 1]` requests a grid. Explicit APA also uses a grid.
- `reg_param_list` sets the HS candidates. SP defaults to no HS; SHS tunes it.
- `selection_rule="one_se"` favors a simpler tree within one standard error of the best;
  `"best"` chooses the highest mean score.
- `scoring` defaults to negative MSE for regression and accuracy for classification.

After fitting, `sp_alpha_` and `reg_param_` hold the chosen strengths;
`cv_params_` and `cv_scores_` show the search. For grouped or time-series splits,
use `GridSearchCV` around a non-CV model with `prefit=False`.

### Automatic HS with GCV

GCV is a cheaper way to choose HS from the fitted tree's training statistics:

```python
model = SHSTreeRegressorCV(
    cv=5, reg_param_list="gcv", max_leaf_nodes=32, random_state=0,
).fit(X_reg, y_reg)
```

Here k-fold CV still chooses pruning; GCV chooses HS for each pruned tree.
For a fixed pruning strength, use `SHSTreeRegressor(sp_alpha=..., reg_param="gcv")`.
GCV needs node-based HS, a single-output squared-error tree, uniform positive
weights, and no active monotonic constraints.

K-fold costs more but scores the whole fit/prune/shrink process on held-out
data. GCV corrects training error for a fixed tree, without accounting for how
that tree was learned, so it can be optimistic. Use a test set for the final
performance estimate. Check `reg_param_` for the strength and `gcv_results_` for the scores.

## Solution paths

A path follows one fitted tree as `sp_alpha` (λ) increases. The structural
solver finds the knots where subtrees disappear, without computing coefficients.
Between these knots, the pruned tree stays the same—that is why structural
knots are enough for pruning CV. The fitted model exposes them in `pruning_path_`.

### Traverse and preview pruned trees

For example, here we fit a 280-split tree to simulated classification data with
two signal features, ten irrelevant features, and 15% random label flips.
Pruning to **3 splits** raises validation accuracy from **73.5% to 84.6%**;
pruning further starts removing useful splits.

![Validation accuracy rises with pruning, then falls; the lower panel shows retained splits and an early-pruning zoom.](assets/structural_path.svg)

Penalty is scaled so `λ / λ_root = 1` leaves just the root.
We choose the penalty on validation data; independent test accuracy rises from
74.2% to 84.65%. Classification error is `1 − accuracy`, so it has the opposite,
U-shaped curve.

![The same tree at four penalties: 280 splits, 3 splits, 2 splits, and root only.](assets/pruned_trees.svg)

A–D match the plot above. Panel A summarizes the large starting subtrees;
B–D show every remaining node. A pruned subtree becomes a leaf at its original
root—nothing moves upward. Leaf labels show pooled training probabilities, without HS.

<details>
<summary>Reproduce the example</summary>

```python
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from imodels.tree.sparse_pruning import (
    fitted_tree_linf_exact_topology_path, materialize_fitted_tree_topology,
)

rng = np.random.default_rng(42)
X = rng.normal(size=(14000, 12))
signal = np.where(X[:, 0] > 0, X[:, 1] > -0.65, X[:, 1] > 0.65)
y = np.logical_xor(signal, rng.random(len(X)) < 0.15).astype(int)
X_train, y_train = X[:2000], y[:2000]
X_val, y_val = X[2000:6000], y[2000:6000]
X_test, y_test = X[6000:], y[6000:]

source = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)
structure = fitted_tree_linf_exact_topology_path(source)
alphas = structure.lambdas[::-1]  # increasing penalty
scores, counts = [], []
for alpha in alphas:
    preview = materialize_fitted_tree_topology(source, structure, float(alpha))
    scores.append(preview.score(X_val, y_val))
    counts.append(len(structure.tree_nodes_at(alpha)))

best = max(range(len(alphas)), key=lambda i: (scores[i], -counts[i]))
selected = materialize_fitted_tree_topology(source, structure, float(alphas[best]))
print(counts[best], scores[best], selected.score(X_test, y_test))
# 3 splits, 0.846 validation accuracy, 0.8465 test accuracy
plot_tree(selected, feature_names=[f"x{i}" for i in range(X.shape[1])])
```

</details>

These helpers also work with a fitted `DecisionTreeRegressor`. Keep the
original tree for previews; at a knot, the disappearing splits are already
removed. Use `len(structure.tree_nodes_at(alpha))` to count splits: preview
copies preserve original node IDs rather than compacting their backing arrays.

### Access the full coefficient path

With `ord=np.inf`, squared-error regression has piecewise-linear coefficients.
Some knots change their slopes without changing the tree:

![A coefficient-only knot changes slopes at lambda 0.5; the structural knot removes both splits at 1.5.](assets/solution_path.svg)

In this two-split example, the coefficients meet at λ = 0.5 but both splits
remain until λ = 1.5. To get all coefficient knots, use `"coefficient_path"`:

```python
from imodels import SPTreeRegressor

model = SPTreeRegressor(
    sp_alpha=1.0, solver="coefficient_path", max_leaf_nodes=32, random_state=0,
).fit(X_reg, y_reg)
beta = model.coef_             # coefficients at sp_alpha=1.0
path = model.coefficient_path_
beta_elsewhere, intercept = path.at(0.5 * path.lambdas[0])
```

The fitted model still predicts at your supplied `sp_alpha`. Querying the path
doesn't change it. Coefficients describe the original split features, before HS;
`coef_node_ids_` maps them to original nodes. CV models expose the same attributes
at their selected penalty, computing coefficients only during the final fit.

### Classification paths

Classification coefficients work the same way, through the same `solver`
argument, but the object you get back is different. Logistic and softmax paths
are curved rather than piecewise linear, so `"coefficient_path"` returns
warm-started samples at the positive structural knots and at your supplied
penalty. `path.exact` is `False` and interpolation between samples is
approximate. Your own penalty is solved rather than interpolated, so `coef_` is
a genuine solution there.

```python
from imodels import SPTreeClassifier

model = SPTreeClassifier(
    sp_alpha=0.02, solver="coefficient_path", max_leaf_nodes=32, random_state=0,
).fit(X_class, y_class)
beta, path = model.coef_, model.coefficient_path_
```

Use `"proximal"` if you only need coefficients at one penalty. At zero penalty,
pure leaves can require infinite logits, so classifier coefficients are left
as `None`. For custom grids and adaptive sampling, see the
[classification path details](optimization/README.md#classification-and-other-losses).

## Solver reference

`solver` selects the backend for the **final fit**, identically in CV and
non-CV models. It also decides which attributes come back:

| `solver` | Regression | Classification | `coef_` | `coefficient_path_` |
| --- | --- | --- | --- | --- |
| `"auto"` (default) | Structural path when eligible; otherwise proximal or APA | Structural path when eligible; otherwise binary APA | `None` | `None` |
| `"topology"` | Structural knots, no coefficients | Structural knots, no coefficients | `None` | `None` |
| `"proximal"` | Coefficients at one penalty | Coefficients at one positive penalty | set | `None` |
| `"coefficient_path"` | Full piecewise-linear coefficient path | Nonlinear coefficient samples | set | set |
| `"hicap"` | Slower reference coefficient path | Raises `ValueError` | set | set |
| `"apa_apg2"` | Approximate point solution | Approximate point solution, binary only | set | `None` |

So a full coefficient path is `solver="coefficient_path"`, on any model, CV or
not. Every solver except `"apa_apg2"` also fills `pruning_path_` with the exact
structural knots; APA leaves it `None`.

### What a fitted model exposes

| Attribute | Holds |
| --- | --- |
| `estimator_` | the pruned CART tree behind every prediction |
| `pruning_path_` | exact structural knots, the penalties at which the tree changes |
| `coef_`, `coef_node_ids_` | penalized coefficients, and the original split IDs they index |
| `intercept_` | intercept of the penalized problem |
| `coefficient_path_` | coefficients across penalties; check `.exact` before interpolating |
| `sp_alpha_`, `reg_param_` | the strengths a CV model chose |
| `solver_` | the backend actually used, never `"auto"` |

Coefficients are available for single trees, not forests. Multiclass `coef_`
has shape `(n_splits, n_classes)`, with the class order in `classes_`.

Three places where attributes are deliberately `None`:

- Structural solvers never compute coefficients, so `coef_` stays `None`.
- A classifier at `sp_alpha=0` leaves `coef_` and `intercept_` as `None`,
  because pure leaves can require infinite logits. `optimization_results_`
  then reports `certificate_scope="structure"` rather than
  `"structure_and_coefficients"`.
- Classifier structural fits leave `intercept_=None`. Regression keeps the
  root mean.

The fast tree paths use `ord=np.inf`, the original fitting statistics, and
zero/default `support_tol`. External prefit trees, OOB data, and active monotonic
constraints need different handling; see the [math note](optimization/README.md).
For large trees, stick with structural pruning unless you need coefficients:
the structural path is near-linear in the split count, while a full coefficient
path grows roughly quadratically and ran about three orders of magnitude slower
on a two-hundred-split tree.

## Implementation and references

- **hiCAP penalty:** [Zhao, Rocha, and Yu (2009)](https://arxiv.org/abs/0909.0411), Section 3.1.2.
- **Structural path:** tree-isotonic pooling ([Pardalos and Xue, 1999](https://doi.org/10.1007/PL00009258)); [our derivation](optimization/README.md#structural-path), which rests on [why the fitted-tree Gram matrix is diagonal](optimization/README.md#why-the-fitted-tree-gram-matrix-is-diagonal).
- **Proximal solver:** nested group updates from [Jenatton et al. (2011)](https://jmlr.org/papers/v12/jenatton11a.html), with [FISTA](https://doi.org/10.1137/080716542) for non-diagonal or classification problems.
- **Regression coefficient path:** combines structural bounds, proximal solves, and affine segments; [derivation](optimization/README.md#full-coefficient-path-for-a-fitted-tree).
- **APA-APG2:** [Shen et al. (2017), Algorithm 2](https://ojs.aaai.org/index.php/AAAI/article/view/10873), building on [Yu (2013)](https://papers.nips.cc/paper_files/paper/2013/hash/49182f81e6a13cf5eaa496d51fea6406-Abstract.html).
- **HS and GCV:** [Agarwal et al. (2022)](https://arxiv.org/abs/2202.00858) and [Golub, Heath, and Wahba (1979)](https://doi.org/10.1080/00401706.1979.10489751).

```text
sparse_pruning/
  sparse_hierarchical_shrinkage.py   SP/SHS models
  _solver.py                       solver selection
  _cv.py                           fold-local structural CV
  fitted_tree.py                   fitted-tree statistics and previews
  optimization/                    solvers and derivations
  optimizations.py                 legacy imports and GCV/subset helpers
```

Shared HS/GCV code lives in `imodels/tree/_hs_gcv.py`.
`tests/sparse_pruning_test.py` is the compact suite; `tests/sparse_pruning/`
contains the extended numerical tests.
