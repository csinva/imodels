# Sparse pruning and hierarchical shrinkage

Prune a CART tree with the [CAP/hiCAP penalty](https://arxiv.org/abs/0909.0411),
optionally followed by hierarchical shrinkage (HS). `sp_alpha` controls pruning
strength; `reg_param` controls shrinkage without changing the retained structure.
The penalty groups each split with all its descendant splits.

[Fit a model](#fit-a-model) · [Tune strengths](#tune-regularization-strengths) ·
[Solution paths](#solution-paths) · [Choose a solver](#solver-reference) ·
[Implementation and references](#implementation-and-references)

## Fit a model

### Regression

| Task | Supply strengths yourself | Select strengths by CV |
| --- | --- | --- |
| Pruning only | `SPTreeRegressor` | `SPTreeRegressorCV` |
| Pruning + HS | `SHSTreeRegressor` | `SHSTreeRegressorCV` |

All classes import directly from `imodels`. Non-CV models take `sp_alpha` and,
for HS, `reg_param`. CV models select those strengths and expose `sp_alpha_`
and `reg_param_` after fitting.

```python
from sklearn.datasets import load_diabetes
from imodels import SPTreeRegressorCV, SHSTreeRegressorCV

X, y = load_diabetes(return_X_y=True)
pruned = SPTreeRegressorCV(cv=5, max_leaf_nodes=32, random_state=0).fit(X, y)
shrunk = SHSTreeRegressorCV(cv=5, max_leaf_nodes=32, random_state=0).fit(X, y)
predictions = shrunk.predict(X)
selected_strengths = shrunk.sp_alpha_, shrunk.reg_param_
```

Defaults are `ord=np.inf, solver="auto"`. For eligible regression trees this
computes the fast structural path: all distinct pruned trees, without solving
for every coefficient. Predictions use retained CART node values, optionally
shrunk by HS—not the penalized coefficients used to decide pruning.

### Binary classification

Use `SPTreeClassifier` / `SPTreeClassifierCV` for pruning, or
`SHSTreeClassifier` / `SHSTreeClassifierCV` to add HS. All provide `predict`
and `predict_proba`. These estimator wrappers currently use APA-APG2 with
logistic loss and numeric-grid CV. Their HS strengths must be numeric; GCV
and multiclass classification are unsupported in the wrappers.
The separate [fitted-tree classification APIs](#classification-paths) below
support binary and multiclass structural paths and coefficient samples.

## Tune regularization strengths

### Cross-validation

All four SP/SHS CV classes accept integer `cv` (default: 3): shuffled KFold for
regression, shuffled StratifiedKFold for classification. Each training fold
grows its own tree, prunes candidates, applies HS if requested, and scores
predictions on held-out rows. The selected configuration is refitted on all data.

- `sp_alpha_list="auto"` evaluates distinct structural states for eligible
  regression trees. Candidates come from training-fold knots, never a full-data
  tree. Other cases, including classification and explicit APA, use a numeric grid.
- A numeric `sp_alpha_list` requests grid CV.
- SHS defaults to a numeric `reg_param_list`, selecting HS by held-out scores;
  SP disables HS by default. `reg_param_list="gcv"` changes only HS selection
  as described below.
- `selection_rule="one_se"` favors the simplest tree within one standard error
  of the best mean score; `"best"` selects the highest mean score.
  `scoring` defaults to negative mean squared error for regression and accuracy
  for classification.

Inspect `cv_params_` and `cv_scores_` for candidates and fold scores, and
`cv_path_mode_` for `"structural"` versus `"grid"`. The final fit uses the
selected absolute penalty, not the nearest full-data knot. Structural CV reuses
scores for identical retained states; custom scorers must depend on predictions
or structure, not the numerical penalty label.

Non-CV classes do not run folds internally. For custom splitters, including
grouped or time-ordered data, use sklearn's `GridSearchCV` around a non-CV
estimator with `prefit=False`, rather than a wrapper with internal shuffled folds.

### Automatic HS with GCV

GCV selects HS from a fitted tree's node statistics. It is optional and does
**not** replace held-out pruning selection in a CV model:

```python
auto_hs = SHSTreeRegressorCV(
    cv=5, reg_param_list="gcv", max_leaf_nodes=32, random_state=0,
).fit(X, y)
```

Here GCV selects HS for each training fold's pruned tree; held-out fold scores
still select `sp_alpha`. HS is reselected after full-data pruning.
Without outer CV, `SHSTreeRegressor(sp_alpha=..., reg_param="gcv")` selects HS
after fixed-penalty pruning. For HS alone, use `HSTreeRegressor(reg_param="gcv")`
or `HSTreeRegressorCV` with numeric strengths for k-fold selection.

GCV requires node-based HS, a single-output tree with `squared_error` or
`friedman_mse`, uniform positive weights, and no active monotonic constraints.
Unsupported configurations, including forests, need numeric strengths or CV.
`reg_param_` stores the selected strength; `gcv_results_` stores scores,
effective degrees of freedom, and diagnostics. An infinite strength means
root-mean predictions. Sparse-HS wrappers accept `reg_param=None` as a GCV
compatibility alias; ordinary HS requires the explicit `"gcv"` value.

### K-fold CV versus GCV

- **K-fold:** evaluates the grow/prune/shrink pipeline on held-out rows and
  supports your prediction metric. It costs fold fits and candidate evaluation;
  each fold trains on less data, and selected strengths can vary with the split.
- **GCV:** usually cheaper for HS selection, without HS-specific fold refits or
  held-out predictions. Its training-error correction accounts for a fixed
  tree's effective degrees of freedom, but not for learning/pruning that tree
  from the same targets. It can therefore be optimistic; it is not exact
  leave-one-out refitting. Using it inside SHS CV still incurs pruning-fold costs.

Use k-fold when held-out selection justifies the computation; GCV is a fast
approximation for eligible trees. Neither guarantees better strengths.
Evaluate the selected model on an untouched test set or an additional outer CV
loop, not its winning tuning score. See sklearn's
[cross-validation guide](https://scikit-learn.org/stable/modules/cross_validation.html).

## Solution paths

A pruning path varies `sp_alpha` (λ) on **one fixed fitted tree**, not `reg_param`.
For squared-error regression with `ord=np.inf`, there are two useful outputs:

- **Structural path:** penalties where the retained tree changes. This is enough
  to traverse pruned trees and select pruning by CV; it is the regression default.
- **Coefficient path:** every penalty where a penalized coefficient changes
  **slope**. Between consecutive knots the coefficients are linear in λ, so the
  full path supports exact interpolation. A coefficient knot need not change
  the tree; interpolating only at structural knots can miss a bend.

![Two-coefficient hiCAP example: at lambda 0.5 slopes change but both splits remain; at lambda 1.5 both splits disappear.](assets/solution_path.svg)

This analytic example minimizes
`0.5 * ||beta - [1, 2]||² + λ * (max(|beta₀|, |beta₁|) + |beta₁|)`.
At λ = 0.5 the two coefficients meet and change slopes, without changing the
tree. At λ = 1.5 both vanish, removing both splits. More generally, a split is
retained while its descendant group is active—even if its own coefficient is zero.

### Traverse and preview pruned trees

This synthetic **classification** example starts with an overfit, 280-split
CART tree. Validation accuracy rises from **73.5% to 84.6%** at just **3 splits**,
then falls to 51.1% when only the root remains. Classification error is
`1 − accuracy`, so its curve has the opposite, U-shaped pattern.

![Overfitting example: validation accuracy rises to 84.6% before declining with excessive pruning. Training accuracy, three independent test evaluations, retained split counts, and an early-pruning zoom are also shown.](assets/structural_path.svg)

The penalty is selected using **validation data only** (ties favor fewer splits).
Independent test accuracy improves from 74.2% to **84.65%** for the selected tree;
the test set does not choose the penalty. This is an illustrative fixed-seed
simulation, not a general performance guarantee. The horizontal axis divides
λ by `λ_root`, the penalty at which only the root remains.

Each A–D marker matches a tree below. Panel A summarizes the large original
subtrees explicitly; B–D show every remaining node.

![The overfit 280-split tree, summarized using subtree counts, followed by the validation-selected 3-split tree, a 2-split tree, and the root-only tree. Original node IDs and positions stay fixed.](assets/pruned_trees.svg)

Pruning removes a whole subtree and turns its root into a leaf—descendants
are **not moved upward**. These classifier previews use pooled CART class
frequencies, without HS or penalized-coefficient predictions. Between structural
knots, these previews stay the same even if coefficients change. This example
uses the [fitted-tree classification APIs](#classification-paths), not the
classifier wrappers' numeric-grid CV.

<details>
<summary>Reproduce the example and preview the selected tree</summary>

```python
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from imodels.tree.sparse_pruning import (
    fitted_tree_linf_exact_topology_path, materialize_fitted_tree_topology,
)

# Two signal features, ten irrelevant features, and 15% random label flips.
rng = np.random.default_rng(42)
X = rng.normal(size=(14000, 12))
signal = np.where(X[:, 0] > 0, X[:, 1] > -0.65, X[:, 1] > 0.65)
y = np.logical_xor(signal, rng.random(len(X)) < 0.15).astype(int)
X_train, y_train = X[:2000], y[:2000]
X_val, y_val = X[2000:6000], y[2000:6000]
X_test, y_test = X[6000:], y[6000:]

source = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)
structure = fitted_tree_linf_exact_topology_path(source)
alphas = structure.lambdas[::-1]  # increasing penalty, including zero
scores, counts = [], []
for alpha in alphas:
    preview = materialize_fitted_tree_topology(source, structure, float(alpha))
    scores.append(preview.score(X_val, y_val))
    counts.append(len(structure.tree_nodes_at(alpha)))

# Freeze the choice before evaluating the independent test set.
best = max(range(len(alphas)), key=lambda i: (scores[i], -counts[i]))
selected = materialize_fitted_tree_topology(source, structure, float(alphas[best]))
print(counts[best], scores[best], selected.score(X_test, y_test))
# 3 splits, 0.846 validation accuracy, 0.8465 test accuracy
plot_tree(selected, feature_names=[f"x{i}" for i in range(X.shape[1])])
```

</details>

Regression wrappers using the structural solver expose `pruning_path_`;
`coef_` and `coefficient_path_` are `None`. The same fitted-tree helpers work
with an eligible `DecisionTreeRegressor`. Keep the original CART tree for
previews; `structure.iter_node_pruning_events()` streams disappearing node IDs
in increasing penalty order.

At a structural knot, the disappearing splits are already removed;
`below=True` previews the state immediately below that knot. Previews leave
`source` unchanged and show original CART values, without HS. They preserve
original node IDs without compacting backing arrays: use
`len(structure.tree_nodes_at(alpha))` for the retained split count, not preview
array sizes. Stream previews rather than storing a copy at every knot.

### Access the full coefficient path

```python
from imodels import SPTreeRegressor

model = SPTreeRegressor(
    sp_alpha=1.0, solver="coefficient_path", max_leaf_nodes=32, random_state=0,
).fit(X, y)
predictions = model.predict(X)      # tree at the supplied sp_alpha=1.0
beta = model.coef_                  # coefficients at that same penalty
path = model.coefficient_path_      # also available with solver="hicap"
beta_elsewhere, intercept = path.at(0.5 * path.lambdas[0])
```

`path.lambdas` and `path.coefficients` store knots, endpoints, and their coefficient
vectors. `path.at(alpha)` interpolates within that range and returns a separate
array plus intercept. It does **not** change `model.coef_`, the fitted tree,
or `predict()`. Fit another model with a different `sp_alpha` to change the
prediction tree; setting the parameter alone does not re-prune.

CV models expose the same attributes at their selected `sp_alpha_`. Eligible
automatic CV still scores structural states when the final-fit solver is
`"coefficient_path"` or `"hicap"`; only the final fit needs the full path.

Coefficients refer to unnormalized local stumps of the original, unpruned tree,
before HS. `coef_node_ids_` identifies those original splits, not the renumbered
nodes of the compacted model. Keep the original tree if reconstructing its
feature matrix. Classification/APA grid paths do not provide this exact
piecewise-linear interpolation contract.

### Classification paths

For an eligible fitted `DecisionTreeClassifier`,
`fitted_tree_linf_exact_topology_path` and `materialize_fitted_tree_topology`
also provide all structural pruning states. The objective is logistic/softmax
loss with a class-range extension of hiCAP, not a squared-error surrogate.
Use the original fitted tree, before HS, with positive leaf/class masses and
no monotonic constraints. Its stored fitting partition includes sample and
class weights; it is not a held-out or OOB objective.

Coefficient solutions are optional and separate from the preview's CART values:

```python
from imodels.tree.sparse_pruning import (
    fitted_tree_linf_classification, fitted_tree_linf_classification_path,
)

# source is an already fitted binary or multiclass DecisionTreeClassifier.
path = fitted_tree_linf_classification_path(
    source, [0.1, 0.03, 0.01], tol=1e-8,
    adaptive_tol=1e-3, max_points=100,  # optional midpoint refinement
)
beta, info = fitted_tree_linf_classification(source, 0.02, return_info=True)
assert info["certified"]  # numerical stationarity check, not an error bound
probabilities = info["leaf_probabilities"]
leaf_ids = info["leaf_node_ids"]       # sorted original tree leaf IDs
# Predictions on X_new: probabilities[np.searchsorted(leaf_ids, source.apply(X_new))]
```

Binary coefficients have shape `(n_points, n_splits)`; multiclass coefficients
have shape `(n_points, n_splits, n_classes)`, with class order in
`path.metadata['classes']`. Intercepts are stored separately. The tree is never
modified. Check `path.status` and `path.diagnostics`: an iteration/refinement
limit returns a partial path. `path.exact` is always false, and `path.at(alpha)`
is **approximate interpolation**, not a new solve. Midpoint refinement is not
a uniform error guarantee. Use the point API when a checked solution is needed.
Penalties must be positive: pure leaves can require infinite logits at zero.

These APIs reuse the exact laminar proximal operator and warm starts. Leaf
aggregation avoids an observation-by-split matrix, but coefficient storage and
explicit descendant groups can still be costly for large, deep trees.
They do not change classifier-wrapper solver choices or CV defaults.

## Solver reference

| `solver` | What it computes | Use when |
| --- | --- | --- |
| `"auto"` | Structural path when eligible; otherwise proximal for infinity-norm regression, APA for other supported objectives | Default for fitting and CV |
| `"topology"` | Exact structural knots; no coefficients | You only need pruned trees |
| `"proximal"` | Certified coefficients at one penalty | You need coefficients, but not their full path |
| `"coefficient_path"` | Complete certified diagonal-tree coefficient path | You need all coefficient knots and interpolation |
| `"hicap"` | Generic, slower, certified coefficient path | Small reference/validation problems |
| `"apa_apg2"` | Approximate point solution | Binary classification or `ord=2` |

Wrapper tree-specific solvers require `ord=np.inf`, zero/default `support_tol`, and an
unconstrained single-output mean-based regression tree on its fitting rows and
weights. Wrappers do not assume this for external `prefit=True` trees or forest
OOB data. Explicitly incompatible choices or failed certificates raise errors.

For `p` splits, the structural path takes `O(p log p)` time and `O(p)` space
after fitting. CV and previews still pay for copies and held-out predictions.
Full coefficient paths additionally store dense coefficient rows and rebuild
active constraints; they are optional because pruning does not need that work.

"Exact" means within floating-point certificate tolerances, not symbolic
arithmetic. `tol` controls those tolerances; for APA it only controls stopping.
`max_iter` caps point iterations or coefficient-path events, depending on the
solver. For compatibility, wrapper `sp_alpha=0` preserves every original split;
strict mathematical path queries omit zero-activation splits even at zero.
See the [optimization guide](optimization/README.md) for numerical contracts.

## Implementation and references

### Algorithms

- **Penalty and hiCAP:** [Zhao, Rocha, and Yu (2009)](https://arxiv.org/abs/0909.0411),
  Section 3.1.2. Our `hicap` follows active linear constraints of the infinity-norm
  formulation; it is independently implemented, not a MATLAB translation.
- **APA-APG2:** accelerated gradient steps plus averaged group-proximal updates
  with decreasing approximation, following Algorithm 2 of
  [Shen et al. (2017)](https://ojs.aaai.org/index.php/AAAI/article/view/10873).
  The precursor is [Yu (2013)](https://papers.nips.cc/paper_files/paper/2013/hash/49182f81e6a13cf5eaa496d51fea6406-Abstract.html).
  Warm starts sample a grid; they do not enumerate exact knots.
- **Proximal:** child-to-parent group updates from
  [Jenatton et al. (2011), Section 3.4](https://jmlr.org/papers/v12/jenatton11a.html),
  inside [FISTA (Beck and Teboulle, 2009)](https://doi.org/10.1137/080716542)
  for general designs. The diagonal-tree specialization uses weighted projections
  to solve each penalty in one sweep.
- **Structural path:** a tree-specific reduction to weighted isotonic pooling.
  Heap-based tree pooling: [Pardalos and Xue (1999)](https://doi.org/10.1007/PL00009258).
  See our [CAP-to-activation derivation](optimization/README.md#why-the-fitted-tree-gram-matrix-is-diagonal).
- **Coefficient path:** our diagonal-tree continuation combines structural bounds,
  exact point solves, and intervals with unchanged active constraints;
  see the [derivation](optimization/README.md#full-coefficient-path-for-a-fitted-tree).
- **HS and GCV:** [Agarwal et al. (2022)](https://arxiv.org/abs/2202.00858) for HS;
  [Golub, Heath, and Wahba (1979)](https://doi.org/10.1080/00401706.1979.10489751)
  for GCV, applied here conditional on the retained tree.

### Code layout

```text
sparse_pruning/
  sparse_hierarchical_shrinkage.py   estimator fitting, pruning, and HS
  _solver.py                       solver validation and dispatch
  _cv.py                           fold-local structural CV
  fitted_tree.py                   sklearn statistics and preview adapters
  optimization/                    mathematical solvers and path results
  optimizations.py                 legacy point imports and GCV/subset helpers
```

Shared HS/GCV mathematics lives in `imodels/tree/_hs_gcv.py`.
`tests/sparse_pruning_test.py` is the compact standalone suite;
`tests/sparse_pruning/` holds extended numerical tests, and
`tests/hs_gcv_test.py` covers shared HS/GCV behavior.
