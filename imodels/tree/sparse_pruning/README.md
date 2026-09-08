# Sparse pruning and hierarchical shrinkage

These estimators prune whole subtrees using the hiCAP descendant-group penalty.
Predictions use the retained CART node values, optionally followed by
hierarchical shrinkage (HS). Penalized coefficients determine pruning; they are
not the prediction weights of the resulting tree.

## Defaults and solver choices

`SPTreeRegressor`, `SHSTreeRegressor`, and their CV variants default to
`ord=np.inf, solver="auto"`. For an unconstrained, single-output regression
tree fitted by the wrapper with `squared_error` or `friedman_mse`, this chooses
the exact **structural** path. It uses fitted node statistics, not a dense
sample-by-split design matrix. Positive `support_tol` disables this shortcut.

| `solver` | Result for an eligible fitted regression tree |
| --- | --- |
| `"auto"` | Structural path; otherwise proximal infinity-regression or APA |
| `"topology"` | Exact structural knots (green); no coefficient calculation |
| `"proximal"` | Certified coefficients at the chosen penalty (dark blue) |
| `"coefficient_path"` | Complete certified coefficient path (orange) |
| `"hicap"` | Generic, slower, certified coefficient-path reference |
| `"apa_apg2"` | Legacy approximate point solver; also supports `ord=2` and binary classification |

Native structural/coefficient paths require the tree's original fitting measure;
the wrappers do not assume this for externally supplied `prefit=True` trees or
forest OOB data. General infinity-regression problems instead use design-based
proximal solves. Explicitly incompatible solver choices raise an error, as do
failed exact-solver certificates; there is no silent numerical fallback.

`max_iter` caps APA/FISTA iterations for iterative point solves, but caps
coefficient-continuation events for `"coefficient_path"` and `"hicap"`.
The native diagonal point solve and structural path are noniterative. `tol`
sets numerical accuracy/certificate tolerances for exact coefficient solvers;
for APA it is a stopping tolerance, not an exactness guarantee. `support_tol`
controls pruning support separately and must be zero/default for native paths.

On eligible fitting data, all exact solvers use the same structural events for
pruning, including right at a knot. Tiny floating-point residuals in generic
hiCAP coefficients cannot change that topology. Elsewhere, coefficient-based
pruning uses a response-scaled numerical threshold by default. `support_tol=0`
requests literal coefficient zero there, which can be sensitive to roundoff.
For compatibility, `sp_alpha=0` always preserves the original topology, including
zero-gain splits.

## Cross-validation

```python
from imodels import SPTreeRegressorCV, SHSTreeRegressorCV

pruned = SPTreeRegressorCV(max_leaf_nodes=128, random_state=0).fit(X, y)
shrunk = SHSTreeRegressorCV(max_leaf_nodes=128, random_state=0).fit(X, y)
auto_hs = SHSTreeRegressorCV(
    max_leaf_nodes=128, reg_param_list="gcv", random_state=0,
).fit(X, y)
```

The default `sp_alpha_list="auto"` fits one tree per training fold and scores
each distinct structural state once per HS choice. Scores are aligned on the
union of training-fold knots. The final tree uses the selected absolute penalty;
it is not snapped to a full-data knot. No full-data tree generates CV candidates.
Custom scorers must depend on predictions/tree structure, not on the numerical
penalty label, for structural-state score reuse to be valid.

Numeric alpha lists still request grid CV. Classification, `ord=2`, custom
legacy solver hooks, and other ineligible cases use the historical numeric grid
when given `"auto"`. `cv_path_mode_` distinguishes `"structural"` from `"grid"`;
`cv_solver_` records the CV algorithm and `solver_` the final-fit algorithm.
`cv_sp_alphas_`, `cv_scores_`, and `cv_params_` describe the evaluated candidates;
structural CV also exposes `cv_path_results_` and `cv_n_pruning_states_`.

With the exact backends, an explicit grid on eligible fitting data reuses the
fitted sufficient statistics and path once per fold. Single-tree generic hiCAP
also reuses its complete path across grid penalties. APA retains its historical
point-solve behavior. The proximal backend constructs its group
operator lazily for interior penalties and reuses it across the grid;
unpenalized/fully pruned endpoints need no proximal solve. This reuse does not
remove the cost of copying candidates or predicting on validation rows.

Numeric HS strengths remain the default for SHS CV. With `reg_param_list="gcv"`,
each training state's HS strength is selected by conditional fixed-tree GCV,
then reselected after full-data pruning. GCV requires uniform positive weights
and node-based shrinkage; it does not account for learning the tree structure.

## Optional coefficients

```python
model = SHSTreeRegressorCV(
    solver="coefficient_path", max_leaf_nodes=128, random_state=0,
).fit(X, y)
beta = model.coef_                  # coefficients at the selected penalty
original_node_ids = model.coef_node_ids_
path = model.coefficient_path_      # all coefficient-direction knots
beta_elsewhere, intercept_elsewhere = path.at(0.5 * path.lambdas[0])
```

Automatic CV still scores green structural states, while the final fit honors
the requested coefficient solver. Coefficients refer to **unnormalized local
stumps of the original, unpruned tree**, before HS; original node IDs are not
the renumbered IDs in the compact pruned tree. Preserve the original tree if
you need to reconstruct its feature matrix. `"topology"` leaves `coef_` and
`coefficient_path_` as `None` and exposes `pruning_path_` instead.

The green path itself takes `O(p log p)` time after fitting and `O(p)` space
for `p` split nodes. CV still pays for held-out prediction and one compact tree
copy per local state; it is not an `O(p log p)` end-to-end procedure. Complete
coefficient paths additionally store dense coefficient rows and rebuild faces;
they are optional precisely because pruning does not need that work.

## Source layout

This is one source folder within imodels, not a separate distribution.
The `sparse_pruning` namespace exposes estimators and fitted-tree helpers;
mathematical APIs have one public home in `sparse_pruning.optimization`.
Historical public point-solver imports from `optimizations` remain supported.

```text
sparse_pruning/
  sparse_hierarchical_shrinkage.py   sklearn estimator lifecycle and mutation
  _solver.py                       solver validation and fitted-tree dispatch
  _cv.py                           fold-local structural-state CV
  fitted_tree.py                   sklearn statistics and preview adapters
  optimization/                    mathematical solvers and path results
  optimizations.py                 old point imports and GCV/subset helpers
```

See [optimization/README.md](optimization/README.md) for numerical contracts and
limitations. Shared HS/GCV mathematics lives in `imodels/tree/_hs_gcv.py`.
Pruning tests live under `tests/sparse_pruning/`; shared HS/GCV tests stay in
`tests/hs_gcv_test.py`.
