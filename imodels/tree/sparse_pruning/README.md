# Sparse pruning and hierarchical shrinkage

Use `SPTree*` to prune a tree, or `SHSTree*` to prune and then apply hierarchical
shrinkage (HS). `sp_alpha` controls pruning; `reg_param` controls HS without
changing the retained structure. Classes ending in `CV` select these strengths;
non-CV classes use the strengths you supply.

Pruning uses the [CAP/hiCAP penalty](https://arxiv.org/abs/0909.0411), with each
group containing a split and all its descendant splits. The resulting model
predicts with retained CART node values, optionally shrunk by HS—not with the
penalized coefficients used to decide pruning.

## Quick start

```python
from sklearn.datasets import load_diabetes
from imodels import SPTreeRegressorCV, SHSTreeRegressorCV

X, y = load_diabetes(return_X_y=True)
pruned = SPTreeRegressorCV(cv=5, max_leaf_nodes=32, random_state=0).fit(X, y)
shrunk = SHSTreeRegressorCV(cv=5, max_leaf_nodes=32, random_state=0).fit(X, y)
predictions = shrunk.predict(X)
selected_strengths = shrunk.sp_alpha_, shrunk.reg_param_
```

For ordinary regression trees, the defaults are `ord=np.inf, solver="auto"`:
fast exact **structural knots**, the penalties where the retained tree changes.
This is all pruning needs; computing a full coefficient path is optional.

## Cross-validation

All four SP/SHS CV classes support k-fold selection through integer `cv`
(default: 3). Regression uses shuffled KFold; classification uses shuffled
StratifiedKFold. Non-CV classes do not run folds internally: use their CV
counterparts, or wrap a non-CV estimator with `prefit=False` in sklearn's
`GridSearchCV` for custom splitters.

For eligible regression trees, `sp_alpha_list="auto"` grows one tree per training
fold and evaluates every distinct pruned structure, once per HS choice.
Candidates come from training-fold knots, never a full-data tree.
SHS defaults to a numeric HS grid; SP disables HS by default.
A numeric `sp_alpha_list` instead requests grid CV. Classification, `ord=2`,
explicit APA, and other ineligible cases also use grid CV.

The default `selection_rule="one_se"` favors the simplest tree within one
standard error of the best mean score; `"best"` chooses the highest mean score.
Regression defaults to negative mean squared error; classification to accuracy.
Set `scoring` to change this. Structural-state score reuse requires scorers
that depend on predictions/tree structure, not the numerical penalty label.

Inspect `sp_alpha_` and `reg_param_` for selected strengths, `cv_scores_` and
`cv_params_` for fold scores and candidates, and `cv_path_mode_` for
`"structural"` versus `"grid"`. The final fit uses the selected absolute penalty,
not the nearest full-data knot. Solver choice and CV selection are separate:
eligible automatic CV still scores structural states if the final-fit solver
is `"coefficient_path"` or `"hicap"`.

## Automatic HS with GCV

GCV is an **optional way to select HS**, not a replacement for outer pruning CV:

```python
auto_hs = SHSTreeRegressorCV(
    cv=5, reg_param_list="gcv", max_leaf_nodes=32, random_state=0,
).fit(X, y)
```

Here GCV selects HS on each training fold's pruned tree; held-out fold scores
still select pruning. HS is reselected after full-data pruning. To select both
strengths by held-out k-fold scores, use a numeric `reg_param_list` instead.
For HS without pruning, `HSTreeRegressorCV` also offers k-fold selection over
numeric HS strengths.

Without outer CV, use `SHSTreeRegressor(reg_param="gcv", sp_alpha=...)` to select
HS after fixed-penalty pruning, or `HSTreeRegressor(reg_param="gcv")` for HS alone.
Sparse-HS wrappers also accept `reg_param=None` as an alias; ordinary HS requires
the explicit `"gcv"` value. `reg_param_` stores the selected strength and
`gcv_results_` stores scores, effective degrees of freedom, and diagnostics.
An infinite selected strength means root-mean predictions.

GCV uses node statistics, without a dense training design. It requires node-based
HS, a single-output regression tree with `squared_error` or `friedman_mse`,
uniform positive weights, and no active monotonic constraints. Classifiers,
forests, and other unsupported configurations need numeric strengths or CV.

### K-fold CV versus GCV: benefits and limitations

- **K-fold CV** evaluates the grow/prune/shrink pipeline on held-out rows and
  supports your chosen prediction metric. The tradeoff is repeated fold fits
  and candidate evaluation. Each fold trains on less data than the final model,
  and selected strengths can vary with the split, especially on small datasets.
- **GCV** is usually cheaper for HS selection: it uses retained node statistics
  without HS-specific fold refits or held-out predictions. It corrects training
  residual error for the **fixed tree's** effective degrees of freedom, but not
  for learning splits or selecting pruning from the same targets. It can
  therefore be optimistic; it is not exact leave-one-out refitting of the tree.

Use k-fold when held-out model selection justifies the computation; use GCV for
fast approximate HS selection on eligible trees. Neither is guaranteed to select
better strengths. GCV inside an SHS CV wrapper still incurs pruning-fold costs.
Do not treat the winning tuning score as an unbiased final performance estimate:
use an untouched test set or an additional outer CV loop. See sklearn's
[cross-validation guide](https://scikit-learn.org/stable/modules/cross_validation.html)
for evaluation and appropriate splitters for grouped or time-ordered data.

## Binary classification

Use `SPTreeClassifierCV` or `SHSTreeClassifierCV`, imported from `imodels`,
for `predict` and `predict_proba`. Their non-CV counterparts take a fixed
`sp_alpha`. Classification uses APA-APG2 with logistic loss and numeric-grid CV;
it does not have the exact squared-error structural/coefficient paths.
HS strengths must be numeric: GCV is regression-only. Multiclass is unsupported.

## Defaults and solver choices

| `solver` | What it computes |
| --- | --- |
| `"auto"` | Structural path for eligible trees; otherwise proximal for infinity-norm regression, APA for other supported objectives |
| `"topology"` | Exact structural knots; no coefficients |
| `"proximal"` | Certified coefficients at the chosen penalty; no full coefficient path |
| `"coefficient_path"` | Complete certified coefficient path using the fitted tree's diagonal structure |
| `"hicap"` | Generic, slower, certified coefficient-path reference |
| `"apa_apg2"` | Approximate point solution; also supports `ord=2` and binary classification |

Tree-specific solvers require `ord=np.inf`, zero/default `support_tol`, and an
unconstrained single-output mean-based regression tree evaluated on its fitting
rows and weights. Wrappers do not assume this for externally supplied
`prefit=True` trees or forest out-of-bag data. Incompatible explicit choices and
failed exact-solver certificates raise errors rather than silently falling back.

## Optional coefficients

**A non-CV model still represents your supplied `sp_alpha`, even when its solver
computes the full path.** Access the extra path without changing that model:

```python
from imodels import SPTreeRegressor

model = SPTreeRegressor(
    sp_alpha=1.0, solver="coefficient_path", max_leaf_nodes=32, random_state=0,
).fit(X, y)
predictions = model.predict(X)      # pruned tree at sp_alpha=1.0
beta = model.coef_                  # penalized coefficients at sp_alpha=1.0
path = model.coefficient_path_      # also available with solver="hicap"
beta_elsewhere, intercept = path.at(0.5 * path.lambdas[0])
```

`path.lambdas` contains coefficient knots; `path.coefficients` contains their
coefficient vectors. `path.at(alpha)` interpolates within the stored range
and returns a separate coefficient array plus intercept. It does **not**
change `model.coef_`, the pruned tree, or `predict()`. To fit the tree at another
penalty, fit another model with a different `sp_alpha`; changing a parameter
alone does not re-prune. CV models expose the same attributes at their selected
`sp_alpha_`.

For squared-error regression with `ord=np.inf`, the full coefficient path is
piecewise linear. A coefficient knot changes a coefficient's **slope**, not
necessarily the retained tree. Structural knots alone are therefore insufficient
for coefficient interpolation.

Coefficients describe unnormalized local stumps of the original, unpruned tree,
before HS. `coef_node_ids_` maps them to original split IDs, not IDs in the
compacted pruned tree. Preserve that original tree to reconstruct its feature
matrix. The structural solver instead exposes `pruning_path_` and leaves
`coef_` and `coefficient_path_` as `None`.

## Numerical behavior and performance

The structural path costs `O(p log p)` time and `O(p)` space after fitting,
where `p` is the number of splits. CV additionally pays for retained-tree copies
and held-out predictions. Complete coefficient paths store dense coefficient
rows and do more work; use them only when you need coefficients.

"Exact" means within floating-point certificate tolerances, not symbolic
arithmetic. `tol` controls those tolerances; for APA it only controls stopping.
`max_iter` caps iterative point updates or coefficient-path events, depending
on the solver. Eligible exact solvers agree on pruning at structural knots.
For compatibility, `sp_alpha=0` preserves all original splits, even zero-gain
ones. See the [optimization guide](optimization/README.md) for full contracts.

## Algorithms and references

- **Penalty and hiCAP:** [Zhao, Rocha, and Yu (2009)](https://arxiv.org/abs/0909.0411),
  Section 3.1.2. Our `hicap` follows active linear constraints of the infinity-norm
  formulation; it is independently implemented, not a MATLAB translation.
- **APA-APG2:** accelerated gradient steps plus averaged group-proximal updates
  with decreasing approximation, following Algorithm 2 of
  [Shen et al. (2017)](https://ojs.aaai.org/index.php/AAAI/article/view/10873).
  The precursor is [Yu (2013)](https://papers.nips.cc/paper_files/paper/2013/hash/49182f81e6a13cf5eaa496d51fea6406-Abstract.html).
  Warm starts sample a penalty grid; they do not enumerate exact knots.
- **Proximal:** child-to-parent group updates from
  [Jenatton et al. (2011), Section 3.4](https://jmlr.org/papers/v12/jenatton11a.html),
  inside [FISTA (Beck and Teboulle, 2009)](https://doi.org/10.1137/080716542)
  for general designs. The diagonal-tree specialization uses weighted
  projections and solves each penalty in one sweep.
- **Structural path:** a tree-specific reduction to weighted isotonic pooling
  (merging blocks to respect parent-child order). Heap-based tree pooling:
  [Pardalos and Xue (1999)](https://doi.org/10.1007/PL00009258).
  Our CAP-to-activation reduction is explained in the
  [tree-specific derivation](optimization/README.md#why-the-fitted-tree-gram-matrix-is-diagonal).
- **Coefficient path:** our diagonal-tree continuation combines structural
  bounds, exact point solves, and intervals with unchanged active constraints;
  see the [derivation](optimization/README.md#full-coefficient-path-for-a-fitted-tree).
- **HS and GCV:** [Agarwal et al. (2022)](https://arxiv.org/abs/2202.00858) for HS;
  [Golub, Heath, and Wahba (1979)](https://doi.org/10.1080/00401706.1979.10489751)
  for the GCV criterion, applied here conditional on the retained tree.

## Source layout

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
The standalone `tests/sparse_pruning_test.py` covers public behavior and core
mathematical invariants; `tests/sparse_pruning/` holds extended development tests.
Shared HS/GCV tests remain in `tests/hs_gcv_test.py`.
