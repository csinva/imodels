# Sparse pruning and hierarchical shrinkage

These estimators prune whole subtrees using the
[CAP/hiCAP penalty](https://arxiv.org/abs/0909.0411):
each group contains a split and all of its descendant splits.
Predictions use the retained CART node values, optionally followed by
hierarchical shrinkage (HS). Penalized coefficients determine pruning; they are
not the prediction weights of the resulting tree.

Use `SPTree*` for pruning alone and `SHSTree*` for pruning with HS. The
`sp_alpha` parameter controls pruning strength; `reg_param` controls HS without
changing the retained structure. Classes ending in `CV` select these strengths
using training/validation folds; non-CV classes accept fixed strengths.

## Quick start

```python
from sklearn.datasets import load_diabetes
from imodels import SPTreeRegressorCV, SHSTreeRegressorCV

X, y = load_diabetes(return_X_y=True)
pruned = SPTreeRegressorCV(max_leaf_nodes=32, random_state=0).fit(X, y)
shrunk = SHSTreeRegressorCV(max_leaf_nodes=32, random_state=0).fit(X, y)
predictions = pruned.predict(X)
```

The following sections describe automatic penalty selection, optional
coefficients, and the separate requirements for binary classification.

## Defaults and solver choices

`SPTreeRegressor`, `SHSTreeRegressor`, and their CV variants default to
`ord=np.inf, solver="auto"`. For an unconstrained, single-output regression
tree fitted by the wrapper with `squared_error` or `friedman_mse`, this chooses
the exact **structural** path. It uses fitted node statistics, not a dense
sample-by-split design matrix. Positive `support_tol` disables this shortcut.

The structural path lists penalties where the retained tree changes. Each such
penalty is a **structural knot**. It is sufficient for comparing pruned trees;
computing the penalized coefficients themselves is optional.

| `solver` | Result for an eligible fitted regression tree |
| --- | --- |
| `"auto"` | Structural path for eligible trees; otherwise `"proximal"` for regression with `ord=np.inf`, or `"apa_apg2"` |
| `"topology"` | Exact structural knots; no coefficient calculation |
| `"proximal"` | Certified coefficients at the chosen penalty; no full coefficient path |
| `"coefficient_path"` | Complete certified coefficient path using the fitted tree's diagonal structure |
| `"hicap"` | Generic, slower, certified coefficient-path reference |
| `"apa_apg2"` | Legacy approximate point solver; also supports `ord=2` and binary classification |

The tree-specific `"topology"` and `"coefficient_path"` solvers require the
samples and weights used to fit the tree. The wrappers do not assume this for
externally supplied `prefit=True` trees or forest out-of-bag data. Other
regression problems with `ord=np.inf` use design-based proximal solves under
`solver="auto"`. Explicitly incompatible solver choices raise an error, as do
failed exact-solver certificates; there is no silent numerical fallback.

## Cross-validation

For eligible regression trees, the default `sp_alpha_list="auto"` fits one
tree per training fold and scores each distinct retained tree once per HS
choice. Scores are aligned on the union of training-fold knots.
The final tree uses the selected absolute penalty;
it is not snapped to a full-data knot. No full-data tree generates CV candidates.
Custom scorers must depend on predictions/tree structure, not on the numerical
penalty label, for structural-state score reuse to be valid.

Regression CV defaults to negative mean squared error and the one-standard-error
rule: choose the simplest tree whose mean score is within one standard error
of the best. Use `selection_rule="best"` to select the highest mean CV score.
Inspect `sp_alpha_` and `reg_param_` for the selected strengths.

Numeric `sp_alpha_list` values request grid CV. Classification, `ord=2`,
`solver="apa_apg2"`, custom point solvers, and other ineligible cases use a
fixed numeric grid when given `"auto"`.
`cv_path_mode_` distinguishes `"structural"` from `"grid"`;
`cv_solver_` records the CV algorithm and `solver_` the final-fit algorithm.
`cv_sp_alphas_`, `cv_scores_`, and `cv_params_` describe the evaluated candidates;
structural CV also exposes `cv_path_results_` and `cv_n_pruning_states_`.

For eligible single-tree regression, exact solvers reuse their setup or complete
path across explicit grid penalties within each fold. APA instead solves each
candidate separately. All CV modes still pay for candidate evaluation and
prediction on validation rows.

Numeric HS strengths remain the default for SHS CV. With `reg_param_list="gcv"`,
each training state's HS strength is selected by conditional fixed-tree GCV,
then reselected after full-data pruning. GCV requires uniform positive weights
and node-based shrinkage; it does not account for learning the tree structure.

## Binary classification

Use `SPTreeClassifierCV` for pruning alone or `SHSTreeClassifierCV` to add
hierarchical shrinkage after pruning. Both can be imported directly from
`imodels` and provide `predict` and `predict_proba`. Their non-CV counterparts,
`SPTreeClassifier` and `SHSTreeClassifier`, take a fixed `sp_alpha`.

Classification uses the APA-APG2 logistic solver and numeric-grid CV, not the
exact structural or coefficient paths for squared-error regression. With
`sp_alpha_list="auto"`, it uses the historical numeric grid. CV defaults to
accuracy; set `scoring` to choose another supported scorer. HS strengths must
be numeric: automatic GCV is regression-only. Multiclass classification is
not supported.

## Automatic HS with GCV

To select HS by GCV within each training fold's pruning state:

```python
auto_hs = SHSTreeRegressorCV(
    max_leaf_nodes=32, reg_param_list="gcv", random_state=0,
).fit(X, y)
```

For a single regression tree, automatic node-based HS is available through
`HSTreeRegressor(reg_param="gcv")` or `SHSTreeRegressor(reg_param="gcv")`.
The latter selects HS after sparse pruning; `reg_param=None` is a compatibility
alias in sparse-HS wrappers, while ordinary `HSTreeRegressor` requires the
explicit `"gcv"` value. Inspect `reg_param_` for the selected strength and
`gcv_results_` for candidate scores, effective degrees of freedom, and search
diagnostics. A selected strength of infinity means root-mean predictions.

GCV uses retained node statistics without constructing a dense training design.
It supports single-output trees with `squared_error` or `friedman_mse` criteria,
uniform positive observation weights, and no active monotonic constraints.
This is **conditional-on-the-fitted-tree GCV**: it does not account for choosing
the tree or pruning it from the same targets, and is not exact leave-one-out
refitting. Ordinary CV remains the default; forests, classifiers, nonuniform
weights, and other shrinkage schemes require explicit strengths or CV.

## Optional coefficients

For squared-error regression with `ord=np.inf`, a complete coefficient path is
piecewise linear in `sp_alpha`. A **coefficient knot** is a penalty where a
coefficient's slope changes, not every point where its value changes. Such a knot need not change
the retained tree, so structural knots alone are insufficient for exact
coefficient interpolation.

```python
model = SHSTreeRegressorCV(
    solver="coefficient_path", max_leaf_nodes=32, random_state=0,
).fit(X, y)
beta = model.coef_                  # coefficients at the selected penalty
original_node_ids = model.coef_node_ids_
path = model.coefficient_path_      # all coefficient-direction knots
beta_elsewhere, intercept_elsewhere = path.at(0.5 * path.lambdas[0])
```

For eligible regression trees, automatic CV still scores structural states
when `solver="coefficient_path"`, while the final fit computes coefficients.
Coefficients refer to **unnormalized local stumps of the original, unpruned
tree**, before HS. Original node IDs are not the renumbered IDs in the compact
pruned tree. Preserve the original tree if
you need to reconstruct its feature matrix. `"topology"` leaves `coef_` and
`coefficient_path_` as `None` and exposes `pruning_path_` instead.

## Numerical behavior and performance

`max_iter` caps APA/FISTA iterations for iterative point solves, but caps
coefficient-path events for `"coefficient_path"` and `"hicap"`. The tree-specific
diagonal point solve and structural path do not use iterative optimization.
`tol` sets numerical accuracy/certificate tolerances for exact coefficient
solvers; for APA it is a stopping tolerance, not an exactness guarantee.
"Exact" and "certified" refer to floating-point numerical tolerances, not
symbolic arithmetic. `support_tol` controls coefficient thresholding separately
and must be zero/default for the tree-specific paths.

On eligible fitting data, all exact solvers use the same structural events for
pruning, including right at a knot. Tiny floating-point residuals in generic
hiCAP coefficients cannot change that topology. Elsewhere, coefficient-based
pruning uses a response-scaled numerical threshold by default. `support_tol=0`
requests literal coefficient zero there, which can be sensitive to roundoff.
For compatibility, `sp_alpha=0` always preserves the original topology, including
zero-gain splits.

The fitted-tree structural path takes `O(p log p)` time after fitting and `O(p)`
space for `p` split nodes. CV still pays for held-out prediction and one compact
tree copy per local state; it is not an `O(p log p)` end-to-end procedure. Complete
coefficient paths additionally store dense coefficient rows and rebuild faces;
they are optional precisely because pruning does not need that work.

## Algorithms and references

The penalty and original hiCAP path algorithm come from Zhao, Rocha, and Yu
(2009), [*The composite absolute penalties family for grouped and hierarchical
variable selection*](https://arxiv.org/abs/0909.0411), especially Section 3.1.2.
The solvers here target that objective through different computational methods:

- `hicap` represents the infinity-norm penalties as linear constraints and
  follows intervals where the active constraints satisfy the optimality (KKT)
  conditions. It is independently implemented from the convex formulation,
  not a translation of the authors' MATLAB code.
- `apa_apg2` combines accelerated loss-gradient steps with an average of
  individual group-proximal updates, decreasing the approximation parameter
  over iterations. It applies Algorithm 2 of
  [Shen et al. (2017), *Adaptive Proximal Average Approximation for Composite
  Convex Minimization*](https://ojs.aaai.org/index.php/AAAI/article/view/10873).
  The proximal-average predecessor is
  [Yu (2013), *Better Approximation and Faster Algorithm Using the Proximal
  Average*](https://papers.nips.cc/paper_files/paper/2013/hash/49182f81e6a13cf5eaa496d51fea6406-Abstract.html).
  Warm-start path functions sample a supplied penalty grid; they do not
  enumerate exact knots.
- `proximal` composes group-proximal maps from children to parents, following
  [Jenatton et al. (2011), *Proximal Methods for Hierarchical Sparse
  Coding*](https://jmlr.org/papers/v12/jenatton11a.html), Section 3.4.
  For a general design, these maps are used inside
  [Beck and Teboulle's FISTA (2009)](https://doi.org/10.1137/080716542).
  Our diagonal-tree specialization rescales coordinates and uses weighted
  projections to solve each penalty in one proximal sweep.
- `topology` converts fitted-tree statistics into activation penalties through
  weighted tree-isotonic pooling: adjacent blocks are merged until their
  values respect the parent-child order. Heap-based tree pooling is described
  by [Pardalos and Xue (1999), *Algorithms for a Class of Isotonic Regression
  Problems*](https://doi.org/10.1007/PL00009258). The reduction from this CAP
  objective to activation penalties is documented in the
  [tree-specific derivation](optimization/README.md#why-the-fitted-tree-gram-matrix-is-diagonal);
  it is not the original hiCAP coefficient-path algorithm.
- `coefficient_path` combines the structural bounds with exact diagonal point
  solves. At each step, it identifies which coefficient constraints are
  active and calculates the full penalty interval where they remain valid.
  This is an implementation-specific diagonal-tree continuation method; see
  the [coefficient-path derivation](optimization/README.md#full-coefficient-path-for-a-fitted-tree).

HS after pruning follows [Agarwal et al. (2022), *Hierarchical
Shrinkage*](https://arxiv.org/abs/2202.00858). Automatic HS applies the
[GCV criterion of Golub, Heath, and Wahba (1979)](https://doi.org/10.1080/00401706.1979.10489751)
to the retained fixed-tree smoother; it does not account for learning or
selecting the tree itself.

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
