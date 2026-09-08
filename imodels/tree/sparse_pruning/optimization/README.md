# Sparse-pruning regularization paths

This folder keeps regularization-path computation separate from estimator
mutation. It provides several deliberately different path contracts for the
normalized objective

```text
loss(beta) + lambda * sum_G ||beta[G]||_q.
```

Numerical implementations live in this directory. The sklearn-specific
adapter lives in [`../fitted_tree.py`](../fitted_tree.py). Estimator users should
start with the [sparse-pruning guide](../README.md); this document describes
the lower-level mathematical APIs, not additional estimator choices.

The homotopy implementation is derived independently from the convex
epigraph/KKT formulation in the
[CAP/hiCAP paper](https://arxiv.org/abs/0909.0411). The historical MATLAB
archives are useful behavioral references, but their noncommercial license is
kept separate from this MIT-licensed implementation.

- `hicap_regression_path` computes the complete, piecewise-affine path for
  squared loss, `q = infinity`, and a rooted laminar group family. It returns
  every coefficient-direction breakpoint from `lambda_max` through zero and
  sets `exact=True` only after KKT certification. An unpenalized intercept and
  relative sample weights are supported. A numerically rank-deficient
  weighted centered design has no unique coefficient path, so that case is
  returned explicitly as `status="nonunique_design"`, `exact=False`, with
  endpoint solutions only.
- `apa_apg_regression_path` and `apa_apg_classification_path` evaluate a
  user-provided lambda grid with descending continuation. They support both
  `q = 2` and `q = infinity`, but remain sampled numerical paths and therefore
  always return `exact=False`. With overlapping groups, the current APA-APG2
  point solver uses its full iteration budget; a coefficient warm start can
  improve the finite-iteration answer but does not itself provide an exact
  breakpoint path or a KKT certificate. The returned path has
  `status="partial"` when any sampled point does not meet the point solver's
  stopping test.
- `laminar_group_linf_regression_path` uses the exact laminar-tree proximal
  map. For a general Gram matrix it runs FISTA with continuous stationarity
  diagnostics; for a positive diagonal Gram matrix, every requested lambda is
  instead solved exactly by one child-to-parent proximal sweep. The stored
  point solutions can be certified (against the retained full Gram when it was
  formed), but an arbitrary grid still has
  `exact=False` because it does not enumerate coefficient-slope knots.
- `tree_group_linf_exact_coefficient_path` accepts a positive Gram diagonal,
  linear scores, and parent indices (one coefficient and descendant group per
  node). It identifies a clipping face with the exact diagonal point solver,
  then derives and certifies that face's affine lambda interval. This
  enumerates structural **and coefficient-only** knots. The initial version
  rebuilds each face and materializes descendant groups and dense coefficient
  rows; its cost differs from the near-linear structural path.
- `fitted_tree_linf_exact_coefficient_path` exposes that coefficient path for
  a fitted single-output regression tree. It extracts sufficient statistics
  from the stored node weights and means and adds the constant root-mean
  intercept. It avoids a training design matrix but still constructs groups
  internally. Its coefficients use **unnormalized** local stumps and follow
  `metadata['tree_node_ids']`.
- `laminar_group_linf_exact_topology_path` targets the path object needed to
  visualize pruned trees. For a positive diagonal Gram and positive group
  weights, one weighted tree-isotonic pass returns every exact zero-support
  group/topology knot. Its result is a compact `TreeTopologyPath`, with tied
  entering-group batches and both at-knot and just-below topology conventions.
  Applying `iter_events()` to a mutable active-node mask traverses the full
  path in linear output space. Coefficients at all knots are optional, because
  materializing `p` coefficients at `K` knots already costs `O(p K)`.
- `fitted_tree_linf_exact_topology_path` is the large-tree specialization for
  a fitted single-output `DecisionTreeRegressor`. It derives each local-stump
  score directly from child weights and child predictions stored by sklearn,
  so it constructs neither an `n`-by-`p` stump matrix nor descendant groups.
  Its post-fit path cost is `O(p log p)` and does not depend on `n`.
- `materialize_fitted_tree_topology` makes a plot-ready deep copy at one exact
  state by turning inactive split nodes into leaves. It never mutates the
  source tree and intentionally keeps original node IDs. The unreachable nodes
  remain allocated, so structural metadata need not describe the visible
  preview and serialized size remains that of the non-compacted backing tree.

## Why the fitted-tree Gram matrix is diagonal

For a tree node `v`, let its two child weights be `L` and `R`. The unnormalized
local stump is `-sqrt(R/L)` on the left child, `sqrt(L/R)` on the right child,
and zero outside the node. Therefore its weighted sum is zero and its squared
weighted norm is `L + R = W_v`. Two node regions are either disjoint or one is
inside one child of the other; in the latter case the ancestor stump is
constant on the descendant region and the descendant's weighted sum is zero.
Consequently, on the rows and weights used to fit the tree,

```text
Z.T @ W @ Z = diag(W_v),       Z.T @ W @ 1 = 0.
```

For the loss normalized by total weight `W_total`, the Gram diagonal is
`W_v / W_total`. Calling `make_stumps(..., normalize=True)` instead makes the
raw weighted Gram the identity (and the normalized-loss Gram
`I / W_total`). Weighted centering leaves this structure unchanged.

This identity is measure-specific. OOB rows, held-out rows, a prefit tree with
different data, changed sample weights, or full-data rows for a
bootstrap-fitted tree generally destroy it. `cache_quadratic=True` safely forms
and checks the Gram once. When the fitted-tree construction guarantees the
same rows and weights, `assume_diagonal_gram=True` is the matrix-free fast path:
it stores only the diagonal in `O(p)` memory. The assumption is explicit
because using it on a nonorthogonal design changes the optimization problem.
Without that flag, the solver forms a dense Gram matrix to verify numerical
orthogonality, so the end-to-end call is not near-linear in a large `p`.

For native-missing-value trees, some sklearn releases can store fitting child
statistics that differ from the partition obtained by reapplying the fitted
tree to the original NaN-containing rows. The fitted-tree shortcut remains
exact for the implicit stored fitting partition. To target a stump matrix
reconstructed from `X`, construct it explicitly and use the design-input API.

For a positive diagonal Gram `D`, set `theta = sqrt(D) * beta`. The hiCAP
objective becomes an ordinary squared-distance proximal problem with the same
laminar groups and shared coordinate weights `1 / sqrt(diag(D))`. The exact
laminar proximal composition therefore solves a complete fixed-lambda
regression problem in one bottom-up pass. No APA smoothing schedule, optimizer
iteration, or warm start is required.

The structural knots admit an even cheaper representation. For each group
`g`, assign the coordinates not owned by a child group to its atom and set
`Q_g` to the sum of their absolute quadratic scores. Weighted decreasing
tree-isotonic regression of `Q_g / group_weight_g` gives activation values
`t_g`; the exact retained topology at lambda is `{g: t_g > lambda}`. Distinct
positive `t_g` values are all zero-support topology knots. The Gram diagonal
affects coefficient magnitudes but not these activation penalties.

For a fitted regression tree, every descendant-group atom contains exactly
the coefficient of its own internal node. If node `v` has child weights
`L_v, R_v`, child predictions `mu_L, mu_R`, and the root has weight `W`, then

```text
h_v = sqrt(L_v * R_v) * (mu_R - mu_L) / W,
D_vv = (L_v + R_v) / W.
```

Thus sklearn's stored node statistics are already sufficient for the exact
structural path. This statement uses the tree-fitting rows and weights and a
mean-based regression criterion; it is not an OOB or held-out-data result.

All group indices are zero-based. The exact solver accepts a design without
an explicit intercept column and uses `fit_intercept=True` by default. The
APA path follows the existing point-solver convention, so an unpenalized
explicit intercept column can be omitted from every group.

In the example below, `X` is a centered local-stump design and `y` is centered;
`tree_regressor` is the corresponding fitted source tree, not the design matrix.

```python
from imodels.tree.sparse_pruning.optimization import (
    apa_apg_regression_path,
    hicap_regression_path,
    laminar_group_linf_exact_topology_path,
    laminar_group_linf_regression_path,
)
from imodels.tree.sparse_pruning.fitted_tree import (
    fitted_tree_linf_exact_topology_path,
    materialize_fitted_tree_topology,
)

exact = hicap_regression_path(X, y, subtree_groups)
beta, intercept = exact.at(lambda_value)

sampled = apa_apg_regression_path(
    X,
    y,
    subtree_groups,
    lambdas=lambda_grid,
    ord="inf",
    assume_diagonal_gram=True,  # only for matched fitted-tree rows/weights
)

exact_points = laminar_group_linf_regression_path(
    X,
    y,
    subtree_groups,
    lambdas=lambda_grid,
    fit_intercept=False,
    assume_diagonal_gram=True,
)

tree_path = laminar_group_linf_exact_topology_path(
    X,
    y,
    subtree_groups,
    fit_intercept=False,
    assume_diagonal_gram=True,
    include_coefficients=False,
)
topology_at_knot = tree_path.topology_at(tree_path.lambdas[0])
topology_just_below = tree_path.topology_at(
    tree_path.lambdas[0], below=True
)

# Scalable traversal: each group/node appears in at most one batch.
active_nodes = set()
# Render the initial no-split state here.
for knot, entering_nodes in tree_path.iter_events():
    active_nodes.update(entering_nodes)  # apply every tied batch atomically
    # Render or update the corresponding tree here.

# Preferred after fitting a large sklearn regression tree: no X_tree matrix.
native_tree_path = fitted_tree_linf_exact_topology_path(tree_regressor)
preview = materialize_fitted_tree_topology(
    tree_regressor, native_tree_path, native_tree_path.lambdas[0]
)
# Plot/save the initial exact-at-knot state once.
for knot, sklearn_node_ids in native_tree_path.iter_node_events():
    preview = materialize_fitted_tree_topology(
        tree_regressor, native_tree_path, knot, below=True
    )
    # Plot/save `preview`, the state immediately below this knot.

# Or initialize the exact lambda=0 topology and increase the penalty.
active_node_ids = set(native_tree_path.tree_nodes_at(0.0))
# Render this lambda-zero state once before removing any batch.
for knot, sklearn_node_ids in native_tree_path.iter_node_pruning_events():
    active_node_ids.difference_update(sklearn_node_ids)
    # Render the state at `knot` after removing the whole tied batch.
```

The fitted-tree constructor costs `O(p log p)` time and `O(p)` storage, and
the delta-event traversal costs `O(p)` total. There are `K + 1` states for `K`
knots. Materializing or drawing every full tree necessarily costs `O(p K)`
work/output (and `O(p K)` memory if all copies are retained), so stream one
copy or figure at a time for `O(p)` live memory. Returned node IDs identify
split nodes; leaves are implicit. The materialization helper displays original
CART node values, not hiCAP-shrunken coefficients. Because it preserves the
original node IDs without compacting the backing arrays, `tree_.node_count`,
`get_depth()`, `get_n_leaves()`, and serialized size can overstate the reachable
preview; use the rendered structure itself for visualization.

The fitted-tree structural API avoids both the design matrix and explicit
groups. The fitted-tree coefficient API also avoids the design, but uses
explicit descendant groups for its point oracle. The generic design/group
point solver remains available for isolated coefficient queries. Its work is
proportional to the total descendant-group memberships—typically
`O(p log p)` for a balanced tree but `O(p^2)` for a degenerate chain. This does
not affect support-only pruning or previews based on the original CART node
values.

If `include_coefficients=True`, row `k` is evaluated exactly at
`tree_path.lambdas[k]` and therefore depicts the strict pre-entry topology.
For the newly entered state, evaluate the one-sweep point solver at any lambda
strictly between that knot and the next one. Coordinates omitted from every
penalty group are unpenalized and are not represented in the group topology.
Likewise, a node with activation value exactly zero is absent even from the
strict lambda-zero topology and does not appear in a positive pruning-event
batch; initialize an ascending traversal with `tree_nodes_at(0.0)` as above.

`RegularizationPath.at` is exact between adjacent knots only when
`path.exact` is true. For an APA path it is ordinary linear interpolation
between sampled solutions.

Exact structural knots are not the same as all coefficient knots. A clipping
face can change a coefficient's slope without changing whether any subtree is
retained. Use the diagonal coefficient homotopy for every coefficient-direction
knot of the fitted-tree training objective, or the generic hiCAP homotopy for
small non-diagonal problems. The exact topology path plus lazy coefficient
queries remains useful when only distinct tree structures need rendering.

## Full coefficient path for a fitted tree

```python
from imodels.tree.sparse_pruning.fitted_tree import (
    fitted_tree_linf_exact_coefficient_path,
)
from imodels.tree.sparse_pruning.optimization import (
    tree_group_linf_exact_coefficient_path,
)

path = fitted_tree_linf_exact_coefficient_path(tree_regressor)
if path.exact and path.status == "complete":
    beta, intercept = path.at(0.5 * path.lambdas[0])
    # beta columns correspond to path.metadata["tree_node_ids"].
    # Prediction: intercept + unnormalized_local_stump_design @ beta.

# Or directly from a diagonal quadratic:
path = tree_group_linf_exact_coefficient_path(
    linear_scores=[1.0, 2.0],
    gram_diagonal=[1.0, 1.0],
    parent_indices=[-1, 0],
)
# Full knots: [1.5, 0.5, 0.0]. Green's structural knots: [1.5, 0.0].
# At lambda=0.5 beta=[1,1]; thereafter beta[0] is constant while beta[1]
# continues to change. Both coefficients stay nonzero.
```

The solver assumes squared loss and positive diagonal curvature and group
weights. The score API defines this quadratic directly; it cannot verify
whether an original design was diagonal. The fitted-tree wrapper inherits
the training-measure and criterion restrictions above. It must not be used
as a held-out/OOB coefficient path by substituting the fitting statistics.

For each positive connected block of equal subtree magnitudes, set
`c = abs(h) / D` and let `A` contain its still-clipped coordinates. Then

```text
q_B(lambda) = (sum_A abs(h) - lambda * sum_B group_weight) / sum_A D
beta_v(lambda) = sign(h_v) * min(c_v, q_B(lambda)).
```

The maximal validity interval follows from the linear clipping,
parent/child, nonnegativity, and dual-multiplier inequalities. Their roots
include merges, splits, saturation, and structural events. Green supplies
zero-group activation thresholds; dark blue identifies a face at an interior
lambda; interval certificates establish coverage between samples. Legacy
hiCAP is used in independent tests, not by the production solver.

`exact=True` is numerical certification, not symbolic rational arithmetic.
The requested tolerance has a machine-precision floor of `1024 * eps`,
reported as `effective_certificate_tolerance`. Independently evaluated
neighboring coefficient-face boundaries use a `512 * eps` relative roundoff
floor (`relative_event_roundoff_tolerance`) plus cancellation-aware uncertainty
estimates from the affine constraint formulas. Each interval records those
absolute boundary uncertainties. Coefficient events below this numerical
resolution are not symbolically distinguished. Known distinct structural
events constrain probes even when their spacing is below the optimization
tolerance. If two such penalties are adjacent floating-point numbers, a
boundary probe is accepted only when its affine interval covers both endpoints;
`n_boundary_probes` reports how often this occurs. Always check `status` and
`exact`: reaching an
event cap or failing to resolve a face returns an explicitly nonexact prefix.
The first version builds dense coefficient rows, requiring `O(p K)` storage
for `p` splits and `K` stored knots. It also rebuilds the point oracle's face
after every event and stores explicit group memberships (potentially
`O(p^2)` for chains). It is not an incremental dynamic-tree implementation.
Once constructed, `path.at(lam)` uses binary search and only the neighboring
coefficient rows, taking `O(log K + p)` time without copying the full path.
Result arrays are owned, read-only snapshots; changing caller arrays cannot
invalidate an existing path. Queries outside the stored range are rejected,
including lambda zero on an incomplete prefix. A partial path can have
certified stored points without having complete coefficient-event coverage;
check `exact`/`status` separately from the point-certificate metadata.

Positive proximal group radii must remain representable. If their calculation
underflows to zero, point APIs reject the unsupported scale instead of dropping
the penalty. During coefficient continuation such a failure returns an
explicitly nonexact prefix. Tree penalty values are computed by a bottom-up
`O(p K)` sweep, and fitted-tree statistics are extracted once per path.

## Classification and other losses

The structural and full affine coefficient paths above solve a squared-error
pruning objective. They are not logistic-loss paths. Binary classification
uses the sampled APA solver; logistic coefficients are generally not
piecewise affine in the penalty. For non-diagonal **squared-loss** problems,
the exact laminar proximal map inside FISTA gives certified requested points,
not a complete knot path.
