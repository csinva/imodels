# Sparse-pruning regularization paths

This note explains the structural and coefficient solvers behind sparse
pruning. For fitting models, examples, and solver options, start with the
[sparse-pruning guide](../README.md).

For regression, the normalized objective is

```text
0.5 * sum_i w_i (y_i - intercept - z_i @ beta)^2 / sum_i w_i
    + lambda * sum_g a_g ||beta[G_g]||_inf,
```

where the groups form a rooted laminar family and `a_g > 0`. In a tree,
`G_g` contains a split and all its descendant splits. A group is retained
while any coefficient in that group is nonzero.

The homotopy implementation is derived from the convex epigraph/KKT
formulation in the [CAP/hiCAP paper](https://arxiv.org/abs/0909.0411).
Historical MATLAB implementations serve as behavioral references; their
noncommercial license is separate from this MIT-licensed implementation.

## Why the fitted-tree Gram matrix is diagonal

At node `v`, let the child weights be `L` and `R`. Its unnormalized local
stump `z_v` is `-sqrt(R/L)` on the left child, `sqrt(L/R)` on the right child,
and zero outside the node. Two identities follow immediately:

```text
sum_i w_i z_iv      = L * (-sqrt(R/L)) + R * sqrt(L/R) = -sqrt(LR) + sqrt(LR) = 0
sum_i w_i z_iv^2    = L * (R/L)        + R * (L/R)     = R + L = W_v
```

so each stump is weighted zero-mean with weighted squared norm `W_v`.

For two distinct nodes `u` and `v`, their regions are either disjoint or
nested, because the tree partitions recursively.

- **Disjoint.** The product `z_iu * z_iv` is zero at every row, so the inner
  product vanishes.
- **Nested**, say `region(v)` inside `region(u)`. Then `z_u` is constant on
  `region(v)`, taking one value `k` on the whole of it, because `region(v)`
  lies entirely within a single child of `u`. Hence
  `sum_i w_i z_iu z_iv = k * sum_{i in region(v)} w_i z_iv = 0`, using the
  zero-mean identity for `z_v`.

Therefore, on the fitting rows and weights,

```text
Z.T @ W @ Z = diag(W_v),       Z.T @ W @ 1 = 0.
```

The same two identities give the linear scores, since
`sum_i w_i z_iv y_i = -sqrt(LR) * mu_L + sqrt(LR) * mu_R`.

With loss normalized by total weight `W_total`, the Gram diagonal is
`D_vv = W_v / W_total`. With `make_stumps(..., normalize=True)`, the raw
weighted Gram is the identity instead. Weighted centering changes neither.

For a fitted mean-based regression tree, stored node statistics also give the
linear scores:

```text
h_v = sqrt(L_v * R_v) * (mu_R - mu_L) / W_total,
D_vv = (L_v + R_v) / W_total.
```

No observation-by-split design matrix is needed. These identities depend on
using the tree's fitting measure: changing rows or weights, including using
held-out or OOB data, generally destroys orthogonality. Native-missing-value
trees use their stored fitting partition, which some sklearn versions may not
reproduce when reapplied to the same NaN-containing rows.

For design-input APIs, `assume_diagonal_gram=True` explicitly asserts this
matched geometry and stores only the diagonal. Without it,
`cache_quadratic=True` forms and checks the full Gram. Use an explicit design
when targeting a different partition or measure.

## Structural path

For positive diagonal `D`, the change of variables
`theta = sqrt(D) * beta` turns the quadratic objective into a squared-distance
proximal problem with coordinate weights `1 / sqrt(D)`. Exact laminar
proximal composition solves a fixed penalty in one child-to-parent sweep.

Finding only the zero groups is cheaper still, and needs neither `D` nor any
coefficient. The result is that the retained set is

```text
retained groups at lambda = {g : t_g > lambda},
```

where `t` is the weighted antitone tree-isotonic regression of `Q_g / a_g` and
`Q_g` is the atom mass defined below. The rest of this section derives that,
because the statement is easy to misread: the obvious candidate formula is not
`t`, and gives a set that is not even a tree.

### 1. The zero-subtree condition does not involve the curvature

Let `S = subtree(v)` and suppose `beta = 0` on `S`. The gradient of the
quadratic part at such a point is `D @ beta - h = -h` on `S`, so `D` cancels
and stationarity on `S` reads

```text
h_u  in  sum over groups g containing u of  lambda * a_g * (subdifferential of ||.||_inf at 0)
```

for every `u` in `S`. Every quantity left is built from `h`, the tree, and the
group weights `a`. This is why the structural path is invariant to `D`:
curvature rescales surviving coefficients, but never decides which subtrees
survive.

### 2. Only groups rooted inside the subtree can help

The groups containing a coordinate `u` are those rooted at its ancestors. If
such a group `g` is *active*, meaning `max abs(beta)` over `G_g` is positive,
its `||.||_inf` subdifferential is supported on the maximizing coordinates,
which lie outside the zero set `S`. An active ancestor group therefore
contributes nothing to `S`.

Let `v` be the root of a *maximal* zero subtree, so its parent is retained.
Every strict-ancestor group of `v` is then active, and only groups rooted
inside `S` can supply mass.

### 3. Feasibility is a transportation problem

Signs are free, so stationarity on `S` is solvable exactly when a nonnegative
flow exists from group budgets to coordinate demands:

```text
demand(u)      = abs(h_u)          for u in S
supply(g)      = lambda * a_g      for g rooted in S
g may serve u  iff u in G_g        (g is an ancestor-or-self of u)
```

By the Gale-Hall criterion this is feasible exactly when
`demand(A) <= supply(N(A))` for every coordinate set `A`, where `N(A)` is the
set of groups able to serve `A`. For any `A`, `N(A)` is the ancestor closure
`T` of `A` inside `S`. That `T` is connected, contains `v`, and satisfies
`demand(A) <= demand(T)` with `supply(N(A)) = supply(T)`. The binding sets are
therefore exactly the connected subsets containing `v`, for which `N(T) = T`:

```text
subtree(v) can be zero   <=>   lambda >= m(v),
m(v) = max of Q(T) / a(T) over connected T inside subtree(v) with v in T.
```

### 4. Why that maximum average is not the activation

`m(v)` is a *conditional* threshold. It assumed in step 2 that the parent of
`v` is retained, so that ancestor budgets were already committed. On its own
it is wrong, because `m` need not decrease down the tree, so
`{v : m(v) > lambda}` need not be ancestor closed.

Take the chain `0 -> 1 -> 2` with `a = 1` and `abs(h) = (1, 1.2, 1.4)`:

```text
m            = (1.2, 1.3, 1.4)     not antitone
t            = (1.2, 1.2, 1.2)     pooled activation
at lambda = 1.25:
  {m > lambda} = {1, 2}            drops the root but keeps its child
  {t > lambda} = {}                the true answer
```

At `lambda = 1.25` the whole tree is zero for every positive `D`, because the
root group's own budget can serve the deepest coordinate: total demand `3.6`
is below total supply `3 * 1.25`. Step 3 never applies here, since no parent
is retained.

### 5. Antitone pooling resolves the interaction

Let `t` solve the weighted antitone tree-isotonic problem

```text
minimize  sum_g a_g * (t_g - Q_g / a_g)^2    subject to   t_parent >= t_child.
```

Three facts connect `t` to the retained set.

1. `t` is antitone, so `{t > lambda}` is ancestor closed and is a valid pruned
   tree.
2. Call `g` a *block root* when it is a forest root or its fitted value is
   strictly below its parent's. Its pooling block never extends above it, so
   `t_g` is determined inside `subtree(g)` alone, where the block-root value of
   antitone tree-isotonic regression is the maximum average over rooted
   connected subsets. Hence `t_g = m(g)` at every block root.
3. Therefore `{t > lambda}` is exactly the retained set:
   - If `t_v <= lambda`, let `b` be the highest ancestor with `t_b <= lambda`.
     Its parent has `t > lambda`, so `b` is a block root and
     `m(b) = t_b <= lambda`. By step 3 the whole of `subtree(b)`, which
     contains `v`, is zero.
   - If `t_v > lambda` but `subtree(v)` were zero, then `v` lies in a maximal
     zero subtree rooted at some ancestor `b`, giving `lambda >= m(b) = t_b`
     and `t_b >= t_v` by antitonicity, contradicting `t_v > lambda`.

The distinct positive values of `t` are exactly the structural knots. Pooling
uses the leftist-heap tree PAVA of Pardalos and Xue, which inserts and removes
each block at most once, for `O(p log p)` time and `O(p)` space.

### 6. Atoms, when a group owns several coordinates

For a fitted tree each group's atom is its own split coefficient, so
`Q_v = abs(h_v)`. In a general laminar family a group may own several
coordinates. The groups containing a coordinate form a chain, so the
coordinates that *only* the groups in an ancestor-closed set `U` can serve are
those whose smallest containing group lies in `U`. Assign each coordinate to
that smallest group, sum `abs(h)` over it to get the atom mass `Q_g`, and the
Hall condition becomes the same comparison of `Q(U)` against `lambda * a(U)`
over rooted connected `U`. Coordinates in no group are unpenalized and never
appear in the group topology.

### Computing the path

`fitted_tree_linf_exact_topology_path` extracts the scores directly from the
tree. For `p` splits it takes `O(p log p)` time and `O(p)` storage after
fitting, without a design matrix or explicit descendant groups.

```python
from imodels.tree.sparse_pruning import (
    fitted_tree_linf_exact_topology_path,
    materialize_fitted_tree_topology,
)

structure = fitted_tree_linf_exact_topology_path(tree)
active = set(structure.tree_nodes_at(0.0))
for lam, removed_node_ids in structure.iter_node_pruning_events():
    active.difference_update(removed_node_ids)
    # Update a drawing, or create one preview:
    preview = materialize_fitted_tree_topology(tree, structure, lam)
```

At a knot the departing group is already zero; `below=True` returns the
state just below it. Apply tied events as a batch. Zero-activation splits are
absent even at lambda zero, so initialize from `tree_nodes_at(0.0)`, not
all source splits. Coordinates outside every penalty group are unpenalized
and do not appear in the group topology.

Delta-event traversal costs `O(p)` total. Drawing every full tree can still
require `O(p K)` output for `K` knots; stream previews instead of keeping
all copies. A preview preserves original CART node values and IDs, not
optimized coefficients. Its backing arrays retain unreachable nodes, so use
the path's reachable split count rather than sklearn's un-compacted metadata.

## Full coefficient path for a fitted tree

A structural event changes which subtrees are retained. A coefficient event
can also change a slope without removing a subtree. Squared-loss hiCAP is
piecewise affine, so enumerating **all coefficient events** allows exact
linear interpolation.

The diagonal homotopy identifies a clipping face with the exact point solver,
then certifies the face's affine interval. For each positive connected block
`B` of equal subtree magnitudes, write `c = abs(h) / D` and let `A`
contain its still-clipped coordinates:

```text
q_B(lambda) = (sum_A abs(h) - lambda * sum_B a_g) / sum_A D
beta_v(lambda) = sign(h_v) * min(c_v, q_B(lambda)).
```

Linear clipping, parent/child, nonnegativity, and dual-multiplier inequalities
bound the interval. Their roots include merges, splits, saturation, and
structural events. Structural activation values constrain the probes;
interval certificates establish coverage between them. Generic hiCAP is an
independent test reference, not part of this solver's production path.

```python
from imodels.tree.sparse_pruning import fitted_tree_linf_exact_coefficient_path
from imodels.tree.sparse_pruning.optimization import (
    tree_group_linf_exact_coefficient_path,
)

path = fitted_tree_linf_exact_coefficient_path(tree_regressor)
if path.exact and path.status == "complete":
    beta, intercept = path.at(0.5 * path.lambdas[0])
    # beta uses unnormalized local stumps; columns follow
    # path.metadata["tree_node_ids"]. The intercept is the root mean.

example = tree_group_linf_exact_coefficient_path(
    linear_scores=[1.0, 2.0],
    gram_diagonal=[1.0, 1.0],
    parent_indices=[-1, 0],
)
# Coefficient knots: [1.5, 0.5, 0.0]; structural path: [1.5, 0.0].
# At 0.5 the coefficients meet at [1, 1]; neither becomes zero.
```

The score API assumes positive diagonal curvature and positive group weights;
it cannot check the original design. The fitted-tree adapter inherits the
fitting-measure restrictions above.

This solver stores dense coefficient rows (`O(p K)`) and rebuilds faces
using explicit descendant groups. Those memberships can cost `O(p^2)` in a
chain, versus `O(p log p)` for a balanced tree. The structural solver avoids
that cost. Once built, `path.at(lam)` takes `O(log K + p)` time.

## Classification and other losses

Binary logistic and multiclass softmax coefficient solves use the exact
laminar proximal map inside accelerated proximal gradient, with backtracking,
an unpenalized intercept, and descending warm starts. Fitted-tree adapters
use leaf class proportions and masses rather than reconstructing the design.
Use `fitted_tree_linf_classification(tree, lam)` for a point solve, or
`fitted_tree_linf_classification_path(tree, lambdas)` for a custom grid;
the latter accepts `adaptive_tol` for midpoint refinement.

For a coefficient matrix `B` (splits by classes), the penalty is

```text
sum_g a_g * max_{v in subtree(g)} (max_c B[v,c] - min_c B[v,c]).
```

Binary contrasts reduce to scalar-logit hiCAP at the same lambda, since a
two-class row `(-b/2, b/2)` has range `abs(b)`.

**Gauge lemma.** Write
`Omega_inf(B) = sum_g a_g max_{v in G_g} max_c abs(B[v,c])`, and let
`Omega_range` be the penalty above. Then

```text
min over row shifts s of  2 * Omega_inf(B - s 1^T)  =  Omega_range(B),
```

attained at the per-row midrange `s_v = (max_c B[v,c] + min_c B[v,c]) / 2`.

*Proof.* For a single row, `min_s 2 max_c abs(B[v,c] - s)` equals
`range_c B[v,:]`, attained at the midrange. Each group term is a maximum over
rows of that per-row quantity, and the rows are shifted independently, so the
per-row minimizer minimizes every group term simultaneously. Sum over groups
with `a_g > 0`.

Softmax loss is invariant to these shifts: shifting row `v` by `s_v` adds
`s_v * z_v` to every class logit, which is a per-observation constant.
Minimizing `loss + lambda * 2 * Omega_inf` over all `B` therefore attains the
same value as minimizing `loss + lambda * Omega_range`, with the minimizer in
the midrange gauge. That lifted problem is what the solver actually optimizes.
Fixing a sum-zero gauge *during* that optimization would constrain `B` and
generally solve a different problem, so returned coefficients are
canonicalized to sum-zero contrasts only after solving.

**Zero-subtree gradient.** Suppose every coefficient inside `subtree(a)`
vanishes. The logits are then constant on the region of each `v` in
`subtree(a)`, and

```text
d Loss / d B[v,c] = sum_i w_i z_iv (p_i[c] - y_i[c]) / W_total.
```

On `region(v)` the probabilities `p_i[c]` equal one constant `pi[c]`, and the
local stump satisfies `sum_i w_i z_iv = 0`, so the probability term cancels:

```text
h_vc = sqrt(L_v * R_v) * (p_right[c] - p_left[c]) / W_total.
```

This is the original CART class-frequency contrast. It depends on neither
`lambda` nor the retained part of the tree, which is what lets the regression
argument carry over unchanged.

**Dual norm.** Step 3 above assigns each row the norm dual to `range`,
evaluated at `h_v`. For any sum-zero `g`,

```text
sup { <g, x> : range(x) <= 1 }  =  0.5 * ||g||_1.
```

*Proof.* Split `g = g+ - g-` with `||g+||_1 = ||g-||_1 = M = 0.5 * ||g||_1`,
which is possible exactly because `g` sums to zero. Transporting that mass
writes `g = sum_k mu_k (e_{c_k} - e_{c'_k})` with `mu_k >= 0` and
`sum_k mu_k = M`, so `<g, x> <= M * range(x)`. Conversely `x = 0.5` on the
support of `g+` and `-0.5` on the support of `g-` has `range(x) = 1` and
attains `M`.

Each `h_v` sums to zero over classes, because both child rows are probability
vectors. Hence `Q_v = 0.5 * sum_c abs(h_vc)`, and the same tree-isotonic
pooling of `Q_v / a_v` gives the classification structural knots. This is the
exact structural path for the logistic and softmax objective, not a
squared-error surrogate. It requires original weighted child-mean
probabilities, positive fitting leaf and class masses, and no active monotonic
clipping. The fitting-partition restrictions from regression apply here too.

Coefficient adapters screen splits inactive throughout the requested penalty
interval. Their forward/adjoint passes cost `O(nodes * classes)` per
loss/gradient evaluation, but explicit group memberships and proximal
projections can still be costly on deep trees. Multiclass samples require
`O(n_points * n_splits * n_classes)` storage.

Unlike squared loss, classification coefficient paths are generally curved;
see [Park and Hastie (2007)](https://doi.org/10.1111/j.1467-9868.2007.00607.x).
These paths always have `exact=False`. Adaptive midpoint refinement checks
interpolation at solved midpoints, not a uniform error bound.
`path.at(lam)` interpolates samples; call the point API to check another
penalty. Direct coefficient APIs require positive penalties because a finite
unpenalized solution need not exist.

SP/SHS classifier wrappers use structural pruning by default for eligible
binary/multiclass trees. Optional coefficients are computed only for the final
fit during automatic structural CV; at zero penalty the wrappers retain the
original tree without finite coefficients.

## API map and numerical checks

The sklearn adapters are in [`fitted_tree.py`](../fitted_tree.py). Generic
design/group APIs live here:

| API | Output |
| --- | --- |
| `hicap_regression_path` | KKT-certified, full squared-loss coefficient path for a rooted laminar family. |
| `tree_group_linf_exact_coefficient_path` | Full coefficient path from positive diagonal curvature and tree scores. |
| `laminar_group_linf_exact_topology_path` | Structural path from a diagonal design; coefficients optional. |
| `laminar_group_linf_regression[_path]` | Exact diagonal point solves, or FISTA for a general Gram; a grid is still sampled. |
| `laminar_group_linf_classification[_path]` | Logistic/softmax point solves and optionally refined samples. |
| `apa_apg_regression_path`, `apa_apg_classification_path` | Warm-started grids for `ord=2` or `ord=inf`; always sampled. |

Group indices are zero-based. Regression APIs can fit an unpenalized
intercept; APA instead follows its point API's explicit-intercept convention.

Check `exact` and `status` separately from point certificates. A complete
sampled path is not a complete breakpoint path. `RegularizationPath.at`
interpolates exactly only when `exact=True`; queries outside the stored
range are rejected. Returned arrays are owned, read-only snapshots.

For diagonal homotopy, `exact=True` means floating-point certification,
not symbolic arithmetic. Tolerances have machine-precision floors and
cancellation-aware boundary uncertainties; unresolved faces, event caps, or
underflowing positive proximal radii return a nonexact prefix. Point APIs
reject unrepresentable positive radii. See
[`diagonal_homotopy.py`](diagonal_homotopy.py) for the reported tolerances
and boundary diagnostics.

Generic hiCAP returns `status="nonunique_design"` and endpoint solutions
when the weighted centered design is numerically rank-deficient.
Classification's `certified` flag checks scaled stationarity and the
proximal dual certificate, not coefficient error or objective suboptimality.
Iteration/refinement limits produce `status="partial"`. With overlapping
groups, APA uses its full iteration budget; warm starts improve finite-budget
solutions but do not supply exact breakpoints or KKT certificates.
