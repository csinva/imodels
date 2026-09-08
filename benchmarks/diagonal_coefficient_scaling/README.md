# Full diagonal-tree hiCAP coefficient paths

The tables and audit notes below describe historical local runs, before source
fingerprints were recorded. They are not measurements of the current release.
See the [benchmark index](../README.md) for provenance and output policy;
regenerate results before making current performance claims.

This benchmark tests the full coefficient-path continuation, not just a grid
of point solutions or a structural-only path. The input is a CART regression
tree fitted to the bundled, real diabetes dataset (442 observations).
Targets are standardized; coefficients use the unnormalized local-stump basis.

## What the colors mean

- Orange: the new continuation discovers coefficient-direction knots and
  returns coefficients at every stored knot, down to lambda zero.
- Green: the existing solver discovers structural events only. It does not
  include slope changes that leave the retained tree structure unchanged.
- Dark blue: the exact diagonal proximal solver evaluates the knot grid that
  orange has already discovered. Its timing excludes grid discovery.
- Pink: the generic hiCAP homotopy independently discovers the full path.
  It is run only on small trees, using a diagonal pseudo-design with exactly
  the same quadratic sufficient statistics as the real fitted-tree problem.

Running the benchmark creates `diagonal_coefficient_scaling.png` and `.svg`;
generated figures and JSON/CSV results are not versioned.

## Recorded results

Medians of three runs, seeds 0--2, one BLAS thread, `min_samples_leaf=2`:

| Internal splits | Stored coefficient points, including zero | New full path (s) | Dark-blue supplied grid (s) | Generic hiCAP full path (s) |
|---:|---:|---:|---:|---:|
| 7 | 14 | 0.0174 | 0.00621 | 0.0395 |
| 15 | 32 | 0.0876 | 0.0323 | 0.184 |
| 31 | 76 | 0.429 | 0.173 | 3.19 |
| 63 | 160 | 1.79 | 0.708 | not run |
| 127 | 280 | 5.92 | 2.40 | not run |

At 31 splits the new complete path was about **7.4 times faster** than the
generic complete hiCAP path. Orange's comparison with dark blue is not an
equal-output speed comparison: orange must discover the supplied grid.

A separate larger-tree run (`large_tree/`, one run, seed 0,
`min_samples_leaf=1`) completed 255 splits and 465 stored points in **19.4 s**.
This is a different leaf-size configuration, not a fourth repeat of the table.
The largest diabetes-tree run (`largest_diabetes_tree/`, requesting 441 splits
with leaf size 1) produced 433 actual splits and 575 stored points, completing
in **31.2 s** with maximum relative coefficient error `1.01e-15`.

All 15 main runs completed. The nine generic-hiCAP comparisons agreed on
canonical knot counts and locations: maximum relative knot distance was
`1.78e-15`, and maximum absolute coefficient difference was `6.69e-14`.
Every returned knot and every interval midpoint was checked against a direct
dark-blue solve. Maximum relative coefficient error across the main runs was
`2.69e-15`, normalized by the largest unpenalized coefficient magnitude.
The 255-split check had maximum relative error `1.01e-15`.

Tree fitting and sufficient-statistic extraction are timed separately and
excluded from solver times. The new full-path time includes its own group/prox
setup and certification; the dark-blue grid time excludes its separately
recorded setup. Green timing starts from extracted scores and parents.
After fitting, the coefficient continuation needs only tree statistics, not
the observation-by-split design matrix. Observation count still affects tree
fitting and any explicit design/statistic construction.

## Algorithm and interpretation

The new solver combines green activation thresholds with dark-blue point
solutions to identify an active clipping face. Affine primal/dual inequalities
give that face's maximal penalty interval. Following their roots finds
coefficient-only events as well as structural ones; it does not merely
interpolate between green knots. Generic hiCAP is a validation reference,
not a runtime dependency of the new solver.

```python
from imodels.tree.sparse_pruning.fitted_tree import (
    fitted_tree_linf_exact_coefficient_path,
)

path = fitted_tree_linf_exact_coefficient_path(fitted_cart)
if not path.exact:
    raise RuntimeError((path.status, path.metadata.get("failure_message")))

beta, intercept = path.at(0.5 * path.lambdas[0])
# Coefficient columns correspond to path.metadata["tree_node_ids"].
```

`path.at` uses only neighboring coefficient rows after binary search, taking
`O(log K + p)` time. Interpolation is justified by the checked affine regions,
not by an assumption that support stays constant between arbitrary samples.

`exact=True` means numerical completion within coefficient tolerance and the
reported event-boundary uncertainties, not symbolic arithmetic or rigorous
directed-rounding interval bounds. Near-coincident coefficient events may be
indistinguishable at that resolution. Known distinct structural events are
preserved, including tested adjacent-float cases. An unresolved face or event
limit returns an explicitly nonexact prefix. Extremely ill-conditioned inputs
can fail: an additional adversarial audit completed 53/160 cases and rejected
107; five interior checks per interval found no completed path exceeding the
default `1e-9` coefficient tolerance (worst `2.02e-10`). This audit is not an
exhaustive correctness proof.

This is a first full-path implementation, not yet an incremental dynamic-tree
solver. It rebuilds each face using point solves, materializes descendant
memberships (quadratic for chains), and stores a dense `K` by `p` coefficient
matrix. The results establish an improvement over generic hiCAP on the tested
trees, not scalability to arbitrary thousand-node trees. When only tree
structures need rendering, green plus lazy dark-blue coefficient evaluation
remains the smaller workload.

The fitted-tree shortcut applies to squared-loss pruning under the fitting
rows/weights. Held-out or OOB rows generally lose the diagonal guarantee.
See the [path API documentation](../../imodels/tree/sparse_pruning/optimization/README.md)
for restrictions, weighted inputs, numerical diagnostics, and direct-statistic
usage.

## Reproduce

```bash
python benchmarks/benchmark_sparse_pruning_diagonal_coefficient_path.py \
  --output-dir benchmarks/diagonal_coefficient_scaling

python benchmarks/benchmark_sparse_pruning_diagonal_coefficient_path.py \
  --node-counts 255 --min-samples-leaf 1 --repeats 1 --legacy-max-nodes 0 \
  --output-dir benchmarks/diagonal_coefficient_scaling/large_tree

python benchmarks/benchmark_sparse_pruning_diagonal_coefficient_path.py \
  --node-counts 441 --min-samples-leaf 1 --repeats 1 --legacy-max-nodes 0 \
  --output-dir benchmarks/diagonal_coefficient_scaling/largest_diabetes_tree
```

The JSON stores individual timings, statuses, environment versions, data digest,
and interval certificates. CSV is a flattened summary; PNG/SVG are generated
from the recorded results. Partial-path timings are excluded from complete-path
comparisons, and failed runs are displayed explicitly.

## Code-review follow-up

The earlier local figures and JSON retain the original benchmark results.
A subsequent review fixed result-array aliasing, endpoint-range checks,
complex-input coercion, unsigned parent-index overflow, and extreme-range
proximal certificates. It also removed duplicate fitted-tree topology work,
added singleton proximal shortcuts, skipped unrequested certificate work, and
replaced repeated descendant penalty calculations with an `O(p K)` sweep.

Rerunning the same three-seed cases after those changes gave median complete
path times of **0.326 s** at 31 splits and **4.78 s** at 127 splits, versus
0.429 s and 5.92 s in the recorded baseline. All knot counts matched and the
largest coefficient error against direct point solves was `2.36e-15`.
The 433-split single-run check completed all 575 points in **26.1 s**, with
error `1.01e-15`. These are observed reruns, not interleaved paired timings.
Review-run artifacts were written to a temporary directory rather than adding
another result bundle to the repository; the commands above reproduce them.
