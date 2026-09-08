# Exact diagonal hiCAP topology and point-path benchmark

The numerical tables below are historical local runs, not measurements of the
current release; they predate source fingerprints. Generated result bundles
are not versioned. See the [benchmark index](../README.md) for current scope,
provenance, and output policy; rerun the commands before reporting new results.

This benchmark uses local-stump designs from real fitted CART regressors.  It
adds three diagonal-specialized workloads to the existing exact-hiCAP versus
APA-APG2 scaling comparison:

- **Tree-native exact structural path** derives scores from the fitted CART
  node statistics and computes every mathematical zero-support topology knot.
  It constructs neither the stump design nor descendant groups.
- **Design-input exact structural path** computes the same knots from a stump
  matrix and explicit laminar groups, including sufficient-statistic setup.
- **Exact diagonal point path** solves the same 20 requested lambda values as
  APA-APG2.  Each value is one exact child-to-parent laminar proximal sweep;
  there is no optimization iteration or warm start.

The California Housing artifact used 1,024 observations for the node sweep,
15 internal nodes for the observation sweep, three seeds, one BLAS thread,
and the same tree/data construction as
`benchmarks/tree_path_scaling/california/results.json`.

## Main result

At 31 internal nodes, median wall times were:

| workload | method | median seconds |
|---|---|---:|
| all structural knots | tree-native fitted-statistics path | 0.000428 |
| all structural knots | design-input tree-isotonic path | 0.000802 |
| 20 exact coefficient points | diagonal laminar prox | 0.0121 |
| all coefficient knots | legacy hiCAP homotopy | 2.53 |
| 20 approximate coefficient points | APA-APG2 | 8.99 |

For the matched 20-point workload, the exact diagonal solver was about **746x
faster** than the capped APA-APG2 run. The APA runs were partial and are not an
equal-accuracy time-to-solution comparison. The topology-only traversal was
about **5,910x faster** than the legacy complete coefficient homotopy, but
those are different outputs and the ratio should be read as the benefit of
targeting the tree events needed for visualization.

Across all three seeds in every small validation case (3, 7, 15, and 31 nodes
at fixed sample size, plus the observation sweep), the direct coefficient
points matched the complete hiCAP path to at most `1.07e-13` absolute error.
Every topology state matched, and per-group activation penalties agreed to at
most `2.61e-15`. The maximum continuous relative stationarity residual over
all new benchmark runs was `2.95e-16`.

At the largest requested tree size, the `min_samples_leaf=2` constraint yielded
430--438 actual internal nodes across seeds. Enumerating all 418--427 exact
structural knots took `0.00397` seconds through the fitted-tree entry point or
`0.00911` seconds from the design input; the 20-point exact coefficient grid
took `0.124` seconds. Timings exclude fitting the source CART tree and the
one-time local-stump design/group construction. The design-input path timing
does include reducing that already-built design to quadratic sufficient
statistics, while the fitted-tree timing includes extraction of the stored
node statistics.

Increasing observations from 128 to 16,384 at fixed 15-node trees left the
fitted-tree path essentially flat (`0.000284` to `0.000334` seconds). The
design-input path changed from `0.000476` to `0.00101` seconds, and the
20-point solve remained near `0.006--0.007` seconds. This reflects the expected
split: reducing `X` to `(diag(X'WX), X'Wy)` depends on observations, while the
tree-native structural path depends only on stored nodes after fitting.

## Interpretation

The exact topology knots and complete coefficient knots are different path
objects.  A coefficient can change slope because a group clipping face changes
without changing whether any subtree is pruned.  For tree visualization, the
structural path is both sufficient and much smaller.  If coefficients at a
particular state are needed, evaluate the one-sweep solver lazily at that knot
or at an interior lambda for the state. The fitted-tree wrappers now support
both native point coefficients (`solver="proximal"`) and complete coefficient
paths (`solver="coefficient_path"`) using stored statistics, without a design
matrix. Coefficient solvers still materialize descendant groups, whose total
memberships can be quadratic for a degenerate chain; green avoids those groups.

The diagonal guarantee holds for local stumps evaluated under the same rows
and weights used to fit the tree.  OOB/held-out pruning rows, changed weights,
and raw indicator bases generally do not retain the diagonal quadratic.
General **squared-loss regression** can use the exact laminar prox inside FISTA
for certified sampled points. Logistic classification instead uses the sampled
APA solver; the regression FISTA API does not implement logistic loss.
Neither case has the fitted-tree exact structural-path guarantee.

The speed ratios above compare historical runs on the same machine and matched
seed/data construction, but the new methods ran sequentially in-process while
the older benchmark isolated solvers in worker processes. Read them as
observed workload ratios, not precision-controlled microbenchmark ratios.

## Reproduction

```bash
python benchmarks/benchmark_sparse_pruning_diagonal_topology.py \
  --dataset california_housing \
  --data-home /path/to/sklearn_data \
  --no-fallback-to-diabetes \
  --axis both \
  --node-counts 3 7 15 31 63 127 255 511 \
  --observation-counts 128 256 512 1024 2048 4096 8192 16384 \
  --fixed-observations 1024 --fixed-nodes 15 \
  --min-samples-leaf 2 --path-points 20 \
  --repeats 3 --threads 1 --seed 20260903 \
  --output-dir benchmarks/diagonal_topology_scaling/california

python benchmarks/plot_sparse_pruning_diagonal_topology.py \
  benchmarks/diagonal_topology_scaling/california/results.json \
  benchmarks/tree_path_scaling/california/results.json \
  benchmarks/diagonal_topology_scaling/california/diagonal_topology_vs_legacy.png
```

The JSON is the durable machine-readable artifact; the CSV contains the flat
run records, and the PNG/SVG are generated solely from those artifacts.
