# Tree-path scaling benchmark

This historical experiment compares the wall-clock cost of the generic exact
hiCAP homotopy path and a descending, warm-started APA-APG2 lambda grid. The
benchmark uses regression trees fitted to real scikit-learn datasets and the
same local-stump representation used by sparse pruning in `imodels`.

## What is held equal

For every case, the runner:

1. selects a deterministic nested subset of the real dataset;
2. fits a CART tree with the requested number of internal nodes;
3. constructs one centered local-stump feature per internal node;
4. constructs one infinity-norm group from each node and all its internal
   descendants;
5. verifies that the centered stump matrix has full column rank; and
6. passes the same design, response, groups, and normalized squared-loss
   objective to both solvers.

The node sweep fixes the observation count. The observation sweep refits the
tree on each nested data subset while holding its number of internal nodes
fixed. Consequently, the latter is a practical end-to-end problem sequence,
not a synthetic experiment that duplicates rows in an unchanged objective.

## Timing contract

- Solver calls are sequential, single-threaded by default, and isolated in
  fresh spawned processes.
- `solver_wall_seconds` excludes tree fitting, stump transformation, process
  startup, plotting, and serialization.
- Tree and transform times are retained separately. The external lambda-grid
  setup needed by APA is also retained as `lambda_setup_seconds`.
- A hard timeout creates a right-censored record instead of silently dropping
  the case.
- Exact hiCAP is counted as a complete path only when it reaches lambda zero
  and passes its independent KKT checks.
- APA timing is a fixed-work comparison: the current point solver commonly
  reaches its iteration limit. Such paths have `solver_status="partial"` and
  remain usable approximate grid results, but they are not convergence- or
  KKT-certified.

## Accuracy contract

The JSON and CSV distinguish two comparisons against exact hiCAP:

- `sample_*` metrics evaluate APA only at the lambda values it actually
  solved. These isolate point-solver error.
- Metrics without the `sample_` prefix evaluate a dense lambda grid using
  linear interpolation between APA samples. These measure the fidelity of the
  finite grid when it is used as an approximation to the full path.

This distinction matters: increasing APA iterations can improve `sample_*`
metrics, but it cannot recover an unsampled hiCAP breakpoint. A denser or
adaptive lambda grid is needed when the transitions themselves matter.

The current fitted-tree specialization is covered by the maintained comparisons
in the [benchmark index](../README.md); this experiment is not a solver-selection
guide for the new defaults. Earlier local result bundles predate source
fingerprints and are not versioned or presented as current-release measurements.

Run `benchmark_sparse_pruning_tree_scaling.py` to create `results.csv` and
`results.json`, then run `plot_sparse_pruning_tree_scaling.py` to generate PNG,
SVG, or PDF figures. The [California notes](california/README.md) preserve
the historical interpretation and experimental conditions.
