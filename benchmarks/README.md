# Sparse-pruning benchmarks

These are developer experiments, not modules imported by the estimators.
For using sparse pruning, start with the [estimator guide](../imodels/tree/sparse_pruning/README.md).

## Maintained solver comparisons

| Entry point | What it measures |
| --- | --- |
| [benchmark_sparse_pruning_diagonal_coefficient_path.py](benchmark_sparse_pruning_diagonal_coefficient_path.py) | Complete coefficient-knot discovery (orange), structural events (green), supplied-grid point solves (dark blue), and small-tree generic hiCAP checks; also creates its plots |
| [benchmark_sparse_pruning_diagonal_topology.py](benchmark_sparse_pruning_diagonal_topology.py) | Real-data node/observation sweeps for native and design-input structural paths, plus a supplied coefficient grid |
| [plot_sparse_pruning_diagonal_topology.py](plot_sparse_pruning_diagonal_topology.py) | Plots the structural benchmark alongside a separately generated legacy scaling run |

An offline smoke comparison using bundled diabetes data:

```bash
python benchmarks/benchmark_sparse_pruning_diagonal_coefficient_path.py \
  --node-counts 3 7 --repeats 1 \
  --output-dir benchmarks/diagonal_coefficient_scaling/smoke
```

See [full coefficient paths](diagonal_coefficient_scaling/README.md) and
[structural scaling](diagonal_topology_scaling/README.md) for larger commands
and interpretation. Fitting, statistic extraction, and solver work are timed
separately. The green result contains fewer events than a full coefficient
path; dark blue receives the grid that orange already discovered. Neither
comparison is an equal-output time-to-solution contest. A capped, inaccurate
APA run is likewise not an equal-accuracy baseline.

## Historical comparisons and research

The remaining scripts preserve the investigations that led to the specialized
solvers. They are useful reference experiments, not additional production
features or evidence that every approach should stay in the runtime package.

| Files | Purpose |
| --- | --- |
| [benchmark_sparse_pruning_paths.py](benchmark_sparse_pruning_paths.py), [plot_sparse_pruning_paths.py](plot_sparse_pruning_paths.py) | Synthetic cold/warm APA grids versus generic hiCAP; the plot command also runs the comparison |
| [benchmark_sparse_pruning_tree_scaling.py](benchmark_sparse_pruning_tree_scaling.py), [plot_sparse_pruning_tree_scaling.py](plot_sparse_pruning_tree_scaling.py) | Real-data generic-hiCAP/APA sweeps with worker timeouts; plotter reads saved results |
| [benchmark_sparse_pruning_adaptive_paths.py](benchmark_sparse_pruning_adaptive_paths.py), [plot_sparse_pruning_adaptive_paths.py](plot_sparse_pruning_adaptive_paths.py) | Fixed versus adaptively refined APA grids; not exact knot enumeration |
| [_sparse_pruning_tree_scaling.py](_sparse_pruning_tree_scaling.py) | Shared datasets, problem construction, comparison metrics, serialization and provenance; not another CLI |
| [experimental/adaptive.py](experimental/adaptive.py), [experimental/topology.py](experimental/topology.py) | Archived adaptive APA and approximate thresholded-topology searches, with their tests alongside |

Historical tables in the result-directory READMEs describe specific earlier
local runs. They predate code fingerprints and are **not current-release
measurements**. Re-run the relevant commands before claiming present-day speed
or accuracy. In particular, the archived positive-threshold topology search
does not prove it found every change; it is distinct from green's exact
mathematical-zero structural path.

## Reproducibility and output hygiene

Real-data runners record dataset identifiers/digests; synthetic runs record
their generation seed and sizes. JSON reports also include arguments, library versions,
Git commit SHA and dirty state, plus a SHA-256 fingerprint of the working
Python sources and packaging files. Per-file hashes distinguish uncommitted
source changes from HEAD; Git fields are null when Git is unavailable. The
source fingerprint excludes generated results and does not archive the source:
retain the corresponding checkout/diff to reproduce a dirty run. Metadata
collection is outside timed solver calls.

Use a fixed thread count, repeated seeds, the same dataset and objective, and
check completion/certification before comparing timings. A root-only tree is
valid: its path has no coefficient columns, and unused point/reference
comparisons are explicitly marked `not_needed_no_splits`, with null timings.
Matplotlib/cache defaults use the operating system's temporary directory and
respect existing `MPLCONFIGDIR`/`XDG_CACHE_HOME` settings. California Housing
requires an existing cache or explicit download permission; diabetes is
bundled with scikit-learn.

Generated JSON/CSV/PNG/SVG bundles under the known result directories are
ignored, not deleted. This includes earlier local figures and the unreviewed
image-generation draft. The contribution consists of source, tests and these
instructions; run the commands to create plots locally. New result locations
outside those directories should also be kept out of source commits.

## Checks

Run supported pruning and shared GCV tests with
`python -m pytest tests/sparse_pruning tests/hs_gcv_test.py`.
The archived implementations keep their checks alongside the experiments:
`python -m pytest benchmarks/experimental`. These tests are not needed to
import or use imodels; they preserve the earlier research for reproducibility.
