# Exact, fixed-grid, and adaptive path comparison

This directory documents a historical local fitted-tree comparison
between exact hiCAP, a fixed geometric APA-APG2 grid, and adaptively refined
APA-APG2. The California Housing run used 1,024 observations, 31
internal tree nodes/local-stump features, seed `20260903`, and one thread.

These results predate source fingerprints and are not current-release
measurements. Generated artifacts are not versioned. The adaptive implementation
is archived in `benchmarks/experimental/adaptive.py`, outside runtime code; it
does not certify complete knot enumeration. See the [benchmark index](../README.md)
for the maintained exact structural/coefficient comparisons.

| method | solver time (s) | native points | maximum coefficient-path error | maximum objective gap |
|---|---:|---:|---:|---:|
| exact hiCAP | 2.316 | 72 | 0 | 0 |
| fixed-grid APA-APG2 | 7.568 | 20 | 3.33% | 2.51e-3 |
| adaptive APA-APG2 | 19.065 | 51 | 1.51% | 3.86e-4 |

Errors are evaluated on a shared 401-point lambda grid.  Coefficient error is
the L2 distance from exact hiCAP divided by the maximum exact coefficient L2
norm over that grid; objective gaps use the analogous path-wide objective
normalizer.  These definitions are invariant to rescaling the response.

The exact path reached lambda zero and passed its knot checks.  The adaptive
path exhausted no refinement interval and all points passed the requested 1%
near-face KKT check, but only the latter is a relaxed backward-error check:
the strict floating-point active-face test did not certify every point.  The
artifact therefore records `point_solutions_certified=false`,
`point_solutions_kkt_accepted=true`, `kkt_acceptance_mode="relaxed_face"`, and
`exact=false`.  Fixed-grid APA is marked partial because its overlapping-group
point solver reached the iteration cap at most positive lambdas.

The fitted-row Gram check found a maximum absolute off-diagonal correlation of
`3.13e-16`, validating the explicitly requested diagonal backend for this
run.  Timings are a single-run algorithm comparison, not uncertainty
estimates; tree fitting, dense evaluation, serialization, and plotting are
excluded.  Exact/adaptive timings include their internal zero-threshold LP,
whereas the fixed-grid timing excludes the shared grid setup (about a few
milliseconds here).

Reproduce the outputs with:

```bash
python benchmarks/benchmark_sparse_pruning_adaptive_paths.py \
    --dataset california_housing \
    --data-home /path/to/sklearn_data \
    --no-fallback-to-diabetes \
    --samples 1024 --nodes 31 --min-samples-leaf 1 \
    --repeats 1 --seed 20260903 --threads 1 \
    --fixed-points 20 --dense-points 401 \
    --max-iter 2000 --tolerance 1e-8 \
    --adaptive-initial-points 5 --adaptive-max-points 81 \
    --adaptive-max-depth 10 \
    --adaptive-coefficient-tolerance 2e-2 \
    --adaptive-objective-tolerance 1e-3 \
    --no-adaptive-support-check --adaptive-kkt-check \
    --adaptive-kkt-tolerance 1e-3 \
    --adaptive-kkt-face-tolerance 1e-2 \
    --assume-diagonal-gram \
    --output-dir benchmarks/adaptive_path_comparison/california_p31
```

The JSON contains the native paths, dense comparison arrays, solver metadata,
and environment/command information.  The CSV is the flat summary used for
the table, and the PNG/SVG files are generated solely from the JSON.
