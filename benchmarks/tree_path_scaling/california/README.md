# California Housing results

Historical experiment: these local measurements predate source fingerprints
and the specialized fitted-tree defaults. They are not current-release
benchmarks or current solver recommendations. Generated bundles are not
versioned; see the [benchmark index](../../README.md) for maintained comparisons.

These results compare exact hiCAP with a warm-started APA-APG2 grid on local
stumps from fitted regression trees. They were measured on an Apple M3 Pro
(11 cores, 18 GB), macOS 14.6, Python 3.11.11, NumPy 2.2.6, SciPy 1.15.3, and
scikit-learn 1.6.1. Numerical libraries were restricted to one thread.

The main run uses three deterministic California Housing subsets/trees per
case, 20 APA lambda points (19 positive plus zero), 2,000 iterations per
positive point, and a 45-second hard timeout. Times below are solver-only
medians in seconds. `APA/exact` below one favors APA. Exact knots are the
median number of certified coefficient-direction breakpoints.

## Scaling the tree at 1,024 observations

| Internal nodes / stump features | Exact full path | Exact knots | APA 20-point grid | APA/exact | APA max coefficient error at solved points | APA max coefficient error between points |
|---:|---:|---:|---:|---:|---:|---:|
| 3 | 0.0092 | 4 | 1.321 | 143.0 | 0.050% | 4.54% |
| 7 | 0.0338 | 12 | 2.477 | 73.3 | 0.068% | 4.99% |
| 15 | 0.179 | 32 | 5.001 | 27.9 | 0.145% | 4.96% |
| 31 | 2.528 | 72 | 8.991 | 3.56 | 0.236% | 4.89% |
| 47 | 20.341 | 120 | 12.425 | 0.61 | 0.254% | 5.56% |
| 63 | failed 3/3 | — | 16.119 | — | — | — |

At 63 nodes, exact hiCAP returned a non-exact prefix in all three repetitions
after 19.5–38.5 seconds. Each failure was explicitly labeled
`PathContinuationFailure`; it is not plotted as a completed full path.

## Scaling observations at 15 internal nodes

| Observations | Exact full path | APA 20-point grid | APA/exact |
|---:|---:|---:|---:|
| 128 | 0.166 | 4.910 | 29.6 |
| 256 | 0.197 | 4.915 | 24.9 |
| 512 | 0.195 | 4.914 | 25.2 |
| 1,024 | 0.184 | 4.964 | 27.0 |
| 2,048 | 0.153 | 5.207 | 34.0 |
| 4,096 | 0.143 | 5.541 | 38.8 |
| 8,192 | 0.177 | 7.787 | 44.0 |
| 16,384 | 0.169 | 10.537 | 62.4 |

For this small hierarchy, exact-path time is nearly flat over this observation
range, while APA increasingly pays for repeated matrix-vector products.

## Is APA's 47-node speed advantage a full-path advantage?

No. Two targeted controls separate optimization effort from lambda-grid
resolution. All three use the same 47-node, 1,024-row tree (seed 20260903).

| APA configuration | APA time | Max coefficient error at solved points | Max coefficient error on interpolated path | Interpolated topology agreement |
|---|---:|---:|---:|---:|
| 20 points × 2,000 iterations | 12.43 s | 0.254% | 5.56% | 58.4% |
| 20 points × 8,000 iterations | 48.87 s | 0.084% | 5.48% | 60.4% |
| 101 points × 2,000 iterations | 65.87 s | 0.300% | 0.954% | 82.0% |

The corresponding exact path took about 20.2 seconds and returned 115 knots.
Four times more APA iterations substantially improves the solved points but
barely changes interpolation error. Increasing the number of lambda samples
improves the reconstructed curve, but removes the raw speed advantage. The
101-point row has more locations at which its solved-point maximum can occur,
so its solved-point maximum is not directly comparable to the 20-point
maximum.

## What the historical generic-hiCAP/APA comparison showed

- Up through roughly 31 internal nodes, generic hiCAP was faster here and
  returned the certified full breakpoint path.
- Around 47 nodes, warm APA was faster for a coarse grid, but generic hiCAP
  was faster than the denser APA approximation in this control.
- At 63 nodes, that generic hiCAP implementation did not complete these runs.
  APA returned approximate sampled solutions. This does not establish a size
  limit for the later diagonal coefficient solver or the structural solver.
- Increasing observations alone did not favor APA at 15 nodes. Tree/group
  size and path complexity drove the crossover much more strongly than sample
  size in these data.

Every APA run in the main experiment reached its fixed iteration cap, so the
records use `solver_status="partial"`; none is presented as convergence- or
KKT-certified. At the actual 20 solved lambda values, topology agreement with
exact hiCAP was 95–100% in the node sweep. On the dense interpolated path it
fell from 95% at 3 nodes to 58% at 47 nodes.

The full per-run records and environment metadata are in `results.csv` and
`results.json`. The PNG/SVG figure plots medians and interquartile bands. This
is a real-data crossover study, not a universal threshold: conditioning, tree
shape, group overlap, requested path density, tolerance, and hardware can all
move the boundary.
