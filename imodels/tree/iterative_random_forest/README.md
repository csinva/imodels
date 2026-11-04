# Iterative random forests: iRF, siRF, and lo-siRF

`IRFClassifier` and `IRFRegressor` fit a sequence of random forests, each
sampling features in proportion to the previous forest's importances. Random
intersection trees (RIT) then find feature combinations that recur across
bootstrap samples. Signed iRF (siRF) and local siRF (lo-siRF) are options and
methods of the same two estimators, not separate classes:

| Analysis | Fit with | Then use |
| --- | --- | --- |
| **iRF**: stable feature sets | default (`signed=False`) | `interactions_`, `interaction_stability_` |
| **siRF**: signed interactions and screens | `signed=True` | `interaction_scores_`, `select_interactions()` |
| **lo-siRF**: per-row importance and ranking | `signed=True` | `local_stability()`, `rank_interactions()` |

`feature_groups` works with all three. Predictions are identical across modes
with the same seed and fitting parameters.

## iRF

```python
from imodels import IRFClassifier, IRFRegressor

model = IRFClassifier(n_estimators=100, n_iterations=5, n_bootstraps=10,
                      random_state=42).fit(X_train, y_train)
probabilities = model.predict_proba(X_test)
stable = {s: v for s, v in model.interaction_stability_.items() if v >= 0.5}

# Continuous targets
model = IRFRegressor(random_state=42).fit(X_train, y_train)
```

Each key of `interaction_stability_` is a sorted tuple of two or more
zero-based feature indices (group IDs when `feature_groups` is set), with no
split directions or thresholds. Each value is the fraction of outer bootstrap
samples that recovered that exact set. It measures stability, not the
probability that the interaction is real.

RIT searches these leaves:

- **Classification:** leaves predicting `interaction_class` (default: the last
  sorted label, e.g. 1 for 0/1 labels).
- **Regression:** all leaves, or only those whose in-bag mean is strictly
  above `leaf_threshold` when one is given.

## siRF: signed interactions

```python
model = IRFClassifier(signed=True, random_state=42).fit(X_train, y_train)

# A high value of feature 0 together with a low value of feature 2.
print(model.interaction_scores_.get(((0, +1), (2, -1))))

# Candidates passing all three bootstrap screens of the siRF paper.
selected = model.select_interactions(threshold=0.5)
```

An interaction is a sorted tuple of signed atoms `(feature_or_group_id,
direction)`: `-1` is the left branch (`<=` the split threshold) and `+1` the
right branch (`>`). Discovery keeps only the first direction for each feature
or group on a path.

The implementation follows the
[revised siRF paper (arXiv:1810.07287v2)](https://arxiv.org/html/1810.07287v2).
Candidates come from the final full-data forest and are evaluated on every
outer bootstrap, so `interactions_` also holds candidates that bootstrap RIT
never recovered (stability zero). Each `interaction_scores_[s]` holds the
bootstrap means of:

- `prevalence`: per-tree fraction of target-class rows whose path contains `s`;
- `precision`: per-tree target-class fraction among out-of-bag (OOB) rows whose
  path contains `s`;
- `fsd`: log prevalence relative to the product of singleton prevalences;
- `mip`: log precision relative to the best subset with one atom removed;

plus `fsd_stability` and `mip_stability`, the fractions of bootstraps with
positive FSD and MIP. `select_interactions` requires recovery, FSD and MIP
stability all at or above the threshold. Per-bootstrap values are in
`bootstrap_interaction_scores_`.

The default 10 bootstraps gives stability in steps of 0.1, which makes a 0.5
screen coarse; use 50 for final analyses when affordable.

Details:

- Precision excludes every outer-bootstrap copy of a row used to fit the tree.
  With `bootstrap=False` there are no OOB rows: fitting warns, and precision
  and MIP are `NaN`. Their zero stability is then not evidence against an
  interaction.
- A tree ratio without support is `NaN` and left out of that mean. Undefined
  bootstrap scores are left out of means but still count in stability
  denominators. A log ratio with one zero argument is infinite; `0/0` is `NaN`.
- Observation weights apply to the scores as well as to forest fitting.

### Signed regression

```python
model = IRFRegressor(signed=True, random_state=42).fit(X_train, y_train)
```

Trees still fit the continuous response. Only discovery and scoring use a
binary target, following R iRF v3:

- The cutoff is `leaf_threshold`, defaulting in signed mode to the median of
  the positive-weight training responses. It is frozen as `leaf_threshold_` and
  reused, unchanged, for bootstraps and held-out ranking.
- Responses are binarized as `y >= leaf_threshold_`; RIT uses leaves whose mean
  is strictly `> leaf_threshold_`.
- The median defines an upper-half target (including ties), which may not match
  the outcome of interest. For sparse signals or a meaningful outcome
  threshold, set `leaf_threshold` from domain knowledge.
- Fitting warns if every training response falls on one side of the cutoff.

## lo-siRF: local importance

```python
model = IRFClassifier(signed=True, random_state=42).fit(X_train, y_train)

local = model.local_stability(X_valid)  # shape (n_rows, n_interactions)
ranking = model.rank_interactions(X_valid, y_valid, random_state=42)
features = model.rank_interactions(X_valid, y_valid, "features", random_state=42)
```

Both methods default to `interactions_`. They also accept a list of
interactions, including single atoms such as `((0, +1),)`, or `"features"`
for both signs of every feature or group (ordered by ID, `-1` before `+1`).
If you only need lo-siRF, fit with `n_bootstraps=0`: candidates are still
discovered, while the stability and score dictionaries stay empty.

**`local_stability`** gives, for each row, the fraction of trees whose path
for that row contains every atom of the interaction. As in the
[lo-siRF authors' code](https://github.com/Yu-Group/losiRF/blob/main/losiRF/R/local_stability.R),
it counts **all visited splits**, including repeated splits in opposite
directions, and every tree regardless of its prediction. Pass
`first_only=True` (to either method) to use discovery's first-split encoding.

**`rank_interactions`** compares mean local stability between target rows
(`interaction_class_`, or `y >= leaf_threshold_` for regression) and the
rest, then permutes the labels with the forest fixed. Each result has the
target-minus-rest `difference`, a two-sided `pvalue`, and `pvalue_bh`
(Benjamini-Hochberg over the interactions in that call), ordered by `pvalue`
then absolute `difference`. The p-value is `(exceedances + 1) /
(n_permutations + 1)`, with ties counted as exceedances, so the default 10,000
permutations give a minimum of 1/10,001.

Pass held-out rows: the methods neither split data nor reuse training
responses, and their scores are conditional on the fitted forest. The lo-siRF
study's GWAS filtering, phenotype binarization, predictive checks, and
aggregation across binarization choices are application-level steps.

## Feature groups

`feature_groups` takes one nonnegative integer group ID per column (IDs need
not be consecutive) and treats interchangeable columns as one feature in
discovery and interpretation. Grouping happens before RIT and before the first
direction is kept, so one group cannot pair with itself during discovery.
Tree fitting and feature reweighting still use individual columns.

```python
model = IRFClassifier(signed=True, feature_groups=[0, 0, 1, 2],
                      random_state=42).fit(X_train, y_train)
ranking = model.rank_interactions(X_valid, y_valid, [((0, +1),), ((0, +1), (1, -1))])
```

## How fitting works

With `n_iterations=K` and `n_bootstraps=B`, iRF fits **K + B forests**:

1. **Reweight.** Fit K forests on the full training data. The first uses
   uniform feature weights; each later forest samples candidate features at
   every node (without replacement) in proportion to its predecessor's
   impurity decrease: Gini for classification, residual sum of squares for
   regression.
2. **Freeze.** Use the Kth forest for prediction. Freeze the weights **used to
   fit it** (not the importances it produces) for interaction discovery.
3. **Bootstrap.** Fit one forest with the frozen weights on each of B outer
   bootstrap samples. Samples are full-size by default and stratified for
   classification. Each tree still bootstraps rows internally.
4. **Search.** Run RIT on each outer forest, sampling paths in proportion to
   how many outer-sample rows (duplicates included) reach their leaf. A set
   gets at most one vote per outer sample, and samples that recover nothing
   still count in the denominator. With `signed=True`, RIT also runs once on
   the Kth forest to produce the candidates.

RIT follows the historical R implementation. Each tree starts by intersecting
two paths. Pairs are recorded immediately and end their branch; empty and
single-feature sets are dropped. Larger sets keep intersecting with new paths
and are recorded once `rit_depth` paths (default 5) have been combined.

Signed RIT samples paths by leaf mass as well, our reading of the
[v2 paper](https://arxiv.org/html/1810.07287v2#S1.SS1.SSS2), which does not
specify precision reweighting.
[R iRF v3](https://github.com/sumbose/iRF/blob/master/R/gRIT.R#L115-L125)
instead weights by `count * precision` (OOB counts by default), a different
sampling distribution.

## Controls and inspection

| Parameter or attribute | Purpose |
| --- | --- |
| `n_estimators`, `n_rit` | Trees per forest and RITs per outer forest (both default to 100) |
| `n_bootstraps=0` | Skip bootstrap stability and scores; signed candidates remain available |
| `bootstrap=False` | Turn off per-tree row bootstrapping (disables siRF precision and MIP) |
| `bootstrap_fraction` | Outer sample size as a fraction of training rows |
| `signed`, `feature_groups` | Signed discovery; aggregation of columns into groups |
| `interaction_class`, `leaf_threshold` | Target class, or regression cutoff (`leaf_threshold_` once fitted) |
| `first_only`, `n_permutations` | lo-siRF method arguments: path encoding; permutation count (default 10,000) |
| `n_jobs=-1` | Fit trees in parallel on all cores (default: one core) |
| `random_state` | An integer gives identical results for any `n_jobs` |
| `feature_weights_history_` | Weights fed into each full-data forest |
| `feature_importances_history_` | Importances produced by each full-data forest |
| `bootstrap_samples_`, `bootstrap_interactions_` | Row indices and recovered sets for each outer sample |

## Scope

- Inputs must be dense, finite and numeric, with a single target. Encode
  categorical features before fitting.
- Multiclass classification is supported, but interactions are searched for
  one class at a time.
- The backend uses NumPy and scikit-learn stumps, so it is slower than a
  compiled forest. Runtime grows with the tree, bootstrap and RIT budgets.

Results will not exactly match the historical R package: the classifier
averages leaf probabilities where R averages hard tree votes, and the two
differ in random number generation and split tie-breaking. The siRF score
formulas follow the v2 paper rather than the R package.

## References

- [Basu et al. (2018), paper and supplement](https://arxiv.org/abs/1706.08457)
- [Historical R implementation](https://github.com/sumbose/iRF/blob/fda5999b10fa878d904c0b891458ea55de467061/R/iRF.R)
- [Revised siRF paper and score definitions](https://arxiv.org/html/1810.07287v2)
- [lo-siRF paper, Methods steps 4.1–4.4](https://pmc.ncbi.nlm.nih.gov/articles/PMC12170338/)
- [Authors' lo-siRF local stability implementation](https://github.com/Yu-Group/losiRF/blob/main/losiRF/R/local_stability.R)
- [Full estimator parameters and attributes](iterative_random_forest.py)
