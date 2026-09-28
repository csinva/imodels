# Iterative random forests

`IRFClassifier` and `IRFRegressor` fit a sequence of random forests, each
sampling features in proportion to the previous forest's importances. Random
intersection trees (RIT) then find feature combinations that recur across
bootstrap samples and score how stable each one is.

## Usage

```python
from imodels import IRFClassifier

model = IRFClassifier(
    n_estimators=100,
    n_iterations=5,
    n_bootstraps=10,
    random_state=42,
).fit(X_train, y_train)

probabilities = model.predict_proba(X_test)
for features, stability in model.interaction_stability_.items():
    if stability >= 0.5:
        print(features, stability)
```

For a continuous target, use `IRFRegressor`:

```python
from imodels import IRFRegressor

model = IRFRegressor(random_state=42).fit(X_train, y_train)
predictions = model.predict(X_test)
```

### Reading `interaction_stability_`

Each key is a sorted tuple of two or more zero-based feature indices, with no
split directions or thresholds. Each value is the fraction of outer bootstrap
samples that recovered that exact set. It measures stability, not the
probability that the interaction is real.

Which leaves RIT searches:

- **Classification:** leaves predicting `interaction_class` (default: the last
  sorted label, e.g. 1 for 0/1 labels).
- **Regression:** all leaves by default. Set `leaf_threshold` to keep only
  leaves whose in-bag mean is strictly above it.

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
   still count in the denominator.

RIT follows the historical R implementation. Each tree starts by intersecting
two paths. Pairs are recorded immediately and end their branch; empty and
single-feature sets are dropped. Larger sets keep intersecting with new paths
and are recorded once `rit_depth` paths (default 5) have been combined.

## Controls and inspection

| Parameter or attribute | Purpose |
| --- | --- |
| `n_estimators`, `n_rit` | Trees per forest and RITs per outer forest (both default to 100) |
| `n_bootstraps=0` | Skip interaction discovery and fit for prediction only |
| `bootstrap=False` | Turn off per-tree row bootstrapping |
| `bootstrap_fraction` | Outer sample size as a fraction of training rows |
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
- Interactions are unsigned feature sets.
- The backend uses NumPy and scikit-learn stumps, so it is slower than a
  compiled forest. Runtime grows with the tree, bootstrap and RIT budgets.

Results will not exactly match the historical R package: the classifier
averages leaf probabilities where R averages hard tree votes, and the two
differ in random number generation and split tie-breaking.

## References

- [Basu et al. (2018), paper and supplement](https://arxiv.org/abs/1706.08457)
- [Historical R implementation](https://github.com/sumbose/iRF/blob/fda5999b10fa878d904c0b891458ea55de467061/R/iRF.R)
- [Full estimator parameters and attributes](iterative_random_forest.py)
