"""Signed interaction scores and post-fit lo-siRF local importance.

Scores follow the revised siRF paper (arXiv:1810.07287v2). Local stability
follows Yu-Group/losiRF: all visited split directions count, including repeated
splits on a feature. Inputs here are validated by the public estimator.
"""

import numpy as np
from sklearn.utils import check_random_state


_LEAF_BLOCK_SIZE = 4096
_TERM_BLOCK_SIZE = 256


def _interaction_terms(candidates):
    """Return candidates, their singletons and their leave-one-out subsets."""
    terms = set()
    for candidate in candidates:
        candidate = tuple(sorted(candidate))
        terms.add(candidate)
        terms.update((atom,) for atom in candidate)
        if len(candidate) > 1:
            terms.update(candidate[:i] + candidate[i + 1:]
                         for i in range(len(candidate)))
    return sorted(terms)


def _tree_statistics(paths, leaf_ids, y, sample_weight, inbag_indices,
                     row_ids, terms, target_class):
    """Return per-term prevalence and OOB precision, shape ``(2, n_terms)``.

    Prevalence uses all outer-sample rows and their observed classes. Precision
    excludes every copy of an original row if any copy was used to fit this
    tree. Eligible outer duplicates retain their multiplicity. A ratio with
    zero denominator is NaN; callers average over supported trees separately
    for each statistic. ``paths`` uses the same first-split encoding as RIT.
    """
    if not terms:
        return np.empty((2, 0))
    target = np.asarray(y) == target_class
    row_ids = np.asarray(row_ids)
    oob = ~np.isin(row_ids, row_ids[inbag_indices])
    n_nodes = max(paths) + 1
    positive_mass = np.bincount(
        leaf_ids, weights=sample_weight * target, minlength=n_nodes)
    oob_mass = np.bincount(
        leaf_ids, weights=sample_weight * oob, minlength=n_nodes)
    positive_oob_mass = np.bincount(
        leaf_ids, weights=sample_weight * target * oob, minlength=n_nodes)
    total_positive = positive_mass.sum()

    sums = np.zeros((3, len(terms)))
    leaf_items = list(paths.items())
    empty = set()
    for start in range(0, len(leaf_items), _LEAF_BLOCK_SIZE):
        block = leaf_items[start:start + _LEAF_BLOCK_SIZE]
        leaves = np.fromiter((leaf for leaf, _ in block), dtype=np.intp)
        masses = np.vstack((positive_mass[leaves], oob_mass[leaves],
                            positive_oob_mass[leaves]))
        atom_leaves = {}
        for i, (_, path) in enumerate(block):
            for atom in path:
                atom_leaves.setdefault(atom, set()).add(i)
        for first in range(0, len(terms), _TERM_BLOCK_SIZE):
            batch = terms[first:first + _TERM_BLOCK_SIZE]
            # At most 1 MiB of Boolean inclusions (8 MiB when matmul casts
            # to float). Set intersections run in C instead of scanning
            # every leaf in Python for every interaction.
            included = np.zeros((len(block), len(batch)), dtype=bool)
            for j, term in enumerate(batch):
                matches = sorted((atom_leaves.get(atom, empty) for atom in term), key=len)
                if matches[0]:
                    included[list(set.intersection(*matches)), j] = True
            sums[:, first:first + len(batch)] += masses @ included

    statistics = np.full((2, len(terms)), np.nan)
    np.divide(sums[0], total_positive, out=statistics[0], where=total_positive > 0)
    np.divide(sums[2], sums[1], out=statistics[1], where=sums[1] > 0)
    return statistics


def _interaction_scores(candidates, terms, statistics):
    """Return columns prevalence, precision, FSD and MIP for each candidate.

    ``statistics`` contains tree-averaged prevalence and precision. FSD is
    log prevalence relative to independent singleton selection; MIP is log
    precision relative to the best leave-one-out subset. Undefined scores
    remain NaN, including 0/0. Positive/0 gives +inf and 0/positive gives -inf.
    Logs are subtracted to avoid underflow in high-order prevalence products.
    """
    indices = {term: j for j, term in enumerate(terms)}
    scores = np.full((len(candidates), 4), np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_statistics = np.log(statistics)
        for j, candidate in enumerate(candidates):
            candidate = tuple(sorted(candidate))
            index = indices[candidate]
            scores[j, :2] = statistics[:, index]
            singleton_indices = [indices[(atom,)] for atom in candidate]
            scores[j, 2] = (log_statistics[0, index]
                            - log_statistics[0, singleton_indices].sum())
            if len(candidate) > 1:
                subset_indices = [indices[candidate[:i] + candidate[i + 1:]]
                                  for i in range(len(candidate))]
                scores[j, 3] = (log_statistics[1, index]
                                - log_statistics[1, subset_indices].max())
    return scores


def _local_stability(trees, X, interactions, feature_groups, *, normalize=True,
                     first_only=False):
    """Fraction of trees using each signed interaction on each row's path.

    All visited directions count by default, as in lo-siRF. ``first_only``
    instead keeps the first split per feature/group, matching signed discovery.
    Trees are evaluated once each; memory does not grow with tree count.
    ``normalize=False`` keeps integer counts for exact permutation comparisons.
    """
    interactions = [frozenset(interaction) for interaction in interactions]
    stability = np.zeros((len(X), len(interactions)), dtype=np.int64)
    if not interactions:
        return stability / len(trees) if normalize else stability
    for tree in trees:
        paths = tree._leaf_paths(signed=True, feature_groups=feature_groups,
                                 first_only=first_only)
        leaves, inverse = np.unique(tree.apply(X), return_inverse=True)
        reached_paths = [paths[leaf] for leaf in leaves]
        for j, interaction in enumerate(interactions):
            present = np.fromiter((interaction <= path for path in reached_paths),
                                  dtype=bool, count=len(leaves))
            stability[:, j] += present[inverse]
    return stability / len(trees) if normalize else stability


def _permutation_ranking(counts, y_target_bool, n_permutations, random_state,
                         n_trees):
    """Return target-minus-other mean LSI and two-sided permutation scores.

    The fitted forest and integer tree counts stay fixed. Labels are
    permuted with their class counts preserved. Ties count as exceedances;
    the (count + 1)/(n_permutations + 1) correction prevents zero scores.
    Comparisons use the integer numerator of the difference in mean LSI,
    preserving ties exactly even with fractional scores and unequal classes.
    """
    if counts.shape[1] == 0:
        return np.empty(0), np.empty(0)
    target = np.asarray(y_target_bool, dtype=bool)
    n_rows = len(target)
    n_positive = int(target.sum())
    n_negative = n_rows - n_positive
    # Bound both products below. Python integers handle unusually large
    # sample/tree counts without silently overflowing their cross products.
    if n_rows ** 2 * int(n_trees) > np.iinfo(np.int64).max:
        counts = counts.astype(object)
    total = counts.sum(axis=0)

    def difference(labels):
        positive = labels @ counts
        return n_rows * positive - n_positive * total

    observed = difference(target)
    threshold = np.abs(observed)
    exceedances = np.ones(counts.shape[1], dtype=np.intp)
    rng = (np.random.RandomState() if random_state is None
           else check_random_state(random_state))
    for _ in range(n_permutations):
        permuted = difference(rng.permutation(target))
        exceedances += np.abs(permuted) >= threshold
    observed = np.asarray(observed / (n_positive * n_negative * n_trees), dtype=float)
    return observed, exceedances / (n_permutations + 1)


def _benjamini_hochberg(pvalues):
    """Return Benjamini-Hochberg adjusted p-values in their original order."""
    pvalues = np.asarray(pvalues, dtype=float)
    order = np.argsort(pvalues)
    ranked = pvalues[order] * len(pvalues) / np.arange(1, len(pvalues) + 1)
    adjusted = np.empty_like(pvalues)
    adjusted[order] = np.minimum(1, np.minimum.accumulate(ranked[::-1])[::-1])
    return adjusted
