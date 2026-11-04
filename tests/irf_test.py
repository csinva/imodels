"""Checks for IRFClassifier and IRFRegressor.

The shared registry suites cover the estimator API. These cover what is specific
to iterative random forests: the schedule of weighted forests, per-node weighted
feature sampling, the leaf masses and search rules of random intersection trees
(RIT), and the split search. `test_numpy_split_search_matches_sklearn_stump_bit_for_bit`
matters most there: the forest splits nodes with a NumPy search that must choose
exactly what a scikit-learn stump would, so a scikit-learn release that changes
its tie-breaking fails here rather than silently changing results.
"""

from collections import Counter
from itertools import combinations, product
from math import prod

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.base import clone
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from imodels import IRFClassifier, IRFRegressor
from imodels.tree.iterative_random_forest import _forest, _interactions, _rit
from imodels.tree.iterative_random_forest import iterative_random_forest as irf_module
from imodels.tree.iterative_random_forest._forest import _WeightedForest, _WeightedTree
from imodels.tree.iterative_random_forest._interactions import (
    _benjamini_hochberg, _interaction_scores, _interaction_terms,
    _permutation_ranking, _tree_statistics,
)
from imodels.tree.iterative_random_forest._rit import _random_intersection_trees


def _small_model(**kwargs):
    params = dict(n_estimators=5, n_iterations=2, n_bootstraps=3,
                  n_rit=8, rit_depth=2, random_state=13)
    params.update(kwargs)
    return IRFClassifier(**params)


def _tree(**kwargs):
    params = dict(max_features=1, max_depth=None, min_samples_split=2,
                  min_samples_leaf=1, bootstrap=False, random_state=0)
    params.update(kwargs)
    return _WeightedTree(**params)


def _and_data(n=300, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.uniform(size=(n, 5))
    y = ((X[:, 0] > 0.5) & (X[:, 1] > 0.5)).astype(float) + 0.05 * rng.normal(size=n)
    return X, y


# the estimators

def test_classifier_finds_and_interaction():
    X = np.tile([[0, 0], [0, 1], [1, 0], [1, 1]], (40, 1))
    y = np.where(np.all(X == 1, axis=1), "positive", "negative")
    model = _small_model(max_features=None, bootstrap=False,
                         n_bootstraps=4, n_rit=20).fit(X, y)
    assert_array_equal(model.predict(X), y)
    assert model.interaction_class_ == "positive"
    assert model.interaction_stability_[(0, 1)] == 1.0


def test_regressor_finds_and_interaction():
    X, y = _and_data()
    model = IRFRegressor(n_estimators=30, n_iterations=3, n_bootstraps=5,
                         leaf_threshold=0.5, random_state=0).fit(X, y)
    assert model.interactions_[0] == (0, 1)
    assert model.interaction_stability_[(0, 1)] == 1.0
    assert set(np.argsort(model.feature_importances_)[-2:]) == {0, 1}
    assert model.score(X, y) > 0.8


@pytest.mark.parametrize("signed", [False, True])
def test_serial_and_parallel_fits_match_without_touching_global_rng(signed):
    rng = np.random.RandomState(9)
    X = rng.normal(size=(60, 4))
    y = (X[:, 0] * X[:, 1] > 0).astype(int)
    np.random.seed(314)
    before = np.random.get_state()
    serial = _small_model(n_jobs=1, signed=signed).fit(X, y)
    parallel = clone(serial).set_params(n_jobs=2).fit(X, y)
    after = np.random.get_state()
    assert before[0] == after[0]
    assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]
    assert_allclose(serial.predict_proba(X), parallel.predict_proba(X), atol=0, rtol=0)
    assert_allclose(serial.feature_weights_history_, parallel.feature_weights_history_)
    assert serial.bootstrap_interactions_ == parallel.bootstrap_interactions_
    assert serial.interaction_stability_ == parallel.interaction_stability_
    if signed:
        assert serial.interaction_scores_.keys() == parallel.interaction_scores_.keys()
        for interaction, scores in serial.interaction_scores_.items():
            assert_allclose(list(scores.values()),
                            list(parallel.interaction_scores_[interaction].values()),
                            atol=0, rtol=0, equal_nan=True)
    for first, second in zip(serial.bootstrap_samples_, parallel.bootstrap_samples_):
        assert_array_equal(first, second)


def test_outer_forests_use_the_bootstrap_samples_and_frozen_weights(monkeypatch):
    # Each full-data forest samples features by its predecessor's importances.
    # The outer forests all use the weights that entered the last one, not the
    # importances it produced.
    calls = []
    forest_paths = irf_module._forest_paths

    def record(forests, datasets, n_jobs=None):
        datasets = list(datasets)
        calls.append((forests, datasets))
        return forest_paths(forests, iter(datasets), n_jobs)

    monkeypatch.setattr(irf_module, "_forest_paths", record)
    X, y = make_classification(n_samples=120, n_features=6, random_state=1)
    model = IRFClassifier(n_estimators=6, n_iterations=3, n_bootstraps=4,
                          n_estimators_bootstrap=2, random_state=0).fit(X, y)

    history = model.feature_weights_history_
    assert_allclose(history[0], np.full(6, 1 / 6))
    for weights, importances in zip(history[1:], model.feature_importances_history_):
        assert_allclose(weights, importances / importances.sum())
    assert_array_equal(model.feature_weights_, history[-1])

    (forests, datasets), = calls
    assert [forest.n_estimators for forest in forests] == [2] * 4
    for (X_outer, _, weights, _, _), indices in zip(datasets, model.bootstrap_samples_):
        assert_array_equal(X_outer, X.astype(np.float32)[indices])
        assert_array_equal(weights, model.feature_weights_)


def test_outer_bootstraps_are_full_size_with_replacement_and_stratified():
    X = np.arange(80).reshape(40, 2)
    y = np.repeat(["common", "rare"], [30, 10])
    model = _small_model(n_estimators=2, max_depth=1, n_bootstraps=4).fit(X, y)
    for indices in model.bootstrap_samples_:
        assert len(indices) == 40
        assert len(np.unique(indices)) < len(indices)
        assert_array_equal(np.unique(y[indices], return_counts=True)[1], [30, 10])


# the weighted forest

def test_feature_weights_set_candidate_probabilities_and_exclude_zero():
    # Every column predicts perfectly, so with one candidate per node the root
    # feature shows which candidate was drawn.
    X = np.tile([[0, 0, 0], [1, 1, 1]], (20, 1))
    y = X[:, 0]
    forest = _WeightedForest(n_estimators=300, max_features=1, max_depth=1,
                             bootstrap=False, random_state=4).fit(
        X, y, feature_weights=np.array([0.9, 0.1, 0.0])
    )
    roots = np.array([tree.feature_[0] for tree in forest.estimators_])
    assert not np.any(roots == 2)
    assert 0.84 < np.mean(roots == 0) < 0.96


def test_candidates_are_redrawn_at_each_node_so_xor_is_learnable():
    # With one candidate per tree, XOR cannot be represented. Drawing again at
    # each node, and keeping the zero-gain first split, can.
    X = np.tile([[0, 0], [0, 1], [1, 0], [1, 1]], (12, 1))
    y = np.logical_xor(X[:, 0], X[:, 1]).astype(int)
    forest = _WeightedForest(n_estimators=40, max_features=1,
                             bootstrap=False, random_state=3).fit(
        X, y, feature_weights=np.array([0.5, 0.5])
    )
    complete = [tree for tree in forest.estimators_
                if np.array_equal(np.argmax(tree.predict_proba(X), axis=1), y)]
    assert complete
    assert set(complete[0].feature_[complete[0].feature_ >= 0]) == {0, 1}


def test_with_every_feature_a_candidate_a_tree_matches_sklearn_cart():
    X, y = make_classification(n_samples=180, n_features=5, n_informative=3,
                               n_redundant=0, random_state=8)
    X = X.astype(np.float32)
    weights = np.linspace(0.4, 2, len(y))
    reference = DecisionTreeClassifier(max_depth=3, min_samples_leaf=3,
                                       random_state=2).fit(X, y, sample_weight=weights)
    actual = _tree(max_features=5, max_depth=3, min_samples_leaf=3).fit(
        X, y, np.full(5, 0.2), weights, n_classes=2
    )
    assert_allclose(actual.predict_proba(X), reference.predict_proba(X))
    assert_allclose(actual.feature_importances_, reference.feature_importances_)


def test_with_every_feature_a_candidate_a_regression_tree_matches_sklearn_cart():
    rng = np.random.RandomState(8)
    X = rng.normal(size=(150, 4)).astype(np.float32)
    y = X[:, 0] * X[:, 1] + rng.normal(size=150)
    weights = np.linspace(0.5, 2, len(y))
    # five-row leaves avoid one-row splits that several features make equally well
    reference = DecisionTreeRegressor(max_depth=3, min_samples_leaf=5,
                                      random_state=0).fit(X, y, sample_weight=weights)
    actual = _tree(max_features=4, max_depth=3, min_samples_leaf=5,
                   task="regression").fit(X, y, np.full(4, 0.25), weights)
    assert_allclose(actual.predict(X), reference.predict(X))
    assert_allclose(actual.feature_importances_, reference.feature_importances_)


def test_numpy_split_search_matches_sklearn_stump_bit_for_bit(monkeypatch):
    # Nodes the NumPy search resolves must match a seeded sklearn stump
    # exactly, including tie-breaking; the rest are fit by the stump itself.
    tie_breaks = []
    feature_order = _forest._sklearn_feature_order
    monkeypatch.setattr(_forest, "_sklearn_feature_order",
                        lambda *args: tie_breaks.append(1) or feature_order(*args))
    rng = np.random.RandomState(0)
    resolved = {False: 0, True: 0}
    for trial in range(1500):
        regression = trial % 2 == 1
        n = int(rng.choice([2, 3, 5, 8, 12, 40, 150]))
        k = int(rng.randint(1, 6))
        X = rng.randn(n, k) if trial % 3 else rng.randint(0, 3, size=(n, k))
        X = X.astype(np.float32)
        weights = (rng.randint(1, 4, size=n).astype(float) if trial % 4 == 0
                   else np.ones(n))
        min_samples_leaf = int(rng.choice([1, 1, 2]))
        seed = rng.randint(np.iinfo(np.int32).max)
        if regression:
            raw = rng.randn(n) if trial % 5 else rng.randint(0, 3, size=n).astype(float)
            if raw.min() == raw.max():
                continue
            y = _forest._rescale_for_split_search(
                raw, np.dot(weights, raw) / weights.sum())
        else:
            y = rng.randint(0, 3, size=n)
            if np.unique(y).size < 2:
                continue
        fast = _forest._fast_split(X, y, weights, regression, 3, min_samples_leaf, seed)
        if fast is _forest._UNRESOLVED:
            continue
        resolved[regression] += 1
        expected = _forest._sklearn_split(X, y, weights, regression, 2,
                                          min_samples_leaf, seed)
        if regression and fast is not None and expected is not None:
            fast, expected = fast[:2], expected[:2]
        assert fast == expected, (trial, fast, expected)
    assert min(resolved.values()) > 600
    assert tie_breaks


def test_rit_leaf_mass_counts_every_outer_row():
    # RIT samples leaves by the outer sample's rows routed to them, including
    # rows the tree's own bootstrap left out, not by the tree's in-bag counts.
    X = np.arange(4, dtype=np.float32).reshape(-1, 1)
    weights = np.array([0.25, 0.5, 1, 2])
    tree = _tree(bootstrap=True)
    [(_, _, mass)] = _forest._fit_tree_paths(
        tree, X, np.zeros(4, dtype=int), np.ones(1), weights, 2)
    # the tree's bootstrap draws rows [0, 3, 1, 0], leaving out row 2
    assert_array_equal(tree.bootstrap_indices_, [0, 3, 1, 0])
    assert tree.terminal_paths()[0][2] == 3.0
    assert mass == 3.75


# random intersection trees

def test_rit_never_returns_singletons():
    assert _random_intersection_trees([(0,)], [1], 10, 2, 2, random_state=0) == set()


def test_rit_saves_pairs_immediately_and_larger_sets_at_full_depth():
    # Seed 2 draws path A twice, so the root is the pair A. It is saved without
    # a third draw, which would have shrunk it to a singleton.
    rng = np.random.RandomState(2)
    assert _random_intersection_trees(
        [(0, 1), (1, 2)], [1, 1], 1, 100, 2, random_state=rng) == {(0, 1)}
    reference = np.random.RandomState(2)
    reference.random_sample(2)
    assert rng.random_sample() == reference.random_sample()
    # Seed 2 samples A, A, B. A larger root is saved only once max_depth paths
    # are combined; one more draw reduces it to the pair {1, 2}.
    kwargs = dict(paths=[(0, 1, 2), (1, 2, 3)], path_weights=[1, 1], n_trees=1,
                  n_children=1, random_state=2)
    assert _random_intersection_trees(max_depth=2, **kwargs) == {(0, 1, 2)}
    assert _random_intersection_trees(max_depth=3, **kwargs) == {(1, 2)}


def test_rit_deduplicates_within_a_replicate():
    assert _random_intersection_trees(
        [(0, 2, 4)], [1], 50, 4, 2, random_state=1) == {(0, 2, 4)}


class _FixedUniforms:
    """Feed one enumerated sequence of draws to RIT."""

    def __init__(self, values):
        self.values = np.asarray(values)
        self.position = 0

    def random_sample(self, size=None):
        count = 1 if size is None else int(np.prod(size))
        end = self.position + count
        assert end <= len(self.values), "RIT consumed too many random draws."
        result = self.values[self.position:end]
        self.position = end
        return float(result[0]) if size is None else result.reshape(size)


def test_rit_matches_exact_probabilities_on_a_weighted_chain(monkeypatch):
    # A={0,1}, B={0,1,2}, C={0,2}, with probabilities 1/4, 1/2, 1/4. The two
    # root draws give A with probability 5/16, C with 5/16, B with 4/16, or a
    # discarded singleton with 2/16. Only B needs a third draw, so B survives
    # with probability 8/64, A and C each with 5/16 + (4/16)(1/4) = 24/64, and
    # nothing with 8/64. Enumerating an unused third draw keeps the masses whole.
    monkeypatch.setattr(_rit, "check_random_state", lambda rng: rng)
    paths = [(0, 1), (0, 1, 2), (0, 2)]
    weights = [1, 2, 1]
    midpoints = [0.125, 0.5, 0.875]
    actual = Counter()
    for sequence in product(range(3), repeat=3):
        rng = _FixedUniforms([midpoints[index] for index in sequence])
        result = _rit._random_intersection_trees(
            paths, weights, n_trees=1, max_depth=3, n_children=1, random_state=rng)
        assert rng.position == (3 if sequence[:2] == (1, 1) else 2)
        actual[frozenset(result)] += prod(weights[index] for index in sequence)
    assert actual == {
        frozenset({(0, 1, 2)}): 8,
        frozenset({(0, 1)}): 24,
        frozenset({(0, 2)}): 24,
        frozenset(): 8,
    }


def test_rit_integer_seed_is_repeatable_without_touching_global_rng():
    before = np.random.get_state()
    kwargs = dict(paths=[(0, 1), (1, 2), (1, 2, 3)], path_weights=[1, 2, 3],
                  n_trees=20, max_depth=3, n_children=2, random_state=14)
    assert _random_intersection_trees(**kwargs) == _random_intersection_trees(**kwargs)
    after = np.random.get_state()
    assert before[0] == after[0]
    assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]


# signed interactions and local importance

def test_signed_grouped_discovery_preserves_predictions_and_candidates(monkeypatch):
    X = np.tile([[0, 0], [0, 1], [1, 0], [1, 1]], (20, 1))
    y = ((X[:, 0] == 1) & (X[:, 1] == 0)).astype(int)
    X = X[:, [0, 0, 1, 1]]
    model = _small_model(signed=True, feature_groups=[7, 7, 2, 2],
                         max_features=None).fit(X, y)
    interaction = ((2, -1), (7, 1))
    assert model.interactions_ == [interaction]
    assert model.interaction_stability_ == {interaction: 1.0}
    unsigned = clone(model).set_params(signed=False).fit(X, y)
    assert unsigned.interactions_ == [(2, 7)]
    assert_array_equal(model.predict_proba(X), unsigned.predict_proba(X))
    assert_array_equal(model.feature_weights_history_, unsigned.feature_weights_history_)
    for scores in [model.interaction_scores_, *model.bootstrap_interaction_scores_]:
        assert all(type(value) is float for value in scores[interaction].values())
    no_bootstrap = clone(model).set_params(n_bootstraps=0).fit(X, y)
    assert no_bootstrap.interactions_ == [interaction]
    assert no_bootstrap.interaction_stability_ == no_bootstrap.interaction_scores_ == {}
    assert_allclose(no_bootstrap.local_stability(X)[:, 0], y)
    assert list(no_bootstrap.rank_interactions(X, y, n_permutations=9)) == [interaction]
    with pytest.raises(ValueError):
        no_bootstrap.select_interactions()
    # Group before choosing the first direction: one group cannot form a pair.
    collapsed = clone(no_bootstrap).set_params(feature_groups=[0] * 4).fit(X, y)
    assert collapsed.interactions_ == []
    assert_allclose(collapsed.local_stability(X[y == 1], [((0, -1), (0, 1))]), 1)
    assert_allclose(collapsed.local_stability(
        X[y == 1], [((0, -1), (0, 1))], first_only=True), 0)
    # Failed bootstrap searches do not remove full-data candidates or denominator rows.
    searches = iter([{interaction}, set(), set()])
    monkeypatch.setattr(IRFClassifier, "_discover_interactions", lambda *args: next(searches))
    missed = clone(model).set_params(n_bootstraps=2).fit(X, y)
    assert missed.interaction_stability_ == {interaction: 0.0}
    assert all(interaction in scores for scores in missed.bootstrap_interaction_scores_)


def test_signed_regression_uses_frozen_cutoff_without_binarizing_predictions():
    X = np.tile([[0, 0], [0, 1], [1, 0], [1, 1]], (8, 1))
    y = np.r_[X.sum(axis=1) - 1., np.full(33, 999.)]
    X = np.vstack([X, np.zeros((33, 2))])
    weights = np.r_[np.ones(32), np.zeros(33)]
    with pytest.warns(UserWarning, match="bootstrap=False"):
        model = IRFRegressor(
            signed=True, n_estimators=3, n_iterations=1, n_bootstraps=2,
            n_rit=20, rit_depth=2, max_features=None, bootstrap=False,
            random_state=1).fit(X, y, sample_weight=weights)
    unsigned = clone(model).set_params(signed=False).fit(X, y, sample_weight=weights)
    assert model.leaf_threshold is None and model.leaf_threshold_ == 0
    assert_array_equal(model.predict(X), unsigned.predict(X))
    assert_allclose(model.predict(X[:32]), y[:32])
    interaction = ((0, 1), (1, 1))
    assert model.interactions_ == [interaction]  # predicted zero is excluded
    for indices, scores in zip(model.bootstrap_samples_, model.bootstrap_interaction_scores_):
        expected = np.sum(y[indices] > 0) / np.sum(y[indices] >= 0)
        assert_allclose(scores[interaction]["prevalence"], expected)
    # Observed zero is included; the evaluation median (50) does not replace the cutoff.
    result = model.rank_interactions(
        [[0, 0], [1, 1], [0, 1], [1, 1]], [-1., 0., 100., 101.],
        n_permutations=9, random_state=0)
    assert_allclose(result[interaction]["difference"], 2 / 3)


@pytest.mark.parametrize("estimator, cutoff, side", [
    (IRFClassifier, None, "inside"),
    (IRFRegressor, -5., "inside"), (IRFRegressor, 5., "outside"),
])
def test_signed_fit_warns_when_scoring_target_has_only_one_side(estimator, cutoff, side):
    X = np.arange(8).reshape(4, 2)
    y = np.ones(4) if estimator is IRFClassifier else [-1., 0., 0., 1.]
    kwargs = {} if estimator is IRFClassifier else {"leaf_threshold": cutoff}
    with pytest.warns(UserWarning, match=f"is {side} the signed scoring target"):
        estimator(signed=True, n_estimators=1, n_iterations=1, n_bootstraps=0,
                  random_state=0, **kwargs).fit(X, y)


def test_local_directions_and_feature_shortcut_include_unused_groups():
    X = np.repeat([[0, 0], [1, 0], [2, 0]], 8, axis=0)
    y = (X[:, 0] == 1).astype(int)
    with pytest.warns(UserWarning, match="bootstrap=False"):
        model = _small_model(signed=True, feature_groups=[9, 4], max_features=None,
                             n_estimators=1, n_iterations=1, n_bootstraps=0,
                             bootstrap=False).fit(X, y)
    features = [((4, -1),), ((4, 1),), ((9, -1),), ((9, 1),)]
    assert_allclose(model.local_stability(X, "features"),
                    np.column_stack([np.zeros((len(X), 2)), X[:, 0] < 2, X[:, 0] > 0]))
    assert model.local_stability([[1, 0]], "features", first_only=True).sum() == 1
    result = model.rank_interactions(X, y, "features", first_only=True,
                                     n_permutations=19, random_state=0)
    assert sorted(result[term]["difference"] for term in features[2:]) == [-.5, .5]
    assert all(result[term]["pvalue_bh"] == 1 for term in features[:2])
    assert all(0 < row["pvalue"] <= row["pvalue_bh"] <= 1 for row in result.values())


def test_signed_score_batches_preserve_weights_and_original_row_oob_exclusion(monkeypatch):
    monkeypatch.setattr(_interactions, "_LEAF_BLOCK_SIZE", 1)
    monkeypatch.setattr(_interactions, "_TERM_BLOCK_SIZE", 2)
    a, b, absent = (0, 1), (1, -1), (2, 1)
    paths = {1: frozenset([a, b]), 3: frozenset([a]),
             4: frozenset([b]), 6: frozenset()}
    weights = np.array([1., 3., 2., 2., 4., 1.])
    kwargs = dict(paths=paths, leaf_ids=np.array([1, 1, 1, 3, 4, 6]),
                  y=np.array([1, 1, 0, 1, 0, 0]), inbag_indices=np.array([0, 2, 5]),
                  row_ids=np.array([10, 10, 11, 12, 13, 14]),
                  terms=[(a,), (b,), (a, b), (absent,)], target_class=1)
    # Both copies of original row 10 are in-bag, leaving no OOB support for a+b.
    expected = [[1, 2 / 3, 2 / 3, 0], [1, 0, np.nan, np.nan]]
    assert_allclose(_tree_statistics(sample_weight=weights, **kwargs), expected)
    assert_allclose(_tree_statistics(sample_weight=13 * weights, **kwargs), expected)


def test_signed_score_formulas_and_unsupported_ratios():
    a, b, c = (0, 1), (1, -1), (2, 1)
    candidates = [(a, b), (a, b, c)]
    terms = _interaction_terms(candidates)
    values = {(a,): (.5, .5), (b,): (.4, .6), (c,): (.8, .7),
              (a, b): (.3, .8), (a, c): (.35, .75),
              (b, c): (.32, .85), (a, b, c): (.25, .9)}
    scores = _interaction_scores(candidates, terms, np.array([values[t] for t in terms]).T)
    assert_allclose(scores, [[.3, .8, np.log(1.5), np.log(.8 / .6)],
                             [.25, .9, np.log(.25 / (.5 * .4 * .8)), np.log(.9 / .85)]])
    scores = _interaction_scores([(a, b)], [(a,), (b,), (a, b)],
                                 np.array([[0, .5, 0], [.5, .7, 0]]))[0]
    assert np.isnan(scores[2]) and np.isneginf(scores[3])  # 0/0 versus 0/positive
    high_order = tuple((j, 1) for j in range(20))
    terms = _interaction_terms([high_order])
    scores = _interaction_scores([high_order], terms, np.full((2, len(terms)), 1e-20))
    assert_allclose(scores[0, 2], -19 * np.log(1e-20))  # null product underflows


@pytest.mark.parametrize("counts,n_trees,difference,pvalues", [
    ([[2, 0, 1], [2, 0, 1], [0, 2, 1], [0, 2, 1]], 2, [1, -1, 0], [3/7, 3/7, 1]),
    ([[0], [0], [2], [0], [2], [2]], 3, [-.5], [7/16]),
])
def test_permutation_ranking_matches_exhaustive_partitions(
        monkeypatch, counts, n_trees, difference, pvalues):
    counts = np.array(counts, dtype=np.int64)
    partitions = list(combinations(range(len(counts)), 2))

    class AllPartitions:
        def __init__(self):
            self.partitions = iter(partitions)

        def permutation(self, labels):
            return np.isin(np.arange(len(counts)), next(self.partitions))

    monkeypatch.setattr(_interactions, "check_random_state", lambda _: AllPartitions())
    actual = _permutation_ranking(counts, np.arange(len(counts)) < 2,
                                  len(partitions), 0, n_trees=n_trees)
    assert_allclose(actual, [difference, pvalues])


def test_permutation_imbalance_overflow_and_bh_ties():
    counts = np.repeat([0, 1], 500).astype(np.int64)[:, None]
    for target in (np.arange(1000) == 0, np.arange(1000) != 0):
        difference, pvalue = _permutation_ranking(counts, target, 99, 0, n_trees=3)
        assert_allclose(abs(difference[0]), 500 / (999 * 3))
        assert_array_equal(pvalue, [1])  # every permutation ties, including opposite tails
    counts = np.array([[0], [0], [2], [2]], dtype=np.int64)
    expected = _permutation_ranking(counts, [True, True, False, False], 19, 3, n_trees=3)
    actual = _permutation_ranking(counts * 10**18, [True, True, False, False],
                                  19, 3, n_trees=3 * 10**18)
    assert_allclose(actual, expected)
    assert_allclose(_benjamini_hochberg([.04, .01, .03, .002, 1., .01]),
                    [.048, .02, .045, .012, 1., .02])


def test_selection_requires_all_screens_and_available_scores():
    X = np.tile([[0, 0], [0, 1], [1, 0], [1, 1]], (20, 1))
    y = np.all(X == 1, axis=1).astype(int)
    model = _small_model(signed=True, max_features=None).fit(X, y)
    interaction = model.interactions_[0]
    for weak in ("recovery", "fsd", "mip"):
        model.interaction_stability_[interaction] = .4 if weak == "recovery" else .8
        for metric in ("fsd", "mip"):
            model.interaction_scores_[interaction][metric + "_stability"] = .4 if weak == metric else .8
        assert model.select_interactions(.5) == []
        assert model.select_interactions(.4) == [interaction]
    with pytest.warns(UserWarning, match="bootstrap=False"):
        model.set_params(bootstrap=False).fit(X, y)
    assert np.isnan(model.interaction_scores_[interaction]["precision"])
    assert np.isnan(model.interaction_scores_[interaction]["mip"])
    assert model.select_interactions(0) == []
