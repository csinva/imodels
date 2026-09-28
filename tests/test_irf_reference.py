"""Small exact oracles for the bootstrap and historical R RIT sampling laws.

These do not establish numerical equivalence to another iRF package. They
enumerate finite sample spaces and use a controlled two-layer bootstrap.
"""

from collections import Counter
import importlib
from itertools import product
from math import prod

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from imodels import IRFClassifier


irf_module = importlib.import_module(
    "imodels.tree.iterative_random_forest.iterative_random_forest"
)
forest_module = importlib.import_module("imodels.tree.iterative_random_forest._forest")
rit_module = importlib.import_module("imodels.tree.iterative_random_forest._rit")


def _fit_outer_forests_serially(monkeypatch):
    """Fit outer forests one at a time with ``fit`` so tests can inspect them.

    The estimator fits all outer trees in one worker pool and keeps only their
    paths; test_irf_forest checks that pool against this serial reference.
    """
    def forest_paths(forests, datasets, n_jobs=None):
        for forest, (X, y, feature_weights, sample_weight, n_classes) in zip(
                forests, datasets):
            forest.fit(X, y, feature_weights=feature_weights,
                       sample_weight=sample_weight, n_classes=n_classes)
            yield [leaf for tree in forest.estimators_
                   for leaf in tree.terminal_paths(X, sample_weight=sample_weight)]

    monkeypatch.setattr(irf_module, "_forest_paths", forest_paths)


class _FixedUniforms:
    """Feed one enumerated RNG outcome to the implementation being checked."""

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


def test_exact_weighted_three_path_chain_distribution(monkeypatch):
    # A={0,1}, B={0,1,2}, C={0,2}, with probabilities 1/4, 1/2, 1/4.
    # The two root draws make A with probability 5/16, C with 5/16, B with
    # 4/16, or a discarded singleton with 2/16. Only B needs a third draw.
    # Thus B survives with probability 8/64; A and C each have probability
    # 5/16+(4/16)*(1/4)=24/64; and an empty result has probability 8/64.
    # Enumerating a third draw even when unused accounts for its full mass.
    monkeypatch.setattr(rit_module, "check_random_state", lambda rng: rng)
    paths = [(0, 1), (0, 1, 2), (0, 2)]
    weights = [1, 2, 1]
    midpoints = [0.125, 0.5, 0.875]
    actual = Counter()
    for sequence in product(range(3), repeat=3):
        rng = _FixedUniforms([midpoints[index] for index in sequence])
        result = rit_module._random_intersection_trees(
            paths, weights, n_trees=1, max_depth=3, n_children=1,
            random_state=rng,
        )
        assert rng.position == (3 if sequence[:2] == (1, 1) else 2)
        actual[frozenset(result)] += prod(weights[index] for index in sequence)
        if sequence == (0, 1, 2):
            # The pair is absorbing: C is not drawn to shrink it to {0}.
            assert result == {(0, 1)}
    assert actual == {
        frozenset({(0, 1, 2)}): 8,
        frozenset({(0, 1)}): 24,
        frozenset({(0, 2)}): 24,
        frozenset(): 8,
    }
    assert sum(actual.values()) == 64


def test_exact_branching_distribution_preserves_shared_ancestor_dependence(monkeypatch):
    # A={0,1,2,3}, B={0,1,4}, and A intersect B is the absorbing pair P.
    # Two root draws, two children, and four grandchildren require at most
    # eight draws. Pad early-stopped histories with unused draws to enumerate
    # all 2^8 equally likely histories. Mixed root draws give P in 128 cases.
    # Given root AA, a child branch lacks an A terminal with probability 5/8;
    # neither branch has one with probability 25/64. Only all-A descendants
    # avoid P (1/64). The other 38/64 give both A and P. Root BB is symmetric.
    monkeypatch.setattr(rit_module, "check_random_state", lambda rng: rng)
    actual = Counter()
    for sequence in product([0.25, 0.75], repeat=8):
        rng = _FixedUniforms(sequence)
        result = rit_module._random_intersection_trees(
            [(0, 1, 2, 3), (0, 1, 4)], [1, 1], n_trees=1, max_depth=4,
            n_children=2, random_state=rng,
        )
        assert rng.position in (2, 4, 6, 8)
        actual[frozenset(result)] += 1
    assert actual == {
        frozenset({(0, 1)}): 178,
        frozenset({(0, 1, 2, 3)}): 1,
        frozenset({(0, 1, 4)}): 1,
        frozenset({(0, 1), (0, 1, 2, 3)}): 38,
        frozenset({(0, 1), (0, 1, 4)}): 38,
    }
    assert sum(actual.values()) == 256


def test_controlled_nested_bootstrap_keeps_duplicate_rows_and_routed_mass(monkeypatch):
    X = np.column_stack([np.arange(6), np.zeros(6)])
    y = np.array([0, 0, 0, 1, 1, 1])
    weights = np.array([1, 0, 2, 4, 8, 16], dtype=float)
    # Zero-weight row 1 is removed first. These fixed outer samples therefore
    # refer to original rows [0,2,3,3,3] and [0,0,3,4,5]. Both retain the
    # active dataset's two negative and three positive observations.
    outer_indices = iter([np.array([0, 1, 2, 2, 2]),
                          np.array([0, 0, 2, 3, 4])])
    fits, rit_inputs = [], []
    original_forest_fit = forest_module._WeightedForest.fit
    original_tree_fit = forest_module._WeightedTree.fit

    def fixed_resample(population, *, replace, n_samples, stratify, random_state):
        assert_array_equal(population, np.arange(5))
        assert replace and n_samples == 5
        assert_array_equal(stratify, y[[0, 2, 3, 4, 5]])
        indices = next(outer_indices)
        assert_array_equal(np.bincount(stratify[indices]), np.bincount(stratify))
        return indices

    def fixed_inner_fit(self, *args, **kwargs):
        # NumPy seed 0 draws [4,0,3,3,3] from five rows. Controlling this layer
        # makes inner training counts visibly different from outer route counts.
        self.random_state = 0
        return original_tree_fit(self, *args, **kwargs)

    def record_fit(self, X_fit, y_fit, **kwargs):
        fits.append((X_fit.copy(), y_fit.copy(), kwargs["sample_weight"].copy()))
        return original_forest_fit(self, X_fit, y_fit, **kwargs)

    def record_rit(paths, masses, **kwargs):
        rit_inputs.append((paths, masses))
        return set()

    monkeypatch.setattr(irf_module, "resample", fixed_resample)
    monkeypatch.setattr(forest_module._WeightedTree, "fit", fixed_inner_fit)
    monkeypatch.setattr(forest_module._WeightedForest, "fit", record_fit)
    _fit_outer_forests_serially(monkeypatch)
    monkeypatch.setattr(irf_module, "_random_intersection_trees", record_rit)
    model = IRFClassifier(n_estimators=1, n_iterations=2, n_bootstraps=2,
                          max_features=None, max_depth=1, random_state=0).fit(
        X, y, sample_weight=weights
    )

    expected_rows = [[0, 2, 3, 4, 5], [0, 2, 3, 4, 5],
                     [0, 2, 3, 3, 3], [0, 0, 3, 4, 5]]
    assert len(fits) == len(expected_rows)
    for (actual_X, actual_y, actual_weights), indices in zip(fits, expected_rows):
        assert_array_equal(actual_X, X[indices])
        assert_array_equal(actual_y, y[indices])
        assert_allclose(actual_weights, weights[indices] / weights.max())
    for actual, expected in zip(model.bootstrap_samples_, expected_rows[2:]):
        assert_array_equal(actual, expected)

    # First outer tree trains on x=[3,0,3,3,3], splitting at 1.5. Its positive
    # leaf receives original row 2 plus all three copies of row 3 when ALL
    # outer rows are routed: mass (2+4+4+4)/16=14/16, not inner mass 4*4/16=1.
    # Row 2's true class is zero, so this also checks predicted-leaf filtering.
    # Second tree splits at 2: outer positive mass is (4+8+16)/16=28/16.
    assert len(rit_inputs) == 2
    assert rit_inputs[0][0] == [frozenset({0})]
    assert rit_inputs[1][0] == [frozenset({0})]
    assert_allclose(rit_inputs[0][1], [14 / 16])
    assert_allclose(rit_inputs[1][1], [28 / 16])
