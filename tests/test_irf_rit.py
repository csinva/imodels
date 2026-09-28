"""Tests of weighted sampling and the historical R iRF RIT conventions."""

import numpy as np
import pytest

from imodels.tree.iterative_random_forest._rit import _random_intersection_trees


def test_depth_counts_sampled_paths_and_larger_sets_need_terminal_depth():
    # Seed 2 samples A, A, B. The root combines two paths; one more draw
    # reduces A to a pair. Returning internal nodes would also emit A.
    kwargs = dict(
        paths=[(0, 1, 2), (1, 2, 3)],
        path_weights=[1, 1],
        n_trees=1,
        n_children=1,
        random_state=2,
    )
    assert _random_intersection_trees(max_depth=2, **kwargs) == {(0, 1, 2)}
    assert _random_intersection_trees(max_depth=3, **kwargs) == {(1, 2)}


def test_pair_root_is_saved_without_drawing_descendants():
    # Seed 2 draws A twice. Its third draw would shrink A to a singleton,
    # but historical R saves the pair and stops after the two root draws.
    rng = np.random.RandomState(2)
    assert _random_intersection_trees(
        [(0, 1), (1, 2)], [1, 1], 1, 100, 2, random_state=rng
    ) == {(0, 1)}
    reference = np.random.RandomState(2)
    reference.random_sample(2)
    assert rng.random_sample() == reference.random_sample()


def test_pair_child_is_saved_without_drawing_descendants():
    rng = np.random.RandomState(2)
    assert _random_intersection_trees(
        [(0, 1, 2), (1, 2, 3)], [1, 1], 1, 100, 1, random_state=rng
    ) == {(1, 2)}
    reference = np.random.RandomState(2)
    reference.random_sample(3)
    assert rng.random_sample() == reference.random_sample()


def test_siblings_intersect_with_their_parent_independently():
    # Seed 2 makes root A intersect children B and A. Saving the first pair
    # must not shrink the parent used for the second child.
    result = _random_intersection_trees(
        [(0, 1, 2), (1, 2, 3)], [1, 1], 1, 3, 2, random_state=2
    )
    assert result == {(1, 2), (0, 1, 2)}


@pytest.mark.parametrize("short_path", [(), (0,)])
def test_sampled_short_paths_are_not_removed_from_the_population(short_path):
    # Seed 1 samples A then the short path, so the root is discarded.
    assert _random_intersection_trees(
        [(0, 1, 2), short_path], [1, 1], 1, 2, 1, random_state=1
    ) == set()
    # Seed 2 samples A twice, then the short path kills the child branch.
    assert _random_intersection_trees(
        [(0, 1, 2), short_path], [1, 1], 1, 3, 1, random_state=2
    ) == set()


def test_empty_branch_does_not_remove_surviving_sibling():
    assert _random_intersection_trees(
        [(0, 1, 2), ()], [1, 1], 1, 3, 2, random_state=2
    ) == {(0, 1, 2)}


def test_sampling_uses_path_mass_for_both_root_draws():
    rng = np.random.RandomState(123)
    count = sum(
        _random_intersection_trees(
            [(0, 1), (2, 3)], [1, 3], 1, 2, 1, random_state=rng
        ) == {(0, 1)}
        for _ in range(2000)
    )
    # Both independent root draws must choose the first path: (1/4)^2.
    assert 0.045 < count / 2000 < 0.08


def test_sampling_is_with_replacement_and_zero_mass_is_never_sampled():
    # Only one path has mass, yet forming a root requires two samples of it.
    assert _random_intersection_trees(
        [(), (9,), (3, 1, 3)], [0, 0, 7], 10, 5, 2, random_state=0
    ) == {(1, 3)}


def test_discoveries_are_deduplicated_within_a_replicate():
    assert _random_intersection_trees(
        [(0, 2, 4)], [1], 50, 4, 2, random_state=1
    ) == {(0, 2, 4)}


@pytest.mark.parametrize(
    "paths, weights, n_trees",
    [([], [], 10), ([(0,)], [0], 10), ([(0,)], [1], 0)],
)
def test_empty_populations_or_no_trees(paths, weights, n_trees):
    assert _random_intersection_trees(
        paths, weights, n_trees, 3, 2, random_state=0
    ) == set()


def test_singletons_are_never_returned():
    assert _random_intersection_trees(
        [(0,)], [1], 10, 2, 2, random_state=0
    ) == set()


def test_large_finite_masses_do_not_overflow():
    assert _random_intersection_trees(
        [(0, 1), (2, 3)], [1e308, 1e308], 30, 2, 1, random_state=2
    ) == {(0, 1), (2, 3)}


def test_integer_seed_is_repeatable_and_does_not_touch_global_rng():
    state_before = np.random.get_state()
    kwargs = dict(
        paths=[(0, 1), (1, 2), (1, 2, 3)],
        path_weights=[1, 2, 3],
        n_trees=20,
        max_depth=3,
        n_children=2,
        random_state=14,
    )
    first = _random_intersection_trees(**kwargs)
    assert first == _random_intersection_trees(**kwargs)
    state_after = np.random.get_state()
    assert state_before[0] == state_after[0]
    np.testing.assert_array_equal(state_before[1], state_after[1])
    assert state_before[2:] == state_after[2:]


@pytest.mark.parametrize("weights", [[-1], [np.nan], [np.inf], [[1]], []])
def test_invalid_path_masses_are_rejected(weights):
    with pytest.raises(ValueError, match="path_weights"):
        _random_intersection_trees([(0,)], weights, 1, 2, 2, random_state=0)


@pytest.mark.parametrize(
    "parameter, value",
    [
        ("n_trees", -1),
        ("n_trees", 1.5),
        ("max_depth", -1),
        ("max_depth", 0),
        ("max_depth", 1),
        ("max_depth", True),
        ("n_children", 0),
        ("n_children", 1.5),
    ],
)
def test_invalid_tree_parameters_are_rejected(parameter, value):
    kwargs = dict(n_trees=1, max_depth=2, n_children=2)
    kwargs[parameter] = value
    with pytest.raises(ValueError, match=parameter):
        _random_intersection_trees([(0,)], [1], random_state=0, **kwargs)
