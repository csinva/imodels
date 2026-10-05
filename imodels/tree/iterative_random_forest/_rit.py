"""Random intersection trees for unsigned iRF interactions."""

from numbers import Integral

import numpy as np
from sklearn.utils import check_random_state


def _random_intersection_trees(
    paths, path_weights, n_trees, max_depth, n_children, random_state=None
):
    """Recover interactions using the historical R iRF RIT convention.

    Follows ``RIT_basic`` in the authors' iRF source at commit
    ``fda5999b10fa878d904c0b891458ea55de467061``. ``paths`` must already be
    filtered to leaves predicting the target class. Their nonnegative
    ``path_weights`` specify sampling mass, normally the number of
    observations in each leaf.

    A root intersects two independently sampled paths. Every child intersects
    its parent's feature set with one fresh path, sampled with replacement.
    ``max_depth`` counts the total paths combined along a surviving branch.
    Pairs are saved immediately and have no descendants; larger sets are
    saved only at the final depth. Empty and singleton intersections are
    discarded. Returned sorted tuples are deduplicated, so each interaction
    contributes at most one vote per outer bootstrap replicate.

    Empty and singleton paths retain their sampling mass: drawing one can
    eliminate a branch. The sampling law matches historical R for integer
    branching, but random draws use NumPy's reproducible RNG rather than R's
    clock-seeded C++ RNG.
    """
    for name, value, minimum in (
        ("n_trees", n_trees, 0),
        ("max_depth", max_depth, 2),
        ("n_children", n_children, 1),
    ):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
            raise ValueError(f"{name} must be an integer >= {minimum}.")
        if value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}.")

    paths = [frozenset(path) for path in paths]
    weights = np.asarray(path_weights, dtype=float)
    if weights.ndim != 1 or len(weights) != len(paths):
        raise ValueError("path_weights must have one entry per path.")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("path_weights must be finite and nonnegative.")
    if not paths or n_trees == 0 or not np.any(weights > 0):
        return set()

    # Scaling first avoids overflow when summing large finite masses. Short
    # paths are deliberately retained, even though they eliminate a branch.
    weights = weights / weights.max()
    cumulative = np.cumsum(weights)
    cumulative /= cumulative[-1]
    cumulative[-1] = 1.0
    rng = check_random_state(random_state)
    interactions = set()

    for _ in range(n_trees):
        first, second = np.searchsorted(
            cumulative, rng.random_sample(2), side="right"
        )
        root = paths[first].intersection(paths[second])
        if len(root) < 2:
            continue
        if len(root) == 2 or max_depth == 2:
            interactions.add(tuple(sorted(root)))
            continue

        frontier = [root]
        for depth in range(3, max_depth + 1):
            if not frontier:
                break
            indices = np.searchsorted(
                cumulative,
                rng.random_sample(len(frontier) * n_children),
                side="right",
            ).reshape(len(frontier), n_children)
            children = []
            for parent, sampled_indices in zip(frontier, indices):
                for index in sampled_indices:
                    intersection = parent.intersection(paths[index])
                    if len(intersection) == 2 or (
                        len(intersection) > 2 and depth == max_depth
                    ):
                        interactions.add(tuple(sorted(intersection)))
                    elif len(intersection) > 2:
                        children.append(intersection)
            frontier = children

    return interactions
