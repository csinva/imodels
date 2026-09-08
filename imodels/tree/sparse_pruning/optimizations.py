"""Compatibility imports and fitting helpers for sparse-pruning estimators.

Numerical solvers live in the optimization package; these imports preserve
historical point-solver entry points without maintaining duplicate code.
"""
from __future__ import annotations

import warnings

import numpy as np

from .optimization.apa_point import hiCAP_classification, hiCAP_regression
from .optimization.tree_prox import proj_l1_ball

__all__ = [
    "hiCAP_classification", "hiCAP_regression", "proj_l1_ball",
    "get_gcv_reg_param", "get_reg_set",
]


def get_gcv_reg_param(
    tree, X=None, y=None, *, sample_weight=None, return_info=False,
):
    """Choose nonnegative node-based HS by conditional fixed-tree GCV.

    Prefer passing a fitted ``DecisionTreeRegressor``; its retained node
    statistics suffice, with no design matrix. The historical raw sklearn
    ``Tree`` argument is also accepted, but requires its fitting ``X, y`` to
    verify that its values and impurities really are response means and
    squared-error variances. Supplied data are validated, never silently
    ignored. Nonuniform weights, forests, and classification are unsupported.

    Returns a scalar, or ``(scalar, diagnostics)`` with ``return_info=True``.
    Infinity denotes the root-mean limit. The score conditions on the fitted
    structure and does not account for learning/pruning it from the targets.
    The input tree is not modified.
    """
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.tree._tree import Tree
    from sklearn.utils.validation import check_array
    from imodels.tree._hs_gcv import select_hs_reg_param

    if (X is None) != (y is None):
        raise ValueError("GCV requires both X and y when fitting data are supplied")
    if isinstance(tree, Tree):
        if X is None:
            raise ValueError("GCV with a raw Tree requires its fitting X and y")
        if tree.n_outputs != 1 or np.any(tree.n_classes != 1):
            raise ValueError("GCV requires a single-output regression tree")
        estimator = DecisionTreeRegressor()
        estimator.tree_ = tree
        estimator.n_outputs_ = 1
        estimator.n_features_in_ = tree.n_features
    else:
        estimator = tree

    selected, info = select_hs_reg_param(
        estimator, sample_weight=sample_weight, y=y
    )
    if X is not None:
        X = check_array(X, dtype=np.float32, accept_sparse="csr")
        if np.iscomplexobj(y):
            raise ValueError("GCV y must contain finite real values")
        y = np.asarray(y, dtype=float)
        fitted_tree = estimator.tree_
        if (
            y.ndim != 1 or y.size != X.shape[0]
            or not np.all(np.isfinite(y))
            or X.shape[1] != estimator.n_features_in_
            or X.shape[0] != info["n_samples"]
        ):
            raise ValueError("GCV X and y must match the tree's fitting observations")

        # A sparse routing matrix avoids the historical n-by-node dense
        # allocation. Unreachable backing-array nodes have zero column counts.
        path = fitted_tree.decision_path(X)
        counts = np.asarray(path.sum(axis=0)).ravel()
        retained = counts > 0
        if not np.array_equal(
            counts[retained], fitted_tree.n_node_samples[retained]
        ):
            raise ValueError("GCV X does not match the tree's fitting node counts")
        centered = y - fitted_tree.value[0, 0, 0]
        means = np.asarray(path.T @ centered).ravel()[retained] / counts[retained]
        variances = (
            np.asarray(path.T @ (centered * centered)).ravel()[retained]
            / counts[retained] - means * means
        )
        stored_means = (
            fitted_tree.value[retained, 0, 0] - fitted_tree.value[0, 0, 0]
        )
        scale = max(1.0, float(np.max(np.abs(centered))))
        if not (
            np.allclose(means, stored_means, rtol=1e-8, atol=1e-10 * scale)
            and np.allclose(
                variances, fitted_tree.impurity[retained],
                rtol=1e-7, atol=1e-10 * scale * scale,
            )
        ):
            raise ValueError(
                "GCV requires unchanged fitting means and squared-error "
                "impurities; X and y do not match this tree"
            )
    return (selected, info) if return_info else selected


def get_reg_set(
    prune_set: str,
    X: np.ndarray,
    y: np.ndarray,
    random_state,
    n_samples: int,
    n_samples_bootstrap: int,
):
    """Deprecated compatibility helper returning a forest pruning subset."""
    warnings.warn(
        "get_reg_set is deprecated; sparse-pruning wrappers now reconstruct "
        "per-tree bootstrap subsets internally.",
        DeprecationWarning,
        stacklevel=2,
    )
    y = np.asarray(y)
    if X.shape[0] != n_samples or len(y) != n_samples:
        raise ValueError("n_samples must match the number of rows in X and y")
    if prune_set == "full":
        return X, y
    if prune_set not in {"ib", "oob"}:
        raise ValueError("prune_set must be one of {'ib', 'oob', 'full'}")

    from sklearn.ensemble._forest import (
        _generate_sample_indices,
        _generate_unsampled_indices,
    )

    if prune_set == "ib":
        indices = _generate_sample_indices(
            random_state, n_samples, n_samples_bootstrap
        )
    else:
        indices = _generate_unsampled_indices(
            random_state, n_samples, n_samples_bootstrap
        )
    return X[indices], y[indices]
