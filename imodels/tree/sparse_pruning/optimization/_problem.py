"""Shared input validation and unpenalized endpoints for path solvers."""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
from numpy.linalg import norm

from ._quadratic import _QuadraticRegressionLoss


def _validate_lambdas(lambdas: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
    """Return the requested and descending-unique regularization strengths."""
    try:
        raw_lambdas = (
            np.asarray(lambdas)
            if isinstance(lambdas, np.ndarray)
            else np.asarray(list(lambdas))
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "lambdas must be a non-empty one-dimensional array of finite "
            "nonnegative values"
        ) from exc
    if raw_lambdas.ndim != 1 or raw_lambdas.size == 0:
        raise ValueError("lambdas must be a non-empty one-dimensional array")
    if raw_lambdas.dtype.kind not in "iuf":
        raise ValueError("lambdas must contain real numeric values, not booleans")

    requested = raw_lambdas.astype(float, copy=True)
    if not np.all(np.isfinite(requested)) or np.any(requested < 0):
        raise ValueError("lambdas must be finite and nonnegative")
    descending_unique = np.unique(requested)[::-1]
    return requested, descending_unique


def _normalize_ord(ord: int | str) -> int | str:
    if isinstance(ord, (bool, np.bool_)) or not np.isscalar(ord):
        raise ValueError("ord must be 2, 'inf', or np.inf")
    if isinstance(ord, str):
        if ord != "inf":
            raise ValueError("ord must be 2, 'inf', or np.inf")
        return "inf"
    if ord == np.inf:
        return "inf"
    if ord == 2:
        return 2
    raise ValueError("ord must be 2, 'inf', or np.inf")


def _validate_data(
    X: np.ndarray, y: np.ndarray, beta_init: np.ndarray | None,
    sample_weight: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate shared real-valued arrays before any numeric conversion."""
    for values, name in (
        (X, "X"), (y, "y"), (beta_init, "beta_init"),
        (sample_weight, "sample_weight"),
    ):
        if np.iscomplexobj(values):
            raise ValueError(f"{name} must contain real values")
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"X must be two-dimensional; got shape {X.shape}")
    if y.ndim != 1:
        raise ValueError(f"y must be one-dimensional; got shape {y.shape}")
    n_samples, n_features = X.shape
    if n_samples == 0 or n_features == 0:
        raise ValueError("X must contain at least one sample and one feature")
    if y.shape[0] != n_samples:
        raise ValueError(
            f"X and y have inconsistent sample counts: {n_samples} != {y.shape[0]}"
        )
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
        raise ValueError("X and y must contain only finite values")
    return X, y


def _validate_groups(
    groups: Iterable[np.ndarray], n_features: int, *, copy: bool,
) -> list[np.ndarray]:
    """Validate feature-index groups, retaining each caller's copy policy."""
    try:
        raw_groups = list(groups)
    except TypeError as exc:
        raise ValueError("groups must be a non-empty iterable of index arrays") from exc
    if not raw_groups:
        raise ValueError("groups must contain at least one group")
    validated = []
    for group_index, group in enumerate(raw_groups):
        group_array = np.asarray(group)
        if group_array.ndim != 1 or group_array.size == 0:
            raise ValueError(f"group {group_index} must be a non-empty 1D array")
        if group_array.dtype.kind not in "iu":
            raise ValueError(f"group {group_index} must contain integer indices")
        group_array = group_array.astype(np.intp, copy=copy)
        if np.unique(group_array).size != group_array.size:
            raise ValueError(f"group {group_index} contains duplicate indices")
        if np.any(group_array < 0) or np.any(group_array >= n_features):
            raise ValueError(
                f"group {group_index} contains an index outside [0, {n_features})"
            )
        validated.append(group_array)
    return validated


def _initial_coefficients(
    beta_init: np.ndarray | None, n_features: int,
) -> np.ndarray:
    """Return a private, finite coefficient initialization."""
    if beta_init is None:
        return np.zeros(n_features, dtype=float)
    beta = np.asarray(beta_init, dtype=float)
    if beta.shape != (n_features,):
        raise ValueError(
            f"beta_init must have shape ({n_features},); got shape {beta.shape}"
        )
    if not np.all(np.isfinite(beta)):
        raise ValueError("beta_init must contain only finite values")
    return beta.copy()


def _validate_sample_weight(
    sample_weight: np.ndarray | None, n_samples: int, *, copy: bool,
) -> np.ndarray:
    """Validate nonnegative weights with a finite, positive total."""
    if sample_weight is None:
        weights = np.ones(n_samples, dtype=float)
    else:
        weights = np.asarray(sample_weight, dtype=float)
        if weights.shape != (n_samples,):
            raise ValueError(
                f"sample_weight must have shape ({n_samples},); got shape {weights.shape}"
            )
        if not np.all(np.isfinite(weights)):
            raise ValueError("sample_weight must contain only finite values")
        if np.any(weights < 0):
            raise ValueError("sample_weight cannot contain negative values")
        if copy:
            weights = weights.copy()
    with np.errstate(over="ignore", invalid="ignore"):
        weight_sum = float(weights.sum())
    if not np.isfinite(weight_sum) or weight_sum <= 0:
        raise ValueError("sample_weight must have a positive finite total weight")
    return weights


def _prepare_problem(
    X: np.ndarray,
    y: np.ndarray,
    groups: Iterable[np.ndarray],
    beta_init: np.ndarray | None,
    sample_weight: np.ndarray | None,
    ord: int | str,
) -> tuple[
    np.ndarray,
    np.ndarray,
    list[np.ndarray],
    np.ndarray,
    np.ndarray,
    int | str,
]:
    """Validate shared path inputs and make private copies of all groups."""
    X, y = _validate_data(X, y, beta_init, sample_weight)
    n_samples, n_features = X.shape
    private_groups = _validate_groups(groups, n_features, copy=True)
    initial_beta = _initial_coefficients(beta_init, n_features)
    weights = _validate_sample_weight(sample_weight, n_samples, copy=True)
    return X, y, private_groups, initial_beta, weights, _normalize_ord(ord)


def _group_penalty(
    beta: np.ndarray, groups: list[np.ndarray], ord: int | str
) -> float:
    if ord == 2:
        return float(sum(norm(beta[group]) for group in groups))
    return float(sum(np.max(np.abs(beta[group])) for group in groups))


def _weighted_least_squares(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    beta_start: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Solve the zero-penalty endpoint, preserving null-space warm components."""
    weight_sum = float(weights.sum())
    sqrt_weights = np.sqrt(weights / weight_sum)
    weighted_X = X * sqrt_weights[:, np.newaxis]
    weighted_residual = (y - X @ beta_start) * sqrt_weights
    delta, _, rank, singular_values = np.linalg.lstsq(
        weighted_X, weighted_residual, rcond=None
    )
    beta = beta_start + delta
    residual = X @ beta - y
    loss = 0.5 * float(np.dot(weights / weight_sum, residual**2))
    step_norm = float(norm(delta))
    info: dict[str, Any] = {
        "converged": True,
        "n_iter": 1,
        "step_norm": step_norm,
        "relative_step_norm": step_norm / max(1.0, float(norm(beta_start))),
        "approximation_parameter": 0.0,
        "objective": loss,
        "loss": loss,
        "penalty": 0.0,
        "lipschitz_constant": float(norm(weighted_X, 2) ** 2),
        "weight_sum": weight_sum,
        "rank": int(rank),
        "singular_values": tuple(float(value) for value in singular_values),
        "solver": "weighted_lstsq",
    }
    return beta, info


def _diagonal_weighted_least_squares(
    quadratic_loss: _QuadraticRegressionLoss,
    beta_start: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Solve a cached diagonal quadratic without factoring the design.

    Coordinates with a positive Gram diagonal have the unique minimizer
    ``linear[j] / diagonal[j]``. A zero diagonal identifies a null-space
    coordinate, so its warm-started value is intentionally preserved, just as
    :func:`_weighted_least_squares` preserves the design's null-space
    component.
    """
    if quadratic_loss.diagonal is None:
        raise ValueError("diagonal least squares requires a diagonal backend")

    diagonal = np.asarray(quadratic_loss.diagonal, dtype=float)
    linear = np.asarray(quadratic_loss.linear, dtype=float)
    if diagonal.shape != beta_start.shape or linear.shape != beta_start.shape:
        raise ValueError("cached quadratic dimensions do not match beta_start")
    if (
        not np.all(np.isfinite(diagonal))
        or not np.all(np.isfinite(linear))
        or np.any(diagonal < 0.0)
    ):
        raise FloatingPointError("cached diagonal quadratic is not finite PSD")

    positive = diagonal > 0.0
    null_space = ~positive
    # For genuine squared-loss sufficient statistics, d_j == 0 implies that
    # the weighted column is zero and hence h_j == 0. A violation would make
    # the diagonal quadratic unbounded rather than define a least-squares
    # null-space coordinate.
    if np.any(linear[null_space] != 0.0):
        raise FloatingPointError(
            "zero Gram diagonal has a nonzero cached linear coefficient"
        )

    beta = np.asarray(beta_start, dtype=float).copy()
    beta[positive] = linear[positive] / diagonal[positive]
    if not np.all(np.isfinite(beta)):
        raise FloatingPointError(
            "cached diagonal least squares produced non-finite coefficients"
        )

    delta = beta - beta_start
    step_norm = float(norm(delta))
    singular_values = np.sort(np.sqrt(diagonal))[::-1]
    loss = quadratic_loss.loss(beta)
    info: dict[str, Any] = {
        "converged": True,
        "n_iter": 1,
        "step_norm": step_norm,
        "relative_step_norm": step_norm
        / max(1.0, float(norm(beta_start))),
        "approximation_parameter": 0.0,
        "objective": loss,
        "loss": loss,
        "penalty": 0.0,
        "lipschitz_constant": quadratic_loss.lipschitz_constant,
        "weight_sum": quadratic_loss.weight_sum,
        "rank": int(np.count_nonzero(positive)),
        "singular_values": tuple(float(value) for value in singular_values),
        "solver": "cached_diagonal_lstsq",
        "quadratic_backend": quadratic_loss.backend,
        "assumed_diagonal_gram": quadratic_loss.assumed_diagonal_gram,
        "gram_max_off_diagonal": quadratic_loss.max_off_diagonal,
        "gram_max_off_diagonal_correlation": (
            quadratic_loss.max_off_diagonal_correlation
        ),
    }
    return beta, info
