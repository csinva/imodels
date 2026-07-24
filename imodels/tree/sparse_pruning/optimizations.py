"""Optimization utilities for SparseForestry models."""
from __future__ import annotations

import warnings

import numpy as np
from numpy.linalg import norm
from scipy.special import expit


def proj_l1_ball(u: np.ndarray, tau: float) -> np.ndarray:
    if tau <= 0:
        return np.zeros_like(u)
    if np.sum(np.abs(u)) <= tau:
        return u.copy()

    abs_u = np.abs(u)
    s = -np.sort(-abs_u)
    css = np.cumsum(s)
    js = np.arange(1, len(s) + 1)
    cond = s - (css - tau) / js
    rho = np.nonzero(cond > 0)[0][-1]
    theta = (css[rho] - tau) / (rho + 1)
    return np.sign(u) * np.maximum(abs_u - theta, 0.0)


def _validate_solver_inputs(
    X: np.ndarray,
    y: np.ndarray,
    groups: list[np.ndarray],
    lam: float,
    beta_init: np.ndarray | None,
    sample_weight: np.ndarray | None,
    gamma1: float,
    a: float,
    max_iter: int,
    tol: float,
    ord: int | str,
) -> tuple[
    np.ndarray,
    np.ndarray,
    list[np.ndarray],
    float,
    np.ndarray,
    np.ndarray,
    int | str,
]:
    """Validate and normalize common APA-APG2 solver inputs."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"X must be two-dimensional; got shape {X.shape}")
    if y.ndim != 1:
        raise ValueError(f"y must be one-dimensional; got shape {y.shape}")
    n, d = X.shape
    if n == 0 or d == 0:
        raise ValueError("X must contain at least one sample and one feature")
    if y.shape[0] != n:
        raise ValueError(
            f"X and y have inconsistent sample counts: {n} != {y.shape[0]}"
        )
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
        raise ValueError("X and y must contain only finite values")

    if (
        isinstance(lam, (bool, np.bool_))
        or not isinstance(lam, (int, float, np.number))
    ):
        raise ValueError("lam must be a positive finite scalar")
    lam = float(lam)
    if not np.isfinite(lam) or lam <= 0:
        raise ValueError("lam must be a positive finite scalar")

    if isinstance(ord, (bool, np.bool_)) or not np.isscalar(ord):
        raise ValueError("ord must be 2, 'inf', or np.inf")
    if isinstance(ord, str):
        if ord != "inf":
            raise ValueError("ord must be 2, 'inf', or np.inf")
        ord = "inf"
    elif ord == np.inf:
        ord = "inf"
    elif ord == 2:
        ord = 2
    else:
        raise ValueError("ord must be 2, 'inf', or np.inf")

    try:
        raw_groups = list(groups)
    except TypeError as exc:
        raise ValueError("groups must be a non-empty iterable of index arrays") from exc
    if not raw_groups:
        raise ValueError("groups must contain at least one group")

    validated_groups: list[np.ndarray] = []
    for group_idx, group in enumerate(raw_groups):
        group_array = np.asarray(group)
        if group_array.ndim != 1 or group_array.size == 0:
            raise ValueError(f"group {group_idx} must be a non-empty 1D array")
        if group_array.dtype.kind not in "iu":
            raise ValueError(f"group {group_idx} must contain integer indices")
        group_array = group_array.astype(np.intp, copy=False)
        if np.unique(group_array).size != group_array.size:
            raise ValueError(f"group {group_idx} contains duplicate indices")
        if np.any(group_array < 0) or np.any(group_array >= d):
            raise ValueError(
                f"group {group_idx} contains an index outside [0, {d})"
            )
        validated_groups.append(group_array)

    if beta_init is None:
        beta = np.zeros(d, dtype=float)
    else:
        beta = np.asarray(beta_init, dtype=float)
        if beta.shape != (d,):
            raise ValueError(
                f"beta_init must have shape ({d},); got shape {beta.shape}"
            )
        if not np.all(np.isfinite(beta)):
            raise ValueError("beta_init must contain only finite values")
        beta = beta.copy()

    if sample_weight is None:
        weights = np.ones(n, dtype=float)
    else:
        weights = np.asarray(sample_weight, dtype=float)
        if weights.shape != (n,):
            raise ValueError(
                f"sample_weight must have shape ({n},); got shape {weights.shape}"
            )
        if not np.all(np.isfinite(weights)):
            raise ValueError("sample_weight must contain only finite values")
        if np.any(weights < 0):
            raise ValueError("sample_weight cannot contain negative values")
        with np.errstate(over="ignore", invalid="ignore"):
            weight_sum = float(weights.sum())
        if not np.isfinite(weight_sum) or weight_sum <= 0:
            raise ValueError(
                "sample_weight must have a positive finite total weight"
            )

    for value, name in ((gamma1, "gamma1"), (a, "a"), (tol, "tol")):
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, float, np.number))
        ):
            raise ValueError(f"{name} must be a positive finite scalar")
        numeric_value = float(value)
        if not np.isfinite(numeric_value) or numeric_value <= 0:
            raise ValueError(f"{name} must be a positive finite scalar")
    if isinstance(max_iter, (bool, np.bool_)) or not isinstance(
        max_iter, (int, np.integer)
    ):
        raise ValueError("max_iter must be a positive integer")
    if max_iter <= 0:
        raise ValueError("max_iter must be a positive integer")

    return X, y, validated_groups, lam, beta, weights, ord


def _proximal_average(
    v: np.ndarray,
    groups: list[np.ndarray],
    gamma: float,
    ord: int | str,
) -> np.ndarray:
    """Evaluate the full-vector proximal average for equally weighted groups."""
    alpha = 1.0 / len(groups)
    # Every component prox is the identity off its group. Starting from v and
    # accumulating only the within-group changes is algebraically equivalent
    # to averaging full vectors, without allocating one dense vector per group.
    beta_next = v.copy()
    for group in groups:
        v_group = v[group]
        if ord == 2:
            group_norm = norm(v_group)
            if group_norm == 0:
                prox_group = np.zeros_like(v_group)
            else:
                prox_group = max(1.0 - gamma / group_norm, 0.0) * v_group
        else:
            prox_group = v_group - proj_l1_ball(v_group, gamma)
        beta_next[group] += alpha * (prox_group - v_group)
    return beta_next


def _solver_info(
    converged: bool,
    n_iter: int,
    step_norm: float,
    relative_step_norm: float,
    approximation_parameter: float,
    objective: float,
    loss: float,
    penalty: float,
    lipschitz_constant: float,
    weight_sum: float,
) -> dict[str, bool | int | float]:
    return {
        "converged": converged,
        "n_iter": n_iter,
        "step_norm": step_norm,
        "relative_step_norm": relative_step_norm,
        "approximation_parameter": approximation_parameter,
        "objective": objective,
        "loss": loss,
        "penalty": penalty,
        "lipschitz_constant": lipschitz_constant,
        "weight_sum": weight_sum,
    }


def _group_penalty(
    beta: np.ndarray, groups: list[np.ndarray], ord: int | str
) -> float:
    if ord == 2:
        return float(sum(norm(beta[group]) for group in groups))
    return float(sum(np.max(np.abs(beta[group])) for group in groups))


def hiCAP_regression(
    X: np.ndarray,
    y: np.ndarray,
    groups: list[np.ndarray],
    lam: float,
    beta_init: np.ndarray | None = None,
    gamma1: float = 1.0,
    a: float = 1.0,
    max_iter: int = 2000,
    tol: float = 1e-6,
    ord: int | str = 2,
    verbose: bool = False,
    sample_weight: np.ndarray | None = None,
    return_info: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, bool | int | float]]:
    """Solve the overlapping-group regression objective with APA-APG2.

    The normalized weighted objective is

    ``0.5 * sum_i w_i (y_i - x_i @ beta)^2 / sum_i w_i
    + lam * sum_G ||beta_G||_ord``.

    Sample weights are therefore relative weights: uniformly rescaling them
    does not change the solution. Internally, both terms are divided by
    ``lam * len(groups)`` so the nonsmooth term is an equally weighted
    proximal average. This is why each component proximal map uses threshold
    ``gamma`` while the resulting full vectors are averaged with weight
    ``1 / len(groups)``.
    """
    X, y, groups, lam, beta, weights, ord = _validate_solver_inputs(
        X=X,
        y=y,
        groups=groups,
        lam=lam,
        beta_init=beta_init,
        sample_weight=sample_weight,
        gamma1=gamma1,
        a=a,
        max_iter=max_iter,
        tol=tol,
        ord=ord,
    )
    K = len(groups)
    weight_sum = float(weights.sum())
    normalized_weights = weights / weight_sum
    weighted_X = X * np.sqrt(normalized_weights)[:, np.newaxis]
    sigma_max = norm(weighted_X, 2)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        Lf = (sigma_max**2) / (lam * K)
    if not np.isfinite(Lf):
        raise FloatingPointError(
            "The weighted design scale produced a non-finite Lipschitz "
            "constant"
        )
    max_step = np.inf if Lf == 0 else 1.0 / Lf
    beta_tilde = beta.copy()
    converged = False
    step_norm = np.inf
    relative_step_norm = np.inf
    for k in range(max_iter):
        tau = 1.0 / (k + a)
        gamma = min(gamma1 * a / (k + a), max_step)
        beta_hat = (1 - tau) * beta + tau * beta_tilde
        residual = X.dot(beta_hat) - y
        grad = (X.T @ (normalized_weights * residual)) / (lam * K)
        v = beta_hat - gamma * grad
        beta_next = _proximal_average(v, groups, gamma, ord)
        if not np.all(np.isfinite(beta_next)):
            raise FloatingPointError("APA-APG2 produced non-finite coefficients")
        beta_tilde += (1.0 / tau) * (2 - gamma * Lf) * (beta_next - beta_hat)

        step_norm = float(norm(beta_next - beta))
        relative_step_norm = step_norm / max(1.0, float(norm(beta)))
        # With more than one component, a small step can merely mean that the
        # iterate has reached the fixed point of the current proximal-average
        # surrogate. APA-APG still has to continue while gamma decreases toward
        # zero in order to approach the original composite objective. For one
        # group the proximal average is exact, so ordinary early stopping is
        # valid.
        if K == 1 and relative_step_norm <= tol:
            if verbose:
                print(f"Converged after {k + 1} iterations")
            beta = beta_next
            converged = True
            break

        beta = beta_next

    residual = X.dot(beta) - y
    loss = 0.5 * float(np.dot(normalized_weights, residual**2))
    penalty = lam * _group_penalty(beta, groups, ord)
    info = _solver_info(
        converged=converged,
        n_iter=k + 1,
        step_norm=step_norm,
        relative_step_norm=relative_step_norm,
        approximation_parameter=float(gamma),
        objective=loss + penalty,
        loss=loss,
        penalty=penalty,
        lipschitz_constant=float(Lf),
        weight_sum=weight_sum,
    )
    return (beta, info) if return_info else beta


def hiCAP_classification(
    X: np.ndarray,
    y: np.ndarray,
    groups: list[np.ndarray],
    lam: float,
    beta_init: np.ndarray | None = None,
    gamma1: float = 1.0,
    a: float = 1.0,
    max_iter: int = 2000,
    tol: float = 1e-6,
    ord: int | str = 2,
    verbose: bool = False,
    sample_weight: np.ndarray | None = None,
    return_info: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, bool | int | float]]:
    """Solve the overlapping-group logistic objective with APA-APG2.

    The objective is mean weighted binary logistic loss plus
    ``lam * sum_G ||beta_G||_ord``, where the loss is divided by the sum of
    weights. Targets may be any finite values in ``[0, 1]``; sparse-pruning
    classifiers pass binary zero/one targets.
    """
    X, y, groups, lam, beta, weights, ord = _validate_solver_inputs(
        X=X,
        y=y,
        groups=groups,
        lam=lam,
        beta_init=beta_init,
        sample_weight=sample_weight,
        gamma1=gamma1,
        a=a,
        max_iter=max_iter,
        tol=tol,
        ord=ord,
    )
    if np.any((y < 0) | (y > 1)):
        raise ValueError("classification y must contain values in [0, 1]")
    K = len(groups)
    weight_sum = float(weights.sum())
    normalized_weights = weights / weight_sum
    weighted_X = X * np.sqrt(normalized_weights)[:, np.newaxis]
    sigma_max = norm(weighted_X, 2)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        Lf = (sigma_max**2) / (4 * lam * K)
    if not np.isfinite(Lf):
        raise FloatingPointError(
            "The weighted design scale produced a non-finite Lipschitz "
            "constant"
        )
    max_step = np.inf if Lf == 0 else 1.0 / Lf
    beta_tilde = beta.copy()
    converged = False
    step_norm = np.inf
    relative_step_norm = np.inf
    for k in range(max_iter):
        tau = 1.0 / (k + a)
        gamma = min(gamma1 * a / (k + a), max_step)
        beta_hat = (1 - tau) * beta + tau * beta_tilde

        probabilities = expit(X.dot(beta_hat))
        grad = (
            X.T @ (normalized_weights * (probabilities - y))
        ) / (lam * K)
        v = beta_hat - gamma * grad
        beta_next = _proximal_average(v, groups, gamma, ord)
        if not np.all(np.isfinite(beta_next)):
            raise FloatingPointError("APA-APG2 produced non-finite coefficients")
        beta_tilde += (1.0 / tau) * (2 - gamma * Lf) * (beta_next - beta_hat)

        step_norm = float(norm(beta_next - beta))
        relative_step_norm = step_norm / max(1.0, float(norm(beta)))
        if K == 1 and relative_step_norm <= tol:
            if verbose:
                print(f"Converged after {k + 1} iterations")
            beta = beta_next
            converged = True
            break

        beta = beta_next

    linear_predictor = X.dot(beta)
    loss = float(
        np.dot(
            normalized_weights,
            np.logaddexp(0.0, linear_predictor) - y * linear_predictor,
        )
    )
    penalty = lam * _group_penalty(beta, groups, ord)
    info = _solver_info(
        converged=converged,
        n_iter=k + 1,
        step_norm=step_norm,
        relative_step_norm=relative_step_norm,
        approximation_parameter=float(gamma),
        objective=loss + penalty,
        loss=loss,
        penalty=penalty,
        lipschitz_constant=float(Lf),
        weight_sum=weight_sum,
    )
    return (beta, info) if return_info else beta


def get_gcv_reg_param(tree, X: np.ndarray, y: np.ndarray) -> float:
    """Deprecated compatibility shim for the disabled automatic-GCV path."""
    warnings.warn(
        "get_gcv_reg_param is deprecated. Automatic GCV shrinkage is disabled "
        "because it has not been validated for sparse-pruned trees; choose "
        "reg_param explicitly.",
        DeprecationWarning,
        stacklevel=2,
    )
    raise NotImplementedError(
        "Automatic GCV shrinkage is disabled for sparse-pruned trees"
    )


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
