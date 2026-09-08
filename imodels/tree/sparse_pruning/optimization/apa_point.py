"""APA-APG2 point solvers for hierarchical CAP penalties."""
from __future__ import annotations

import numpy as np
from numpy.linalg import norm
from scipy.special import expit

from ._quadratic import _QuadraticRegressionLoss
from ._problem import (
    _group_penalty, _initial_coefficients, _normalize_ord, _validate_data,
    _validate_groups, _validate_sample_weight,
)
from .tree_prox import proj_l1_ball


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
    X, y = _validate_data(X, y, beta_init, sample_weight)
    n, d = X.shape

    if (
        isinstance(lam, (bool, np.bool_))
        or np.iscomplexobj(lam)
        or not isinstance(lam, (int, float, np.number))
    ):
        raise ValueError("lam must be a positive finite scalar")
    lam = float(lam)
    if not np.isfinite(lam) or lam <= 0:
        raise ValueError("lam must be a positive finite scalar")

    ord = _normalize_ord(ord)
    validated_groups = _validate_groups(groups, d, copy=False)
    beta = _initial_coefficients(beta_init, d)
    weights = _validate_sample_weight(sample_weight, n, copy=False)

    for value, name in ((gamma1, "gamma1"), (a, "a"), (tol, "tol")):
        if (
            isinstance(value, (bool, np.bool_))
            or np.iscomplexobj(value)
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


def _hiCAP_regression_core(
    gradient_function,
    loss_function,
    loss_lipschitz_constant: float,
    weight_sum: float,
    groups: list[np.ndarray],
    lam: float,
    beta_init: np.ndarray,
    gamma1: float,
    a: float,
    max_iter: int,
    tol: float,
    ord: int | str,
    verbose: bool = False,
    backend_diagnostics: dict[str, bool | float | str | None] | None = None,
) -> tuple[np.ndarray, dict[str, bool | int | float | str | None]]:
    """Run the shared APA-APG2 loop for a normalized quadratic loss."""

    beta = np.asarray(beta_init, dtype=float).copy()
    K = len(groups)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        Lf = loss_lipschitz_constant / (lam * K)
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
        grad = gradient_function(beta_hat) / (lam * K)
        v = beta_hat - gamma * grad
        beta_next = _proximal_average(v, groups, gamma, ord)
        if not np.all(np.isfinite(beta_next)):
            raise FloatingPointError("APA-APG2 produced non-finite coefficients")
        beta_tilde += (1.0 / tau) * (2 - gamma * Lf) * (
            beta_next - beta_hat
        )

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

    objective_loss = float(loss_function(beta))
    penalty = lam * _group_penalty(beta, groups, ord)
    info: dict[str, bool | int | float | str | None] = _solver_info(
        converged=converged,
        n_iter=k + 1,
        step_norm=step_norm,
        relative_step_norm=relative_step_norm,
        approximation_parameter=float(gamma),
        objective=objective_loss + penalty,
        loss=objective_loss,
        penalty=penalty,
        lipschitz_constant=float(Lf),
        weight_sum=weight_sum,
    )
    if backend_diagnostics is not None:
        info.update(backend_diagnostics)
    return beta, info


def _hiCAP_regression_quadratic(
    loss: _QuadraticRegressionLoss,
    groups: list[np.ndarray],
    lam: float,
    beta_init: np.ndarray,
    gamma1: float,
    a: float,
    max_iter: int,
    tol: float,
    ord: int | str,
    verbose: bool = False,
) -> tuple[np.ndarray, dict[str, bool | int | float | str | None]]:
    """Run APA-APG2 using cached quadratic sufficient statistics."""

    return _hiCAP_regression_core(
        gradient_function=lambda beta: loss.matvec(beta) - loss.linear,
        loss_function=loss.loss,
        loss_lipschitz_constant=loss.lipschitz_constant,
        weight_sum=loss.weight_sum,
        groups=groups,
        lam=lam,
        beta_init=beta_init,
        gamma1=gamma1,
        a=a,
        max_iter=max_iter,
        tol=tol,
        ord=ord,
        verbose=verbose,
        backend_diagnostics={
            "quadratic_backend": loss.backend,
            "assumed_diagonal_gram": loss.assumed_diagonal_gram,
            "gram_max_off_diagonal": loss.max_off_diagonal,
            "gram_max_off_diagonal_correlation": (
                loss.max_off_diagonal_correlation
            ),
        },
    )


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
    weight_sum = float(weights.sum())
    normalized_weights = weights / weight_sum
    weighted_X = X * np.sqrt(normalized_weights)[:, np.newaxis]
    loss_lipschitz_constant = float(norm(weighted_X, 2) ** 2)

    def gradient_function(value):
        residual = X @ value - y
        return X.T @ (normalized_weights * residual)

    def loss_function(value):
        residual = X @ value - y
        return 0.5 * float(np.dot(normalized_weights, residual**2))

    beta, info = _hiCAP_regression_core(
        gradient_function=gradient_function,
        loss_function=loss_function,
        loss_lipschitz_constant=loss_lipschitz_constant,
        weight_sum=weight_sum,
        groups=groups,
        lam=lam,
        beta_init=beta,
        gamma1=float(gamma1),
        a=float(a),
        max_iter=int(max_iter),
        tol=float(tol),
        ord=ord,
        verbose=verbose,
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
