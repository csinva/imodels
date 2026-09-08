"""Sample regularization paths built from the APA-APG2 point solvers.

The functions in this module use continuation: requested positive
regularization strengths are deduplicated, sorted from largest to smallest,
and each solution initializes the next point solve.  The returned paths are
sampled numerical paths, not exact breakpoint paths.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any

import numpy as np
from numpy.linalg import norm
from scipy.optimize import minimize

from ._quadratic import _QuadraticRegressionLoss, _make_quadratic_regression_loss
from .apa_point import (
    _hiCAP_regression_quadratic,
    hiCAP_classification,
    hiCAP_regression,
)
from ._problem import (
    _diagonal_weighted_least_squares,
    _group_penalty,
    _normalize_ord,
    _prepare_problem,
    _validate_lambdas,
    _weighted_least_squares,
)
from ._result import RegularizationPath


PointSolver = Callable[..., tuple[np.ndarray, Mapping[str, Any]]]


def _weighted_logistic_fit(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    beta_start: np.ndarray,
    max_iter: int,
    tol: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Numerically solve the unpenalized logistic endpoint at lambda zero."""
    weight_sum = float(weights.sum())
    normalized_weights = weights / weight_sum

    def objective_and_gradient(beta: np.ndarray) -> tuple[float, np.ndarray]:
        linear_predictor = X @ beta
        loss = float(
            np.dot(
                normalized_weights,
                np.logaddexp(0.0, linear_predictor) - y * linear_predictor,
            )
        )
        # Written without exp(linear_predictor), so extreme values remain
        # finite and agree with the point solver's stable logistic gradient.
        probabilities = np.empty_like(linear_predictor)
        nonnegative = linear_predictor >= 0
        probabilities[nonnegative] = 1.0 / (
            1.0 + np.exp(-linear_predictor[nonnegative])
        )
        exp_values = np.exp(linear_predictor[~nonnegative])
        probabilities[~nonnegative] = exp_values / (1.0 + exp_values)
        gradient = X.T @ (normalized_weights * (probabilities - y))
        return loss, gradient

    result = minimize(
        objective_and_gradient,
        beta_start,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": int(max_iter), "ftol": float(tol), "gtol": float(tol)},
    )
    beta = np.asarray(result.x, dtype=float)
    loss = float(result.fun)
    if beta.shape != beta_start.shape or not np.all(np.isfinite(beta)):
        raise FloatingPointError(
            "unpenalized logistic optimization produced non-finite coefficients"
        )
    if not np.isfinite(loss):
        raise FloatingPointError(
            "unpenalized logistic optimization produced a non-finite loss"
        )

    step_norm = float(norm(beta - beta_start))
    weighted_X = X * np.sqrt(normalized_weights)[:, np.newaxis]
    info: dict[str, Any] = {
        "converged": bool(result.success),
        "n_iter": int(result.nit),
        "step_norm": step_norm,
        "relative_step_norm": step_norm / max(1.0, float(norm(beta_start))),
        "approximation_parameter": 0.0,
        "objective": loss,
        "loss": loss,
        "penalty": 0.0,
        "lipschitz_constant": float(norm(weighted_X, 2) ** 2 / 4.0),
        "weight_sum": weight_sum,
        "optimizer_status": int(result.status),
        "message": str(result.message),
        "solver": "weighted_logistic_lbfgs",
    }
    return beta, info


def _sampled_path(
    *,
    X: np.ndarray,
    y: np.ndarray,
    groups: Iterable[np.ndarray],
    lambdas: Iterable[float],
    beta_init: np.ndarray | None,
    gamma1: float,
    a: float,
    max_iter: int,
    tol: float,
    ord: int | str,
    verbose: bool,
    sample_weight: np.ndarray | None,
    point_solver: PointSolver | None,
    problem: str,
    cache_quadratic: bool = False,
    assume_diagonal_gram: bool = False,
) -> RegularizationPath:
    requested_lambdas, path_lambdas = _validate_lambdas(lambdas)
    X, y, private_groups, warm_beta, weights, ord = _prepare_problem(
        X=X,
        y=y,
        groups=groups,
        beta_init=beta_init,
        sample_weight=sample_weight,
        ord=ord,
    )
    if isinstance(max_iter, (bool, np.bool_)) or not isinstance(
        max_iter, (int, np.integer)
    ):
        raise ValueError("max_iter must be a positive integer")
    if max_iter <= 0:
        raise ValueError("max_iter must be a positive integer")
    for value, name in ((gamma1, "gamma1"), (a, "a"), (tol, "tol")):
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, float, np.number))
            or not np.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"{name} must be a positive finite scalar")
    if problem == "classification" and np.any((y < 0) | (y > 1)):
        raise ValueError("classification y must contain values in [0, 1]")
    if not isinstance(cache_quadratic, (bool, np.bool_)):
        raise ValueError("cache_quadratic must be a boolean")
    if not isinstance(assume_diagonal_gram, (bool, np.bool_)):
        raise ValueError("assume_diagonal_gram must be a boolean")
    if assume_diagonal_gram and (
        problem != "regression" or not cache_quadratic or point_solver is not None
    ):
        raise ValueError(
            "assume_diagonal_gram=True requires cache_quadratic=True and the "
            "default regression point solver"
        )

    quadratic_loss = None
    if problem == "regression" and point_solver is None and cache_quadratic:
        quadratic_loss = _make_quadratic_regression_loss(
            X,
            y,
            weights,
            assume_diagonal_gram=assume_diagonal_gram,
        )
    if point_solver is None:
        point_solver = (
            hiCAP_regression
            if problem == "regression"
            else hiCAP_classification
        )

    coefficients: list[np.ndarray] = []
    penalties: list[float] = []
    diagnostics: list[Mapping[str, Any]] = []
    previous_lambda: float | None = None
    all_points_converged = True

    for lam in path_lambdas:
        lam = float(lam)
        warm_start_norm = float(norm(warm_beta))
        if lam > 0:
            if quadratic_loss is None:
                result = point_solver(
                    X=X,
                    y=y,
                    groups=private_groups,
                    lam=lam,
                    beta_init=warm_beta,
                    gamma1=gamma1,
                    a=a,
                    max_iter=max_iter,
                    tol=tol,
                    ord=ord,
                    verbose=verbose,
                    sample_weight=weights,
                    return_info=True,
                )
            else:
                result = _hiCAP_regression_quadratic(
                    quadratic_loss,
                    private_groups,
                    lam,
                    warm_beta,
                    float(gamma1),
                    float(a),
                    int(max_iter),
                    float(tol),
                    ord,
                    verbose,
                )
            if not isinstance(result, tuple) or len(result) != 2:
                raise TypeError(
                    "point_solver must return (coefficients, diagnostics) when "
                    "return_info=True"
                )
            beta, raw_info = result
            beta = np.asarray(beta, dtype=float)
            if beta.shape != warm_beta.shape or not np.all(np.isfinite(beta)):
                raise FloatingPointError(
                    "APA-APG2 point solver returned invalid coefficients"
                )
            info = dict(raw_info)
            if quadratic_loss is not None:
                normalized_weights = weights / float(weights.sum())
                residual = X @ beta - y
                direct_loss = 0.5 * float(
                    normalized_weights @ (residual**2)
                )
                info["loss"] = direct_loss
                info["objective"] = direct_loss + float(info["penalty"])
            all_points_converged = all_points_converged and bool(
                info.get("converged", False)
            )
        elif problem == "regression":
            if quadratic_loss is not None and quadratic_loss.diagonal is not None:
                beta, info = _diagonal_weighted_least_squares(
                    quadratic_loss, warm_beta
                )
            else:
                beta, info = _weighted_least_squares(
                    X, y, weights, warm_beta
                )
        else:
            beta, info = _weighted_logistic_fit(
                X, y, weights, warm_beta, max_iter=max_iter, tol=tol
            )
            all_points_converged = all_points_converged and bool(
                info["converged"]
            )

        info.update(
            {
                "lambda": lam,
                "warm_start_norm": warm_start_norm,
                "warm_started_from_lambda": previous_lambda,
                "unscaled_penalty": _group_penalty(beta, private_groups, ord),
            }
        )
        warm_beta = np.asarray(beta, dtype=float).copy()
        coefficients.append(warm_beta)
        penalties.append(float(info["unscaled_penalty"]))
        diagnostics.append(info)
        previous_lambda = lam

    metadata = {
        "problem": problem,
        "ord": ord,
        "warm_started": True,
        "lambda_order": "descending_unique",
        "requested_lambdas": tuple(float(value) for value in requested_lambdas),
        "duplicates_removed": int(requested_lambdas.size - path_lambdas.size),
        "zero_lambda_solver": (
            (
                "cached_diagonal_lstsq"
                if quadratic_loss is not None
                and quadratic_loss.diagonal is not None
                else "weighted_lstsq"
            )
            if problem == "regression"
            else "weighted_logistic_lbfgs"
        ),
        "cached_sufficient_statistics": quadratic_loss is not None,
        "quadratic_backend": (
            None if quadratic_loss is None else quadratic_loss.backend
        ),
        "assumed_diagonal_gram": (
            False
            if quadratic_loss is None
            else quadratic_loss.assumed_diagonal_gram
        ),
        "gram_max_off_diagonal": (
            None
            if quadratic_loss is None
            else quadratic_loss.max_off_diagonal
        ),
        "gram_max_off_diagonal_correlation": (
            None
            if quadratic_loss is None
            else quadratic_loss.max_off_diagonal_correlation
        ),
    }
    return RegularizationPath(
        lambdas=path_lambdas,
        coefficients=np.vstack(coefficients),
        penalties=np.asarray(penalties),
        diagnostics=tuple(diagnostics),
        method="apa-apg2-warm-start",
        exact=False,
        status="complete" if all_points_converged else "partial",
        metadata=metadata,
    )


def apa_apg_regression_path(
    X: np.ndarray,
    y: np.ndarray,
    groups: Iterable[np.ndarray],
    lambdas: Iterable[float],
    beta_init: np.ndarray | None = None,
    gamma1: float = 1.0,
    a: float = 1.0,
    max_iter: int = 2000,
    tol: float = 1e-6,
    ord: int | str = 2,
    verbose: bool = False,
    sample_weight: np.ndarray | None = None,
    *,
    cache_quadratic: bool = True,
    assume_diagonal_gram: bool = False,
    point_solver: PointSolver | None = None,
) -> RegularizationPath:
    """Compute a descending, warm-started sampled regression path.

    Positive path points are solved by :func:`hiCAP_regression`.  If zero is
    requested, its coefficient vector is a weighted least-squares solution;
    a cached diagonal problem solves that endpoint coordinate-wise without a
    design factorization. No zero value is passed to the APA-APG2 point
    solver. The returned status is ``"partial"`` if any positive point does
    not meet the point solver's stopping test.

    With ``cache_quadratic=True``, the default solver computes weighted
    quadratic sufficient statistics once for the entire path and detects a
    diagonal Gram matrix. If the caller already knows that ``X.T @ W @ X`` is
    diagonal, ``assume_diagonal_gram=True`` avoids materializing the full Gram
    matrix and stores only its diagonal. This assumption is valid only for the
    exact rows and weights that establish the orthogonality. ``point_solver``
    is an advanced injection point used for benchmarking and testing solvers
    with the same call contract as ``hiCAP_regression``; an injected solver
    bypasses the quadratic cache.
    """
    return _sampled_path(
        X=X,
        y=y,
        groups=groups,
        lambdas=lambdas,
        beta_init=beta_init,
        gamma1=gamma1,
        a=a,
        max_iter=max_iter,
        tol=tol,
        ord=ord,
        verbose=verbose,
        sample_weight=sample_weight,
        point_solver=point_solver,
        problem="regression",
        cache_quadratic=cache_quadratic,
        assume_diagonal_gram=assume_diagonal_gram,
    )


def apa_apg_classification_path(
    X: np.ndarray,
    y: np.ndarray,
    groups: Iterable[np.ndarray],
    lambdas: Iterable[float],
    beta_init: np.ndarray | None = None,
    gamma1: float = 1.0,
    a: float = 1.0,
    max_iter: int = 2000,
    tol: float = 1e-6,
    ord: int | str = 2,
    verbose: bool = False,
    sample_weight: np.ndarray | None = None,
    *,
    point_solver: PointSolver | None = None,
) -> RegularizationPath:
    """Compute a descending, warm-started sampled logistic path.

    Positive path points are solved by :func:`hiCAP_classification`.  If zero
    is requested, weighted unpenalized logistic loss is minimized with
    L-BFGS. A nonconverged positive point or zero endpoint is retained with
    ``status='partial'`` and its optimizer diagnostics, which is important for
    separable data where a finite unpenalized logistic minimizer need not
    exist.

    ``point_solver`` is an advanced injection point used for benchmarking and
    testing solvers with the same call contract as ``hiCAP_classification``.
    """
    return _sampled_path(
        X=X,
        y=y,
        groups=groups,
        lambdas=lambdas,
        beta_init=beta_init,
        gamma1=gamma1,
        a=a,
        max_iter=max_iter,
        tol=tol,
        ord=ord,
        verbose=verbose,
        sample_weight=sample_weight,
        point_solver=point_solver,
        problem="classification",
    )


__all__ = ["apa_apg_regression_path", "apa_apg_classification_path"]
