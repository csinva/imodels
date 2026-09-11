r"""Exact-proximal solvers for laminar group-:math:`\ell_\infty` regression.

The APA-APG2 routines in :mod:`imodels.tree.sparse_pruning.optimizations`
replace an overlapping-group proximal map by a sequence of increasingly
accurate smooth approximations.  A laminar family needs no approximation:
its Euclidean proximal map is one composition of group proximal maps, ordered
from children to parents.

There is an additional useful specialization for local tree stumps.  When the
weighted Gram matrix is diagonal, write ``D = X.T @ W @ X`` and transform
``theta = sqrt(D) * beta``.  The complete penalized regression problem is then
one coordinate-weighted laminar proximal map.  Thus every requested positive
lambda is solved by one bottom-up sweep, without an iterative optimizer or a
warm start.  For a non-diagonal Gram matrix this module uses accelerated
proximal gradient with the same exact laminar proximal map.

The path function stores one-sweep exact diagonal solutions, or general-Gram
solutions certified to a requested numerical tolerance, at the requested
lambdas.  It deliberately returns ``exact=False``: arbitrary sampled lambdas
do not enumerate every coefficient-slope breakpoint.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.linalg import norm

from ._quadratic import (
    _QuadraticRegressionLoss,
    _make_quadratic_regression_loss,
)
from ._result import RegularizationPath
from ._problem import (
    _diagonal_weighted_least_squares,
    _prepare_problem,
    _validate_lambdas,
    _weighted_least_squares,
)
from .tree_prox import (
    LaminarGroupLinfProx,
    _max_componentwise_relative_residual,
    _stable_l2_norm,
)


def _positive_integer(value: Any, name: str) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or int(value) < 1
    ):
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _positive_scalar(value: Any, name: str) -> float:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, float, np.number))
        or not np.isfinite(value)
        or float(value) <= 0.0
    ):
        raise ValueError(f"{name} must be a positive finite scalar")
    return float(value)


def _group_penalty(beta: np.ndarray, prox: LaminarGroupLinfProx) -> float:
    return float(
        sum(
            weight * np.max(np.abs(beta[group]))
            for group, weight in zip(prox.groups, prox.group_weights)
        )
    )


def _matvec(quadratic: _QuadraticRegressionLoss, beta: np.ndarray) -> np.ndarray:
    return quadratic.matvec(beta)


def _certificate_matvec(
    quadratic: _QuadraticRegressionLoss, beta: np.ndarray
) -> np.ndarray:
    """Apply the supplied full Gram when an automatic diagonal surrogate exists."""
    if quadratic.gram is not None:
        return quadratic.gram @ beta
    return quadratic.matvec(beta)


def _certificate_loss(
    quadratic: _QuadraticRegressionLoss, beta: np.ndarray
) -> float:
    """Evaluate loss against the supplied Gram rather than a truncated surrogate."""
    gram_beta = _certificate_matvec(quadratic, beta)
    value = 0.5 * float(
        beta @ gram_beta
        - 2.0 * quadratic.linear @ beta
        + quadratic.response_squared
    )
    scale = 0.5 * (
        abs(float(beta @ gram_beta))
        + 2.0 * abs(float(quadratic.linear @ beta))
        + abs(quadratic.response_squared)
    )
    if value < -128.0 * np.finfo(float).eps * max(1.0, scale):
        raise FloatingPointError("cached full-Gram loss became materially negative")
    return max(0.0, value)


def _zero_penalty_certificate(
    quadratic: _QuadraticRegressionLoss,
    beta: np.ndarray,
    tolerance: float,
) -> dict[str, float | bool]:
    gradient = _certificate_matvec(quadratic, beta) - quadratic.linear
    residual_norm, residual_scale, relative_residual = _relative_stationarity(
        gradient,
        gradient,
        np.zeros_like(beta),
        quadratic.linear,
    )
    certified = bool(relative_residual <= tolerance)
    loss = _certificate_loss(quadratic, beta)
    return {
        "converged": certified,
        "certified": certified,
        "loss": loss,
        "objective": loss,
        "stationarity_residual": residual_norm,
        "stationarity_scale": residual_scale,
        "relative_stationarity_residual": relative_residual,
        "certificate_uses_full_gram": quadratic.gram is not None,
    }


def _relative_stationarity(
    residual: np.ndarray,
    gradient: np.ndarray,
    penalty_subgradient: np.ndarray,
    linear: np.ndarray,
) -> tuple[float, float, float]:
    residual_norm = _stable_l2_norm(residual)
    scale = max(
        _stable_l2_norm(gradient),
        _stable_l2_norm(penalty_subgradient),
        _stable_l2_norm(linear),
        np.nextafter(0.0, 1.0),
    )
    if not np.isfinite(residual_norm) or not np.isfinite(scale):
        relative_residual = np.inf
    else:
        relative_residual = max(
            residual_norm / scale,
            _max_componentwise_relative_residual(
                residual,
                gradient,
                penalty_subgradient,
                linear,
            ),
        )
    return residual_norm, scale, relative_residual


def _prox_certificate_passes(
    prox_info: Mapping[str, Any], tolerance: float
) -> bool:
    """Check primal-dual gap and feasibility without masking a negative gap."""
    threshold = max(tolerance, 128.0 * np.finfo(float).eps)
    raw_relative_gap = float(
        prox_info.get("raw_relative_duality_gap", np.inf)
    )
    relative_violation = float(
        prox_info.get("max_relative_dual_l1_violation", np.inf)
    )
    relative_moreau_residual = float(
        prox_info.get("relative_moreau_residual", np.inf)
    )
    return bool(
        np.isfinite(raw_relative_gap)
        and abs(raw_relative_gap) <= threshold
        and np.isfinite(relative_violation)
        and relative_violation <= threshold
        and np.isfinite(relative_moreau_residual)
        and relative_moreau_residual <= threshold
    )


@dataclass
class _PointResult:
    beta: np.ndarray
    diagnostic: dict[str, Any]


class _LaminarQuadraticSolver:
    """Reusable exact-prox solver for one cached quadratic problem."""

    def __init__(
        self,
        quadratic: _QuadraticRegressionLoss,
        groups: Sequence[np.ndarray],
        group_weights: float | Sequence[float] | None,
        *,
        max_iter: int,
        tolerance: float,
        restart: bool,
    ) -> None:
        self.quadratic = quadratic
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.restart = restart
        self.euclidean_prox = LaminarGroupLinfProx(
            groups, quadratic.linear.size, group_weights
        )

        diagonal = quadratic.diagonal
        self.diagonal_prox: LaminarGroupLinfProx | None = None
        self.sqrt_diagonal: np.ndarray | None = None
        if diagonal is not None and np.all(np.asarray(diagonal) > 0.0):
            sqrt_diagonal = np.sqrt(np.asarray(diagonal, dtype=float))
            coordinate_weights = 1.0 / sqrt_diagonal
            self.diagonal_prox = self.euclidean_prox._with_coordinate_weights(
                coordinate_weights
            )
            self.sqrt_diagonal = sqrt_diagonal

    @property
    def uses_diagonal_closed_form(self) -> bool:
        return self.diagonal_prox is not None

    @property
    def prox_for_penalty(self) -> LaminarGroupLinfProx:
        return (
            self.diagonal_prox
            if self.diagonal_prox is not None
            else self.euclidean_prox
        )

    def penalty(self, beta: np.ndarray) -> float:
        return _group_penalty(beta, self.prox_for_penalty)

    def solve(self, lam: float, beta_init: np.ndarray) -> _PointResult:
        if lam <= 0.0:
            raise ValueError("the proximal point solver requires lambda > 0")
        if self.diagonal_prox is not None:
            return self._solve_diagonal(lam)
        return self._solve_accelerated(lam, beta_init)

    def _solve_diagonal(self, lam: float) -> _PointResult:
        diagonal = np.asarray(self.quadratic.diagonal, dtype=float)
        sqrt_diagonal = self.sqrt_diagonal
        prox = self.diagonal_prox
        if sqrt_diagonal is None or prox is None:  # pragma: no cover
            raise RuntimeError("diagonal solver was not initialized")

        transformed_center = self.quadratic.linear / sqrt_diagonal
        theta, prox_info = prox(transformed_center, lam, return_info=True)
        beta = theta / sqrt_diagonal

        gradient = (
            _certificate_matvec(self.quadratic, beta)
            - self.quadratic.linear
        )
        # Moreau displacement in transformed coordinates, mapped back to the
        # beta coordinates, is a member of lambda * partial Omega(beta).
        transformed_displacement = np.asarray(
            prox_info.get(
                "_dual_displacement", transformed_center - theta
            ),
            dtype=float,
        )
        penalty_subgradient = sqrt_diagonal * transformed_displacement
        stationarity = gradient + penalty_subgradient
        residual_norm, residual_scale, relative_residual = _relative_stationarity(
            stationarity,
            gradient,
            penalty_subgradient,
            self.quadratic.linear,
        )
        exact_diagonal = (
            self.quadratic.gram is None
            or self.quadratic.max_off_diagonal == 0.0
        )
        strong_convexity = (
            float(np.min(diagonal)) if exact_diagonal else None
        )
        distance_bound = (
            residual_norm / strong_convexity
            if strong_convexity is not None
            else None
        )
        objective_gap_bound = (
            residual_norm**2 / (2.0 * strong_convexity)
            if strong_convexity is not None
            else None
        )
        penalty = self.penalty(beta)
        loss = _certificate_loss(self.quadratic, beta)
        certified = bool(
            relative_residual <= self.tolerance
            and _prox_certificate_passes(prox_info, self.tolerance)
        )
        diagnostic: dict[str, Any] = {
            "converged": certified,
            "certified": certified,
            "n_iter": 1,
            "solver": "diagonal_one_sweep_laminar_prox",
            "loss": loss,
            "unscaled_penalty": penalty,
            "objective": loss + lam * penalty,
            "stationarity_residual": residual_norm,
            "stationarity_scale": residual_scale,
            "relative_stationarity_residual": relative_residual,
            "coefficient_distance_upper_bound": distance_bound,
            "objective_gap_upper_bound": objective_gap_bound,
            "strong_convexity": strong_convexity,
            "lipschitz_constant": self.quadratic.lipschitz_constant,
            "prox_duality_gap": float(prox_info["duality_gap"]),
            "prox_relative_duality_gap": float(
                prox_info["relative_duality_gap"]
            ),
            "prox_raw_relative_duality_gap": float(
                prox_info["raw_relative_duality_gap"]
            ),
            "prox_max_dual_l1_violation": float(
                prox_info["max_dual_l1_violation"]
            ),
            "prox_max_relative_dual_l1_violation": float(
                prox_info["max_relative_dual_l1_violation"]
            ),
            "prox_relative_moreau_residual": float(
                prox_info["relative_moreau_residual"]
            ),
            "quadratic_backend": self.quadratic.backend,
            "assumed_diagonal_gram": self.quadratic.assumed_diagonal_gram,
            "certificate_uses_full_gram": self.quadratic.gram is not None,
        }
        return _PointResult(beta=np.asarray(beta), diagnostic=diagnostic)

    def _solve_accelerated(
        self, lam: float, beta_init: np.ndarray
    ) -> _PointResult:
        lipschitz = float(self.quadratic.lipschitz_constant)
        if not np.isfinite(lipschitz) or lipschitz < 0.0:
            raise ValueError(
                "accelerated proximal gradient requires a nonnegative finite "
                "quadratic Lipschitz constant"
            )
        if lipschitz == 0.0:
            beta = np.zeros_like(beta_init, dtype=float)
            gradient = -np.asarray(self.quadratic.linear, dtype=float)
            residual_norm, residual_scale, relative_residual = (
                _relative_stationarity(
                    gradient,
                    gradient,
                    np.zeros_like(beta),
                    self.quadratic.linear,
                )
            )
            certified = bool(relative_residual <= self.tolerance)
            loss = _certificate_loss(self.quadratic, beta)
            return _PointResult(
                beta=beta,
                diagnostic={
                    "converged": certified,
                    "certified": certified,
                    "n_iter": 0,
                    "solver": "zero_quadratic",
                    "loss": loss,
                    "unscaled_penalty": 0.0,
                    "objective": loss,
                    "stationarity_residual": residual_norm,
                    "stationarity_scale": residual_scale,
                    "relative_stationarity_residual": relative_residual,
                    "coefficient_distance_upper_bound": None,
                    "objective_gap_upper_bound": None,
                    "strong_convexity": 0.0,
                    "lipschitz_constant": 0.0,
                    "prox_duality_gap": 0.0,
                    "prox_relative_duality_gap": 0.0,
                    "prox_raw_relative_duality_gap": 0.0,
                    "prox_max_dual_l1_violation": 0.0,
                    "prox_max_relative_dual_l1_violation": 0.0,
                    "prox_relative_moreau_residual": 0.0,
                    "quadratic_backend": self.quadratic.backend,
                    "assumed_diagonal_gram": (
                        self.quadratic.assumed_diagonal_gram
                    ),
                    "certificate_uses_full_gram": (
                        self.quadratic.gram is not None
                    ),
                },
            )
        step = 1.0 / lipschitz
        beta = np.asarray(beta_init, dtype=float).copy()
        extrapolated = beta.copy()
        momentum_parameter = 1.0
        last_prox_info: Mapping[str, Any] = {}
        residual_norm = np.inf
        residual_scale = np.finfo(float).tiny
        relative_residual = np.inf
        gradient_at_beta = _matvec(self.quadratic, beta) - self.quadratic.linear
        penalty_subgradient = np.zeros_like(beta)
        iteration = 0

        for iteration in range(1, self.max_iter + 1):
            gradient_at_extrapolated = (
                _matvec(self.quadratic, extrapolated) - self.quadratic.linear
            )
            prox_center = extrapolated - step * gradient_at_extrapolated
            next_beta, last_prox_info = self.euclidean_prox(
                prox_center, step * lam, return_info=True
            )
            prox_displacement = np.asarray(
                last_prox_info.get(
                    "_dual_displacement", prox_center - next_beta
                ),
                dtype=float,
            )
            penalty_subgradient = prox_displacement / step
            gradient_at_beta = (
                _matvec(self.quadratic, next_beta) - self.quadratic.linear
            )
            stationarity = gradient_at_beta + penalty_subgradient
            residual_norm, residual_scale, relative_residual = (
                _relative_stationarity(
                    stationarity,
                    gradient_at_beta,
                    penalty_subgradient,
                    self.quadratic.linear,
                )
            )
            beta_step = next_beta - beta
            beta = np.asarray(next_beta)
            if (
                relative_residual <= self.tolerance
                and _prox_certificate_passes(
                    last_prox_info, self.tolerance
                )
            ):
                break

            next_momentum = 0.5 * (
                1.0 + np.sqrt(1.0 + 4.0 * momentum_parameter**2)
            )
            next_extrapolated = beta + (
                (momentum_parameter - 1.0) / next_momentum
            ) * beta_step
            if self.restart and float(
                np.dot(extrapolated - beta, beta_step)
            ) > 0.0:
                momentum_parameter = 1.0
                extrapolated = beta.copy()
            else:
                momentum_parameter = next_momentum
                extrapolated = next_extrapolated

        penalty = self.penalty(beta)
        loss = self.quadratic.loss(beta)
        certified = bool(
            relative_residual <= self.tolerance
            and _prox_certificate_passes(last_prox_info, self.tolerance)
        )
        diagnostic = {
            "converged": certified,
            "certified": certified,
            "n_iter": iteration,
            "solver": "fista_exact_laminar_prox",
            "loss": loss,
            "unscaled_penalty": penalty,
            "objective": loss + lam * penalty,
            "stationarity_residual": residual_norm,
            "stationarity_scale": residual_scale,
            "relative_stationarity_residual": relative_residual,
            "coefficient_distance_upper_bound": None,
            "objective_gap_upper_bound": None,
            "strong_convexity": None,
            "lipschitz_constant": lipschitz,
            "prox_duality_gap": float(last_prox_info.get("duality_gap", np.nan)),
            "prox_relative_duality_gap": float(
                last_prox_info.get("relative_duality_gap", np.nan)
            ),
            "prox_raw_relative_duality_gap": float(
                last_prox_info.get("raw_relative_duality_gap", np.nan)
            ),
            "prox_max_dual_l1_violation": float(
                last_prox_info.get("max_dual_l1_violation", np.nan)
            ),
            "prox_max_relative_dual_l1_violation": float(
                last_prox_info.get(
                    "max_relative_dual_l1_violation", np.nan
                )
            ),
            "prox_relative_moreau_residual": float(
                last_prox_info.get("relative_moreau_residual", np.nan)
            ),
            "quadratic_backend": self.quadratic.backend,
            "assumed_diagonal_gram": self.quadratic.assumed_diagonal_gram,
            "certificate_uses_full_gram": self.quadratic.gram is not None,
        }
        return _PointResult(beta=beta.copy(), diagnostic=diagnostic)


@dataclass
class _PreparedRegression:
    X: np.ndarray
    y: np.ndarray
    weights: np.ndarray
    groups: list[np.ndarray]
    initial_beta: np.ndarray
    x_mean: np.ndarray
    y_mean: float
    quadratic: _QuadraticRegressionLoss
    solver: _LaminarQuadraticSolver

    def intercept(self, beta: np.ndarray) -> float:
        return self.y_mean - float(self.x_mean @ beta)


def _prepare_regression(
    X: np.ndarray,
    y: np.ndarray,
    groups: Iterable[np.ndarray],
    *,
    sample_weight: np.ndarray | None,
    fit_intercept: bool,
    beta_init: np.ndarray | None,
    group_weights: float | Sequence[float] | None,
    max_iter: int,
    tolerance: float,
    assume_diagonal_gram: bool,
    restart: bool,
) -> _PreparedRegression:
    X, y, private_groups, initial_beta, weights, _ = _prepare_problem(
        X=X,
        y=y,
        groups=groups,
        beta_init=beta_init,
        sample_weight=sample_weight,
        ord="inf",
    )
    normalized_weights = weights / float(weights.sum())
    if fit_intercept:
        x_mean = normalized_weights @ X
        y_mean = float(normalized_weights @ y)
        # Preserve exact constants: mean roundoff would otherwise invent a
        # tiny nonzero column and spoil its componentwise stationarity check.
        positive = weights > 0
        reference = int(np.argmax(positive))
        constant = np.all((X == X[reference]) | ~positive[:, None], axis=0)
        x_mean[constant] = X[reference, constant]
        if np.all((y == y[reference]) | ~positive):
            y_mean = float(y[reference])
        X_work = X - x_mean
        y_work = y - y_mean
    else:
        x_mean = np.zeros(X.shape[1], dtype=float)
        y_mean = 0.0
        X_work = X
        y_work = y
    quadratic = _make_quadratic_regression_loss(
        X_work,
        y_work,
        weights,
        assume_diagonal_gram=assume_diagonal_gram,
    )
    solver = _LaminarQuadraticSolver(
        quadratic,
        private_groups,
        group_weights,
        max_iter=max_iter,
        tolerance=tolerance,
        restart=restart,
    )
    return _PreparedRegression(
        X=X_work,
        y=y_work,
        weights=weights,
        groups=private_groups,
        initial_beta=initial_beta,
        x_mean=np.asarray(x_mean),
        y_mean=y_mean,
        quadratic=quadratic,
        solver=solver,
    )


def laminar_group_linf_regression(
    X: np.ndarray,
    y: np.ndarray,
    groups: Iterable[np.ndarray],
    lam: float,
    beta_init: np.ndarray | None = None,
    *,
    sample_weight: np.ndarray | None = None,
    fit_intercept: bool = True,
    group_weights: float | Sequence[float] | None = None,
    max_iter: int = 10_000,
    tol: float = 1e-8,
    assume_diagonal_gram: bool = False,
    restart: bool = True,
    return_info: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
    r"""Solve one laminar group-:math:`\ell_\infty` regression problem.

    The minimized objective is normalized weighted squared error plus
    ``lam * sum_g group_weights[g] * max(abs(beta[groups[g]]))``.  With
    ``fit_intercept=True`` the intercept is profiled out by weighted centering
    and returned in the diagnostics under ``"intercept"``.

    A numerically diagonal Gram matrix uses the exact one-sweep solver.
    ``assume_diagonal_gram=True`` skips construction and verification of the
    full Gram matrix; callers must establish diagonality for these exact rows
    and weights. When the full Gram was formed, the stationarity certificate
    is evaluated against that supplied Gram rather than the machine-scale
    diagonal surrogate.
    """
    if (
        isinstance(lam, (bool, np.bool_))
        or not isinstance(lam, (int, float, np.number))
        or not np.isfinite(lam)
        or float(lam) < 0.0
    ):
        raise ValueError("lam must be a finite nonnegative scalar")
    if not isinstance(fit_intercept, (bool, np.bool_)):
        raise ValueError("fit_intercept must be a boolean")
    if not isinstance(assume_diagonal_gram, (bool, np.bool_)):
        raise ValueError("assume_diagonal_gram must be a boolean")
    if not isinstance(restart, (bool, np.bool_)):
        raise ValueError("restart must be a boolean")
    if not isinstance(return_info, (bool, np.bool_)):
        raise ValueError("return_info must be a boolean")
    max_iter = _positive_integer(max_iter, "max_iter")
    tolerance = _positive_scalar(tol, "tol")
    problem = _prepare_regression(
        X,
        y,
        groups,
        sample_weight=sample_weight,
        fit_intercept=bool(fit_intercept),
        beta_init=beta_init,
        group_weights=group_weights,
        max_iter=max_iter,
        tolerance=tolerance,
        assume_diagonal_gram=bool(assume_diagonal_gram),
        restart=bool(restart),
    )
    lam = float(lam)
    if lam == 0.0:
        if problem.quadratic.diagonal is not None:
            beta, diagnostic = _diagonal_weighted_least_squares(
                problem.quadratic, problem.initial_beta
            )
        else:
            beta, diagnostic = _weighted_least_squares(
                problem.X,
                problem.y,
                problem.weights,
                problem.initial_beta,
            )
        diagnostic = dict(diagnostic)
        diagnostic.update(
            _zero_penalty_certificate(
                problem.quadratic, beta, tolerance
            )
        )
        diagnostic["unscaled_penalty"] = problem.solver.penalty(beta)
    else:
        point = problem.solver.solve(lam, problem.initial_beta)
        beta, diagnostic = point.beta, point.diagnostic
    diagnostic.update(
        {
            "lambda": lam,
            "intercept": problem.intercept(beta),
            "fit_intercept": bool(fit_intercept),
        }
    )
    if return_info:
        return beta, diagnostic
    return beta


def laminar_group_linf_regression_path(
    X: np.ndarray,
    y: np.ndarray,
    groups: Iterable[np.ndarray],
    lambdas: Iterable[float],
    beta_init: np.ndarray | None = None,
    *,
    sample_weight: np.ndarray | None = None,
    fit_intercept: bool = True,
    group_weights: float | Sequence[float] | None = None,
    max_iter: int = 10_000,
    tol: float = 1e-8,
    assume_diagonal_gram: bool = False,
    restart: bool = True,
) -> RegularizationPath:
    """Solve exact-proximal point problems on a descending lambda grid.

    Every stored point has a continuous stationarity diagnostic.  On a
    positive diagonal Gram matrix, each positive point is a non-iterative
    one-sweep solution.  On a general Gram matrix, descending warm starts feed
    an accelerated proximal-gradient solve.

    ``RegularizationPath.exact`` remains false because a user-supplied grid
    need not contain every coefficient-path knot.  Check
    ``metadata['point_solutions_certified']`` for fixed-lambda accuracy. If an
    automatically accepted numerical diagonal has off-diagonal roundoff, each
    point is certified against the retained full Gram.
    """
    requested_lambdas, path_lambdas = _validate_lambdas(lambdas)
    if not isinstance(fit_intercept, (bool, np.bool_)):
        raise ValueError("fit_intercept must be a boolean")
    if not isinstance(assume_diagonal_gram, (bool, np.bool_)):
        raise ValueError("assume_diagonal_gram must be a boolean")
    if not isinstance(restart, (bool, np.bool_)):
        raise ValueError("restart must be a boolean")
    max_iter = _positive_integer(max_iter, "max_iter")
    tolerance = _positive_scalar(tol, "tol")
    problem = _prepare_regression(
        X,
        y,
        groups,
        sample_weight=sample_weight,
        fit_intercept=bool(fit_intercept),
        beta_init=beta_init,
        group_weights=group_weights,
        max_iter=max_iter,
        tolerance=tolerance,
        assume_diagonal_gram=bool(assume_diagonal_gram),
        restart=bool(restart),
    )

    coefficients: list[np.ndarray] = []
    intercepts: list[float] = []
    penalties: list[float] = []
    diagnostics: list[dict[str, Any]] = []
    warm_beta = problem.initial_beta.copy()
    previous_lambda: float | None = None

    for raw_lambda in path_lambdas:
        lam = float(raw_lambda)
        warm_start_norm = float(norm(warm_beta))
        if lam == 0.0:
            if problem.quadratic.diagonal is not None:
                beta, raw_diagnostic = _diagonal_weighted_least_squares(
                    problem.quadratic, warm_beta
                )
            else:
                beta, raw_diagnostic = _weighted_least_squares(
                    problem.X, problem.y, problem.weights, warm_beta
                )
            diagnostic = dict(raw_diagnostic)
            diagnostic.update(
                _zero_penalty_certificate(
                    problem.quadratic, beta, tolerance
                )
            )
        else:
            point = problem.solver.solve(lam, warm_beta)
            beta, diagnostic = point.beta, point.diagnostic
        penalty = problem.solver.penalty(beta)
        diagnostic.update(
            {
                "lambda": lam,
                "intercept": problem.intercept(beta),
                "unscaled_penalty": penalty,
                "warm_start_norm": warm_start_norm,
                "warm_started_from_lambda": previous_lambda,
            }
        )
        warm_beta = np.asarray(beta, dtype=float).copy()
        coefficients.append(warm_beta)
        intercepts.append(problem.intercept(warm_beta))
        penalties.append(penalty)
        diagnostics.append(diagnostic)
        previous_lambda = lam

    points_certified = all(
        diagnostic.get("certified") is True for diagnostic in diagnostics
    )
    diagonal_closed_form = problem.solver.uses_diagonal_closed_form
    metadata = {
        "problem": "regression",
        "ord": "inf",
        "fit_intercept": bool(fit_intercept),
        "point_solutions_certified": points_certified,
        "point_solution_kind": (
            "diagonal_one_sweep_exact_prox"
            if diagonal_closed_form
            else "iterative_exact_prox"
        ),
        "warm_started": not diagonal_closed_form,
        "lambda_order": "descending_unique",
        "requested_lambdas": tuple(float(value) for value in requested_lambdas),
        "duplicates_removed": int(requested_lambdas.size - path_lambdas.size),
        "quadratic_backend": problem.quadratic.backend,
        "assumed_diagonal_gram": problem.quadratic.assumed_diagonal_gram,
        "gram_max_off_diagonal": problem.quadratic.max_off_diagonal,
        "gram_max_off_diagonal_correlation": (
            problem.quadratic.max_off_diagonal_correlation
        ),
        "certificate_uses_full_gram": problem.quadratic.gram is not None,
        "coefficient_knots_enumerated": False,
    }
    return RegularizationPath(
        lambdas=path_lambdas,
        coefficients=np.vstack(coefficients),
        intercepts=np.asarray(intercepts),
        penalties=np.asarray(penalties),
        diagnostics=tuple(diagnostics),
        method=(
            "laminar-diagonal-one-sweep"
            if diagonal_closed_form
            else "laminar-exact-prox-fista"
        ),
        exact=False,
        status="complete" if points_certified else "partial",
        metadata=metadata,
    )


__all__ = [
    "laminar_group_linf_regression",
    "laminar_group_linf_regression_path",
]
