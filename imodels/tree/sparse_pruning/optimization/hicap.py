r"""Exact regularization paths for hierarchical infinity-CAP regression.

This module implements the path from the convex formulation, rather than from
the historical MATLAB implementation.  For a laminar collection of groups it
solves

.. math::

    \frac{1}{2}\sum_i \bar w_i (y_i - b - x_i^T\beta)^2
    + \lambda \sum_g \|\beta_g\|_\infty,

where ``bar(w) = w / sum(w)``.  The infinity norms are represented by
epigraph variables.  Conditional on an active epigraph face, the KKT system is
linear in ``lambda``; following the intervals on which its primal and dual
variables remain feasible therefore gives the complete piecewise-linear path.

The loss must be strictly convex in the coefficient vector.  A rank-deficient
design can have a set-valued coefficient path, so in that case this module
returns the two unambiguous endpoint solutions with ``exact=False`` and an
explicit ``"nonunique_design"`` status.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from scipy.linalg import lstsq
from scipy.optimize import LinearConstraint, linprog, minimize

from ._result import RegularizationPath


@dataclass(frozen=True)
class _Problem:
    """Centered weighted least-squares problem and its epigraph."""

    H: np.ndarray
    h: np.ndarray
    groups: tuple[np.ndarray, ...]
    A: np.ndarray
    row_group: np.ndarray
    row_feature: np.ndarray
    row_sign: np.ndarray
    x_mean: np.ndarray
    y_mean: float
    fit_intercept: bool
    weight_sum: float

    @property
    def p(self) -> int:
        return self.H.shape[0]

    @property
    def n_groups(self) -> int:
        return len(self.groups)


@dataclass(frozen=True)
class _Region:
    """One affine KKT region, valid for ``lower <= lambda <= upper``."""

    z_constant: np.ndarray
    z_slope: np.ndarray
    dual_constant: np.ndarray
    dual_slope: np.ndarray
    basis: np.ndarray
    lower: float
    upper: float
    condition: float

    def z_at(self, lam: float) -> np.ndarray:
        return self.z_constant + float(lam) * self.z_slope


class _PathFailure(RuntimeError):
    pass


def _validate_inputs(
    X: np.ndarray,
    y: np.ndarray,
    groups: Iterable[Sequence[int]],
    sample_weight: np.ndarray | None,
    fit_intercept: bool,
) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, ...], np.ndarray, float]:
    for values, name in ((X, "X"), (y, "y"), (sample_weight, "sample_weight")):
        if np.iscomplexobj(values):
            raise ValueError(f"{name} must contain real values")
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"X must be two-dimensional; got shape {X.shape}")
    if y.ndim != 1:
        raise ValueError(f"y must be one-dimensional; got shape {y.shape}")
    n, p = X.shape
    if n == 0 or p == 0:
        raise ValueError("X must contain at least one sample and one feature")
    if y.shape[0] != n:
        raise ValueError("X and y have inconsistent sample counts")
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
        raise ValueError("X and y must contain only finite values")
    if not isinstance(fit_intercept, (bool, np.bool_)):
        raise ValueError("fit_intercept must be a boolean")

    try:
        raw_groups = list(groups)
    except TypeError as exc:
        raise ValueError("groups must be a non-empty iterable") from exc
    if not raw_groups:
        raise ValueError("groups must contain at least one group")

    validated: list[np.ndarray] = []
    keys: set[tuple[int, ...]] = set()
    for group_number, group in enumerate(raw_groups):
        raw = np.asarray(group)
        if raw.ndim != 1 or raw.size == 0:
            raise ValueError(f"group {group_number} must be a non-empty 1D array")
        if raw.dtype.kind not in "iu":
            raise ValueError(f"group {group_number} must contain integer indices")
        group_array = np.sort(raw.astype(np.intp, copy=False))
        if np.unique(group_array).size != group_array.size:
            raise ValueError(f"group {group_number} contains duplicate indices")
        if np.any(group_array < 0) or np.any(group_array >= p):
            raise ValueError(
                f"group {group_number} contains an index outside [0, {p})"
            )
        key = tuple(int(index) for index in group_array)
        if key in keys:
            raise ValueError("groups must not contain duplicate groups")
        keys.add(key)
        validated.append(group_array)

    root = set(range(p))
    if sum(set(group.tolist()) == root for group in validated) != 1:
        raise ValueError(
            "laminar subtree groups must contain exactly one root group "
            "covering every feature"
        )
    group_sets = [set(group.tolist()) for group in validated]
    for left in range(len(group_sets)):
        for right in range(left + 1, len(group_sets)):
            intersection = group_sets[left] & group_sets[right]
            if intersection and not (
                group_sets[left] <= group_sets[right]
                or group_sets[right] <= group_sets[left]
            ):
                raise ValueError(
                    "groups must be laminar: every pair must be disjoint or nested"
                )

    if sample_weight is None:
        weights = np.ones(n, dtype=float)
    else:
        weights = np.asarray(sample_weight, dtype=float)
        if weights.shape != (n,):
            raise ValueError(f"sample_weight must have shape ({n},)")
        if not np.all(np.isfinite(weights)):
            raise ValueError("sample_weight must contain only finite values")
        if np.any(weights < 0):
            raise ValueError("sample_weight cannot contain negative values")
    with np.errstate(over="ignore", invalid="ignore"):
        weight_sum = float(np.sum(weights))
    if not np.isfinite(weight_sum) or weight_sum <= 0:
        raise ValueError("sample_weight must have a positive finite total weight")
    return X, y, tuple(validated), weights / weight_sum, weight_sum


def _make_problem(
    X: np.ndarray,
    y: np.ndarray,
    groups: tuple[np.ndarray, ...],
    normalized_weight: np.ndarray,
    weight_sum: float,
    fit_intercept: bool,
) -> _Problem:
    if fit_intercept:
        x_mean = normalized_weight @ X
        y_mean = float(normalized_weight @ y)
        X_work = X - x_mean
        y_work = y - y_mean
    else:
        x_mean = np.zeros(X.shape[1], dtype=float)
        y_mean = 0.0
        X_work = X
        y_work = y
    H = X_work.T @ (normalized_weight[:, None] * X_work)
    H = 0.5 * (H + H.T)
    h = X_work.T @ (normalized_weight * y_work)

    # Rows encode s * beta_j - t_g <= 0.  Keeping this explicit makes both
    # active-face continuation and primal certification straightforward.
    row_group: list[int] = []
    row_feature: list[int] = []
    row_sign: list[float] = []
    for group_number, group in enumerate(groups):
        for feature in group:
            for sign in (1.0, -1.0):
                row_group.append(group_number)
                row_feature.append(int(feature))
                row_sign.append(sign)
    row_group_array = np.asarray(row_group, dtype=np.intp)
    row_feature_array = np.asarray(row_feature, dtype=np.intp)
    row_sign_array = np.asarray(row_sign, dtype=float)
    A = np.zeros((len(row_group), X.shape[1] + len(groups)), dtype=float)
    A[np.arange(len(row_group)), row_feature_array] = row_sign_array
    A[np.arange(len(row_group)), X.shape[1] + row_group_array] = -1.0
    return _Problem(
        H=H,
        h=h,
        groups=groups,
        A=A,
        row_group=row_group_array,
        row_feature=row_feature_array,
        row_sign=row_sign_array,
        x_mean=np.asarray(x_mean, dtype=float),
        y_mean=y_mean,
        fit_intercept=bool(fit_intercept),
        weight_sum=weight_sum,
    )


def _penalty(problem: _Problem, beta: np.ndarray) -> float:
    return float(sum(np.max(np.abs(beta[group])) for group in problem.groups))


def _epigraph(beta: np.ndarray, groups: tuple[np.ndarray, ...]) -> np.ndarray:
    return np.asarray([np.max(np.abs(beta[group])) for group in groups])


def _lambda_max(problem: _Problem, tolerance: float) -> float:
    """Find the smallest lambda for which beta=0 satisfies the KKT system."""

    n_rows = problem.A.shape[0]
    p, m = problem.p, problem.n_groups
    # mu_r is the mass placed on signed feature r in its group's l1 ball.
    # Stationarity requires signed masses to sum to h and every group to have
    # total mass lambda.  At beta=0 unused mass can always be added as a
    # cancelling +/- pair, making equality equivalent to the usual <= form.
    equality = np.zeros((p + m, n_rows + 1), dtype=float)
    equality[problem.row_feature, np.arange(n_rows)] = problem.row_sign
    equality[p + problem.row_group, np.arange(n_rows)] = 1.0
    equality[p:, -1] = -1.0
    rhs = np.r_[problem.h, np.zeros(m)]
    objective = np.zeros(n_rows + 1, dtype=float)
    objective[-1] = 1.0
    result = linprog(
        objective,
        A_eq=equality,
        b_eq=rhs,
        bounds=[(0.0, None)] * (n_rows + 1),
        method="highs-ds",
        options={
            "primal_feasibility_tolerance": max(1e-10, min(1e-7, tolerance)),
            "dual_feasibility_tolerance": max(1e-10, min(1e-7, tolerance)),
        },
    )
    if not result.success or result.x is None:
        raise _PathFailure(f"could not compute lambda_max: {result.message}")
    lam = float(result.x[-1])
    residual = np.max(np.abs(equality @ result.x - rhs))
    scale = max(1.0, float(np.max(np.abs(problem.h))))
    if residual > 50.0 * tolerance * scale or lam < -tolerance:
        raise _PathFailure("lambda_max linear program failed its residual check")
    return max(0.0, lam)


def _candidate_rows(
    problem: _Problem, beta: np.ndarray, tie_tolerance: float
) -> np.ndarray:
    t = _epigraph(beta, problem.groups)
    slack = (
        problem.row_sign * beta[problem.row_feature]
        - t[problem.row_group]
    )
    scale = np.maximum(1.0, t[problem.row_group])
    return np.flatnonzero(np.abs(slack) <= tie_tolerance * scale)


def _dual_basis(
    problem: _Problem,
    beta: np.ndarray,
    lam: float,
    tolerance: float,
    tie_tolerance: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return a basic nonnegative subgradient decomposition at ``beta``."""

    if lam <= 0:
        raise _PathFailure("a positive lambda is required to recover a dual face")
    p, m = problem.p, problem.n_groups
    residual_gradient = problem.h - problem.H @ beta
    candidates = _candidate_rows(problem, beta, tie_tolerance)
    if candidates.size == 0:
        raise _PathFailure("no tight epigraph constraints were found")

    D = np.zeros((p + m, candidates.size), dtype=float)
    D[problem.row_feature[candidates], np.arange(candidates.size)] = (
        problem.row_sign[candidates]
    )
    D[p + problem.row_group[candidates], np.arange(candidates.size)] = 1.0
    rhs = np.r_[residual_gradient / lam, np.ones(m)]
    # Dual simplex returns a basic feasible solution.  Its positive columns
    # form the independent constraint basis needed by the KKT continuation.
    result = linprog(
        np.zeros(candidates.size, dtype=float),
        A_eq=D,
        b_eq=rhs,
        bounds=[(0.0, None)] * candidates.size,
        method="highs-ds",
        options={
            "primal_feasibility_tolerance": max(1e-10, min(1e-7, tolerance)),
            "dual_feasibility_tolerance": max(1e-10, min(1e-7, tolerance)),
        },
    )
    if not result.success or result.x is None:
        # SLSQP can stop with a coefficient error of a few ulps in objective
        # space even when its active face is unambiguous.  Use a Chebyshev
        # residual LP to identify a nearby basic dual face, then solve that
        # face's KKT equations exactly below.  This relaxation is never used
        # as a path solution or certificate.
        relaxed_objective = np.r_[np.zeros(candidates.size), 1.0]
        relaxed_A = np.block(
            [
                [D, -np.ones((D.shape[0], 1))],
                [-D, -np.ones((D.shape[0], 1))],
            ]
        )
        relaxed_rhs = np.r_[rhs, -rhs]
        result = linprog(
            relaxed_objective,
            A_ub=relaxed_A,
            b_ub=relaxed_rhs,
            bounds=[(0.0, None)] * (candidates.size + 1),
            method="highs-ds",
        )
        if not result.success or result.x is None:
            raise _PathFailure(
                f"could not recover a KKT subgradient: {result.message}"
            )
        alpha_all = result.x[:-1]
        residual = float(np.max(np.abs(D @ alpha_all - rhs)))
        recovery_limit = max(1e-2, 1000.0 * tolerance) * max(
            1.0, float(np.max(np.abs(rhs)))
        )
        if residual > recovery_limit:
            raise _PathFailure(
                "the fixed-lambda solution is not close to a certifiable dual face"
            )
    else:
        alpha_all = result.x
        residual = float(np.max(np.abs(D @ alpha_all - rhs)))
    positive_tolerance = max(1e-11, tolerance * 0.01)
    selected = np.flatnonzero(alpha_all > positive_tolerance)
    if selected.size == 0:
        raise _PathFailure("dual recovery returned an empty active basis")
    basis = candidates[selected]
    alpha = alpha_all[selected]

    # A degenerate LP may report tiny dependent basic variables.  They cannot
    # define an invertible KKT face; fail explicitly instead of perturbing the
    # mathematical penalty or silently dropping dual mass.
    C = problem.A[basis]
    rank = np.linalg.matrix_rank(C, tol=max(tolerance, 1e-12))
    if rank != C.shape[0]:
        raise _PathFailure("the active dual face is degenerate")
    return basis, alpha, residual


def _affine_region(
    problem: _Problem,
    basis: np.ndarray,
    tolerance: float,
) -> _Region:
    p, m = problem.p, problem.n_groups
    dimension = p + m
    C = problem.A[basis]
    Q = np.zeros((dimension, dimension), dtype=float)
    Q[:p, :p] = problem.H
    KKT = np.block(
        [[Q, C.T], [C, np.zeros((C.shape[0], C.shape[0]), dtype=float)]]
    )
    rhs_constant = np.r_[problem.h, np.zeros(m + C.shape[0])]
    rhs_slope = np.r_[np.zeros(p), -np.ones(m), np.zeros(C.shape[0])]
    solution_constant, _, rank, singular_values = lstsq(
        KKT, rhs_constant, cond=max(tolerance * 0.01, 1e-13), lapack_driver="gelsy"
    )
    solution_slope, _, slope_rank, _ = lstsq(
        KKT, rhs_slope, cond=max(tolerance * 0.01, 1e-13), lapack_driver="gelsy"
    )
    if rank != KKT.shape[0] or slope_rank != KKT.shape[0]:
        raise _PathFailure("the active-face KKT system is singular")
    kkt_residual = max(
        float(np.max(np.abs(KKT @ solution_constant - rhs_constant))),
        float(np.max(np.abs(KKT @ solution_slope - rhs_slope))),
    )
    kkt_scale = max(1.0, float(np.max(np.abs(problem.h))))
    if kkt_residual > 100.0 * tolerance * kkt_scale:
        raise _PathFailure("the active-face KKT solve failed its residual check")

    z_constant = solution_constant[:dimension]
    z_slope = solution_slope[:dimension]
    dual_constant = solution_constant[dimension:]
    dual_slope = solution_slope[dimension:]

    lower, upper = 0.0, np.inf
    primal_constant = problem.A @ z_constant
    primal_slope = problem.A @ z_slope
    slope_tolerance = max(1e-14, tolerance * 1e-3)
    for constant, slope in zip(primal_constant, primal_slope):
        if slope > slope_tolerance:
            upper = min(upper, -constant / slope)
        elif slope < -slope_tolerance:
            lower = max(lower, -constant / slope)
        elif constant > 100.0 * tolerance:
            raise _PathFailure("an affine KKT face is primal infeasible")
    for constant, slope in zip(dual_constant, dual_slope):
        # Active multipliers must remain nonnegative.
        if slope > slope_tolerance:
            lower = max(lower, -constant / slope)
        elif slope < -slope_tolerance:
            upper = min(upper, -constant / slope)
        elif constant < -100.0 * tolerance:
            raise _PathFailure("an affine KKT face is dual infeasible")
    lower = max(0.0, float(lower))
    upper = float(upper)
    if upper < lower - 100.0 * tolerance * max(1.0, abs(lower), abs(upper)):
        raise _PathFailure("an affine KKT face has an empty lambda interval")

    # gelsy does not promise singular values, hence use a direct condition
    # estimate only for diagnostics (never for deciding the solution).
    del singular_values
    condition = float(np.linalg.cond(KKT))
    return _Region(
        z_constant=z_constant,
        z_slope=z_slope,
        dual_constant=dual_constant,
        dual_slope=dual_slope,
        basis=np.asarray(basis, dtype=np.intp),
        lower=lower,
        upper=upper,
        condition=condition,
    )


def _solve_fixed_lambda(
    problem: _Problem,
    lam: float,
    tolerance: float,
    max_iter: int,
    beta_start: np.ndarray | None = None,
) -> np.ndarray:
    """Convex epigraph-QP oracle used only to identify the next face."""

    p, m = problem.p, problem.n_groups
    if lam <= tolerance:
        beta, _, rank, _ = lstsq(
            problem.H,
            problem.h,
            cond=max(tolerance * 0.01, 1e-13),
            lapack_driver="gelsy",
        )
        if rank != p:
            raise _PathFailure("least-squares endpoint is nonunique")
        return np.asarray(beta, dtype=float)

    if beta_start is None:
        beta_start = np.zeros(p, dtype=float)
    else:
        beta_start = np.asarray(beta_start, dtype=float)
    z_start = np.r_[beta_start, _epigraph(beta_start, problem.groups)]
    linear = np.r_[-problem.h, lam * np.ones(m)]
    # SLSQP's stopping tests are absolute in objective units.  Scaling by the
    # current regularization strength prevents it from declaring convergence
    # at a pre-event solution when lambda (and the objective gain from a newly
    # active coefficient) is small.
    objective_scale = 1.0 / max(lam, tolerance)

    def objective(z: np.ndarray) -> float:
        return float(
            objective_scale * (0.5 * z[:p] @ problem.H @ z[:p] + linear @ z)
        )

    def gradient(z: np.ndarray) -> np.ndarray:
        result = linear.copy()
        result[:p] += problem.H @ z[:p]
        return objective_scale * result

    constraint = LinearConstraint(problem.A, -np.inf, 0.0)
    result = minimize(
        objective,
        z_start,
        jac=gradient,
        constraints=(constraint,),
        method="SLSQP",
        options={"ftol": max(1e-13, tolerance * 0.01), "maxiter": max_iter},
    )
    if not result.success or result.x is None or not np.all(np.isfinite(result.x)):
        raise _PathFailure(f"fixed-lambda QP failed: {result.message}")
    beta = np.asarray(result.x[:p], dtype=float)
    z = np.r_[beta, _epigraph(beta, problem.groups)]
    violation = max(0.0, float(np.max(problem.A @ z)))
    if violation > 100.0 * tolerance:
        raise _PathFailure("fixed-lambda QP failed primal feasibility")
    return beta


def _region_below(
    problem: _Problem,
    upper: float,
    initial_gap: float,
    lambda_max: float,
    tolerance: float,
    tie_tolerance: float,
    oracle_max_iter: int,
    beta_start: np.ndarray | None,
) -> tuple[_Region, np.ndarray, int]:
    """Find the first active region immediately below an event knot."""

    scale = max(1.0, lambda_max)
    coverage_tolerance = 20.0 * tolerance * scale
    event_tolerance = max(1e-12, tolerance * 0.1) * scale
    gap = min(max(initial_gap, 1e-9 * scale), max(upper * 0.25, 1e-9 * scale))
    last_error: Exception | None = None
    for attempt in range(15):
        next_gap_factor = 0.1
        probe = max(0.0, upper - gap)
        if probe <= tolerance * scale:
            probe = max(0.0, upper * 0.5)
        try:
            beta = _solve_fixed_lambda(
                problem,
                probe,
                tolerance=tolerance,
                max_iter=oracle_max_iter,
                beta_start=beta_start,
            )
            # Expand the tie tolerance only when the QP's numerical solution
            # leaves the exact dual face just outside the initial threshold.
            region: _Region | None = None
            for multiplier in (1.0, 10.0, 100.0, 1000.0):
                try:
                    basis, _, _ = _dual_basis(
                        problem,
                        beta,
                        probe,
                        tolerance=tolerance,
                        tie_tolerance=tie_tolerance * multiplier,
                    )
                    candidate = _affine_region(problem, basis, tolerance)
                    is_face_above_event = (
                        candidate.lower >= upper - event_tolerance
                        and candidate.upper >= upper - coverage_tolerance
                    )
                    if is_face_above_event or (
                        candidate.lower - coverage_tolerance
                        <= probe
                        <= candidate.upper + coverage_tolerance
                    ):
                        region = candidate
                        break
                except _PathFailure as exc:
                    last_error = exc
            if region is None:
                raise _PathFailure("could not certify the oracle's active face")
            if region.lower >= upper - event_tolerance:
                # The fixed-lambda oracle remained on the face above the
                # event because the probe was too close for its stopping
                # tolerance.  Move farther below; this is distinct from
                # skipping a narrow region, whose upper endpoint would lie
                # strictly below ``upper`` and therefore asks for a smaller
                # gap.
                next_gap_factor = 10.0
                last_error = _PathFailure(
                    "the face oracle did not cross the current event"
                )
            elif region.upper >= upper - coverage_tolerance:
                refined_beta = region.z_at(probe)[: problem.p]
                return region, refined_beta, attempt + 1
        except _PathFailure as exc:
            last_error = exc
        gap *= next_gap_factor
        gap = min(gap, max(upper * 0.5, 1e-9 * scale))
    message = "could not identify the active face directly below a path event"
    if last_error is not None:
        message += f": {last_error}"
    raise _PathFailure(message)


def _kkt_diagnostic(
    problem: _Problem,
    beta: np.ndarray,
    lam: float,
    tolerance: float,
    tie_tolerance: float,
) -> dict[str, float | int | bool]:
    gradient = problem.H @ beta - problem.h
    if lam <= tolerance * max(1.0, float(np.max(np.abs(problem.h)))):
        residual = float(np.max(np.abs(gradient)))
        return {
            "kkt_residual": residual,
            "stationarity_residual": residual,
            "primal_residual": 0.0,
            "n_active_constraints": int(
                _candidate_rows(problem, beta, tie_tolerance).size
            ),
            "certified": residual <= 100.0 * tolerance,
        }
    try:
        basis, alpha, dual_residual = _dual_basis(
            problem,
            beta,
            lam,
            tolerance=tolerance,
            tie_tolerance=tie_tolerance * 10.0,
        )
        stationarity = gradient.copy()
        np.add.at(
            stationarity,
            problem.row_feature[basis],
            lam * alpha * problem.row_sign[basis],
        )
        stationarity_residual = float(np.max(np.abs(stationarity)))
        residual = max(stationarity_residual, float(dual_residual * lam))
        return {
            "kkt_residual": residual,
            "stationarity_residual": stationarity_residual,
            "primal_residual": 0.0,
            "dual_equality_residual": float(dual_residual),
            "n_active_constraints": int(basis.size),
            "certified": residual
            <= 100.0 * tolerance * max(1.0, lam, np.max(np.abs(problem.h))),
        }
    except _PathFailure:
        return {
            "kkt_residual": np.inf,
            "stationarity_residual": np.inf,
            "primal_residual": 0.0,
            "n_active_constraints": 0,
            "certified": False,
        }


def _endpoint_path_for_nonunique_design(
    problem: _Problem,
    lambda_max: float,
    rank: int,
    tolerance: float,
) -> RegularizationPath:
    beta_ls, _, _, _ = lstsq(
        problem.H,
        problem.h,
        cond=max(tolerance * 0.01, 1e-13),
        lapack_driver="gelsy",
    )
    beta_ls = np.asarray(beta_ls, dtype=float)
    if lambda_max <= tolerance:
        lambdas = np.asarray([0.0])
        coefficients = beta_ls[None, :]
        events = ("nonunique_least_squares",)
    else:
        lambdas = np.asarray([lambda_max, 0.0])
        coefficients = np.vstack([np.zeros(problem.p), beta_ls])
        events = ("lambda_max", "nonunique_least_squares")
    intercepts = problem.y_mean - coefficients @ problem.x_mean
    penalties = np.asarray([_penalty(problem, beta) for beta in coefficients])
    diagnostics = tuple(
        {
            "kkt_residual": np.nan,
            "certified": False,
            "reason": "rank-deficient weighted design",
        }
        for _ in range(lambdas.size)
    )
    return RegularizationPath(
        lambdas=lambdas,
        coefficients=coefficients,
        intercepts=intercepts,
        penalties=penalties,
        events=events,
        diagnostics=diagnostics,
        method="hicap-homotopy",
        exact=False,
        status="nonunique_design",
        metadata={
            "rank": int(rank),
            "n_features": problem.p,
            "lambda_max": lambda_max,
            "endpoint_only": True,
            "weight_sum": problem.weight_sum,
        },
    )


def hicap_regression_path(
    X: np.ndarray,
    y: np.ndarray,
    groups: Iterable[Sequence[int]],
    *,
    sample_weight: np.ndarray | None = None,
    fit_intercept: bool = True,
    tolerance: float = 1e-8,
    tie_tolerance: float = 1e-7,
    max_events: int = 10_000,
    oracle_max_iter: int = 2_000,
) -> RegularizationPath:
    """Compute the complete hiCAP infinity-norm regression path.

    Parameters
    ----------
    X, y:
        Design matrix (without an explicit intercept column) and response.
    groups:
        Zero-based feature-index groups forming a rooted laminar family.  The
        unique root group must contain all columns of ``X``.
    sample_weight:
        Nonnegative relative observation weights.  Uniform rescaling leaves
        the result unchanged.
    fit_intercept:
        If true, fit an unpenalized intercept by weighted centering.
    tolerance, tie_tolerance:
        Numerical KKT and active-face tolerances.
    max_events:
        Safety limit on the number of affine path regions.
    oracle_max_iter:
        Iteration limit for the convex QP used to identify each new face.

    Returns
    -------
    RegularizationPath
        Knots are ordered by decreasing lambda.  ``exact`` is true only when
        every affine region reached lambda zero and all knot KKT checks pass.

    Notes
    -----
    The implementation assumes a full-column-rank weighted, centered design.
    Rank-deficient input returns endpoint solutions with
    ``status='nonunique_design'`` and ``exact=False`` because there is no
    canonical coefficient path without an additional selection rule.
    """

    for value, name in ((tolerance, "tolerance"), (tie_tolerance, "tie_tolerance")):
        if (
            isinstance(value, (bool, np.bool_))
            or np.iscomplexobj(value)
            or not isinstance(value, (int, float, np.number))
            or not np.isfinite(value)
            or float(value) <= 0
        ):
            raise ValueError(f"{name} must be a positive finite scalar")
    tolerance = float(tolerance)
    tie_tolerance = float(tie_tolerance)
    integer_parameters = (
        (max_events, "max_events"),
        (oracle_max_iter, "oracle_max_iter"),
    )
    for value, name in integer_parameters:
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)
        ) or int(value) <= 0:
            raise ValueError(f"{name} must be a positive integer")
    max_events = int(max_events)
    oracle_max_iter = int(oracle_max_iter)

    X, y, validated_groups, normalized_weight, weight_sum = _validate_inputs(
        X, y, groups, sample_weight, fit_intercept
    )
    problem = _make_problem(
        X,
        y,
        validated_groups,
        normalized_weight,
        weight_sum,
        bool(fit_intercept),
    )
    eigenvalues = np.linalg.eigvalsh(problem.H)
    spectral_scale = max(1.0, float(np.max(np.abs(eigenvalues))))
    rank = int(np.sum(eigenvalues > tolerance * spectral_scale))

    try:
        lambda_max = _lambda_max(problem, tolerance)
    except _PathFailure as exc:
        raise RuntimeError(str(exc)) from exc
    if rank < problem.p:
        return _endpoint_path_for_nonunique_design(
            problem, lambda_max, rank, tolerance
        )

    lambda_scale = max(1.0, lambda_max)
    if lambda_max <= tolerance * lambda_scale:
        beta = _solve_fixed_lambda(
            problem, 0.0, tolerance=tolerance, max_iter=oracle_max_iter
        )
        intercept = problem.y_mean - float(problem.x_mean @ beta)
        diagnostic = _kkt_diagnostic(
            problem, beta, 0.0, tolerance, tie_tolerance
        )
        return RegularizationPath(
            lambdas=np.asarray([0.0]),
            coefficients=beta[None, :],
            intercepts=np.asarray([intercept]),
            penalties=np.asarray([_penalty(problem, beta)]),
            events=("least_squares",),
            diagnostics=(diagnostic,),
            method="hicap-homotopy",
            exact=bool(diagnostic["certified"]),
            status="complete" if diagnostic["certified"] else "kkt_failure",
            metadata={
                "lambda_max": 0.0,
                "n_regions": 0,
                "rank": rank,
                "condition": float(np.linalg.cond(problem.H)),
                "weight_sum": weight_sum,
            },
        )

    regions: list[_Region] = []
    probe_attempts = 0
    status = "complete"
    failure_message: str | None = None
    try:
        initial_gap = max(lambda_max * 1e-5, tolerance * lambda_scale * 10.0)
        region, beta_probe, attempts = _region_below(
            problem,
            upper=lambda_max,
            initial_gap=initial_gap,
            lambda_max=lambda_max,
            tolerance=tolerance,
            tie_tolerance=tie_tolerance,
            oracle_max_iter=oracle_max_iter,
            beta_start=np.zeros(problem.p),
        )
        probe_attempts += attempts
        regions.append(region)
        current_upper = max(0.0, region.lower)
        previous_width = max(lambda_max - current_upper, tolerance * lambda_scale)
        while current_upper > tolerance * lambda_scale:
            if len(regions) >= max_events:
                raise _PathFailure(f"path exceeded max_events={max_events}")
            next_region, beta_probe, attempts = _region_below(
                problem,
                upper=current_upper,
                initial_gap=max(
                    previous_width * 1e-2, tolerance * lambda_scale * 10.0
                ),
                lambda_max=lambda_max,
                tolerance=tolerance,
                tie_tolerance=tie_tolerance,
                oracle_max_iter=oracle_max_iter,
                beta_start=beta_probe,
            )
            probe_attempts += attempts
            next_lower = max(0.0, next_region.lower)
            if next_lower >= current_upper - max(
                1e-12, tolerance * 0.1
            ) * lambda_scale:
                raise _PathFailure("active-set continuation made no progress")
            regions.append(next_region)
            previous_width = current_upper - next_lower
            current_upper = next_lower
    except _PathFailure as exc:
        status = "numerical_failure"
        failure_message = str(exc)

    # Convert contiguous affine regions into knots.  In the unlikely event of
    # a numerical failure, retain the certified prefix and mark it non-exact.
    if not regions:
        coefficients = np.zeros((1, problem.p), dtype=float)
        lambdas = np.asarray([lambda_max])
        events: tuple[str, ...] = ("lambda_max",)
        conditions: list[float] = []
    else:
        lambda_list = [lambda_max]
        coefficient_list = [regions[0].z_at(lambda_max)[: problem.p]]
        conditions = [regions[0].condition]
        for region_number, region in enumerate(regions):
            lower = max(0.0, region.lower)
            beta_lower = region.z_at(lower)[: problem.p]
            if lower >= lambda_list[-1] - tolerance * lambda_scale:
                continue
            # Adjacent bases can pivot without changing the primal affine
            # segment.  Keep only actual coefficient-direction breakpoints.
            if region_number + 1 < len(regions):
                left_slope = region.z_slope[: problem.p]
                right_slope = regions[region_number + 1].z_slope[: problem.p]
                if np.allclose(
                    left_slope,
                    right_slope,
                    rtol=100.0 * tolerance,
                    atol=100.0 * tolerance,
                ):
                    continue
            lambda_list.append(lower)
            coefficient_list.append(beta_lower)
            conditions.append(region.condition)
        lambdas = np.asarray(lambda_list, dtype=float)
        coefficients = np.asarray(coefficient_list, dtype=float)
        event_list = ["lambda_max"]
        event_list.extend("active_face_change" for _ in range(max(0, len(lambdas) - 2)))
        if lambdas[-1] <= tolerance * lambda_scale:
            lambdas[-1] = 0.0
            event_list.append("least_squares")
        elif len(lambdas) > 1:
            event_list.append("path_stopped")
        events = tuple(event_list)

    # Clamp the theoretically zero start, and independently solve the zero
    # endpoint so accumulated continuation error cannot leak into it.
    coefficients[0] = 0.0
    if lambdas[-1] == 0.0:
        coefficients[-1] = _solve_fixed_lambda(
            problem, 0.0, tolerance=tolerance, max_iter=oracle_max_iter
        )
    intercepts = problem.y_mean - coefficients @ problem.x_mean
    penalties = np.asarray([_penalty(problem, beta) for beta in coefficients])
    diagnostics = tuple(
        _kkt_diagnostic(
            problem,
            beta,
            float(lam),
            tolerance=tolerance,
            tie_tolerance=tie_tolerance,
        )
        for lam, beta in zip(lambdas, coefficients)
    )
    all_certified = all(bool(item["certified"]) for item in diagnostics)
    reached_zero = bool(lambdas[-1] == 0.0)
    exact = status == "complete" and reached_zero and all_certified
    if status == "complete" and not all_certified:
        status = "kkt_failure"
    elif status == "complete" and not reached_zero:
        status = "numerical_failure"
    metadata: dict[str, object] = {
        "lambda_max": lambda_max,
        "n_regions": len(regions),
        "rank": rank,
        "condition": float(np.linalg.cond(problem.H)),
        "max_kkt_residual": float(
            max(float(item["kkt_residual"]) for item in diagnostics)
        ),
        "max_active_kkt_condition": float(max(conditions, default=np.nan)),
        "probe_attempts": probe_attempts,
        "weight_sum": weight_sum,
        "normalized_weight_objective": True,
        "penalty": "sum_group_linf",
    }
    if failure_message is not None:
        metadata["failure_message"] = failure_message
    return RegularizationPath(
        lambdas=lambdas,
        coefficients=coefficients,
        intercepts=intercepts,
        penalties=penalties,
        events=events,
        diagnostics=diagnostics,
        method="hicap-homotopy",
        exact=exact,
        status=status,
        metadata=metadata,
    )


__all__ = ["hicap_regression_path"]
