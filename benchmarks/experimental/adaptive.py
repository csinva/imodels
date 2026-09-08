"""Historical adaptive APA-APG2 experiment, outside the imodels runtime.

The exact squared-loss infinity-CAP path is piecewise affine in lambda.  This
module approximates that path by solving interval midpoints and comparing them
with the affine chord between already solved endpoints.  It remains a
numerical sampled path: midpoint validation cannot prove that an interval
contains no pair of compensating knots, so ``exact`` is always false.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
from numpy.linalg import norm

from imodels.tree.sparse_pruning.optimization.apa_point import (
    _hiCAP_regression_quadratic,
    hiCAP_regression,
)
from imodels.tree.sparse_pruning.optimization._result import RegularizationPath
from imodels.tree.sparse_pruning.optimization._quadratic import _make_quadratic_regression_loss
from imodels.tree.sparse_pruning.optimization.apa import PointSolver
from imodels.tree.sparse_pruning.optimization._problem import (
    _diagonal_weighted_least_squares,
    _group_penalty,
    _prepare_problem,
    _weighted_least_squares,
)
from imodels.tree.sparse_pruning.optimization.diagnostics import (
    group_linf_lambda_max,
    group_linf_quadratic_kkt_diagnostic,
)


@dataclass
class _AdaptivePoint:
    lam: float
    beta: np.ndarray
    diagnostic: dict[str, Any]
    event: str


def _finite_positive(value: Any, name: str) -> float:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, float, np.number))
        or not np.isfinite(value)
        or float(value) <= 0
    ):
        raise ValueError(f"{name} must be a positive finite scalar")
    return float(value)


def _finite_nonnegative(value: Any, name: str) -> float:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, float, np.number))
        or not np.isfinite(value)
        or float(value) < 0
    ):
        raise ValueError(f"{name} must be a nonnegative finite scalar")
    return float(value)


def _positive_integer(value: Any, name: str) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or int(value) < 1
    ):
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _secant_prediction(
    target_lambda: float,
    previous: _AdaptivePoint,
    before_previous: _AdaptivePoint | None,
) -> tuple[np.ndarray, str]:
    if before_previous is None:
        return previous.beta.copy(), "previous"
    denominator = previous.lam - before_previous.lam
    if denominator == 0:
        return previous.beta.copy(), "previous"
    prediction = previous.beta + (
        (target_lambda - previous.lam)
        / denominator
        * (previous.beta - before_previous.beta)
    )
    prediction_step = float(norm(prediction - previous.beta))
    scale = max(1.0, float(norm(previous.beta)))
    if not np.all(np.isfinite(prediction)) or prediction_step > 10.0 * scale:
        return previous.beta.copy(), "previous_fallback"
    return prediction, "secant"


def apa_apg_adaptive_regression_path(
    X: np.ndarray,
    y: np.ndarray,
    groups: Iterable[np.ndarray],
    *,
    sample_weight: np.ndarray | None = None,
    fit_intercept: bool = True,
    beta_init: np.ndarray | None = None,
    lambda_max: float | None = None,
    minimum_ratio: float = 1e-3,
    initial_points: int = 8,
    coefficient_tolerance: float = 1e-2,
    objective_tolerance: float = 1e-4,
    support_tolerance: float | None = 1e-3,
    lambda_tolerance: float = 1e-8,
    max_points: int = 257,
    max_depth: int = 12,
    predictor: str = "secant",
    gamma1: float = 1.0,
    a: float = 1.0,
    max_iter: int = 2_000,
    tol: float = 1e-6,
    cache_quadratic: bool = True,
    assume_diagonal_gram: bool = False,
    kkt_tolerance: float | None = None,
    kkt_face_tolerance: float = 1e-6,
    verbose: bool = False,
    point_solver: PointSolver | None = None,
) -> RegularizationPath:
    """Compute a quality-controlled approximate infinity-CAP path.

    A small geometric anchor grid is solved first. Each interval is probed at
    its arithmetic lambda midpoint, because the target path is affine in
    lambda between knots. Intervals are recursively split when the solved
    midpoint differs from the endpoint chord in coefficients, objective, or
    thresholded support.

    ``kkt_tolerance`` optionally runs an independent fixed-point
    subdifferential LP at every stored point. ``point_solutions_certified`` is
    true only when every point passes a strict-face KKT check. If
    ``kkt_face_tolerance`` is positive, a separate relaxed-face check may accept
    near-ties for refinement and is reported as
    ``point_solutions_kkt_accepted``; it is not called a certificate. Neither
    check nor midpoint refinement certifies that every breakpoint was found,
    so the returned path always has ``exact=False``. ``predictor`` controls the
    descending initial anchor solves. Refinement points use the upper endpoint
    as an independent warm start, rather than the chord whose accuracy is being
    tested.
    """

    if not isinstance(fit_intercept, (bool, np.bool_)):
        raise ValueError("fit_intercept must be a boolean")
    if not isinstance(cache_quadratic, (bool, np.bool_)):
        raise ValueError("cache_quadratic must be a boolean")
    if not isinstance(assume_diagonal_gram, (bool, np.bool_)):
        raise ValueError("assume_diagonal_gram must be a boolean")
    if assume_diagonal_gram and (
        not cache_quadratic or point_solver is not None
    ):
        raise ValueError(
            "assume_diagonal_gram=True requires cache_quadratic=True and the "
            "default regression point solver"
        )
    if predictor not in {"previous", "secant"}:
        raise ValueError("predictor must be 'previous' or 'secant'")
    initial_points = _positive_integer(initial_points, "initial_points")
    max_points = _positive_integer(max_points, "max_points")
    max_depth = _positive_integer(max_depth, "max_depth")
    max_iter = _positive_integer(max_iter, "max_iter")
    if initial_points < 2:
        raise ValueError("initial_points must be at least two")
    if max_points < initial_points + 1:
        raise ValueError(
            "max_points must accommodate all anchors and lambda zero"
        )
    if not np.isfinite(minimum_ratio) or not 0 < minimum_ratio < 1:
        raise ValueError("minimum_ratio must lie strictly between zero and one")
    coefficient_tolerance = _finite_positive(
        coefficient_tolerance, "coefficient_tolerance"
    )
    objective_tolerance = _finite_positive(
        objective_tolerance, "objective_tolerance"
    )
    if support_tolerance is not None:
        support_tolerance = _finite_positive(
            support_tolerance, "support_tolerance"
        )
    lambda_tolerance = _finite_positive(lambda_tolerance, "lambda_tolerance")
    gamma1 = _finite_positive(gamma1, "gamma1")
    a = _finite_positive(a, "a")
    tol = _finite_positive(tol, "tol")
    kkt_face_tolerance = _finite_nonnegative(
        kkt_face_tolerance, "kkt_face_tolerance"
    )
    if kkt_tolerance is not None:
        kkt_tolerance = _finite_positive(kkt_tolerance, "kkt_tolerance")

    X, y, private_groups, initial_beta, weights, ord_value = _prepare_problem(
        X=X,
        y=y,
        groups=groups,
        beta_init=beta_init,
        sample_weight=sample_weight,
        ord="inf",
    )
    del ord_value
    normalized_weights = weights / float(weights.sum())
    if fit_intercept:
        x_mean = normalized_weights @ X
        y_mean = float(normalized_weights @ y)
        X_work = X - x_mean
        y_work = y - y_mean
    else:
        x_mean = np.zeros(X.shape[1], dtype=float)
        y_mean = 0.0
        X_work = X
        y_work = y

    # Quadratic statistics are useful for cached point iterations and KKT
    # diagnostics. A direct, non-KKT run must honor ``cache_quadratic=False``
    # and avoid allocating them.
    needs_quadratic = bool(
        (point_solver is None and cache_quadratic) or kkt_tolerance is not None
    )
    quadratic_loss = (
        _make_quadratic_regression_loss(
            X_work,
            y_work,
            weights,
            assume_diagonal_gram=assume_diagonal_gram,
        )
        if needs_quadratic
        else None
    )
    linear = (
        quadratic_loss.linear
        if quadratic_loss is not None
        else X_work.T @ (normalized_weights * y_work)
    )

    weighted_X = X_work * np.sqrt(normalized_weights)[:, np.newaxis]
    if quadratic_loss is not None and quadratic_loss.diagonal is not None:
        gram_diagonal = np.maximum(quadratic_loss.diagonal, 0.0)
        singular_values = np.sqrt(gram_diagonal)
    else:
        singular_values = np.linalg.svd(weighted_X, compute_uv=False)
    largest_singular_value = float(np.max(singular_values, initial=0.0))
    rank_tolerance = max(weighted_X.shape) * np.finfo(float).eps * (
        largest_singular_value
    )
    design_rank = int(np.count_nonzero(singular_values > rank_tolerance))
    coefficient_path_unique = design_rank == X_work.shape[1]

    zero_solution_lambda = group_linf_lambda_max(
        linear, private_groups, tolerance=tol
    )
    automatic_lambda_max = lambda_max is None
    if automatic_lambda_max:
        if not np.isfinite(zero_solution_lambda):
            raise ValueError(
                "automatic lambda_max is infinite because a feature with a "
                "nonzero loss score is not covered by any penalty group"
            )
        lambda_max_value = zero_solution_lambda
    else:
        if (
            isinstance(lambda_max, (bool, np.bool_))
            or not isinstance(lambda_max, (int, float, np.number))
            or not np.isfinite(lambda_max)
            or float(lambda_max) < 0
        ):
            raise ValueError("lambda_max must be a nonnegative finite scalar")
        lambda_max_value = float(lambda_max)
    lambda_comparison_tolerance = (
        64.0
        * np.finfo(float).eps
        * max(
            abs(lambda_max_value),
            abs(zero_solution_lambda)
            if np.isfinite(zero_solution_lambda)
            else 0.0,
            np.finfo(float).tiny,
        )
    )
    upper_endpoint_is_zero = bool(
        np.isfinite(zero_solution_lambda)
        and lambda_max_value + lambda_comparison_tolerance
        >= zero_solution_lambda
    )

    def loss(beta: np.ndarray) -> float:
        if quadratic_loss is not None:
            return quadratic_loss.loss(beta)
        residual = X_work @ beta - y_work
        return 0.5 * float(normalized_weights @ (residual**2))

    def reduced_objective_terms(
        lam: float, beta: np.ndarray
    ) -> tuple[float, float]:
        if quadratic_loss is not None:
            quadratic_term = 0.5 * float(
                beta @ quadratic_loss.matvec(beta)
            )
        else:
            prediction = X_work @ beta
            quadratic_term = 0.5 * float(
                normalized_weights @ (prediction**2)
            )
        linear_term = float(linear @ beta)
        penalty_term = lam * _group_penalty(beta, private_groups, "inf")
        value = quadratic_term - linear_term + penalty_term
        scale = (
            abs(quadratic_term)
            + abs(linear_term)
            + abs(penalty_term)
        )
        return value, scale

    def add_kkt(diagnostic: dict[str, Any], lam: float, beta: np.ndarray) -> None:
        if kkt_tolerance is None:
            return
        if quadratic_loss is None:  # pragma: no cover - guarded above
            raise RuntimeError("KKT diagnostics require quadratic statistics")
        gram_or_diagonal = (
            quadratic_loss.diagonal
            if quadratic_loss.gram is None
            else quadratic_loss.gram
        )
        kkt = group_linf_quadratic_kkt_diagnostic(
            gram_or_diagonal,
            quadratic_loss.linear,
            private_groups,
            lam,
            beta,
            tolerance=kkt_tolerance,
            face_tolerance=kkt_face_tolerance,
        )
        diagnostic.update({f"kkt_{key}": value for key, value in kkt.items()})

    def solve_positive(
        lam: float,
        start: np.ndarray,
        *,
        source: str,
        depth: int,
        init_method: str,
        parent_interval: tuple[float, float] | None = None,
    ) -> _AdaptivePoint:
        solver = hiCAP_regression if point_solver is None else point_solver
        if point_solver is None and cache_quadratic:
            if quadratic_loss is None:  # pragma: no cover - guarded above
                raise RuntimeError("quadratic cache was not prepared")
            result = _hiCAP_regression_quadratic(
                quadratic_loss,
                private_groups,
                lam,
                start,
                gamma1,
                a,
                max_iter,
                tol,
                "inf",
                verbose,
            )
        else:
            result = solver(
                X=X_work,
                y=y_work,
                groups=private_groups,
                lam=lam,
                beta_init=start,
                gamma1=gamma1,
                a=a,
                max_iter=max_iter,
                tol=tol,
                ord="inf",
                verbose=verbose,
                sample_weight=weights,
                return_info=True,
            )
        if not isinstance(result, tuple) or len(result) != 2:
            raise TypeError(
                "point_solver must return (coefficients, diagnostics) when "
                "return_info=True"
            )
        beta, raw_diagnostic = result
        beta = np.asarray(beta, dtype=float)
        if beta.shape != initial_beta.shape or not np.all(np.isfinite(beta)):
            raise FloatingPointError(
                "APA-APG2 point solver returned invalid coefficients"
            )
        diagnostic = dict(raw_diagnostic)
        residual = X_work @ beta - y_work
        direct_loss = 0.5 * float(normalized_weights @ (residual**2))
        penalty = _group_penalty(beta, private_groups, "inf")
        diagnostic.update(
            {
                "lambda": lam,
                "source": source,
                "depth": depth,
                "init_method": init_method,
                "initial_norm": float(norm(start)),
                "loss": direct_loss,
                "unscaled_penalty": penalty,
                "objective": direct_loss + lam * penalty,
                "parent_interval": parent_interval,
            }
        )
        add_kkt(diagnostic, lam, beta)
        return _AdaptivePoint(lam, beta.copy(), diagnostic, source)

    if quadratic_loss is not None and quadratic_loss.diagonal is not None:
        beta_zero, zero_diagnostic = _diagonal_weighted_least_squares(
            quadratic_loss, initial_beta
        )
    else:
        beta_zero, zero_diagnostic = _weighted_least_squares(
            X_work, y_work, weights, initial_beta
        )
    coefficient_reference_scale = max(
        float(norm(beta_zero)), np.finfo(float).tiny
    )
    support_reference_scale = max(
        float(norm(beta_zero, ord=np.inf)), np.finfo(float).tiny
    )
    effective_support_tolerance = (
        None
        if support_tolerance is None
        else support_tolerance * support_reference_scale
    )

    if lambda_max_value == 0.0:
        diagnostic = dict(zero_diagnostic)
        diagnostic.update({"lambda": 0.0, "source": "least_squares"})
        add_kkt(diagnostic, 0.0, beta_zero)
        intercept = y_mean - float(x_mean @ beta_zero)
        kkt_certified = (
            kkt_tolerance is not None
            and diagnostic.get("kkt_certified") is True
        )
        kkt_accepted = (
            kkt_tolerance is not None
            and diagnostic.get("kkt_accepted") is True
        )
        point_accuracy_accepted = (
            kkt_accepted if kkt_tolerance is not None else True
        )
        return RegularizationPath(
            lambdas=np.asarray([0.0]),
            coefficients=beta_zero[None, :],
            intercepts=(np.asarray([intercept]) if fit_intercept else None),
            penalties=np.asarray(
                [_group_penalty(beta_zero, private_groups, "inf")]
            ),
            events=("least_squares",),
            diagnostics=(diagnostic,),
            method="apa-apg2-adaptive-warm-start",
            exact=False,
            status=(
                "nonunique_design"
                if not coefficient_path_unique
                else "complete" if point_accuracy_accepted else "partial"
            ),
            metadata={
                "adaptive_tolerance_met": True,
                "point_solutions_certified": kkt_certified,
                "point_solutions_kkt_accepted": kkt_accepted,
                "kkt_acceptance_mode": (
                    "not_run"
                    if kkt_tolerance is None
                    else diagnostic.get("kkt_acceptance_mode", "rejected")
                ),
                "kkt_face_relaxation_used": bool(
                    kkt_tolerance is not None and kkt_face_tolerance > 0.0
                ),
                "point_accuracy_accepted": point_accuracy_accepted,
                "lambda_max": 0.0,
                "zero_solution_lambda": zero_solution_lambda,
                "n_initial_points": 1,
                "n_refinement_points": 0,
                "design_rank": design_rank,
                "n_features": int(X_work.shape[1]),
                "coefficient_path_unique": coefficient_path_unique,
                "quadratic_backend": (
                    None if quadratic_loss is None else quadratic_loss.backend
                ),
                "assumed_diagonal_gram": (
                    False
                    if quadratic_loss is None
                    else quadratic_loss.assumed_diagonal_gram
                ),
            },
        )

    positive_anchors = np.geomspace(
        lambda_max_value,
        lambda_max_value * minimum_ratio,
        num=initial_points,
    )
    points: dict[float, _AdaptivePoint] = {}
    if upper_endpoint_is_zero:
        zero_loss = loss(np.zeros_like(initial_beta))
        start_diagnostic: dict[str, Any] = {
            "converged": True,
            "n_iter": 0,
            "lambda": lambda_max_value,
            "source": "lambda_max",
            "depth": 0,
            "init_method": "analytic_zero",
            "initial_norm": 0.0,
            "loss": zero_loss,
            "unscaled_penalty": 0.0,
            "objective": zero_loss,
            "parent_interval": None,
        }
        zero_beta = np.zeros_like(initial_beta)
        add_kkt(start_diagnostic, lambda_max_value, zero_beta)
        points[float(lambda_max_value)] = _AdaptivePoint(
            float(lambda_max_value), zero_beta, start_diagnostic, "lambda_max"
        )
    else:
        points[float(lambda_max_value)] = solve_positive(
            float(lambda_max_value),
            initial_beta,
            source="anchor",
            depth=0,
            init_method="user",
        )

    previous = points[float(lambda_max_value)]
    before_previous: _AdaptivePoint | None = None
    for raw_lambda in positive_anchors[1:]:
        lam = float(raw_lambda)
        if predictor == "secant":
            start, init_method = _secant_prediction(
                lam, previous, before_previous
            )
        else:
            start, init_method = previous.beta.copy(), "previous"
        point = solve_positive(
            lam,
            start,
            source="anchor",
            depth=0,
            init_method=init_method,
        )
        points[lam] = point
        before_previous, previous = previous, point

    zero_diagnostic = dict(zero_diagnostic)
    zero_penalty = _group_penalty(beta_zero, private_groups, "inf")
    zero_diagnostic.update(
        {
            "lambda": 0.0,
            "source": "least_squares",
            "depth": 0,
            "init_method": "previous",
            "unscaled_penalty": zero_penalty,
            "parent_interval": None,
        }
    )
    add_kkt(zero_diagnostic, 0.0, beta_zero)
    points[0.0] = _AdaptivePoint(
        0.0, beta_zero.copy(), zero_diagnostic, "least_squares"
    )

    anchors = sorted(points, reverse=True)
    pending = deque(
        (anchors[index], anchors[index + 1], 0)
        for index in range(len(anchors) - 1)
    )
    accepted_intervals: list[dict[str, Any]] = []
    unresolved_intervals: list[dict[str, Any]] = []
    refinement_points = 0
    maximum_coefficient_error = 0.0
    maximum_objective_error = 0.0

    while pending:
        upper, lower, depth = pending.popleft()
        width = upper - lower
        width_limit = lambda_tolerance * lambda_max_value
        if width <= width_limit:
            unresolved_intervals.append(
                {
                    "upper": upper,
                    "lower": lower,
                    "depth": depth,
                    "reason": "lambda_resolution",
                }
            )
            continue
        midpoint = 0.5 * (upper + lower)
        if midpoint in points or midpoint == upper or midpoint == lower:
            unresolved_intervals.append(
                {
                    "upper": upper,
                    "lower": lower,
                    "depth": depth,
                    "reason": "floating_point_resolution",
                }
            )
            continue
        if len(points) >= max_points:
            unresolved_intervals.append(
                {
                    "upper": upper,
                    "lower": lower,
                    "depth": depth,
                    "reason": "max_points",
                }
            )
            unresolved_intervals.extend(
                {
                    "upper": pending_upper,
                    "lower": pending_lower,
                    "depth": pending_depth,
                    "reason": "max_points",
                }
                for pending_upper, pending_lower, pending_depth in pending
            )
            pending.clear()
            break

        upper_point = points[upper]
        lower_point = points[lower]
        upper_weight = (midpoint - lower) / width
        chord = (
            upper_weight * upper_point.beta
            + (1.0 - upper_weight) * lower_point.beta
        )
        midpoint_point = solve_positive(
            midpoint,
            upper_point.beta,
            source="refinement",
            depth=depth + 1,
            init_method="upper_endpoint",
            parent_interval=(upper, lower),
        )
        points[midpoint] = midpoint_point
        refinement_points += 1

        difference = midpoint_point.beta - chord
        coefficient_error = float(norm(difference)) / max(
            np.sqrt(np.finfo(float).eps) * coefficient_reference_scale,
            float(norm(midpoint_point.beta)),
            float(norm(chord)),
        )
        midpoint_reduced_objective, midpoint_objective_scale = (
            reduced_objective_terms(midpoint, midpoint_point.beta)
        )
        chord_reduced_objective, chord_objective_scale = (
            reduced_objective_terms(midpoint, chord)
        )
        signed_objective_excess = (
            chord_reduced_objective - midpoint_reduced_objective
        )
        objective_scale = max(
            midpoint_objective_scale,
            chord_objective_scale,
            np.finfo(float).tiny,
        )
        objective_error = max(0.0, signed_objective_excess) / objective_scale
        solver_objective_error = (
            max(0.0, -signed_objective_excess) / objective_scale
        )
        if effective_support_tolerance is None:
            support_match = True
        else:
            midpoint_support = (
                np.abs(midpoint_point.beta) > effective_support_tolerance
            )
            chord_support = np.abs(chord) > effective_support_tolerance
            support_match = bool(
                np.array_equal(midpoint_support, chord_support)
            )
        midpoint_kkt_certified = (
            None
            if kkt_tolerance is None
            else midpoint_point.diagnostic.get("kkt_certified") is True
        )
        midpoint_kkt_accepted = (
            True
            if kkt_tolerance is None
            else midpoint_point.diagnostic.get("kkt_accepted") is True
        )
        maximum_coefficient_error = max(
            maximum_coefficient_error, coefficient_error
        )
        maximum_objective_error = max(maximum_objective_error, objective_error)
        passes = (
            coefficient_error <= coefficient_tolerance
            and objective_error <= objective_tolerance
            and support_match
            and midpoint_kkt_accepted
            and solver_objective_error <= objective_tolerance
        )
        midpoint_point.diagnostic.update(
            {
                "chord_coefficient_error": coefficient_error,
                "chord_objective_error": objective_error,
                "signed_chord_objective_excess": signed_objective_excess,
                "solver_objective_error": solver_objective_error,
                "chord_support_match": support_match,
                "midpoint_kkt_certified": midpoint_kkt_certified,
                "midpoint_kkt_accepted": midpoint_kkt_accepted,
                "interval_validated": passes,
            }
        )
        interval = {
            "upper": upper,
            "lower": lower,
            "midpoint": midpoint,
            "depth": depth + 1,
            "coefficient_error": coefficient_error,
            "objective_error": objective_error,
            "signed_chord_objective_excess": signed_objective_excess,
            "solver_objective_error": solver_objective_error,
            "support_match": support_match,
            "midpoint_kkt_certified": midpoint_kkt_certified,
            "midpoint_kkt_accepted": midpoint_kkt_accepted,
        }
        if passes:
            interval["reason"] = "midpoint_validated"
            accepted_intervals.append(interval)
        elif not midpoint_kkt_accepted:
            interval["reason"] = "midpoint_kkt_failure"
            unresolved_intervals.append(interval)
        elif solver_objective_error > objective_tolerance:
            interval["reason"] = "midpoint_solver_objective_failure"
            unresolved_intervals.append(interval)
        elif depth + 1 >= max_depth:
            interval["reason"] = "max_depth"
            unresolved_intervals.append(interval)
        else:
            pending.append((upper, midpoint, depth + 1))
            pending.append((midpoint, lower, depth + 1))

    ordered_points = [points[lam] for lam in sorted(points, reverse=True)]
    lambdas = np.asarray([point.lam for point in ordered_points])
    coefficients = np.vstack([point.beta for point in ordered_points])
    penalties = np.asarray(
        [_group_penalty(beta, private_groups, "inf") for beta in coefficients]
    )
    intercepts = (
        y_mean - coefficients @ x_mean if fit_intercept else None
    )
    diagnostics = tuple(point.diagnostic for point in ordered_points)
    kkt_certified = (
        kkt_tolerance is not None
        and all(item.get("kkt_certified") is True for item in diagnostics)
    )
    kkt_accepted = (
        kkt_tolerance is not None
        and all(item.get("kkt_accepted") is True for item in diagnostics)
    )
    solver_converged = all(
        item.get("converged") is True for item in diagnostics
    )
    adaptive_tolerance_met = not unresolved_intervals
    point_solutions_certified = bool(kkt_certified)
    point_solutions_kkt_accepted = bool(kkt_accepted)
    point_accuracy_accepted = (
        kkt_accepted if kkt_tolerance is not None else solver_converged
    )
    kkt_acceptance_mode = (
        "not_run"
        if kkt_tolerance is None
        else "strict"
        if point_solutions_certified
        else "relaxed_face"
        if point_solutions_kkt_accepted
        else "rejected"
    )
    metadata: dict[str, Any] = {
        "problem": "regression",
        "ord": "inf",
        "fit_intercept": bool(fit_intercept),
        "lambda_max": lambda_max_value,
        "zero_solution_lambda": zero_solution_lambda,
        "lambda_minimum_ratio": float(minimum_ratio),
        "validation_scheme": "arithmetic_midpoint_chord",
        "coefficient_tolerance": coefficient_tolerance,
        "objective_tolerance": objective_tolerance,
        "support_tolerance": support_tolerance,
        "effective_support_tolerance": effective_support_tolerance,
        "support_tolerance_scale": (
            None
            if support_tolerance is None
            else "relative_to_linf_least_squares"
        ),
        "lambda_tolerance": lambda_tolerance,
        "initial_points": initial_points,
        "n_initial_points": initial_points + 1,
        "n_refinement_points": refinement_points,
        "max_points": max_points,
        "max_depth": max_depth,
        "predictor": predictor,
        "cached_sufficient_statistics": bool(
            point_solver is None and cache_quadratic
        ),
        "quadratic_statistics_available": quadratic_loss is not None,
        "quadratic_backend": (
            None if quadratic_loss is None else quadratic_loss.backend
        ),
        "assumed_diagonal_gram": (
            False
            if quadratic_loss is None
            else quadratic_loss.assumed_diagonal_gram
        ),
        "gram_max_off_diagonal": (
            None if quadratic_loss is None else quadratic_loss.max_off_diagonal
        ),
        "gram_max_off_diagonal_correlation": (
            None
            if quadratic_loss is None
            else quadratic_loss.max_off_diagonal_correlation
        ),
        "design_rank": design_rank,
        "n_features": int(X_work.shape[1]),
        "coefficient_path_unique": coefficient_path_unique,
        "adaptive_tolerance_met": adaptive_tolerance_met,
        "point_solutions_certified": point_solutions_certified,
        "point_solutions_kkt_accepted": point_solutions_kkt_accepted,
        "kkt_acceptance_mode": kkt_acceptance_mode,
        "kkt_face_relaxation_used": bool(
            kkt_tolerance is not None and kkt_face_tolerance > 0.0
        ),
        "point_accuracy_accepted": point_accuracy_accepted,
        "solver_stopping_tests_met": solver_converged,
        "kkt_tolerance": kkt_tolerance,
        "accepted_intervals": tuple(accepted_intervals),
        "unresolved_intervals": tuple(unresolved_intervals),
        "maximum_probed_coefficient_error": maximum_coefficient_error,
        "maximum_probed_objective_error": maximum_objective_error,
    }
    return RegularizationPath(
        lambdas=lambdas,
        coefficients=coefficients,
        intercepts=intercepts,
        penalties=penalties,
        events=tuple(point.event for point in ordered_points),
        diagnostics=diagnostics,
        method="apa-apg2-adaptive-warm-start",
        exact=False,
        status=(
            "nonunique_design"
            if not coefficient_path_unique
            else (
                "complete"
                if adaptive_tolerance_met and point_accuracy_accepted
                else "partial"
            )
        ),
        metadata=metadata,
    )


__all__ = ["apa_apg_adaptive_regression_path"]
