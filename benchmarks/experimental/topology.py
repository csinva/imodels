"""Historical thresholded topology bisection, retained for benchmark comparisons.

This approximate traversal is not part of the imodels estimator runtime.
Use the exact structural path in imodels.tree.sparse_pruning for pruning.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from imodels.tree.sparse_pruning.optimization._result import RegularizationPath
from imodels.tree.sparse_pruning.optimization._problem import _diagonal_weighted_least_squares
from imodels.tree.sparse_pruning.optimization.topology import _nonnegative_scalar
from imodels.tree.sparse_pruning.optimization.proximal import (
    _positive_integer, _positive_scalar, _prepare_regression, _zero_penalty_certificate,
)


@dataclass(frozen=True)
class _TopologyPoint:
    lam: float
    beta: np.ndarray
    intercept: float
    support: tuple[int, ...]
    topology: tuple[int, ...]
    diagnostic: dict[str, Any]


def _subset(left: tuple[int, ...], right: tuple[int, ...]) -> bool:
    """Whether sorted integer tuple ``left`` is a subset of ``right``."""
    return set(left).issubset(right)


def _automatic_lambda_upper(problem: Any) -> float:
    """Safe all-zero upper lambda using the positive laminar root groups."""
    prox = problem.solver.euclidean_prox
    linear = np.asarray(problem.quadratic.linear)
    covered = np.zeros(linear.size, dtype=bool)
    bounds: list[float] = []
    for position in np.flatnonzero(prox.parents < 0):
        group = prox.groups[int(position)]
        weight = float(prox.group_weights[int(position)])
        relevant = np.any(linear[group] != 0.0)
        if relevant and weight <= 0.0:
            raise ValueError(
                "automatic lambda_upper requires a positive weight on every "
                "root group carrying a nonzero loss score"
            )
        covered[group] = True
        if weight > 0.0:
            bounds.append(float(np.sum(np.abs(linear[group]))) / weight)
    if np.any((~covered) & (linear != 0.0)):
        raise ValueError(
            "automatic lambda_upper is unavailable because a feature with a "
            "nonzero loss score is not covered by a root penalty group"
        )
    return max(bounds, default=0.0)


def laminar_group_linf_topology_path(
    X: np.ndarray,
    y: np.ndarray,
    groups: Iterable[np.ndarray],
    *,
    sample_weight: np.ndarray | None = None,
    fit_intercept: bool = True,
    group_weights: float | Sequence[float] | None = None,
    lambda_upper: float | None = None,
    support_tolerance: float = 0.0,
    lambda_tolerance: float = 1e-6,
    max_evaluations: int = 10_000,
    solver_tolerance: float = 1e-10,
    assume_diagonal_gram: bool = False,
) -> RegularizationPath:
    """Bracket every thresholded tree-topology event on a diagonal path.

    Parameters
    ----------
    groups:
        A laminar family.  In the tree-pruning use case there is one group per
        internal node containing that node and all its internal descendants.
    lambda_upper:
        Optional upper end of the traversed interval.  The automatic value is
        a safe all-zero bound obtained from the root groups.  A supplied value
        may start inside the path.
    support_tolerance:
        A coefficient is active when its absolute value is strictly greater
        than this fixed threshold.  The returned topology contains every
        group with at least one active descendant coefficient, matching the
        hierarchy-safe pruning rule.
    lambda_tolerance:
        Event brackets have width at most this fraction of ``lambda_upper``.

    Notes
    -----
    The routine is available only when the weighted Gram matrix is positive
    diagonal (verified numerically, unless ``assume_diagonal_gram=True``).
    That restriction is essential: support can leave and re-enter for a
    general correlated design.

    ``path.exact`` is false because event locations are brackets and
    coefficient-slope knots with unchanged topology are intentionally omitted.
    ``metadata['topology_event_coverage_complete']`` reports whether all
    changes were localized to the requested lambda resolution; it does not
    claim that several events inside one terminal bracket were separated.
    """
    if not isinstance(fit_intercept, (bool, np.bool_)):
        raise ValueError("fit_intercept must be a boolean")
    if not isinstance(assume_diagonal_gram, (bool, np.bool_)):
        raise ValueError("assume_diagonal_gram must be a boolean")
    support_tolerance = _nonnegative_scalar(
        support_tolerance, "support_tolerance"
    )
    lambda_tolerance = _positive_scalar(lambda_tolerance, "lambda_tolerance")
    solver_tolerance = _positive_scalar(solver_tolerance, "solver_tolerance")
    max_evaluations = _positive_integer(max_evaluations, "max_evaluations")
    if max_evaluations < 2:
        raise ValueError("max_evaluations must be at least two")

    problem = _prepare_regression(
        X,
        y,
        groups,
        sample_weight=sample_weight,
        fit_intercept=bool(fit_intercept),
        beta_init=None,
        group_weights=group_weights,
        max_iter=1,
        tolerance=solver_tolerance,
        assume_diagonal_gram=bool(assume_diagonal_gram),
        restart=True,
    )
    diagonal = problem.quadratic.diagonal
    if (
        diagonal is None
        or not np.all(np.asarray(diagonal) > 0.0)
        or not problem.solver.uses_diagonal_closed_form
    ):
        raise ValueError(
            "topology traversal requires a positive diagonal weighted Gram "
            "matrix"
        )

    if lambda_upper is None:
        upper = _automatic_lambda_upper(problem)
        automatic_upper = True
    else:
        upper = _nonnegative_scalar(lambda_upper, "lambda_upper")
        automatic_upper = False
    lambda_scale = max(upper, np.finfo(float).tiny)
    width_limit = lambda_tolerance * lambda_scale

    cache: dict[float, _TopologyPoint] = {}

    def solve(lam: float) -> _TopologyPoint:
        key = float(lam)
        cached = cache.get(key)
        if cached is not None:
            return cached
        if len(cache) >= max_evaluations:
            raise RuntimeError("maximum topology-path evaluations reached")
        if key == 0.0:
            beta, raw_diagnostic = _diagonal_weighted_least_squares(
                problem.quadratic, problem.initial_beta
            )
            diagnostic = dict(raw_diagnostic)
            diagnostic.update(
                _zero_penalty_certificate(
                    problem.quadratic, beta, solver_tolerance
                )
            )
        else:
            result = problem.solver.solve(key, problem.initial_beta)
            beta, diagnostic = result.beta, dict(result.diagnostic)
        active = np.abs(beta) > support_tolerance
        support = tuple(int(value) for value in np.flatnonzero(active))
        topology = tuple(
            group_number
            for group_number, group in enumerate(problem.groups)
            if np.any(active[group])
        )
        diagnostic.update(
            {
                "lambda": key,
                "intercept": problem.intercept(beta),
                "support": support,
                "topology": topology,
            }
        )
        point = _TopologyPoint(
            lam=key,
            beta=np.asarray(beta).copy(),
            intercept=problem.intercept(beta),
            support=support,
            topology=topology,
            diagnostic=diagnostic,
        )
        cache[key] = point
        return point

    upper_point = solve(upper)
    lower_point = solve(0.0) if upper != 0.0 else upper_point
    if automatic_upper and upper_point.topology:
        raise RuntimeError("automatic lambda upper bound did not produce zero")
    if not _subset(upper_point.topology, lower_point.topology):
        raise RuntimeError(
            "diagonal topology monotonicity failed at the path endpoints"
        )

    pending = deque([(upper_point, lower_point, 0)])
    constant_intervals: list[dict[str, Any]] = []
    event_brackets: list[dict[str, Any]] = []
    unresolved_intervals: list[dict[str, Any]] = []
    maximum_depth = 0

    while pending:
        high, low, depth = pending.popleft()
        maximum_depth = max(maximum_depth, depth)
        if high.topology == low.topology:
            constant_intervals.append(
                {
                    "upper": high.lam,
                    "lower": low.lam,
                    "topology": high.topology,
                    "reason": "equal_endpoint_topology",
                }
            )
            continue
        if not _subset(high.topology, low.topology):
            unresolved_intervals.append(
                {
                    "upper": high.lam,
                    "lower": low.lam,
                    "reason": "nonmonotone_numerical_signature",
                }
            )
            continue
        width = high.lam - low.lam
        if width <= width_limit or high.lam == low.lam:
            event_brackets.append(
                {
                    "upper": high.lam,
                    "lower": low.lam,
                    "width": width,
                    "upper_topology": high.topology,
                    "lower_topology": low.topology,
                    "activated_groups": tuple(
                        sorted(set(low.topology) - set(high.topology))
                    ),
                    "activated_features": tuple(
                        sorted(set(low.support) - set(high.support))
                    ),
                    "resolution_limited": True,
                }
            )
            continue
        if len(cache) >= max_evaluations:
            unresolved_intervals.append(
                {
                    "upper": high.lam,
                    "lower": low.lam,
                    "reason": "max_evaluations",
                }
            )
            unresolved_intervals.extend(
                {
                    "upper": pending_high.lam,
                    "lower": pending_low.lam,
                    "reason": "max_evaluations",
                }
                for pending_high, pending_low, _ in pending
                if pending_high.topology != pending_low.topology
            )
            pending.clear()
            break

        midpoint_lambda = 0.5 * (high.lam + low.lam)
        if midpoint_lambda in (high.lam, low.lam):
            unresolved_intervals.append(
                {
                    "upper": high.lam,
                    "lower": low.lam,
                    "reason": "floating_point_resolution",
                }
            )
            continue
        midpoint = solve(midpoint_lambda)
        if not (
            _subset(high.topology, midpoint.topology)
            and _subset(midpoint.topology, low.topology)
        ):
            unresolved_intervals.append(
                {
                    "upper": high.lam,
                    "lower": low.lam,
                    "midpoint": midpoint_lambda,
                    "reason": "nonmonotone_numerical_signature",
                }
            )
            continue
        if high.topology == midpoint.topology:
            constant_intervals.append(
                {
                    "upper": high.lam,
                    "lower": midpoint.lam,
                    "topology": high.topology,
                    "reason": "equal_endpoint_topology",
                }
            )
        else:
            pending.append((high, midpoint, depth + 1))
        if midpoint.topology == low.topology:
            constant_intervals.append(
                {
                    "upper": midpoint.lam,
                    "lower": low.lam,
                    "topology": low.topology,
                    "reason": "equal_endpoint_topology",
                }
            )
        else:
            pending.append((midpoint, low, depth + 1))

    # Keep the first solved point inside each newly encountered topology.  It
    # is an exact coefficient solution and a valid representative tree state;
    # event_brackets carry the more precise boundary information.
    ordered = sorted(cache.values(), key=lambda point: point.lam, reverse=True)
    representatives: list[_TopologyPoint] = []
    previous_topology: tuple[int, ...] | None = None
    for point in ordered:
        if point.topology != previous_topology:
            representatives.append(point)
            previous_topology = point.topology
    if representatives[-1].lam != 0.0:
        representatives.append(lower_point)

    point_certification = all(
        point.diagnostic.get("certified") is True for point in cache.values()
    )
    topology_event_coverage_complete = not unresolved_intervals
    status = (
        "complete"
        if topology_event_coverage_complete and point_certification
        else "partial"
    )
    return RegularizationPath(
        lambdas=np.asarray([point.lam for point in representatives]),
        coefficients=np.vstack([point.beta for point in representatives]),
        intercepts=np.asarray([point.intercept for point in representatives]),
        penalties=np.asarray(
            [problem.solver.penalty(point.beta) for point in representatives]
        ),
        events=tuple(
            "lambda_upper" if index == 0 else "topology_change"
            for index in range(len(representatives))
        ),
        diagnostics=tuple(point.diagnostic for point in representatives),
        method="laminar-diagonal-topology-traversal",
        exact=False,
        status=status,
        metadata={
            "problem": "regression",
            "ord": "inf",
            "fit_intercept": bool(fit_intercept),
            "path_scope": "thresholded_tree_topology_events",
            "coefficient_knots_enumerated": False,
            # Coverage means every change is contained in a returned narrow
            # bracket. Several events closer than the requested resolution
            # may share one bracket, so this is not exact state enumeration.
            "topology_event_coverage_complete": (
                topology_event_coverage_complete
            ),
            "topology_states_enumerated": False,
            "event_locations_exact": False,
            "event_lambda_bracket_tolerance": width_limit,
            "relative_lambda_tolerance": lambda_tolerance,
            "support_tolerance": support_tolerance,
            "lambda_upper": upper,
            "automatic_lambda_upper": automatic_upper,
            "n_point_evaluations": len(cache),
            "n_representative_topologies": len(representatives),
            "n_event_brackets": len(event_brackets),
            "maximum_bisection_depth": maximum_depth,
            "point_solutions_certified": point_certification,
            "event_brackets": tuple(
                sorted(event_brackets, key=lambda item: item["upper"], reverse=True)
            ),
            "constant_topology_intervals": tuple(constant_intervals),
            "unresolved_intervals": tuple(unresolved_intervals),
            "quadratic_backend": problem.quadratic.backend,
            "assumed_diagonal_gram": problem.quadratic.assumed_diagonal_gram,
            "monotonicity_guarantee": (
                "positive_diagonal_gram_fixed_threshold"
            ),
        },
    )


__all__ = ["laminar_group_linf_topology_path"]
