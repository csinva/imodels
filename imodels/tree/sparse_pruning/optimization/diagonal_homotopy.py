r"""Complete coefficient paths for positive diagonal descendant-tree hiCAP.

The point oracle is the exact laminar proximal composition. Its output
identifies a tree-isotonic face; affine primal and dual inequalities then
certify the *whole* interval of that face. Point probing is used to identify
the next face, never to substitute sampling for interval coverage.

This reference implementation rebuilds a face after each event, stores
explicit descendant memberships, and returns dense coefficient rows. It
uses neither generic QP/LP solvers nor a dense Gram/KKT factorization.
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ._result import RegularizationPath
from .topology import (
    _child_before_parent_tree,
    _positive_node_weights,
    tree_group_linf_exact_topology_path,
)
from .tree_prox import LaminarGroupLinfProx


_LD = np.longdouble
_EPS = np.finfo(float).eps
_LEPS = np.finfo(_LD).eps


class _PathFailure(RuntimeError):
    """An unresolved interval must not be reported as an exact path."""


def _real_vector(value, name):
    raw = np.asarray(value)
    if raw.ndim != 1 or raw.dtype.kind in "bc":
        raise ValueError(f"{name} must be a finite real vector")
    try:
        result = raw.astype(float, copy=True)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{name} must be a finite real vector") from exc
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite real vector")
    return result


@dataclass
class _Face:
    lower: float
    upper: float
    beta_constant: np.ndarray
    beta_slope: np.ndarray
    max_residual: float
    n_blocks: int
    lower_roundoff: float = 0.0
    upper_roundoff: float = 0.0

    def at(self, lam):
        return self.beta_constant + _LD(lam) * self.beta_slope


class _TreeProblem:
    def __init__(self, scores, diagonal, parents, weights, tolerance):
        self.parents = parents
        self.order, _ = _child_before_parent_tree(parents)
        self.p = len(parents)
        self.tolerance = tolerance
        self.sign = np.sign(scores)
        self.score_scale = float(np.max(np.abs(scores)))
        self.weight_scale = float(np.max(weights))
        self.lambda_scale = self.score_scale / self.weight_scale
        c_original = np.abs(scores.astype(_LD)) / diagonal.astype(_LD)
        self.beta_scale = float(np.max(c_original))
        if not np.isfinite(self.beta_scale) or self.beta_scale == 0:
            raise ValueError("unpenalized coefficients must be representable")
        if not np.isfinite(self.lambda_scale) or self.lambda_scale == 0:
            raise ValueError("penalty scale must be representable")
        self.h = np.abs(scores.astype(_LD)) / _LD(self.score_scale)
        self.d = (diagonal.astype(_LD) / _LD(self.score_scale)
                  * _LD(self.beta_scale))
        self.w = weights.astype(_LD) / _LD(self.weight_scale)
        self.c = self.h / self.d
        self.sqrt_d = np.sqrt(np.asarray(self.d, dtype=float))
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            coordinate_weights = 1.0 / self.sqrt_d
            self.center = np.asarray(self.h, dtype=float) / self.sqrt_d
        if (not np.all(np.isfinite(coordinate_weights))
                or np.any(coordinate_weights <= 0)
                or not np.all(np.isfinite(self.center))):
            raise ValueError("diagonal scaling is outside supported numeric range")

        groups = [[node] for node in range(self.p)]
        for node in self.order:
            parent = int(parents[node])
            if parent >= 0:
                groups[parent].extend(groups[node])
        self.memberships = sum(len(g) for g in groups)
        self.prox = LaminarGroupLinfProx(
            groups, self.p, np.asarray(self.w, dtype=float),
            coordinate_weights=coordinate_weights,
        )
        topology = tree_group_linf_exact_topology_path(
            np.asarray(self.h, dtype=float), parents,
            group_weights=np.asarray(self.w, dtype=float),
        )
        self.activation = np.asarray(topology.activation_lambdas, dtype=_LD)
        self.structural_ascending = topology.lambdas[::-1]
        self.lambda_max = float(topology.lambdas[0])
        self.oracle_calls = 0
        self.boundary_probes = 0

    def point(self, lam):
        self.oracle_calls += 1
        if lam == 0:
            return self.c.copy(), 0.0
        try:
            theta, info = self.prox(self.center, float(lam), return_info=True)
        except (ValueError, FloatingPointError) as exc:
            # Inputs were validated at construction; a point-dependent range
            # failure must retain the already traversed prefix, not escape as
            # an exception or silently remove an unrepresentable penalty.
            raise _PathFailure(f"diagonal point oracle failed: {exc}") from exc
        residual = max(
            abs(float(info["raw_relative_duality_gap"])),
            float(info["max_relative_dual_l1_violation"]),
            float(info["relative_moreau_residual"]),
        )
        if not np.isfinite(residual) or residual > max(self.tolerance, 512 * _EPS):
            raise _PathFailure("the diagonal point oracle failed its certificate")
        return np.asarray(theta / self.sqrt_d, dtype=_LD), residual

    def face(self, probe):
        beta, oracle_residual = self.point(probe)
        q = beta.copy()
        for node in self.order:
            parent = int(self.parents[node])
            if parent >= 0:
                q[parent] = max(q[parent], q[node])
        positive = _LD(probe) < self.activation

        # A probe well inside a face resolves equality to machine precision.
        # Algebraic interval validation below rejects a mistaken near-tie.
        face_epsilon = 64 * _EPS * max(float(np.max(q)), 1e-12)
        block = np.arange(self.p)

        def root(node):
            while block[node] != node:
                block[node] = block[block[node]]
                node = int(block[node])
            return node

        for node in self.order:
            parent = int(self.parents[node])
            if (parent >= 0 and positive[node] and positive[parent]
                    and abs(q[parent] - q[node]) <= face_epsilon):
                block[root(node)] = root(parent)
        for node in range(self.p):
            block[node] = root(node)

        capped = positive & (self.c > q + face_epsilon)
        # c=0 nodes have identically zero beta even when their group is positive.
        capped &= self.h > 0
        qa = np.zeros(self.p, dtype=_LD)
        qb = np.zeros(self.p, dtype=_LD)
        block_nodes = {}
        for node in np.flatnonzero(positive):
            block_nodes.setdefault(int(block[node]), []).append(int(node))
        for members in block_nodes.values():
            members = np.asarray(members, dtype=int)
            active = members[capped[members]]
            curvature = np.sum(self.d[active], dtype=_LD)
            if curvature <= 0:
                raise _PathFailure("a positive face block has no active curvature")
            qa[members] = np.sum(self.h[active], dtype=_LD) / curvature
            qb[members] = -np.sum(self.w[members], dtype=_LD) / curvature

        ba = np.where(capped, qa, np.where(positive, self.c, _LD(0)))
        bb = np.where(capped, qb, _LD(0))
        fit_residual = float(np.max(np.abs(ba + _LD(probe) * bb - beta)))
        if fit_residual > max(self.tolerance, 512 * _EPS):
            raise _PathFailure("reconstructed face does not match point oracle")

        ra = np.where(capped, self.d * qa - self.h, _LD(0))
        rb = np.where(capped, self.d * qb + self.w, self.w)
        sum_a, sum_b = ra.copy(), rb.copy()
        scale_a = np.where(capped, np.abs(self.d * qa) + self.h, _LD(0))
        scale_b = np.where(capped, np.abs(self.d * qb) + self.w, self.w)
        internal = np.zeros(self.p, dtype=bool)
        for node in self.order:
            parent = int(self.parents[node])
            if parent >= 0 and positive[node] and block[node] == block[parent]:
                internal[node] = True
                sum_a[parent] += sum_a[node]
                sum_b[parent] += sum_b[node]
                scale_a[parent] += scale_a[node]
                scale_b[parent] += scale_b[node]

        # Constraint quantities are affine in lambda. Keep their signs/roots,
        # including narrow regions; solver tolerance does not merge events.
        inequalities = []
        def constraint(a, b, a_scale=None, b_scale=None):
            inequalities.append((
                a, b, abs(a) if a_scale is None else a_scale,
                abs(b) if b_scale is None else b_scale,
            ))

        for node in range(self.p):
            if positive[node]:
                constraint(qa[node], qb[node])
                if capped[node]:
                    constraint(self.c[node] - qa[node], -qb[node],
                               self.c[node] + abs(qa[node]))
                else:
                    constraint(qa[node] - self.c[node], qb[node],
                               self.c[node] + abs(qa[node]))
                constraint(self.activation[node], _LD(-1))
                if not internal[node]:
                    check_a = abs(sum_a[node]) / max(scale_a[node], _LD(1e-300))
                    check_b = abs(sum_b[node]) / max(scale_b[node], _LD(1e-300))
                    if max(check_a, check_b) > max(self.tolerance, 512 * _EPS):
                        raise _PathFailure("block stationarity failed")
            else:
                constraint(-self.activation[node], _LD(1))
            parent = int(self.parents[node])
            if parent >= 0 and block[parent] != block[node]:
                constraint(qa[parent] - qa[node], qb[parent] - qb[node],
                           abs(qa[parent]) + abs(qa[node]),
                           abs(qb[parent]) + abs(qb[node]))
            if internal[node]:
                a, b = -sum_a[node], -sum_b[node]
                if abs(a) <= 64 * _LEPS * scale_a[node]:
                    a = _LD(0)
                if abs(b) <= 64 * _LEPS * scale_b[node]:
                    b = _LD(0)
                constraint(a, b, scale_a[node], scale_b[node])

        lower, upper = _LD(0), _LD(np.inf)
        lower_roundoff = upper_roundoff = _LD(0)
        for a, b, a_scale, b_scale in inequalities:
            if b != 0:
                event = -a / b
                # Event locations from cancellation can be less accurate
                # than their primal coefficients. Account for the size of
                # the terms used to form a and b, not just the tiny root.
                roundoff = 64 * _EPS * (a_scale + abs(event) * b_scale) / abs(b)
                if not np.isfinite(event) or not np.isfinite(roundoff):
                    raise _PathFailure("face event arithmetic is not representable")
            if b > 0:
                if event > lower:
                    lower, lower_roundoff = event, roundoff
                elif event == lower:
                    lower_roundoff = max(lower_roundoff, roundoff)
            elif b < 0:
                if event < upper:
                    upper, upper_roundoff = event, roundoff
                elif event == upper:
                    upper_roundoff = max(upper_roundoff, roundoff)
            elif a < -128 * _LEPS * max(abs(a), _LD(1e-300)):
                raise _PathFailure("constant face inequality is infeasible")

        # A dual event and the activation formula can describe the same
        # boundary but round to adjacent floats. Canonicalize to the known
        # structural value before choosing the next interior probe. Otherwise
        # a spurious one-ULP interval can have no representable interior.
        # Restrict to values at/below the current probe: distinct close structural
        # events on opposite sides of the probe must never be combined.
        knots = self.structural_ascending
        last = bisect_right(knots, probe)
        if last:
            index = min(bisect_left(knots, lower), last - 1)
            nearest = _LD(knots[index])
            if index and abs(lower - knots[index - 1]) < abs(lower - nearest):
                nearest = _LD(knots[index - 1])
            if abs(lower - nearest) <= max(
                lower_roundoff, 512 * _EPS * max(abs(lower), abs(nearest))
            ):
                lower = nearest
        lambda_epsilon = 256 * _EPS * max(abs(float(probe)), 1e-300)
        if (not np.isfinite(lower) or upper < lower - lambda_epsilon
                or probe < lower - lambda_epsilon
                or probe > upper + lambda_epsilon):
            raise _PathFailure("probe is outside the reconstructed face interval")

        # Endpoint feasibility implies feasibility throughout an affine face.
        violation = 0.0
        for endpoint in (lower, min(upper, _LD(self.lambda_max))):
            for a, b, _, _ in inequalities:
                value = a + b * endpoint
                scale = max(abs(a) + abs(b * endpoint), _LD(1e-300))
                violation = max(violation, float(max(_LD(0), -value) / scale))
        residual = max(oracle_residual, fit_residual, violation)
        if residual > max(self.tolerance, 512 * _EPS):
            raise _PathFailure("affine interval failed endpoint feasibility")
        return _Face(float(lower), float(upper), ba, bb, residual, len(block_nodes),
                     float(lower_roundoff), float(upper_roundoff))

    def below(self, upper, previous_width, boundary_roundoff=0.0):
        # Known structural events give a safe probe ceiling even when two
        # events are far closer than the optimization tolerance. Never probe
        # past a distinct structural boundary merely because it is numerically tiny.
        index = bisect_left(self.structural_ascending, upper) - 1
        next_structural = float(self.structural_ascending[max(index, 0)])
        if next_structural > 0 and np.nextafter(upper, 0.0) <= next_structural:
            # Adjacent floats have no representable interior. At the lower
            # endpoint, strict activation still identifies the support inside
            # this tiny interval (excluding the groups that enter below it).
            # Accept only when the recovered affine inequalities cover both
            # endpoints; otherwise leave the path explicitly incomplete.
            self.boundary_probes += 1
            face = self.face(next_structural)
            coverage_eps = max(512 * _EPS * abs(upper),
                               boundary_roundoff + face.upper_roundoff)
            if face.lower == next_structural and face.upper >= upper - coverage_eps:
                return face, 1
            raise _PathFailure("adjacent-float structural interval is unresolved")
        max_gap = min(upper * 0.5, (upper - next_structural) * 0.5)
        gap = min(max_gap, max(previous_width * 0.05, upper * 1e-5))
        last_error = "no interior probe"
        attempts = 0
        # Different initial scales recover from probes numerically on a knot;
        # interval upper bounds detect and reject skipped narrow regions.
        tried = set()
        for _ in range(100):
            probe = float(_LD(upper) - _LD(gap))
            if probe <= 0 or probe >= upper or probe in tried:
                gap *= 0.37
                continue
            tried.add(probe)
            attempts += 1
            try:
                face = self.face(probe)
                # Independently evaluated neighboring formulas can differ by
                # more than a few ulps when an event formula subtracts nearly
                # equal terms. Keep this roundoff allowance independent of
                # optimization tolerance; the structural probe ceiling above still
                # prevents stepping across a known structural event.
                coverage_eps = max(512 * _EPS * abs(upper),
                                   boundary_roundoff + face.upper_roundoff)
                if face.upper < upper - coverage_eps:
                    last_error = "probe skipped a face; uncovered lambda interval"
                    gap *= 0.1
                elif face.lower >= upper:
                    last_error = "probe did not resolve the face below the event"
                    gap = min(max_gap, gap * 3.0)
                else:
                    return face, attempts
            except _PathFailure as exc:
                last_error = str(exc)
                # Try a farther probe as well as a closer one; recovery of its
                # upper boundary is still required before it may be accepted.
                gap = min(max_gap, gap * 2.3) if attempts % 3 == 0 else gap * 0.23
        raise _PathFailure(f"could not certify the adjacent face: {last_error}")


def tree_group_linf_exact_coefficient_path(
    linear_scores: Sequence[float],
    gram_diagonal: Sequence[float],
    parent_indices: Sequence[int],
    *,
    group_weights: float | Sequence[float] | None = None,
    tolerance: float = 1e-9,
    max_events: int = 10_000,
) -> RegularizationPath:
    r"""Trace the full squared-loss hiCAP coefficient path from tree statistics.

    Solve ``0.5*sum(D*beta**2) - h@beta + lambda*sum_v w_v
    *max(abs(beta[subtree(v)]))`` for every lambda from its zero threshold
    through zero. There is one coefficient and one descendant group per node.
    Parents may describe a forest in arbitrary order; roots have parent -1.
    Curvatures and group weights must be positive. The score API defines a
    diagonal problem directly; it makes no claim about an external design.

    ``exact=True`` reports coverage by affine KKT regions within numerical
    coefficient tolerance and recorded event-boundary uncertainty. It is not
    symbolic or directed-rounding certification. Event limits or unresolved
    intervals retain a nonexact prefix.
    An empty forest/all-zero score vector returns one zero-lambda solution.

    This first implementation stores explicit descendant memberships and dense
    coefficient rows (O(p*K) output memory). It re-evaluates the point oracle
    and rebuilds the face after each event, rather than maintaining dynamic
    block updates.
    """
    scores = _real_vector(linear_scores, "linear_scores")
    diagonal = _real_vector(gram_diagonal, "gram_diagonal")
    order, _ = _child_before_parent_tree(parent_indices)
    parents = np.asarray(parent_indices, dtype=int)
    p = len(scores)
    if diagonal.shape != scores.shape or len(order) != p:
        raise ValueError("scores, diagonal, and parents must have equal lengths")
    if np.any(diagonal <= 0):
        raise ValueError("gram_diagonal must be strictly positive")
    weights = _positive_node_weights(group_weights, p)
    if (isinstance(tolerance, (bool, np.bool_)) or not np.isscalar(tolerance)
            or np.iscomplexobj(tolerance) or not np.isfinite(tolerance)
            or tolerance <= 0):
        raise ValueError("tolerance must be a finite positive scalar")
    if (isinstance(max_events, (bool, np.bool_))
            or not isinstance(max_events, (int, np.integer)) or max_events < 1):
        raise ValueError("max_events must be a positive integer")
    tolerance = float(tolerance)
    metadata = {
        "problem": "diagonal_quadratic_from_tree_statistics",
        "ord": "inf", "path_scope": "complete_coefficient_direction_path",
        "algorithm": "diagonal_tree_isotonic_face_rebuild",
        "coefficient_path_unique": True,
        "n_features": p, "n_groups": p,
        "parent_indices": tuple(int(x) for x in parents),
        "group_weights": tuple(float(x) for x in weights),
        "exactness_condition": "caller_supplied_positive_diagonal_problem",
        "descendant_groups_materialized": bool(p),
        "dense_coefficients_materialized": True,
        "tolerance": tolerance, "max_events": int(max_events),
        "effective_certificate_tolerance": max(tolerance, 1024 * _EPS),
        "relative_event_roundoff_tolerance": 512 * _EPS,
        "event_roundoff_model": "relative_floor_and_affine_cancellation_scales",
        "generic_qp_or_lp_used": False,
    }
    if p == 0 or not np.any(scores):
        metadata.update({
            "coefficient_knots_enumerated": True,
            "coefficient_event_coverage_complete": True,
            "point_solutions_certified": True, "segment_regions_certified": True,
            "n_regions": 0, "n_oracle_calls": 0, "reached_zero": True,
            "n_boundary_probes": 0, "probe_attempts": 0,
            "n_group_memberships": 0, "descendant_groups_materialized": False,
            "structural_lambdas": (0.0,), "structural_knots_covered": True,
            "lambda_max": 0.0,
        })
        return RegularizationPath(
            np.array([0.0]), np.zeros((1, p)), penalties=np.zeros(1),
            diagnostics=({"certified": True},), events=("least_squares",),
            method="diagonal-tree-coefficient-homotopy", exact=True,
            metadata=metadata,
        )

    problem = _TreeProblem(scores, diagonal, parents, weights, tolerance)
    upper = problem.lambda_max
    lambda_values = [upper]
    coefficient_rows = [np.zeros(p, dtype=_LD)]
    previous_width = upper
    boundary_roundoff = 0.0
    previous_slope = None
    regions = []
    status, failure = "complete", None
    probe_attempts = 0
    try:
        while upper > 0:
            if len(regions) >= max_events:
                status = "event_limit"
                break
            face, attempts = problem.below(upper, previous_width, boundary_roundoff)
            probe_attempts += attempts
            beta_upper = face.at(upper)
            continuity = float(np.max(np.abs(beta_upper - coefficient_rows[-1])))
            if continuity > max(tolerance, 1024 * _EPS):
                raise _PathFailure("adjacent affine regions are not continuous")
            lower = max(0.0, face.lower)
            if lower >= upper:
                raise _PathFailure("continuation made no forward progress")
            # Remove only roundoff-level collinear pivots, not small events
            # according to the user's optimization tolerance.
            if previous_slope is not None:
                slope_scale = np.maximum(
                    np.maximum(np.abs(previous_slope), np.abs(face.beta_slope)),
                    _LD(1e-300),
                )
                if np.all(np.abs(previous_slope - face.beta_slope)
                          <= 512 * _EPS * slope_scale):
                    lambda_values.pop()
                    coefficient_rows.pop()
            lambda_values.append(lower)
            coefficient_rows.append(face.at(lower))
            regions.append({
                "upper": upper * problem.lambda_scale,
                "lower": lower * problem.lambda_scale,
                "certified": True, "n_blocks": face.n_blocks,
                "max_relative_residual": max(face.max_residual, continuity),
                "lower_boundary_roundoff": face.lower_roundoff * problem.lambda_scale,
                "upper_boundary_roundoff": face.upper_roundoff * problem.lambda_scale,
            })
            previous_width, upper = upper - lower, lower
            boundary_roundoff = face.lower_roundoff
            previous_slope = face.beta_slope
    except _PathFailure as exc:
        status, failure = "numerical_failure", str(exc)

    reached_zero = lambda_values[-1] == 0.0
    diagnostics = []
    for lam, beta in zip(lambda_values, coefficient_rows):
        try:
            expected, residual = problem.point(lam)
            error = float(np.max(np.abs(beta - expected)))
            certified = error <= max(tolerance, 1024 * _EPS)
        except _PathFailure:
            error, residual, certified = np.inf, np.inf, False
        diagnostics.append({
            "certified": bool(certified),
            "relative_coefficient_oracle_error": error,
            "relative_prox_certificate_residual": residual,
        })
    points_certified = all(x["certified"] for x in diagnostics)
    if status == "complete" and not points_certified:
        status = "kkt_failure"
    if reached_zero and points_certified:
        coefficient_rows[-1] = problem.c.copy()
    lambdas = np.asarray(lambda_values) * problem.lambda_scale
    coefficients = np.asarray(coefficient_rows, dtype=float)
    coefficients *= problem.sign * problem.beta_scale
    coefficients[0] = 0.0
    if reached_zero and points_certified:
        coefficients[-1] = scores / diagonal
    structural_lambdas = problem.structural_ascending[::-1] * problem.lambda_scale
    ascending = lambdas[::-1]
    positions = np.searchsorted(ascending, structural_lambdas)
    lower_neighbors = ascending[np.maximum(positions - 1, 0)]
    upper_neighbors = ascending[np.minimum(positions, len(ascending) - 1)]
    structural_errors = np.minimum(np.abs(lower_neighbors - structural_lambdas),
                                   np.abs(upper_neighbors - structural_lambdas))
    structural_covered = bool(np.all(
        structural_errors <= 1024 * _EPS * np.maximum(structural_lambdas, 1e-300)
    ))
    if status == "complete" and not structural_covered:
        status, failure = "numerical_failure", "a structural event is missing"
    exact = status == "complete" and reached_zero and points_certified
    metadata.update({
        "n_regions": len(regions), "probe_attempts": probe_attempts,
        "n_oracle_calls": problem.oracle_calls,
        "n_boundary_probes": problem.boundary_probes,
        "n_group_memberships": problem.memberships,
        "coefficient_knots_enumerated": bool(exact),
        "coefficient_event_coverage_complete": bool(exact),
        "point_solutions_certified": bool(points_certified),
        "segment_regions_certified": True,
        "reached_zero": bool(reached_zero),
        "structural_knots_covered": bool(structural_covered),
        "structural_lambdas": tuple(float(x) for x in structural_lambdas),
        "interval_certificates": tuple(regions),
        "max_relative_interval_residual": max(
            (x["max_relative_residual"] for x in regions), default=0.0
        ),
        "lambda_max": float(lambdas[0]),
        "coefficient_storage_bytes": int(coefficients.nbytes),
    })
    if failure is not None:
        metadata["failure_message"] = failure
    events = ["lambda_max"] + ["coefficient_face_change"] * (len(lambdas) - 1)
    if reached_zero:
        events[-1] = "least_squares"
    elif len(events) > 1:
        events[-1] = "path_stopped"
    # One bottom-up sweep evaluates every subtree norm at every knot. Avoid
    # revisiting explicit descendant memberships, which costs O(K*p**2)
    # on a chain; this postprocessing costs O(K*p).
    subtree_norms = np.abs(coefficients)
    for node in problem.order:
        parent = int(parents[node])
        if parent >= 0:
            np.maximum(subtree_norms[:, parent], subtree_norms[:, node],
                       out=subtree_norms[:, parent])
    penalties = subtree_norms @ weights
    return RegularizationPath(
        lambdas=lambdas, coefficients=coefficients, penalties=penalties,
        events=events, diagnostics=tuple(diagnostics), exact=exact, status=status,
        method="diagonal-tree-coefficient-homotopy", metadata=metadata,
    )


__all__ = ["tree_group_linf_exact_coefficient_path"]
