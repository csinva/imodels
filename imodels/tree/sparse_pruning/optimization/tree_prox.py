r"""Exact proximal maps for laminar sums of group :math:`\ell_\infty` norms.

For a laminar family of groups (every pair is disjoint or nested), the
Euclidean proximal map

.. math::

    \operatorname{prox}_{s\Omega}(u), \qquad
    \Omega(\beta) = \sum_g w_g\max_{j\in g} a_j|\beta_j|,

is the composition of the individual group proximal maps in an order where
children precede their ancestors.  The positive coordinate weights ``a_j``
default to one.  An individual weighted infinity-norm proximal map is obtained
from Moreau's identity by subtracting the projection onto the dual weighted
:math:`\ell_1` ball.

``LaminarGroupLinfProx`` separates one-time validation and ordering from the
proximal evaluation.  This matters in accelerated proximal-gradient methods,
where the same group structure is used at every iteration.
"""
from __future__ import annotations

import math
from copy import copy
from collections.abc import Iterable, Sequence
from fractions import Fraction
from typing import Any

import numpy as np

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


def _validate_feature_count(n_features: int) -> int:
    if isinstance(n_features, (bool, np.bool_)) or not isinstance(
        n_features, (int, np.integer)
    ):
        raise ValueError("n_features must be a positive integer")
    n_features = int(n_features)
    if n_features <= 0:
        raise ValueError("n_features must be a positive integer")
    return n_features


def _validate_groups(
    groups: Iterable[Sequence[int]], n_features: int
) -> tuple[list[np.ndarray], list[int]]:
    try:
        raw_groups = list(groups)
    except TypeError as exc:
        raise ValueError("groups must be a non-empty iterable") from exc
    if not raw_groups:
        raise ValueError("groups must contain at least one group")

    validated: list[np.ndarray] = []
    group_sizes: list[int] = []
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
        if np.any(group_array < 0) or np.any(group_array >= n_features):
            raise ValueError(
                f"group {group_number} contains an index outside "
                f"[0, {n_features})"
            )
        key = tuple(int(index) for index in group_array)
        if key in keys:
            raise ValueError("groups must not contain duplicate groups")
        keys.add(key)
        group_array = group_array.copy()
        group_array.flags.writeable = False
        validated.append(group_array)
        group_sizes.append(group_array.size)
    return validated, group_sizes


def _validate_group_weights(
    group_weights: float | Sequence[float] | None, n_groups: int
) -> np.ndarray:
    if group_weights is None:
        weights = np.ones(n_groups, dtype=float)
    else:
        raw = np.asarray(group_weights)
        if raw.ndim == 0:
            if raw.dtype.kind in "bc":
                raise ValueError("group_weights must be finite nonnegative scalars")
            try:
                scalar = float(raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "group_weights must be finite nonnegative scalars"
                ) from exc
            weights = np.full(n_groups, scalar, dtype=float)
        else:
            if (
                raw.ndim != 1
                or raw.shape != (n_groups,)
                or raw.dtype.kind == "c"
            ):
                raise ValueError(
                    f"group_weights must be a scalar or have shape ({n_groups},)"
                )
            if raw.dtype.kind == "b":
                raise ValueError("group_weights must be finite nonnegative scalars")
            try:
                weights = raw.astype(float, copy=True)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "group_weights must be finite nonnegative scalars"
                ) from exc
    if not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("group_weights must be finite and nonnegative")
    return weights


def _validate_coordinate_weights(
    coordinate_weights: float | Sequence[float] | None, n_features: int
) -> np.ndarray:
    if coordinate_weights is None:
        weights = np.ones(n_features, dtype=float)
    else:
        raw = np.asarray(coordinate_weights)
        if raw.ndim == 0:
            if raw.dtype.kind in "bc":
                raise ValueError(
                    "coordinate_weights must be finite positive scalars"
                )
            try:
                scalar = float(raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "coordinate_weights must be finite positive scalars"
                ) from exc
            weights = np.full(n_features, scalar, dtype=float)
        else:
            if (
                raw.ndim != 1
                or raw.shape != (n_features,)
                or raw.dtype.kind == "c"
            ):
                raise ValueError(
                    "coordinate_weights must be a scalar or have shape "
                    f"({n_features},)"
                )
            if raw.dtype.kind == "b":
                raise ValueError(
                    "coordinate_weights must be finite positive scalars"
                )
            try:
                weights = raw.astype(float, copy=True)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "coordinate_weights must be finite positive scalars"
                ) from exc
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError("coordinate_weights must be finite and positive")
    return weights


def _scaled_weighted_l1_norm(
    absolute: np.ndarray,
    coordinate_weights: np.ndarray,
    radius: float,
) -> tuple[float, float, float]:
    """Scale ``sum(abs / a)`` and ``radius`` by a shared log reference."""
    with np.errstate(divide="ignore", invalid="ignore"):
        log_terms = np.log(absolute) - np.log(coordinate_weights)
    log_radius = math.log(radius)
    reference = max(log_radius, float(np.max(log_terms, initial=-np.inf)))
    with np.errstate(under="ignore", invalid="ignore"):
        scaled_terms = np.exp(log_terms - reference)
        scaled_radius = math.exp(log_radius - reference)
    scaled_norm = math.fsum(float(value) for value in scaled_terms)
    return reference, scaled_norm, scaled_radius


def _two_component_sum(values: Iterable[float]) -> tuple[float, float]:
    """Return a rounded sum and its accumulated first-order roundoff."""
    total = 0.0
    errors: list[float] = []
    for raw_value in values:
        value = float(raw_value)
        updated = total + value
        if abs(total) >= abs(value):
            errors.append((total - updated) + value)
        else:
            errors.append((value - updated) + total)
        total = updated
    return total, math.fsum(errors)


def _compensated_add(
    total: float, correction: float, value: float
) -> tuple[float, float]:
    """Neumaier update for a two-component running nonnegative sum."""
    updated = total + value
    if abs(total) >= abs(value):
        roundoff = (total - updated) + value
    else:
        roundoff = (value - updated) + total
    return updated, correction + roundoff


def _stable_l2_norm(value: np.ndarray) -> float:
    """Return an overflow- and underflow-safe Euclidean norm."""
    absolute = np.abs(np.asarray(value, dtype=float))
    scale = float(np.max(absolute, initial=0.0))
    if scale == 0.0:
        return 0.0
    if not np.isfinite(scale):
        return np.inf
    scaled = absolute / scale
    with np.errstate(over="ignore", invalid="ignore"):
        result = scale * math.sqrt(float(scaled @ scaled))
    return float(result)


def _max_componentwise_relative_residual(
    residual: np.ndarray, *references: np.ndarray
) -> float:
    """Scale every residual coordinate by the corresponding input magnitudes."""
    absolute_residual = np.abs(np.asarray(residual, dtype=float))
    component_scale = np.zeros_like(absolute_residual)
    for reference in references:
        component_scale = np.maximum(
            component_scale, np.abs(np.asarray(reference, dtype=float))
        )
    ratios = np.zeros_like(absolute_residual)
    positive_scale = component_scale > 0.0
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        ratios[positive_scale] = (
            absolute_residual[positive_scale] / component_scale[positive_scale]
        )
    ratios[~positive_scale & (absolute_residual != 0.0)] = np.inf
    return float(np.max(ratios, initial=0.0))


def _exact_weighted_l1_projection(
    value: np.ndarray,
    coordinate_weights: np.ndarray,
    radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Rare exact-rational fallback for an ambiguous floating-point face."""
    absolute = np.abs(value)
    exact_absolute = [Fraction.from_float(float(item)) for item in absolute]
    exact_weights = [
        Fraction.from_float(float(item)) for item in coordinate_weights
    ]
    exact_inverse = [Fraction(1, 1) / item for item in exact_weights]
    exact_radius = Fraction.from_float(float(radius))
    order = sorted(
        range(value.size),
        key=lambda index: exact_absolute[index] * exact_weights[index],
        reverse=True,
    )

    if sum(
        exact_absolute[index] * exact_inverse[index]
        for index in range(value.size)
    ) <= exact_radius:
        return value.copy(), np.zeros_like(value)

    first_sum = Fraction(0, 1)
    second_sum = Fraction(0, 1)
    face: list[int] = []
    for index in order:
        coefficient = exact_absolute[index]
        inverse = exact_inverse[index]
        condition = (
            coefficient * second_sum
            - inverse * first_sum
            + inverse * exact_radius
        )
        if condition < 0:
            break
        face.append(index)
        first_sum += coefficient * inverse
        second_sum += inverse * inverse

    if not face:  # pragma: no cover - positive radius outside the ball
        raise RuntimeError("failed to identify the weighted l1 projection face")
    threshold = max(Fraction(0, 1), (first_sum - exact_radius) / second_sum)
    projected_absolute = np.zeros_like(absolute)
    result_absolute = absolute.copy()
    for index in face:
        cap = threshold * exact_inverse[index]
        projected = max(Fraction(0, 1), exact_absolute[index] - cap)
        projected_absolute[index] = float(projected)
        result_absolute[index] = float(min(exact_absolute[index], cap))
    signs = np.sign(value)
    return signs * projected_absolute, signs * result_absolute


def _product_lost(*factors: float, result: float) -> bool:
    """Whether nonzero finite factors rounded to a zero or non-finite product."""
    return bool(
        all(factor != 0.0 and np.isfinite(factor) for factor in factors)
        and (result == 0.0 or not np.isfinite(result))
    )


def _project_singleton_l1_ball(
    value: np.ndarray, coordinate_weight: float, radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Project a one-coordinate ball without sorting or prefix bookkeeping.

    Its dual interval has radius ``radius * coordinate_weight``. Resolve
    cancellation and subnormal products with the existing exact fallback;
    this also preserves tiny nonzero primal values at a rounded boundary.
    """
    coefficient = float(value[0])
    absolute = abs(coefficient)
    if absolute == 0.0:
        return value.copy(), np.zeros_like(value)
    cap = radius * coordinate_weight
    if not math.isfinite(cap):
        # Both factors are finite and positive, so an overflowing interval
        # contains every representable coefficient.
        return value.copy(), np.zeros_like(value)
    if coordinate_weight != 1.0 and (
        cap < np.finfo(float).tiny
        or abs(absolute - cap)
        <= 64.0 * np.finfo(float).eps * max(absolute, cap)
    ):
        return _exact_weighted_l1_projection(
            value, np.array([coordinate_weight]), radius
        )
    projection = math.copysign(min(absolute, cap), coefficient)
    result = math.copysign(max(0.0, absolute - cap), coefficient)
    return np.array([projection]), np.array([result])


def _project_weighted_l1_ball(
    value: np.ndarray, coordinate_weights: np.ndarray, radius: float
) -> tuple[np.ndarray, np.ndarray]:
    r"""Project onto ``sum_j |q_j| / a_j <= radius``.

    The KKT solution has

    .. math::

        q_j = \operatorname{sign}(v_j)
              \max\{|v_j| - \nu/a_j, 0\}.

    Sorting the breakpoints ``|v_j| * a_j`` identifies the active coordinates
    and gives ``nu`` in closed form on that active set.
    """
    if radius <= 0:
        return np.zeros_like(value), value.copy()
    if value.size == 1:
        return _project_singleton_l1_ball(
            value, float(coordinate_weights[0]), radius
        )
    absolute = np.abs(value)
    _, scaled_norm, scaled_radius = _scaled_weighted_l1_norm(
        absolute, coordinate_weights, radius
    )
    norm_difference = scaled_norm - scaled_radius
    comparison_scale = max(scaled_norm, scaled_radius, np.finfo(float).tiny)
    if norm_difference <= 0.0:
        # A log-domain comparison can round a just-outside point onto the
        # boundary. Resolve only this narrow ambiguity exactly; points safely
        # inside retain the inexpensive return path.
        if norm_difference < -128.0 * np.finfo(float).eps * comparison_scale:
            return value.copy(), np.zeros_like(value)
        return _exact_weighted_l1_projection(
            value, coordinate_weights, radius
        )

    # Work in breakpoint order, scaling reciprocal weights by the largest
    # reciprocal in the *current prefix*. An inactive coordinate encountered
    # later must not make all earlier face predicates underflow. The exact
    # active predicates form one prefix, so stop at the first strict failure.
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        inverse_weights = 1.0 / coordinate_weights
    if not np.all(np.isfinite(inverse_weights)) or np.any(inverse_weights <= 0.0):
        return _exact_weighted_l1_projection(value, coordinate_weights, radius)

    with np.errstate(divide="ignore", invalid="ignore"):
        log_breakpoints = np.log(absolute) + np.log(coordinate_weights)
    order = np.argsort(-log_breakpoints, kind="stable")
    ordered_absolute = absolute[order]
    ordered_inverse = inverse_weights[order]

    inverse_reference = 0.0
    first_sum = first_error = 0.0
    second_sum = second_error = 0.0
    face_size = 0
    use_exact_fallback = False
    for position, (coefficient_raw, inverse_raw) in enumerate(
        zip(ordered_absolute, ordered_inverse)
    ):
        coefficient = float(coefficient_raw)
        inverse = float(inverse_raw)
        if inverse_reference == 0.0:
            inverse_reference = inverse
        elif inverse > inverse_reference:
            factor = inverse_reference / inverse
            factor_squared = factor * factor
            if (
                factor == 0.0
                or _product_lost(factor, factor, result=factor_squared)
            ):
                use_exact_fallback = True
                break
            old_values = (first_sum, first_error, second_sum, second_error)
            first_sum *= factor
            first_error *= factor
            second_sum *= factor_squared
            second_error *= factor_squared
            if any(
                old != 0.0 and updated == 0.0
                for old, updated in zip(
                    old_values,
                    (first_sum, first_error, second_sum, second_error),
                )
            ):
                use_exact_fallback = True
                break
            inverse_reference = inverse

        scaled_inverse = inverse / inverse_reference
        scaled_radius = radius / inverse_reference
        if (
            not np.isfinite(scaled_inverse)
            or scaled_inverse <= 0.0
            or not np.isfinite(scaled_radius)
            or (scaled_radius == 0.0 and radius != 0.0)
        ):
            use_exact_fallback = True
            break

        if position > 0:
            terms = [
                coefficient * second_sum,
                coefficient * second_error,
                -scaled_inverse * first_sum,
                -scaled_inverse * first_error,
                scaled_inverse * scaled_radius,
            ]
            factor_pairs = [
                (coefficient, second_sum),
                (coefficient, second_error),
                (-scaled_inverse, first_sum),
                (-scaled_inverse, first_error),
                (scaled_inverse, scaled_radius),
            ]
            if any(
                _product_lost(first, second, result=term)
                for (first, second), term in zip(factor_pairs, terms)
            ) or not np.all(np.isfinite(terms)):
                use_exact_fallback = True
                break
            condition = math.fsum(terms)
            condition_scale = math.fsum(abs(term) for term in terms)
            if abs(condition) <= (
                128.0 * np.finfo(float).eps * condition_scale
            ):
                use_exact_fallback = True
                break
            if condition < 0.0:
                break

        first_term = coefficient * scaled_inverse
        second_term = scaled_inverse * scaled_inverse
        if (
            _product_lost(coefficient, scaled_inverse, result=first_term)
            or _product_lost(scaled_inverse, scaled_inverse, result=second_term)
        ):
            use_exact_fallback = True
            break
        first_sum, first_error = _compensated_add(
            first_sum, first_error, first_term
        )
        second_sum, second_error = _compensated_add(
            second_sum, second_error, second_term
        )
        if not np.all(
            np.isfinite([first_sum, first_error, second_sum, second_error])
        ):
            use_exact_fallback = True
            break
        face_size = position + 1

    if use_exact_fallback or face_size == 0:
        return _exact_weighted_l1_projection(value, coordinate_weights, radius)

    face = order[:face_size]
    face_absolute = absolute[face]
    face_inverse_scale = float(np.max(inverse_weights[face]))
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        face_inverse = inverse_weights[face] / face_inverse_scale
        face_radius = radius / face_inverse_scale
        first_terms = face_absolute * face_inverse
        second_terms = face_inverse * face_inverse
    if (
        not np.all(np.isfinite(face_inverse))
        or np.any(face_inverse <= 0.0)
        or not np.isfinite(face_radius)
        or (face_radius == 0.0 and radius != 0.0)
        or any(
            _product_lost(float(coefficient), float(inverse), result=float(term))
            for coefficient, inverse, term in zip(
                face_absolute, face_inverse, first_terms
            )
        )
        or any(
            _product_lost(float(inverse), float(inverse), result=float(term))
            for inverse, term in zip(face_inverse, second_terms)
        )
    ):
        return _exact_weighted_l1_projection(value, coordinate_weights, radius)

    first_sum, first_error = _two_component_sum(first_terms)
    second_sum, second_error = _two_component_sum(second_terms)
    denominator = second_sum + second_error
    if denominator <= 0.0 or not np.isfinite(denominator):
        return _exact_weighted_l1_projection(value, coordinate_weights, radius)

    projected_face = np.asarray(
        [
            max(
                0.0,
                min(
                    float(coefficient),
                    math.fsum(
                        [
                            float(coefficient * second_sum),
                            float(coefficient * second_error),
                            float(-inverse * first_sum),
                            float(-inverse * first_error),
                            float(inverse * face_radius),
                        ]
                    )
                    / denominator,
                ),
            )
            for coefficient, inverse in zip(face_absolute, face_inverse)
        ],
        dtype=float,
    )
    if not np.all(np.isfinite(projected_face)):
        return _exact_weighted_l1_projection(value, coordinate_weights, radius)

    # Enforce the representable dual boundary by correcting a coordinate with
    # the largest reciprocal weight first.
    boundary_residual = face_radius - math.fsum(
        float(projected * inverse)
        for projected, inverse in zip(projected_face, face_inverse)
    )
    for local_index in np.argsort(-face_inverse, kind="stable"):
        inverse = float(face_inverse[int(local_index)])
        previous = float(projected_face[int(local_index)])
        updated = min(
            float(face_absolute[int(local_index)]),
            max(0.0, previous + boundary_residual / inverse),
        )
        projected_face[int(local_index)] = updated
        boundary_residual -= (updated - previous) * inverse
        if abs(boundary_residual) <= (
            32.0
            * np.finfo(float).eps
            * max(abs(face_radius), np.finfo(float).tiny)
        ):
            break

    threshold = max(
        0.0,
        math.fsum([first_sum, first_error, -face_radius]) / denominator,
    )
    direct_projection = np.zeros_like(value)
    direct_projection[face] = np.sign(value[face]) * projected_face
    result_absolute = absolute.copy()
    result_absolute[face] = np.minimum(
        face_absolute, threshold * face_inverse
    )
    direct_result = np.sign(value) * result_absolute
    moreau_residual = value - direct_result - direct_projection
    if _max_componentwise_relative_residual(
        moreau_residual, value, direct_result, direct_projection
    ) > 2048.0 * np.finfo(float).eps * max(1, value.size):
        return _exact_weighted_l1_projection(value, coordinate_weights, radius)
    return direct_projection, direct_result


def _project_unit_l1_ball(
    value: np.ndarray, radius: float
) -> tuple[np.ndarray, np.ndarray]:
    """Use the vectorized unit-weight projection, with a certified fallback."""
    if value.size == 1:
        return _project_singleton_l1_ball(value, 1.0, radius)
    try:
        projected = proj_l1_ball(value, radius)
    except (IndexError, OverflowError):
        return _project_weighted_l1_ball(
            value, np.ones_like(value), radius
        )
    with np.errstate(over="ignore", invalid="ignore"):
        value_norm = float(np.sum(np.abs(value)))
        projected_norm = float(np.sum(np.abs(projected)))
    if not np.isfinite(value_norm) or not np.isfinite(projected_norm):
        return _project_weighted_l1_ball(
            value, np.ones_like(value), radius
        )
    if value_norm > radius:
        boundary_error = abs(projected_norm - radius) / max(
            projected_norm, radius, np.nextafter(0.0, 1.0)
        )
        if boundary_error > (
            512.0 * np.finfo(float).eps * max(1, value.size)
        ):
            return _project_weighted_l1_ball(
                value, np.ones_like(value), radius
            )
    result = value - projected
    if not np.all(np.isfinite(result)):
        return _project_weighted_l1_ball(
            value, np.ones_like(value), radius
        )
    return projected, result


def _weighted_l1_feasibility(
    value: np.ndarray,
    coordinate_weights: np.ndarray,
    radius: float,
) -> tuple[float, float]:
    """Return absolute and relative weighted-l1 radius violations stably."""
    if value.size == 1:
        absolute_scalar = abs(float(value[0]))
        if absolute_scalar == 0.0:
            return 0.0, 0.0
        inverse_scalar = 1.0 / float(coordinate_weights[0])
        direct_radius_scalar = radius / inverse_scalar
        if math.isfinite(inverse_scalar) and direct_radius_scalar > 0.0:
            violation_scalar = max(0.0, absolute_scalar - direct_radius_scalar)
            relative_scalar = violation_scalar / max(
                absolute_scalar, direct_radius_scalar, np.finfo(float).tiny
            )
            return violation_scalar * inverse_scalar, relative_scalar
        # Extreme reciprocals and unrepresentable scaled radii retain the
        # logarithmic fallback below, as in the general vector calculation.
    absolute = np.abs(value)
    support = absolute > 0.0
    if not np.any(support):
        return 0.0, 0.0
    supported_absolute = absolute[support]
    supported_weights = coordinate_weights[support]
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        inverse = 1.0 / supported_weights
        inverse_scale = float(np.max(inverse, initial=0.0))
        scaled_inverse = inverse / inverse_scale
        direct_radius = radius / inverse_scale
        direct_norm = float(supported_absolute @ scaled_inverse)
    if (
        np.all(np.isfinite(inverse))
        and inverse_scale > 0.0
        and np.all(scaled_inverse > 0.0)
        and np.isfinite(direct_radius)
        and np.isfinite(direct_norm)
        and direct_radius > 0.0
        and direct_norm > 0.0
    ):
        direct_violation = max(0.0, direct_norm - direct_radius)
        relative_violation = direct_violation / max(
            direct_radius, direct_norm, np.finfo(float).tiny
        )
        if direct_violation == 0.0:
            return 0.0, relative_violation
        with np.errstate(over="ignore", invalid="ignore"):
            absolute_violation = float(direct_violation * inverse_scale)
        return (
            absolute_violation if np.isfinite(absolute_violation) else np.inf,
            relative_violation,
        )

    reference, scaled_norm, scaled_radius = _scaled_weighted_l1_norm(
        supported_absolute, supported_weights, radius
    )
    scaled_violation = max(0.0, scaled_norm - scaled_radius)
    relative_violation = scaled_violation / max(
        scaled_radius, scaled_norm, np.finfo(float).tiny
    )
    if scaled_violation == 0.0:
        absolute_violation = 0.0
    else:
        log_violation = reference + math.log(scaled_violation)
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            absolute_violation = float(np.exp(log_violation))
        if not np.isfinite(absolute_violation):
            absolute_violation = np.inf
    return absolute_violation, relative_violation


def _stable_three_factor_product(first: float, second: float, third: float) -> float:
    """Multiply three nonnegative finite floats without intermediate overflow."""
    if first == 0.0 or second == 0.0 or third == 0.0:
        return 0.0
    mantissa = 1.0
    exponent = 0
    for value in (first, second, third):
        fraction, power = np.frexp(value)
        mantissa *= float(fraction)
        exponent += int(power)
    mantissa, adjustment = np.frexp(mantissa)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        product = np.ldexp(mantissa, exponent + int(adjustment))
    return float(product)


def _scaled_group_penalty(
    radius: float,
    coordinate_weights: np.ndarray,
    absolute_value: np.ndarray,
) -> float:
    """Return ``radius * max(a * abs(beta))`` without unsafe association."""
    if absolute_value.size == 1:
        coefficient = float(absolute_value[0])
        if coefficient == 0.0:
            return 0.0
        coordinate_weight = float(coordinate_weights[0])
        weighted_scalar = coordinate_weight * coefficient
        direct_scalar = radius * weighted_scalar
        if math.isfinite(direct_scalar) and weighted_scalar >= np.finfo(float).tiny:
            return direct_scalar
        return _stable_three_factor_product(radius, coordinate_weight, coefficient)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        weighted_absolute = coordinate_weights * absolute_value
        direct = float(radius * np.max(weighted_absolute))
    underflowed_inner_product = bool(
        np.any(
            (absolute_value != 0.0)
            & (weighted_absolute < np.finfo(float).tiny)
        )
    )
    # Even a nonzero subnormal inner product may have lost many significant
    # bits before multiplication by a large radius brings it back to normal
    # scale. Rescale all three factors together in that case as well.
    if np.isfinite(direct) and not underflowed_inner_product:
        return direct
    return max(
        _stable_three_factor_product(
            radius, float(coordinate_weight), float(coefficient)
        )
        for coordinate_weight, coefficient in zip(
            coordinate_weights, absolute_value
        )
    )


def _laminar_leaf_to_root_order(
    groups: list[np.ndarray], group_sizes: list[int], n_features: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return a leaf-to-root order and direct parents, or reject crossings.

    For a laminar family, the groups containing any one coordinate form a
    chain.  Once groups are sorted by cardinality, the immediate group after a
    given group in that chain must be identical for every coordinate of the
    group.  Checking this condition validates laminarity without all
    :math:`O(m^2)` pair comparisons.
    """
    n_groups = len(groups)
    order = np.array(
        sorted(range(n_groups), key=lambda index: (group_sizes[index], index)),
        dtype=np.intp,
    )
    memberships: list[list[int]] = [[] for _ in range(n_features)]
    for ordered_index, source_index in enumerate(order):
        for feature in groups[int(source_index)]:
            memberships[int(feature)].append(ordered_index)

    # Parent indices refer to positions in ``order``.  -2 means that no
    # coordinate of the group has been examined; -1 denotes a root.
    parents = np.full(n_groups, -2, dtype=np.intp)
    for chain in memberships:
        for position, child in enumerate(chain):
            candidate = chain[position + 1] if position + 1 < len(chain) else -1
            previous = int(parents[child])
            if previous == -2:
                parents[child] = candidate
            elif previous != candidate:
                source_child = int(order[child])
                conflicting = candidate if candidate >= 0 else previous
                source_conflicting = int(order[conflicting])
                raise ValueError(
                    "groups must be laminar: group "
                    f"{source_child} overlaps group {source_conflicting} "
                    "without containment"
                )

    # Every non-empty group occurs in at least one membership chain.
    if np.any(parents == -2):  # pragma: no cover - guarded by group validation
        raise RuntimeError("internal error while constructing the group hierarchy")
    return order, parents


class LaminarGroupLinfProx:
    r"""Compiled exact prox for a weighted laminar group-infinity penalty.

    Parameters
    ----------
    groups:
        Non-empty feature-index groups.  Every two groups must be disjoint or
        nested.  Duplicate groups and repeated indices within a group are
        rejected.
    n_features:
        Length of vectors on which this operator acts.
    group_weights:
        A finite nonnegative scalar, broadcast to every group, or one weight
        per input group.  The default is one.
    coordinate_weights:
        A finite positive scalar, broadcast to every coordinate, or one
        weight ``a_j`` per coordinate.  Group ``g`` then penalizes
        ``max(a[group] * abs(beta[group]))``.  The default is one.
    require_leaf_to_root:
        If false (the default), groups are reordered safely.  If true, reject
        input in which an ancestor occurs before one of its descendants.

    Notes
    -----
    Validation and hierarchy construction happen only once.  Calling the
    compiled object costs :math:`O(\sum_g |g|)` plus the cost of the
    :math:`\ell_1` projections.
    """

    def __init__(
        self,
        groups: Iterable[Sequence[int]],
        n_features: int,
        group_weights: float | Sequence[float] | None = None,
        coordinate_weights: float | Sequence[float] | None = None,
        *,
        require_leaf_to_root: bool = False,
    ) -> None:
        n_features = _validate_feature_count(n_features)
        if not isinstance(require_leaf_to_root, (bool, np.bool_)):
            raise ValueError("require_leaf_to_root must be a boolean")
        validated, sizes = _validate_groups(groups, n_features)
        weights = _validate_group_weights(group_weights, len(validated))
        coordinate_weights_array = _validate_coordinate_weights(
            coordinate_weights, n_features
        )
        order, parents = _laminar_leaf_to_root_order(
            validated, sizes, n_features
        )

        if require_leaf_to_root:
            for child, parent in enumerate(parents):
                if parent >= 0 and order[child] > order[parent]:
                    raise ValueError(
                        "groups must be supplied in leaf-to-root order when "
                        "require_leaf_to_root=True"
                    )

        self._n_features = n_features
        self._groups = tuple(validated[int(index)] for index in order)
        self._group_weights = np.asarray(weights[order], dtype=float)
        self._coordinate_weights = coordinate_weights_array
        self._unit_coordinate_weights = bool(
            np.all(self._coordinate_weights == 1.0)
        )
        self._source_order = order
        self._parents = parents
        for array in (
            self._group_weights,
            self._coordinate_weights,
            self._source_order,
            self._parents,
        ):
            array.flags.writeable = False

    def _with_coordinate_weights(self, coordinate_weights) -> LaminarGroupLinfProx:
        """Reuse the immutable hierarchy with a different diagonal geometry."""
        weights = _validate_coordinate_weights(coordinate_weights, self.n_features)
        weights.flags.writeable = False
        operator = copy(self)
        operator._coordinate_weights = weights
        operator._unit_coordinate_weights = bool(np.all(weights == 1.0))
        return operator

    @property
    def n_features(self) -> int:
        """Number of coordinates expected by the proximal map."""
        return self._n_features

    @property
    def groups(self) -> tuple[np.ndarray, ...]:
        """Validated groups in the leaf-to-root composition order."""
        return self._groups

    @property
    def group_weights(self) -> np.ndarray:
        """Group weights in the leaf-to-root composition order."""
        return self._group_weights

    @property
    def coordinate_weights(self) -> np.ndarray:
        """Positive coordinate weights in feature order."""
        return self._coordinate_weights

    @property
    def source_order(self) -> np.ndarray:
        """Input-group index for every group in composition order."""
        return self._source_order

    @property
    def parents(self) -> np.ndarray:
        """Direct-parent positions in composition order; roots have -1."""
        return self._parents

    def __call__(
        self,
        value: np.ndarray,
        scale: float,
        *,
        return_info: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
        r"""Evaluate :math:`\operatorname{prox}_{\mathrm{scale}\,\Omega}`.

        With ``return_info=True``, the returned dictionary contains a dual
        certificate assembled from the per-group :math:`\ell_1` projections.
        The certificate is useful as a continuous implementation check; its
        duality gap is zero at an exact proximal solution up to roundoff.
        A positive scale and group weight must have a representable positive
        product; overflow and underflow of that group radius are rejected.
        """
        raw_value = np.asarray(value)
        if raw_value.shape != (self._n_features,) or raw_value.dtype.kind == "c":
            raise ValueError(
                f"value must be real with shape ({self._n_features},); "
                f"got shape {raw_value.shape}"
            )
        try:
            value_array = raw_value.astype(float, copy=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("value must contain finite real numbers") from exc
        if not np.all(np.isfinite(value_array)):
            raise ValueError("value must contain only finite values")
        if (
            isinstance(scale, (bool, np.bool_))
            or not isinstance(scale, (int, float, np.number))
            or np.iscomplexobj(scale)
        ):
            raise ValueError("scale must be a finite nonnegative scalar")
        scale = float(scale)
        if not np.isfinite(scale) or scale < 0:
            raise ValueError("scale must be a finite nonnegative scalar")
        if not isinstance(return_info, (bool, np.bool_)):
            raise ValueError("return_info must be a boolean")

        if scale == 0.0:
            if not return_info:
                return value_array.copy()
            return value_array.copy(), {
                "primal_objective": 0.0,
                "dual_objective": 0.0,
                "duality_gap": 0.0,
                "raw_duality_gap": 0.0,
                "raw_relative_duality_gap": 0.0,
                "relative_duality_gap": 0.0,
                "max_dual_l1_violation": 0.0,
                "max_relative_dual_l1_violation": 0.0,
                "moreau_residual": 0.0,
                "relative_moreau_residual": 0.0,
                "n_groups": len(self._groups),
                "_dual_displacement": np.zeros_like(value_array),
            }

        result = value_array.copy()
        dual_displacement = np.zeros_like(result) if return_info else None
        max_dual_l1_violation = 0.0
        max_relative_dual_l1_violation = 0.0
        for group, weight in zip(self._groups, self._group_weights):
            with np.errstate(over="ignore", invalid="ignore"):
                radius = scale * float(weight)
            if not np.isfinite(radius):
                raise ValueError("scale times every group weight must be finite")
            if radius == 0:
                if weight > 0.0:
                    raise ValueError(
                        "scale times a positive group weight underflowed; "
                        "rescale the penalty to a representable positive radius"
                    )
                continue
            group_value = result[group]
            group_coordinate_weights = self._coordinate_weights[group]
            if self._unit_coordinate_weights:
                dual_block, group_result = _project_unit_l1_ball(
                    group_value, radius
                )
            else:
                dual_block, group_result = _project_weighted_l1_ball(
                    group_value, group_coordinate_weights, radius
                )
            result[group] = group_result
            if return_info:
                dual_displacement[group] += dual_block
                violation, relative_violation = _weighted_l1_feasibility(
                    dual_block, group_coordinate_weights, radius
                )
                max_dual_l1_violation = max(max_dual_l1_violation, violation)
                max_relative_dual_l1_violation = max(
                    max_relative_dual_l1_violation,
                    relative_violation,
                )

        if not return_info:
            return result

        displacement = dual_displacement
        moreau_residual = value_array - result - displacement
        moreau_residual_norm = _stable_l2_norm(moreau_residual)
        relative_moreau_residual = _max_componentwise_relative_residual(
            moreau_residual,
            value_array,
            result,
            displacement,
        )
        # Form the scaled penalty group by group.  Computing Omega(result)
        # first can overflow even when ``scale * Omega(result)`` is finite.
        penalty = float(
            sum(
                _scaled_group_penalty(
                    scale * float(weight),
                    self._coordinate_weights[group],
                    np.abs(result[group]),
                )
                for group, weight in zip(self._groups, self._group_weights)
                if weight > 0.0
            )
        )

        # The proximal point can still be correct when an extreme, but finite,
        # input makes a reported objective overflow. Keep that situation
        # explicit (an infinite objective cannot certify the point) without
        # leaking runtime warnings from optional diagnostics.
        with np.errstate(over="ignore", invalid="ignore"):
            primal_displacement = value_array - result
            primal_displacement_norm = _stable_l2_norm(primal_displacement)
            displacement_norm = _stable_l2_norm(displacement)
            primal_displacement_squared = float(
                primal_displacement_norm * primal_displacement_norm
            )
            displacement_squared = float(displacement_norm * displacement_norm)
            moreau_residual_squared = float(
                moreau_residual_norm * moreau_residual_norm
            )
            primal_objective = 0.5 * primal_displacement_squared + penalty
            dual_objective = float(
                displacement @ (value_array - 0.5 * displacement)
            )
            result_dual_pairing = float(result @ displacement)
            # This identity remains valid even if the independently rounded
            # primal and dual points do not satisfy Moreau consistency exactly.
            raw_duality_gap = (
                0.5 * moreau_residual_squared
                + penalty
                - result_dual_pairing
            )
        duality_gap = (
            max(0.0, raw_duality_gap)
            if np.isfinite(raw_duality_gap)
            else np.inf
        )
        objective_scale = max(
            np.finfo(float).tiny,
            abs(primal_objective),
            abs(dual_objective),
            abs(penalty),
            abs(result_dual_pairing),
        )
        info: dict[str, Any] = {
            "primal_objective": primal_objective,
            "dual_objective": dual_objective,
            "duality_gap": duality_gap,
            "raw_duality_gap": raw_duality_gap,
            "raw_relative_duality_gap": (
                raw_duality_gap / objective_scale
                if np.isfinite(raw_duality_gap)
                and np.isfinite(objective_scale)
                else np.nan
            ),
            "relative_duality_gap": duality_gap / objective_scale,
            "max_dual_l1_violation": max(0.0, max_dual_l1_violation),
            "max_relative_dual_l1_violation": max(
                0.0, max_relative_dual_l1_violation
            ),
            "moreau_residual": moreau_residual_norm,
            "relative_moreau_residual": relative_moreau_residual,
            "n_groups": len(self._groups),
            "_dual_displacement": displacement.copy(),
        }
        return result, info

    def prox(
        self,
        value: np.ndarray,
        scale: float,
        *,
        return_info: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
        """Named alias for :meth:`__call__`."""
        return self(value, scale, return_info=return_info)


def prox_laminar_group_linf(
    value: np.ndarray,
    groups: Iterable[Sequence[int]],
    scale: float,
    group_weights: float | Sequence[float] | None = None,
    coordinate_weights: float | Sequence[float] | None = None,
    *,
    require_leaf_to_root: bool = False,
    return_info: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
    """Validate, compile, and evaluate a laminar group-infinity proximal map.

    Use :class:`LaminarGroupLinfProx` directly when evaluating the same group
    structure repeatedly.
    """
    raw_value = np.asarray(value)
    if raw_value.ndim != 1:
        raise ValueError(f"value must be one-dimensional; got shape {raw_value.shape}")
    operator = LaminarGroupLinfProx(
        groups,
        raw_value.size,
        group_weights,
        coordinate_weights,
        require_leaf_to_root=require_leaf_to_root,
    )
    return operator(raw_value, scale, return_info=return_info)


__all__ = ["LaminarGroupLinfProx", "prox_laminar_group_linf"]
