r"""Independent optimality diagnostics for sum-of-group-:math:`\ell_\infty`.

For the convex quadratic objective

.. math::

    F(\beta) = \tfrac12 \beta^T H\beta - h^T\beta
      + \lambda \sum_g \|\beta_g\|_\infty,

optimality is equivalent to ``0`` belonging to its subdifferential.  This
module checks that condition with a small linear program over *all* eligible
group subgradients.  It deliberately does not reuse hiCAP's private active
basis recovery: an independent basic face is necessary for homotopy
continuation, but not for certifying a fixed coefficient vector, and can fail
at an otherwise valid degenerate optimum.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
from scipy.optimize import linprog


def _positive_scalar(value: Any, name: str) -> float:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, float, np.number))
    ):
        raise ValueError(f"{name} must be a positive finite scalar")
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite scalar")
    return value


def _nonnegative_scalar(value: Any, name: str) -> float:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, float, np.number))
    ):
        raise ValueError(f"{name} must be a nonnegative finite scalar")
    value = float(value)
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a nonnegative finite scalar")
    return value


def _validate_groups(
    groups: Iterable[Sequence[int]], n_features: int
) -> tuple[np.ndarray, ...]:
    try:
        raw_groups = list(groups)
    except TypeError as exc:
        raise ValueError("groups must be a non-empty iterable") from exc
    if not raw_groups:
        raise ValueError("groups must contain at least one group")

    validated: list[np.ndarray] = []
    for number, raw_group in enumerate(raw_groups):
        group = np.asarray(raw_group)
        if group.ndim != 1 or group.size == 0:
            raise ValueError(f"group {number} must be a non-empty 1D array")
        if group.dtype.kind not in "iu":
            raise ValueError(f"group {number} must contain integer indices")
        group = group.astype(np.intp, copy=False)
        if np.unique(group).size != group.size:
            raise ValueError(f"group {number} contains duplicate indices")
        if np.any(group < 0) or np.any(group >= n_features):
            raise ValueError(
                f"group {number} contains an index outside [0, {n_features})"
            )
        validated.append(np.asarray(group, dtype=np.intp))
    return tuple(validated)


def _validate_linear_and_groups(
    linear: np.ndarray, groups: Iterable[Sequence[int]]
) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    linear = np.asarray(linear, dtype=float)
    if linear.ndim != 1 or linear.size == 0:
        raise ValueError("linear must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(linear)):
        raise ValueError("linear must contain only finite values")
    return linear, _validate_groups(groups, linear.size)


def _validate_quadratic(
    gram: np.ndarray,
    linear: np.ndarray,
    beta: np.ndarray,
    groups: Iterable[Sequence[int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[np.ndarray, ...]]:
    gram = np.asarray(gram, dtype=float)
    linear = np.asarray(linear, dtype=float)
    beta = np.asarray(beta, dtype=float)
    if gram.ndim == 1:
        if gram.size == 0:
            raise ValueError("gram diagonal must be non-empty")
        p = gram.size
    elif gram.ndim == 2 and gram.shape[0] > 0 and gram.shape[0] == gram.shape[1]:
        p = gram.shape[0]
    else:
        raise ValueError(
            "gram must be a non-empty 1D diagonal or a non-empty square matrix"
        )
    if linear.shape != (p,) or beta.shape != (p,):
        raise ValueError(f"linear and beta must both have shape ({p},)")
    if not all(np.all(np.isfinite(value)) for value in (gram, linear, beta)):
        raise ValueError("gram, linear, and beta must contain only finite values")
    if gram.ndim == 1:
        eigenvalues = gram
    else:
        symmetry_scale = max(
            float(np.max(np.abs(gram))), np.finfo(float).tiny
        )
        symmetry_error = float(np.max(np.abs(gram - gram.T)))
        if symmetry_error > 1e-10 * symmetry_scale:
            raise ValueError("gram must be symmetric")
        # Ignore harmless roundoff asymmetry when forming the gradient.
        gram = 0.5 * (gram + gram.T)
        eigenvalues = np.linalg.eigvalsh(gram)
    spectral_scale = max(
        float(np.max(np.abs(eigenvalues))), np.finfo(float).tiny
    )
    psd_roundoff = (
        64.0 * np.finfo(float).eps * max(1, p) * spectral_scale
    )
    if float(np.min(eigenvalues)) < -psd_roundoff:
        raise ValueError("gram must be positive semidefinite")
    return gram, linear, beta, _validate_groups(groups, p)


def _linprog_options(tolerance: float) -> dict[str, float]:
    feasibility = max(1e-10, min(1e-7, tolerance))
    return {
        "primal_feasibility_tolerance": feasibility,
        "dual_feasibility_tolerance": feasibility,
    }


def group_linf_lambda_max(
    linear: np.ndarray,
    groups: Iterable[Sequence[int]],
    *,
    tolerance: float = 1e-8,
) -> float:
    r"""Return the exact zero-solution threshold for a group-:math:`\ell_\infty` penalty.

    ``linear`` is :math:`h` in the quadratic loss
    ``0.5 * beta @ H @ beta - linear @ beta``.  The result is the smallest
    nonnegative lambda for which beta zero satisfies the KKT condition.  The
    groups may overlap; no laminarity assumption is needed for this helper.

    A failed or numerically inconsistent LP raises ``RuntimeError`` rather
    than returning an uncertified threshold.
    """

    tolerance = _positive_scalar(tolerance, "tolerance")
    linear, groups = _validate_linear_and_groups(linear, groups)
    if not np.any(linear):
        return 0.0

    covered = np.zeros(linear.size, dtype=bool)
    for group in groups:
        covered[group] = True
    if np.any((~covered) & (linear != 0.0)):
        # An unpenalized coordinate with a nonzero loss gradient prevents zero
        # from being optimal at every finite regularization strength.
        return np.inf

    # HiGHS feasibility tolerances have an absolute floor. Normalize first so
    # thresholds remain homogeneous when y (and hence h) is extremely small.
    linear_scale = float(np.max(np.abs(linear)))
    scaled_linear = linear / linear_scale

    p = linear.size
    feature_rows: list[int] = []
    group_rows: list[int] = []
    signs: list[float] = []
    for group_number, group in enumerate(groups):
        for feature in group:
            feature_rows.extend((int(feature), int(feature)))
            group_rows.extend((group_number, group_number))
            signs.extend((1.0, -1.0))
    n_masses = len(feature_rows)

    # mu contains the positive/negative signed mass for every group-feature
    # membership. Each group's l1 mass is at most lambda.
    equality = np.zeros((p, n_masses + 1), dtype=float)
    equality[np.asarray(feature_rows), np.arange(n_masses)] = signs
    upper = np.zeros((len(groups), n_masses + 1), dtype=float)
    upper[np.asarray(group_rows), np.arange(n_masses)] = 1.0
    upper[:, -1] = -1.0
    objective = np.zeros(n_masses + 1, dtype=float)
    objective[-1] = 1.0
    result = linprog(
        objective,
        A_ub=upper,
        b_ub=np.zeros(len(groups)),
        A_eq=equality,
        b_eq=scaled_linear,
        bounds=[(0.0, None)] * (n_masses + 1),
        method="highs-ds",
        options=_linprog_options(tolerance),
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"group-Linf lambda_max LP failed: {result.message}")

    scaled_lam = max(0.0, float(result.x[-1]))
    equality_residual = float(
        np.max(np.abs(equality @ result.x - scaled_linear))
    )
    group_violation = max(0.0, float(np.max(upper @ result.x)))
    if max(equality_residual, group_violation) > 50.0 * tolerance:
        raise RuntimeError("group-Linf lambda_max LP failed its residual check")
    return scaled_lam * linear_scale


def _base_diagnostic(
    *,
    lam: float,
    tolerance: float,
    face_tolerance: float,
    gradient_norm: float,
    scale: float,
    kkt_residual: float,
    certified: bool,
    lp_success: bool,
    lp_status: int | None,
    lp_message: str,
    n_variables: int,
    n_active_variables: int,
    n_zero_groups: int,
    subgradient_residual: float,
    face_residual: float,
) -> dict[str, Any]:
    relative = float(kkt_residual / scale)
    return {
        "lambda": lam,
        "kkt_residual": float(kkt_residual),
        "relative_kkt_residual": relative,
        # Short alias is convenient for adaptive-path consumers.
        "relative_residual": relative,
        "stationarity_residual": float(kkt_residual),
        "subgradient_feasibility_residual": float(subgradient_residual),
        "face_residual": float(face_residual),
        "gradient_norm_inf": float(gradient_norm),
        "scale": float(scale),
        "certified": bool(certified),
        "lp_success": bool(lp_success),
        "lp_status": lp_status,
        "lp_message": str(lp_message),
        "tolerance": tolerance,
        "face_tolerance": face_tolerance,
        "n_subgradient_variables": int(n_variables),
        "n_active_subgradient_variables": int(n_active_variables),
        "n_zero_groups": int(n_zero_groups),
    }


def _group_linf_gradient_kkt_at_face(
    gradient: np.ndarray,
    groups: tuple[np.ndarray, ...],
    lam: float,
    beta: np.ndarray,
    *,
    tolerance: float,
    face_tolerance: float,
    scale: float,
) -> dict[str, Any]:
    """Solve the stationarity LP for one chosen numerical face.

    This private helper deliberately knows nothing about certification.  A
    positive ``face_tolerance`` enlarges the true subdifferential of the
    supplied floating-point coefficient vector, so its result is suitable only
    for a relaxed acceptance test.  The public diagnostic below always reruns
    this helper with ``face_tolerance=0`` before setting ``certified=True``.
    """

    gradient_norm = float(np.max(np.abs(gradient)))
    memberships = np.zeros(beta.size, dtype=int)
    for group in groups:
        memberships[group] += 1
    scale = max(float(scale), np.finfo(float).tiny)

    if lam == 0.0:
        return _base_diagnostic(
            lam=lam,
            tolerance=tolerance,
            face_tolerance=face_tolerance,
            gradient_norm=gradient_norm,
            scale=scale,
            kkt_residual=gradient_norm,
            certified=gradient_norm <= tolerance * scale,
            lp_success=True,
            lp_status=None,
            lp_message="lambda is zero; no subgradient LP is required",
            n_variables=0,
            n_active_variables=0,
            n_zero_groups=sum(not np.any(beta[group]) for group in groups),
            subgradient_residual=0.0,
            face_residual=0.0,
        )

    columns: list[np.ndarray] = []
    blocks: list[list[int]] = []
    zero_blocks: list[int] = []
    face_slacks: list[float] = []
    coefficient_scale = max(
        float(np.max(np.abs(beta))), np.finfo(float).tiny
    )
    for group_number, group in enumerate(groups):
        values = beta[group]
        absolute = np.abs(values)
        group_norm = float(np.max(absolute))
        block: list[int] = []
        if group_norm <= face_tolerance * coefficient_scale:
            zero_blocks.append(group_number)
            for feature in group:
                for sign in (1.0, -1.0):
                    column = np.zeros(beta.size, dtype=float)
                    column[int(feature)] = sign
                    columns.append(column)
                    face_slacks.append(group_norm / coefficient_scale)
                    block.append(len(columns) - 1)
        else:
            # This is relative to the group itself. A global absolute floor
            # would incorrectly put zero coordinates on the face of a tiny but
            # nonzero group, enlarging its true subdifferential.
            face_width = face_tolerance * group_norm
            candidates = np.flatnonzero(
                (group_norm - absolute <= face_width) & (absolute > 0.0)
            )
            # The exact argmax is always eligible, including when tolerance=0.
            if candidates.size == 0:
                candidates = np.asarray([int(np.argmax(absolute))])
            for local_feature in candidates:
                column = np.zeros(beta.size, dtype=float)
                column[int(group[local_feature])] = np.sign(values[local_feature])
                columns.append(column)
                face_slacks.append(
                    (group_norm - float(absolute[local_feature])) / group_norm
                )
                block.append(len(columns) - 1)
        blocks.append(block)

    contributions = np.column_stack(columns)
    n_weights = contributions.shape[1]
    # The last nonnegative variable is the infinity-norm stationarity error.
    objective = np.zeros(n_weights + 1, dtype=float)
    objective[-1] = 1.0
    lp_scale = max(
        gradient_norm,
        lam * float(np.max(memberships)),
        np.finfo(float).tiny,
    )
    scaled_lam = lam / lp_scale
    scaled_gradient = gradient / lp_scale
    positive = np.column_stack(
        (scaled_lam * contributions, -np.ones(beta.size, dtype=float))
    )
    negative = np.column_stack(
        (-scaled_lam * contributions, -np.ones(beta.size, dtype=float))
    )
    upper_rows: list[np.ndarray] = [positive, negative]
    upper_bounds: list[np.ndarray] = [-scaled_gradient, scaled_gradient]
    zero_set = set(zero_blocks)
    for group_number in zero_blocks:
        row = np.zeros(n_weights + 1, dtype=float)
        row[blocks[group_number]] = 1.0
        upper_rows.append(row[None, :])
        upper_bounds.append(np.ones(1, dtype=float))

    nonzero_groups = [
        number for number in range(len(groups)) if number not in zero_set
    ]
    if nonzero_groups:
        equality = np.zeros((len(nonzero_groups), n_weights + 1), dtype=float)
        for row_number, group_number in enumerate(nonzero_groups):
            equality[row_number, blocks[group_number]] = 1.0
        equality_rhs: np.ndarray | None = np.ones(len(nonzero_groups))
    else:
        equality = None
        equality_rhs = None

    result = linprog(
        objective,
        A_ub=np.vstack(upper_rows),
        b_ub=np.concatenate(upper_bounds),
        A_eq=equality,
        b_eq=equality_rhs,
        bounds=[(0.0, None)] * (n_weights + 1),
        method="highs-ds",
        options=_linprog_options(tolerance),
    )
    if not result.success or result.x is None:
        return _base_diagnostic(
            lam=lam,
            tolerance=tolerance,
            face_tolerance=face_tolerance,
            gradient_norm=gradient_norm,
            scale=scale,
            kkt_residual=np.inf,
            certified=False,
            lp_success=False,
            lp_status=int(result.status),
            lp_message=str(result.message),
            n_variables=n_weights,
            n_active_variables=0,
            n_zero_groups=len(zero_blocks),
            subgradient_residual=np.inf,
            face_residual=np.inf,
        )

    weights = np.asarray(result.x[:-1], dtype=float)
    total_subgradient = contributions @ weights
    stationarity_residual = float(
        np.max(np.abs(gradient + lam * total_subgradient))
    )
    group_mass_residual = 0.0
    for group_number, block in enumerate(blocks):
        mass = float(np.sum(weights[block]))
        if group_number in zero_set:
            group_mass_residual = max(group_mass_residual, max(0.0, mass - 1.0))
        else:
            group_mass_residual = max(group_mass_residual, abs(mass - 1.0))
    subgradient_residual = lam * group_mass_residual
    active = np.flatnonzero(weights > max(1e-12, tolerance * 0.01))
    face_residual = (
        float(np.max(np.asarray(face_slacks)[active])) if active.size else 0.0
    )
    kkt_residual = max(stationarity_residual, subgradient_residual)
    certified = bool(
        kkt_residual <= tolerance * scale
        and face_residual <= face_tolerance + np.finfo(float).eps
    )
    diagnostic = _base_diagnostic(
        lam=lam,
        tolerance=tolerance,
        face_tolerance=face_tolerance,
        gradient_norm=gradient_norm,
        scale=scale,
        kkt_residual=kkt_residual,
        certified=certified,
        lp_success=True,
        lp_status=int(result.status),
        lp_message=str(result.message),
        n_variables=n_weights,
        n_active_variables=int(active.size),
        n_zero_groups=len(zero_blocks),
        subgradient_residual=subgradient_residual,
        face_residual=face_residual,
    )
    diagnostic["stationarity_residual"] = stationarity_residual
    diagnostic["lp_objective"] = max(0.0, float(result.fun)) * lp_scale
    return diagnostic


def _group_linf_gradient_kkt_diagnostic(
    gradient: np.ndarray,
    groups: tuple[np.ndarray, ...],
    lam: float,
    beta: np.ndarray,
    *,
    tolerance: float,
    face_tolerance: float,
    scale: float,
) -> dict[str, Any]:
    """Return a strict certificate plus an optional relaxed-face diagnostic."""

    strict = _group_linf_gradient_kkt_at_face(
        gradient,
        groups,
        lam,
        beta,
        tolerance=tolerance,
        face_tolerance=0.0,
        scale=scale,
    )
    relaxed = (
        strict
        if face_tolerance == 0.0
        else _group_linf_gradient_kkt_at_face(
            gradient,
            groups,
            lam,
            beta,
            tolerance=tolerance,
            face_tolerance=face_tolerance,
            scale=scale,
        )
    )

    # ``certified`` and the unprefixed residual fields always describe the true
    # subdifferential of the coefficient vector exactly as supplied.  A relaxed
    # face can be useful for iterative solvers whose theoretically tied values
    # differ numerically, but it is a backward-error heuristic rather than a
    # KKT certificate for that vector.
    diagnostic = dict(strict)
    relaxed_accepted = bool(relaxed["certified"])
    diagnostic.update(
        {
            "face_tolerance": face_tolerance,
            "strict_face_tolerance": 0.0,
            "face_relaxation_used": bool(face_tolerance > 0.0),
            "accepted": relaxed_accepted,
            "relaxed_accepted": relaxed_accepted,
            "acceptance_mode": (
                "strict"
                if strict["certified"]
                else "relaxed_face"
                if relaxed_accepted
                else "rejected"
            ),
            "relaxed_kkt_residual": float(relaxed["kkt_residual"]),
            "relative_relaxed_kkt_residual": float(
                relaxed["relative_kkt_residual"]
            ),
            "relaxed_relative_residual": float(
                relaxed["relative_kkt_residual"]
            ),
            "relaxed_stationarity_residual": float(
                relaxed["stationarity_residual"]
            ),
            "relaxed_subgradient_feasibility_residual": float(
                relaxed["subgradient_feasibility_residual"]
            ),
            "relaxed_face_residual": float(relaxed["face_residual"]),
            "relaxed_lp_success": bool(relaxed["lp_success"]),
            "relaxed_lp_status": relaxed["lp_status"],
            "relaxed_lp_message": str(relaxed["lp_message"]),
            "relaxed_n_subgradient_variables": int(
                relaxed["n_subgradient_variables"]
            ),
            "relaxed_n_active_subgradient_variables": int(
                relaxed["n_active_subgradient_variables"]
            ),
            "relaxed_n_zero_groups": int(relaxed["n_zero_groups"]),
        }
    )
    return diagnostic


def group_linf_quadratic_kkt_diagnostic(
    gram: np.ndarray,
    linear: np.ndarray,
    groups: Iterable[Sequence[int]],
    lam: float,
    beta: np.ndarray,
    *,
    tolerance: float = 1e-7,
    face_tolerance: float = 1e-8,
) -> dict[str, Any]:
    r"""Diagnose a fixed beta for a quadratic sum-group-:math:`\ell_\infty` problem.

    The returned ``kkt_residual`` is the minimum infinity-norm stationarity
    error over the *true* subgradient of every group at the supplied
    floating-point ``beta``. Accordingly, ``certified`` is always a strict-face
    numerical certificate at the requested stationarity tolerance.

    A positive ``face_tolerance`` additionally treats small groups as zero and
    admits coordinates that are numerically tied for a group's maximum. That
    enlarged face is reported separately through ``relaxed_kkt_residual`` and
    ``relaxed_accepted``; it is useful as a backward-error heuristic, but it is
    not a KKT certificate for the coefficient vector as supplied. ``accepted``
    aliases this relaxed acceptance decision, and ``acceptance_mode`` records
    whether acceptance was strict, relaxed, or rejected.

    LP failures are returned as structured diagnostics with infinite residual
    and ``lp_success=False``. Invalid inputs still raise ``ValueError``.
    ``gram`` may be either a positive-semidefinite square matrix or its
    nonnegative diagonal supplied as a one-dimensional array. The latter
    avoids materializing a :math:`p \times p` matrix for diagonal designs.
    """

    tolerance = _positive_scalar(tolerance, "tolerance")
    face_tolerance = _nonnegative_scalar(face_tolerance, "face_tolerance")
    lam = _nonnegative_scalar(lam, "lam")
    gram, linear, beta, groups = _validate_quadratic(
        gram, linear, beta, groups
    )
    smooth_gradient = gram * beta if gram.ndim == 1 else gram @ beta
    gradient = smooth_gradient - linear
    memberships = np.zeros(beta.size, dtype=int)
    for group in groups:
        memberships[group] += 1
    scale = max(
        float(np.max(np.abs(linear))),
        float(np.max(np.abs(smooth_gradient))),
        lam * float(np.max(memberships)),
        np.finfo(float).tiny,
    )
    return _group_linf_gradient_kkt_diagnostic(
        gradient,
        groups,
        lam,
        beta,
        tolerance=tolerance,
        face_tolerance=face_tolerance,
        scale=scale,
    )


def _validate_regression_inputs(
    X: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("X must be a non-empty two-dimensional array")
    if y.shape != (X.shape[0],):
        raise ValueError(f"y must have shape ({X.shape[0]},)")
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
        raise ValueError("X and y must contain only finite values")
    if sample_weight is None:
        weight = np.ones(X.shape[0], dtype=float)
    else:
        weight = np.asarray(sample_weight, dtype=float)
        if weight.shape != (X.shape[0],):
            raise ValueError(f"sample_weight must have shape ({X.shape[0]},)")
        if not np.all(np.isfinite(weight)) or np.any(weight < 0):
            raise ValueError("sample_weight must be finite and nonnegative")
    with np.errstate(over="ignore", invalid="ignore"):
        weight_sum = float(np.sum(weight))
    if not np.isfinite(weight_sum) or weight_sum <= 0:
        raise ValueError("sample_weight must have a positive finite total")
    return X, y, weight / weight_sum, weight_sum


def group_linf_regression_kkt_diagnostic(
    X: np.ndarray,
    y: np.ndarray,
    groups: Iterable[Sequence[int]],
    lam: float,
    beta: np.ndarray,
    *,
    sample_weight: np.ndarray | None = None,
    fit_intercept: bool = False,
    intercept: float | None = None,
    tolerance: float = 1e-7,
    face_tolerance: float = 1e-8,
) -> dict[str, Any]:
    """Convenient normalized weighted-regression wrapper for the KKT check.

    If ``fit_intercept=True`` and ``intercept`` is omitted, the optimal
    unpenalized intercept conditional on ``beta`` is used.  The returned
    residual then checks the profiled-intercept problem solved by the path
    routines.  If an intercept is supplied, its stationarity condition is
    included in both the strict certificate and relaxed acceptance test. The
    strict-versus-relaxed face semantics are the same as in
    :func:`group_linf_quadratic_kkt_diagnostic`.
    """

    X, y, normalized_weight, weight_sum = _validate_regression_inputs(
        X, y, sample_weight
    )
    beta = np.asarray(beta, dtype=float)
    if beta.shape != (X.shape[1],) or not np.all(np.isfinite(beta)):
        raise ValueError(f"beta must be a finite array of shape ({X.shape[1]},)")
    if not isinstance(fit_intercept, (bool, np.bool_)):
        raise ValueError("fit_intercept must be a boolean")
    if fit_intercept:
        if intercept is None:
            x_mean = normalized_weight @ X
            y_mean = float(normalized_weight @ y)
            intercept_value = y_mean - float(x_mean @ beta)
            # Form profiled sufficient statistics from centered data. This is
            # algebraically equivalent to the uncentered expression but avoids
            # catastrophic cancellation when feature offsets are large.
            X_work = X - x_mean
            y_work = y - y_mean
        else:
            if (
                isinstance(intercept, (bool, np.bool_))
                or not isinstance(intercept, (int, float, np.number))
            ):
                raise ValueError("intercept must be a finite scalar")
            intercept_value = float(intercept)
            if not np.isfinite(intercept_value):
                raise ValueError("intercept must be a finite scalar")
            X_work = X
            y_work = y - intercept_value
    else:
        if intercept is not None:
            if (
                isinstance(intercept, (bool, np.bool_))
                or not isinstance(intercept, (int, float, np.number))
                or not np.isfinite(intercept)
                or float(intercept) != 0.0
            ):
                raise ValueError(
                    "intercept must be zero/None when fit_intercept=False"
                )
        intercept_value = 0.0
        X_work = X
        y_work = y

    tolerance = _positive_scalar(tolerance, "tolerance")
    face_tolerance = _nonnegative_scalar(face_tolerance, "face_tolerance")
    lam = _nonnegative_scalar(lam, "lam")
    validated_groups = _validate_groups(groups, X.shape[1])
    fitted_work = X_work @ beta
    residual = fitted_work - y_work
    smooth_gradient = X_work.T @ (normalized_weight * fitted_work)
    linear = X_work.T @ (normalized_weight * y_work)
    # Form the gradient from residuals rather than subtracting two potentially
    # large sufficient-statistic terms.
    gradient = X_work.T @ (normalized_weight * residual)
    memberships = np.zeros(beta.size, dtype=int)
    for group in validated_groups:
        memberships[group] += 1
    scale = max(
        float(np.max(np.abs(linear))),
        float(np.max(np.abs(smooth_gradient))),
        lam * float(np.max(memberships)),
        np.finfo(float).tiny,
    )
    diagnostic = _group_linf_gradient_kkt_diagnostic(
        gradient,
        validated_groups,
        lam,
        beta,
        tolerance=tolerance,
        face_tolerance=face_tolerance,
        scale=scale,
    )
    intercept_residual = (
        abs(float(normalized_weight @ residual)) if fit_intercept else 0.0
    )
    combined = max(float(diagnostic["kkt_residual"]), intercept_residual)
    relaxed_combined = max(
        float(diagnostic["relaxed_kkt_residual"]), intercept_residual
    )
    scale = float(diagnostic["scale"])
    certified = bool(
        diagnostic["certified"]
        and intercept_residual <= tolerance * scale
    )
    relaxed_accepted = bool(
        diagnostic["relaxed_accepted"]
        and intercept_residual <= tolerance * scale
    )
    diagnostic.update(
        {
            "kkt_residual": combined,
            "relative_kkt_residual": combined / scale,
            "relative_residual": combined / scale,
            "relaxed_kkt_residual": relaxed_combined,
            "relative_relaxed_kkt_residual": relaxed_combined / scale,
            "relaxed_relative_residual": relaxed_combined / scale,
            "scale": scale,
            "certified": certified,
            "accepted": relaxed_accepted,
            "relaxed_accepted": relaxed_accepted,
            "acceptance_mode": (
                "strict"
                if certified
                else "relaxed_face"
                if relaxed_accepted
                else "rejected"
            ),
            "fit_intercept": bool(fit_intercept),
            "intercept": intercept_value,
            "intercept_stationarity_residual": intercept_residual,
            "weight_sum": weight_sum,
        }
    )
    return diagnostic


def group_linf_regression_lambda_max(
    X: np.ndarray,
    y: np.ndarray,
    groups: Iterable[Sequence[int]],
    *,
    sample_weight: np.ndarray | None = None,
    fit_intercept: bool = False,
    tolerance: float = 1e-8,
) -> float:
    """Return the zero-coefficient threshold for weighted regression."""

    X, y, normalized_weight, _ = _validate_regression_inputs(X, y, sample_weight)
    if not isinstance(fit_intercept, (bool, np.bool_)):
        raise ValueError("fit_intercept must be a boolean")
    if fit_intercept:
        X_work = X - normalized_weight @ X
        y_work = y - float(normalized_weight @ y)
    else:
        X_work = X
        y_work = y
    linear = X_work.T @ (normalized_weight * y_work)
    return group_linf_lambda_max(linear, groups, tolerance=tolerance)


__all__ = [
    "group_linf_lambda_max",
    "group_linf_quadratic_kkt_diagnostic",
    "group_linf_regression_kkt_diagnostic",
    "group_linf_regression_lambda_max",
]
