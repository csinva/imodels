"""Exact-proximal logistic and softmax fits at positive pruning penalties.

Binary coefficients describe the log odds of the second class. Multiclass
coefficients use the gauge-invariant penalty ``sum_g a_g max_{v in g}
(max_c B[v,c] - min_c B[v,c])``. Internally this is minimized as twice the
laminar infinity penalty over *free* class logits. Optimizing the rowwise
common shifts makes these objectives equivalent; fixing those shifts during
optimization would generally solve a different problem.

Paths contain samples with stationarity diagnostics, not coefficient knots.
Logistic and softmax paths are curved; unpenalized endpoints can be infinite.
"""
from __future__ import annotations

import warnings
from collections.abc import Iterable
from typing import Any

import numpy as np
from scipy.special import expit, logsumexp

from ._problem import _validate_lambdas
from ._result import RegularizationPath
from .proximal import _positive_integer, _prox_certificate_passes
from .tree_prox import (
    LaminarGroupLinfProx, _stable_l2_norm, _validate_group_weights,
)


def _finite_array(value: Any, name: str) -> np.ndarray:
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must contain finite real values")
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain finite real values") from exc
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain finite real values")
    return result


def _positive_scalar(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{name} must be a positive finite real scalar")
    try:
        result = float(value)
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a positive finite real scalar") from exc
    if not np.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be a positive finite real scalar")
    return result


class _DenseDesign:
    def __init__(self, X: np.ndarray):
        self.X = _finite_array(X, "X")
        if self.X.ndim != 2 or self.X.shape[0] == 0:
            raise ValueError("X must be a nonempty two-dimensional design")
        self.shape = self.X.shape

    def matmat(self, coefficients: np.ndarray) -> np.ndarray:
        return self.X @ coefficients

    def rmatmat(self, residuals: np.ndarray) -> np.ndarray:
        return self.X.T @ residuals


def _targets(y: np.ndarray, n: int, weights: np.ndarray):
    raw = np.asarray(y)
    if raw.ndim == 1 and raw.shape == (n,):
        if np.iscomplexobj(raw) or (
            raw.dtype.kind in "f" and not np.all(np.isfinite(raw))
        ):
            raise ValueError("class labels must be finite and noncomplex")
        try:
            classes, codes = np.unique(raw, return_inverse=True)
        except (TypeError, ValueError) as exc:
            raise ValueError("class labels must be mutually comparable") from exc
        if any(label is None or isinstance(label, (complex, np.complexfloating))
               or (isinstance(label, (float, np.floating)) and not np.isfinite(label))
               for label in classes):
            raise ValueError("class labels must be finite and noncomplex")
        probabilities = np.eye(len(classes), dtype=float)[codes]
    elif raw.ndim == 2 and raw.shape[0] == n:
        probabilities = _finite_array(raw, "y")
        row_sums = probabilities.sum(axis=1)
        if np.any(probabilities < 0) or not np.allclose(
            row_sums, 1.0, rtol=1e-10, atol=1e-12
        ):
            raise ValueError("two-dimensional y must contain class-probability rows")
        probabilities = probabilities / row_sums[:, None]
        classes = np.arange(raw.shape[1])
    else:
        raise ValueError("y must be labels (n,) or class probabilities (n, K)")
    mass = weights @ probabilities
    if len(classes) < 2 or np.any(mass <= 0):
        raise ValueError("at least two classes with positive effective mass are required")
    return probabilities, classes, mass


class _LaminarClassificationSolver:
    """Reuse one loss and proximal operator across a sampled path.

    ``design`` supplies ``shape=(n,p)``, ``matmat(B)`` and ``rmatmat(R)`` for
    two-dimensional arrays. Its optional positive ``coefficient_scale`` scalar
    or length-p vector changes the optimization metric, not the objective.
    This permits tree-specific linear-time operators without a dense design.
    """

    def __init__(self, design, y, groups, *, sample_weight=None,
                 group_weights=None, max_iter=5000, tol=1e-8,
                 fit_intercept=True):
        self.design = design
        self.n_samples, self.n_features = design.shape
        if self.n_samples < 1 or self.n_features < 0:
            raise ValueError("design must have positive rows and nonnegative columns")
        self.max_iter = _positive_integer(max_iter, "max_iter")
        self.tol = _positive_scalar(tol, "tol")
        if not isinstance(fit_intercept, (bool, np.bool_)):
            raise ValueError("fit_intercept must be a boolean")
        self.fit_intercept = bool(fit_intercept)
        weights = (np.ones(self.n_samples) if sample_weight is None
                   else _finite_array(sample_weight, "sample_weight"))
        if weights.shape != (self.n_samples,) or np.any(weights < 0):
            raise ValueError("sample_weight must have shape (n,) and be nonnegative")
        largest = np.max(weights)
        if largest <= 0:
            raise ValueError("sample_weight must have positive total mass")
        weights = weights / largest
        self.weights = weights / weights.sum()
        probabilities, self.classes, self.class_mass = _targets(
            y, self.n_samples, self.weights
        )
        self.binary = len(self.classes) == 2
        self.dimension = 1 if self.binary else len(self.classes)
        self.target = probabilities[:, 1:2] if self.binary else probabilities
        scale = getattr(design, "coefficient_scale", None)
        if scale is None and isinstance(design, _DenseDesign):
            maximum = np.max(np.abs(design.X), axis=0)
            safe_maximum = np.where(maximum > 0, maximum, 1.0)
            rms = maximum * np.sqrt(self.weights @ (design.X / safe_maximum)**2)
            scale = 1.0 / np.maximum(rms, np.sqrt(np.finfo(float).tiny))
            scale[rms == 0] = 1.0
        scale = _finite_array(1.0 if scale is None else scale, "coefficient_scale")
        if scale.ndim == 0:
            scale = np.full(self.n_features, float(scale))
        if scale.shape != (self.n_features,) or np.any(scale <= 0):
            raise ValueError("coefficient_scale must be positive with shape (p,)")
        self.scale = scale[:, None]
        raw_groups = list(groups)
        if self.n_features:
            base = LaminarGroupLinfProx(raw_groups, self.n_features, group_weights)
            covered = np.zeros(self.n_features, dtype=bool)
            for group, weight in zip(base.groups, base.group_weights):
                if weight > 0:
                    covered[group] = True
            if not np.all(covered):
                raise ValueError("every coefficient must belong to a positive-weight group")
            self.groups, self.group_weights = base.groups, base.group_weights
            expanded = [(g[:, None] * self.dimension + np.arange(self.dimension)).ravel()
                        for g in self.groups]
            self.prox = LaminarGroupLinfProx(
                expanded, self.n_features * self.dimension,
                self.group_weights * (1.0 if self.binary else 2.0),
                coordinate_weights=np.repeat(scale, self.dimension),
            )
        else:
            if raw_groups:
                raise ValueError("an intercept-only design requires empty groups")
            _validate_group_weights(group_weights, 0)
            self.groups, self.group_weights, self.prox = (), np.empty(0), None
        self.lipschitz = 0.25

    def _multiply(self, value, *, transpose=False):
        expected = (self.n_features if transpose else self.n_samples, self.dimension)
        operation = self.design.rmatmat if transpose else self.design.matmat
        raw = operation(value)
        if np.iscomplexobj(raw):
            raise FloatingPointError("classification design returned complex values")
        result = np.asarray(raw, dtype=float)
        if result.shape != expected or not np.all(np.isfinite(result)):
            raise FloatingPointError("classification design returned invalid or nonfinite values")
        return result

    def _loss_gradient(self, theta, intercept, *, gradient=True):
        logits = self._multiply(theta * self.scale) + intercept
        if not np.all(np.isfinite(logits)):
            raise FloatingPointError("classification logits overflowed")
        if self.binary:
            losses = (self.target * np.logaddexp(0.0, -logits)
                      + (1.0 - self.target) * np.logaddexp(0.0, logits)).ravel()
            probabilities = expit(logits)
        else:
            shifted = logits - logits.max(axis=1, keepdims=True)
            normalizer = logsumexp(shifted, axis=1, keepdims=True)
            losses = (normalizer - np.sum(self.target * shifted, axis=1, keepdims=True)).ravel()
            probabilities = np.exp(shifted - normalizer)
        loss = float(self.weights @ losses)
        if not np.isfinite(loss):
            raise FloatingPointError("classification loss overflowed")
        if not gradient:
            return loss
        residual = self.weights[:, None] * (probabilities - self.target)
        grad = self._multiply(residual, transpose=True) * self.scale
        grad_intercept = residual.sum(axis=0) if self.fit_intercept else np.zeros(self.dimension)
        return loss, grad, grad_intercept

    def _penalty(self, theta):
        beta = theta * self.scale
        magnitude = np.abs(beta[:, 0]) if self.binary else np.ptp(beta, axis=1)
        return float(sum(w * np.max(magnitude[g])
                         for g, w in zip(self.groups, self.group_weights)))

    def _initial(self, beta, intercept):
        shape = ((self.n_features,) if self.binary
                 else (self.n_features, self.dimension))
        beta = np.zeros(shape) if beta is None else _finite_array(beta, "beta_init")
        if beta.shape != shape:
            raise ValueError(f"beta_init must have shape {shape}")
        beta = beta.reshape(self.n_features, self.dimension).copy()
        if not self.binary:
            # Minimize the lifted infinity penalty without changing any logits.
            beta -= (0.5 * beta.max(axis=1, keepdims=True)
                     + 0.5 * beta.min(axis=1, keepdims=True))
        if intercept is None:
            intercept = np.log(self.class_mass)
            intercept = (np.array([intercept[1] - intercept[0]]) if self.binary
                         else intercept - intercept.mean())
            if not self.fit_intercept:
                intercept = np.zeros(self.dimension)
        else:
            intercept = _finite_array(intercept, "intercept_init")
            expected = () if self.binary else (self.dimension,)
            if intercept.shape != expected:
                raise ValueError(f"intercept_init must have shape {expected}")
            intercept = intercept.reshape(self.dimension).copy()
            if not self.fit_intercept and np.any(intercept != 0):
                raise ValueError("intercept_init must be zero when fit_intercept=False")
        if not self.binary:
            intercept -= intercept.mean()
        return beta / self.scale, intercept

    def _certificate(self, theta, intercept, lam, lipschitz):
        # Certify a penalty-minimizing gauge of the returned class contrasts.
        representative = theta.copy()
        if not self.binary:
            representative -= (0.5 * representative.max(axis=1, keepdims=True)
                               + 0.5 * representative.min(axis=1, keepdims=True))
        loss, gradient, grad_intercept = self._loss_gradient(representative, intercept)
        prox_info = {}
        if self.prox is None:
            mapping = gradient
            prox_certified = True
        else:
            updated, prox_info = self.prox(
                (representative - gradient / lipschitz).ravel(), lam / lipschitz,
                return_info=True,
            )
            mapping = (representative - updated.reshape(representative.shape)) * lipschitz
            prox_certified = _prox_certificate_passes(prox_info, self.tol)
        original = np.concatenate([(mapping / self.scale).ravel(), grad_intercept])
        scaled = np.concatenate([mapping.ravel(), grad_intercept])
        original_gradient = np.concatenate([(gradient / self.scale).ravel(), grad_intercept])
        scaled_gradient = np.concatenate([gradient.ravel(), grad_intercept])
        residual = _stable_l2_norm(original)
        residual_scale = max(1.0, _stable_l2_norm(original_gradient))
        scaled_residual = _stable_l2_norm(scaled)
        scaled_scale = max(1.0, _stable_l2_norm(scaled_gradient))
        relative = max(residual / residual_scale, scaled_residual / scaled_scale)
        penalty = self._penalty(representative)
        return {
            "converged": bool(relative <= self.tol and prox_certified),
            "certified": bool(relative <= self.tol and prox_certified),
            "loss": loss, "unscaled_penalty": penalty, "objective": loss + lam * penalty,
            "stationarity_residual": residual, "stationarity_scale": residual_scale,
            "relative_stationarity_residual": relative,
            "scaled_stationarity_residual": scaled_residual,
            "stationarity_kind": "proximal_gradient_mapping",
            "coefficient_distance_upper_bound": None, "objective_gap_upper_bound": None,
            "prox_certified": prox_certified,
            "prox_raw_relative_duality_gap": prox_info.get("raw_relative_duality_gap", 0.0),
            "prox_max_relative_dual_l1_violation": prox_info.get("max_relative_dual_l1_violation", 0.0),
            "prox_relative_moreau_residual": prox_info.get("relative_moreau_residual", 0.0),
        }

    def solve(self, lam, beta_init=None, intercept_init=None):
        lam = _positive_scalar(lam, "lam (zero may have infinite classification coefficients)")
        theta, intercept = self._initial(beta_init, intercept_init)
        extrapolated, extrapolated_intercept = theta.copy(), intercept.copy()
        momentum, lipschitz = 1.0, self.lipschitz
        diagnostic = self._certificate(theta, intercept, lam, lipschitz)
        iteration, backtracks = 0, 0
        for iteration in range(1, self.max_iter + 1):
            if diagnostic["converged"]:
                iteration -= 1
                break
            value, gradient, grad_intercept = self._loss_gradient(
                extrapolated, extrapolated_intercept
            )
            for _ in range(100):
                center = extrapolated - gradient / lipschitz
                candidate = (center if self.prox is None else self.prox(
                    center.ravel(), lam / lipschitz
                ).reshape(center.shape))
                candidate_intercept = extrapolated_intercept - grad_intercept / lipschitz
                delta, delta_intercept = (candidate - extrapolated,
                                           candidate_intercept - extrapolated_intercept)
                candidate_loss = self._loss_gradient(candidate, candidate_intercept, gradient=False)
                upper = (value + np.sum(gradient * delta) + grad_intercept @ delta_intercept
                         + 0.5 * lipschitz * (np.sum(delta**2) + delta_intercept @ delta_intercept))
                if candidate_loss <= upper + 32 * np.finfo(float).eps * max(1.0, abs(value)):
                    break
                lipschitz *= 2.0
                backtracks += 1
                if not np.isfinite(lipschitz):
                    raise FloatingPointError("classification backtracking overflowed")
            else:
                raise FloatingPointError("classification backtracking failed")
            change, intercept_change = candidate - theta, candidate_intercept - intercept
            restart = (np.sum((extrapolated - candidate) * change)
                       + (extrapolated_intercept - candidate_intercept) @ intercept_change) > 0
            theta, intercept = candidate, candidate_intercept
            next_momentum = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * momentum**2))
            factor = 0.0 if restart else (momentum - 1.0) / next_momentum
            extrapolated = theta + factor * change
            extrapolated_intercept = intercept + factor * intercept_change
            momentum = 1.0 if restart else next_momentum
            if iteration % 10 == 0 or iteration == self.max_iter:
                diagnostic = self._certificate(theta, intercept, lam, lipschitz)
        self.lipschitz = lipschitz
        beta = theta * self.scale
        if self.binary:
            beta, intercept_out = beta[:, 0], float(intercept[0])
        else:
            beta -= beta.mean(axis=1, keepdims=True)
            intercept_out = intercept - intercept.mean()
        diagnostic.update({
            "lambda": lam, "intercept": intercept_out, "classes": self.classes.copy(),
            "fit_intercept": self.fit_intercept, "n_iter": iteration,
            "n_backtracks": backtracks, "lipschitz_constant": lipschitz,
            "solver": "classification_fista_exact_laminar_prox",
        })
        return beta.copy(), diagnostic


def laminar_group_linf_classification(
    X: np.ndarray, y: np.ndarray, groups: Iterable[np.ndarray], lam: float, *,
    sample_weight=None, group_weights=None, beta_init=None, intercept_init=None,
    max_iter: int = 5000, tol: float = 1e-8, fit_intercept: bool = True,
    return_info: bool = False,
):
    """Minimize weighted mean logistic/softmax loss plus ``lam * penalty``.

    ``y`` contains labels or class-probability rows; the latter support exact
    aggregation of repeated design rows. Every class must have positive weight.
    Groups must be laminar and cover every feature with positive penalty weight.
    Binary output has shape ``(p,)``; multiclass output ``(p,K)`` has zero row
    means. ``return_info=True`` returns ``(coefficients, diagnostics)`` with the
    unpenalized intercept and convergence certificate. A certificate bounds a
    proximal-gradient residual, not coefficient error or objective suboptimality.
    """
    if not isinstance(return_info, (bool, np.bool_)):
        raise ValueError("return_info must be a boolean")
    solver = _LaminarClassificationSolver(
        _DenseDesign(X), y, groups, sample_weight=sample_weight,
        group_weights=group_weights, max_iter=max_iter, tol=tol,
        fit_intercept=fit_intercept,
    )
    beta, info = solver.solve(lam, beta_init, intercept_init)
    if return_info:
        return beta, info
    if not info["converged"]:
        warnings.warn("classification solver did not reach the requested tolerance", RuntimeWarning)
    return beta


def _classification_path(solver, lambdas, *, beta_init=None, intercept_init=None,
                         adaptive_tol=None, max_points=100):
    requested, ordered = _validate_lambdas(lambdas)
    if np.any(ordered <= 0):
        raise ValueError(
            "classification paths require positive lambdas; "
            "zero may have infinite coefficients"
        )
    max_points = _positive_integer(max_points, "max_points")
    if adaptive_tol is not None:
        adaptive_tol = _positive_scalar(adaptive_tol, "adaptive_tol")
        if ordered.size > max_points:
            raise ValueError("max_points must include all requested lambdas")
    points = {}
    for lam in ordered:
        beta_init, info = solver.solve(float(lam), beta_init, intercept_init)
        intercept_init = info["intercept"]
        points[float(lam)] = (beta_init, info)
    pending = list(zip(ordered[:-1], ordered[1:]))[::-1] if adaptive_tol else []
    adaptive_complete, tested_intervals = True, 0
    while pending:
        high, low = pending.pop()
        middle = 0.5 * high + 0.5 * low
        if middle == high or middle == low:
            continue
        if len(points) >= max_points:
            adaptive_complete = False
            break
        left, right = points[high], points[low]
        beta_guess = 0.5 * left[0] + 0.5 * right[0]
        intercept_guess = 0.5 * left[1]["intercept"] + 0.5 * right[1]["intercept"]
        beta, info = solver.solve(middle, beta_guess, intercept_guess)
        points[middle] = beta, info
        error = _stable_l2_norm(np.concatenate([
            (beta - beta_guess).ravel(), np.atleast_1d(info["intercept"] - intercept_guess),
        ])) / max(1.0, _stable_l2_norm(np.concatenate([
            beta.ravel(), np.atleast_1d(info["intercept"]),
        ])))
        tested_intervals += 1
        info["midpoint_relative_deviation"] = error
        if error > adaptive_tol:
            pending.extend([(middle, low), (high, middle)])
    ordered = np.array(sorted(points, reverse=True))
    betas, diagnostics = zip(*(points[lam] for lam in ordered))
    certified = all(info["certified"] for info in diagnostics)
    return RegularizationPath(
        lambdas=ordered, coefficients=np.stack(betas),
        intercepts=np.asarray([info["intercept"] for info in diagnostics]),
        penalties=np.array([info["unscaled_penalty"] for info in diagnostics]),
        diagnostics=diagnostics, method="classification_exact_prox", exact=False,
        status="complete" if certified and adaptive_complete else "partial",
        metadata={
            "classes": solver.classes.copy(), "requested_lambdas": requested,
            "certified": certified, "fit_intercept": solver.fit_intercept,
            "point_solutions_certified": certified, "warm_started": True,
            "adaptive_tol": adaptive_tol, "adaptive_complete": adaptive_complete,
            "max_points": max_points, "tested_intervals": tested_intervals,
            "adaptive_criterion": (
                "relative midpoint coefficient/intercept deviation; not a global error bound"
            ),
        },
    )


def laminar_group_linf_classification_path(
    X: np.ndarray, y: np.ndarray, groups: Iterable[np.ndarray], lambdas, *,
    sample_weight=None, group_weights=None, beta_init=None, intercept_init=None,
    max_iter: int = 5000, tol: float = 1e-8, fit_intercept: bool = True,
    adaptive_tol: float | None = None, max_points: int = 100,
) -> RegularizationPath:
    """Warm-start positive penalties in descending order, optionally refining.

    With ``adaptive_tol``, solved arithmetic midpoints are compared with linear
    interpolation of endpoint coefficients and intercepts. Intervals exceeding
    the relative tolerance are subdivided until accepted or ``max_points`` is
    reached. This diagnostic is not a uniform interpolation-error bound. Every
    evaluated midpoint is retained. ``status='partial'`` means a point failed
    its stationarity tolerance or adaptive refinement exhausted its point budget.
    """
    solver = _LaminarClassificationSolver(
        _DenseDesign(X), y, groups, sample_weight=sample_weight,
        group_weights=group_weights, max_iter=max_iter, tol=tol,
        fit_intercept=fit_intercept,
    )
    return _classification_path(
        solver, lambdas, beta_init=beta_init, intercept_init=intercept_init,
        adaptive_tol=adaptive_tol, max_points=max_points,
    )
