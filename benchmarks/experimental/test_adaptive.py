"""Tests for quality-controlled adaptive APA-APG2 path sampling."""
from __future__ import annotations

import numpy as np
import pytest

from benchmarks.experimental.adaptive import (
    _AdaptivePoint,
    _secant_prediction,
    apa_apg_adaptive_regression_path,
)


def _prox_linf(value: np.ndarray, radius: float) -> np.ndarray:
    """Return prox_{radius ||.||_inf}(value) for a small test oracle."""

    if radius <= 0.0:
        return value.copy()
    absolute = np.abs(value)
    if radius >= float(np.sum(absolute)):
        return np.zeros_like(value)
    ordered = np.sort(absolute)[::-1]
    cumulative = np.cumsum(ordered)
    candidates = (cumulative - radius) / np.arange(1, value.size + 1)
    active = np.flatnonzero(ordered > candidates)
    threshold = float(candidates[active[-1]])
    l1_projection = np.sign(value) * np.maximum(absolute - threshold, 0.0)
    return value - l1_projection


def test_one_feature_adaptive_path_matches_analytic_path_and_kkt() -> None:
    X = np.asarray([[-2.0], [-0.5], [1.0], [3.0]])
    y = np.asarray([-2.2, -0.1, 2.4, 6.2])
    weight = np.asarray([1.0, 2.0, 3.0, 4.0])
    path = apa_apg_adaptive_regression_path(
        X,
        y,
        [np.asarray([0])],
        sample_weight=weight,
        initial_points=3,
        coefficient_tolerance=1e-7,
        objective_tolerance=1e-10,
        kkt_tolerance=1e-7,
        tol=1e-10,
        max_points=24,
    )

    normalized_weight = weight / weight.sum()
    x_mean = float(normalized_weight @ X[:, 0])
    y_mean = float(normalized_weight @ y)
    centered_x = X[:, 0] - x_mean
    centered_y = y - y_mean
    curvature = float(normalized_weight @ centered_x**2)
    score = float(normalized_weight @ (centered_x * centered_y))

    assert path.status == "complete"
    assert not path.exact
    assert path.metadata["quadratic_backend"] == "diagonal"
    assert path.metadata["point_solutions_certified"]
    assert all(item["kkt_certified"] for item in path.diagnostics)
    for lam in np.linspace(0.0, abs(score), 21):
        beta, intercept = path.at(float(lam))
        expected = np.sign(score) * max(abs(score) - lam, 0.0) / curvature
        np.testing.assert_allclose(beta, [expected], rtol=2e-8, atol=2e-9)
        assert intercept == pytest.approx(y_mean - x_mean * expected, abs=2e-9)


def test_internal_knots_trigger_refinement_and_zero_bypasses_point_solver() -> None:
    target = np.asarray([3.0, 2.0, 0.5])
    X = np.sqrt(3.0) * np.eye(3)
    y = X @ target
    calls: list[float] = []

    def analytic_solver(**kwargs):
        lam = float(kwargs["lam"])
        assert lam > 0.0
        calls.append(lam)
        return _prox_linf(target, lam), {"converged": True, "n_iter": 1}

    path = apa_apg_adaptive_regression_path(
        X,
        y,
        [np.arange(3)],
        fit_intercept=False,
        initial_points=2,
        minimum_ratio=1e-3,
        coefficient_tolerance=2e-3,
        objective_tolerance=1e-5,
        support_tolerance=1e-9,
        max_points=100,
        max_depth=12,
        point_solver=analytic_solver,
    )

    assert path.status == "complete"
    assert path.metadata["n_refinement_points"] > path.metadata["initial_points"]
    assert calls and all(lam > 0.0 for lam in calls)
    evaluation = np.linspace(0.0, float(np.sum(target)), 201)
    errors = []
    for lam in evaluation:
        estimated, _ = path.at(float(lam))
        errors.append(np.linalg.norm(estimated - _prox_linf(target, float(lam))))
    assert max(errors) < 0.015


def test_positive_user_lambda_max_is_not_collapsed_and_direct_mode_avoids_gram(
    monkeypatch,
) -> None:
    import benchmarks.experimental.adaptive as adaptive_module

    def cache_must_not_run(*args, **kwargs):
        raise AssertionError("the quadratic cache was unexpectedly prepared")

    monkeypatch.setattr(
        adaptive_module, "_make_quadratic_regression_loss", cache_must_not_run
    )
    calls: list[float] = []

    def analytic_solver(**kwargs):
        lam = float(kwargs["lam"])
        calls.append(lam)
        return np.asarray([1.0 - lam]), {"converged": True, "n_iter": 1}

    path = apa_apg_adaptive_regression_path(
        np.ones((3, 1)),
        np.ones(3),
        [np.asarray([0])],
        fit_intercept=False,
        lambda_max=1e-12,
        minimum_ratio=0.1,
        initial_points=2,
        coefficient_tolerance=1e-8,
        objective_tolerance=1e-8,
        max_points=12,
        cache_quadratic=False,
        point_solver=analytic_solver,
    )

    assert path.lambdas[0] == pytest.approx(1e-12)
    assert path.lambdas[-1] == 0.0
    assert calls and all(lam > 0.0 for lam in calls)
    assert not path.metadata["quadratic_statistics_available"]


def test_assumed_diagonal_adaptive_path_supports_matrix_free_kkt() -> None:
    X = np.sqrt(2.0) * np.eye(2)
    target = np.asarray([1.0, 0.3])
    path = apa_apg_adaptive_regression_path(
        X,
        X @ target,
        [np.arange(2)],
        fit_intercept=False,
        initial_points=3,
        coefficient_tolerance=1e-4,
        objective_tolerance=1e-8,
        support_tolerance=None,
        max_points=40,
        tol=1e-10,
        kkt_tolerance=1e-6,
        assume_diagonal_gram=True,
    )

    assert path.status == "complete"
    assert path.metadata["quadratic_backend"] == "diagonal"
    assert path.metadata["assumed_diagonal_gram"] is True
    assert path.metadata["gram_max_off_diagonal"] is None
    assert path.metadata["point_solutions_certified"]
    assert all(item["kkt_certified"] for item in path.diagnostics)


def test_resolution_limited_intervals_are_unresolved_not_validated() -> None:
    calls: list[float] = []

    def solver(**kwargs):
        calls.append(float(kwargs["lam"]))
        return np.asarray([max(1.0 - kwargs["lam"], 0.0)]), {
            "converged": True,
            "n_iter": 1,
        }

    path = apa_apg_adaptive_regression_path(
        np.ones((3, 1)),
        np.ones(3),
        [np.asarray([0])],
        fit_intercept=False,
        lambda_max=1.0,
        minimum_ratio=0.1,
        initial_points=2,
        lambda_tolerance=2.0,
        point_solver=solver,
    )

    assert path.status == "partial"
    assert not path.metadata["adaptive_tolerance_met"]
    assert path.metadata["accepted_intervals"] == ()
    assert {
        item["reason"] for item in path.metadata["unresolved_intervals"]
    } == {"lambda_resolution"}
    # The exact zero-solution upper anchor bypasses APA and only the other
    # positive anchor is solved; no midpoint is silently called validated.
    assert len(calls) == 1


def test_failed_midpoint_kkt_prevents_interval_acceptance() -> None:
    X = np.sqrt(2.0) * np.eye(2)
    target = np.asarray([1.0, 0.3])
    y = X @ target

    def inaccurate_solver(**kwargs):
        return np.zeros(2), {"converged": True, "n_iter": 1}

    path = apa_apg_adaptive_regression_path(
        X,
        y,
        [np.arange(2)],
        fit_intercept=False,
        initial_points=2,
        max_points=12,
        kkt_tolerance=1e-8,
        point_solver=inaccurate_solver,
    )

    assert path.status == "partial"
    assert not path.metadata["point_solutions_certified"]
    assert any(
        item["reason"] == "midpoint_kkt_failure"
        for item in path.metadata["unresolved_intervals"]
    )


def test_relaxed_kkt_acceptance_does_not_claim_point_certification(
    monkeypatch,
) -> None:
    import benchmarks.experimental.adaptive as adaptive_module

    def relaxed_only_diagnostic(*args, **kwargs):
        return {
            "certified": False,
            "accepted": True,
            "relaxed_accepted": True,
            "acceptance_mode": "relaxed_face",
            "kkt_residual": 1.0,
            "relative_kkt_residual": 1.0,
            "relaxed_kkt_residual": 0.0,
            "relative_relaxed_kkt_residual": 0.0,
        }

    monkeypatch.setattr(
        adaptive_module,
        "group_linf_quadratic_kkt_diagnostic",
        relaxed_only_diagnostic,
    )

    def affine_solver(**kwargs):
        lam = float(kwargs["lam"])
        return np.asarray([max(1.0 - lam, 0.0)]), {
            "converged": True,
            "n_iter": 1,
        }

    path = apa_apg_adaptive_regression_path(
        np.ones((3, 1)),
        np.ones(3),
        [np.asarray([0])],
        fit_intercept=False,
        initial_points=2,
        coefficient_tolerance=1e-8,
        objective_tolerance=1e-8,
        max_points=12,
        kkt_tolerance=1e-6,
        kkt_face_tolerance=1e-2,
        point_solver=affine_solver,
    )

    assert path.status == "complete"
    assert path.metadata["point_solutions_kkt_accepted"]
    assert not path.metadata["point_solutions_certified"]
    assert path.metadata["kkt_acceptance_mode"] == "relaxed_face"
    assert path.metadata["kkt_face_relaxation_used"]
    assert all(item["kkt_accepted"] for item in path.diagnostics)
    assert not any(item["kkt_certified"] for item in path.diagnostics)


def test_rank_deficient_design_is_reported_as_nonunique() -> None:
    x = np.linspace(-1.0, 1.0, 8)
    X = np.column_stack((x, x))
    y = 2.0 * x

    def solver(**kwargs):
        return kwargs["beta_init"], {"converged": True, "n_iter": 1}

    path = apa_apg_adaptive_regression_path(
        X,
        y,
        [np.asarray([0, 1])],
        fit_intercept=False,
        lambda_max=1.0,
        minimum_ratio=0.1,
        initial_points=2,
        lambda_tolerance=2.0,
        point_solver=solver,
    )

    assert path.status == "nonunique_design"
    assert path.metadata["design_rank"] == 1
    assert not path.metadata["coefficient_path_unique"]


def test_intercept_translation_and_relative_weight_scaling_are_invariant() -> None:
    X = np.asarray([[-2.0], [-0.5], [1.0], [3.0]])
    y = np.asarray([-2.2, -0.1, 2.4, 6.2])
    weight = np.asarray([1.0, 2.0, 3.0, 4.0])
    kwargs = {
        "groups": [np.asarray([0])],
        "initial_points": 3,
        "coefficient_tolerance": 1e-6,
        "objective_tolerance": 1e-9,
        "support_tolerance": None,
        "max_points": 24,
        "tol": 1e-10,
    }
    base = apa_apg_adaptive_regression_path(
        X, y, sample_weight=weight, **kwargs
    )
    x_shift = 7.0
    y_shift = -3.0
    translated = apa_apg_adaptive_regression_path(
        X + x_shift,
        y + y_shift,
        sample_weight=11.0 * weight,
        **kwargs,
    )

    np.testing.assert_allclose(translated.lambdas, base.lambdas, atol=1e-11)
    np.testing.assert_allclose(
        translated.coefficients, base.coefficients, atol=2e-9
    )
    np.testing.assert_allclose(
        translated.intercepts,
        base.intercepts + y_shift - x_shift * base.coefficients[:, 0],
        atol=2e-9,
    )


def test_automatic_lambda_max_rejects_uncovered_nonzero_score() -> None:
    with pytest.raises(ValueError, match="not covered"):
        apa_apg_adaptive_regression_path(
            np.eye(2),
            np.asarray([1.0, 2.0]),
            [np.asarray([0])],
            fit_intercept=False,
        )


def test_secant_predictor_is_exact_for_affine_path_and_has_safe_fallback() -> None:
    before = _AdaptivePoint(3.0, np.asarray([1.0, -1.0]), {}, "anchor")
    previous = _AdaptivePoint(2.0, np.asarray([2.0, -0.5]), {}, "anchor")
    predicted, method = _secant_prediction(1.0, previous, before)
    np.testing.assert_allclose(predicted, [3.0, 0.0])
    assert method == "secant"

    extreme = _AdaptivePoint(
        3.0, np.asarray([-1e6, 1e6]), {}, "anchor"
    )
    predicted, method = _secant_prediction(1.0, previous, extreme)
    np.testing.assert_array_equal(predicted, previous.beta)
    assert method == "previous_fallback"
