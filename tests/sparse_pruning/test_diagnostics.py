"""Focused tests for independent sum-group-Linf optimality diagnostics."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from imodels.tree.sparse_pruning.optimization import diagnostics
from imodels.tree.sparse_pruning.optimization.diagnostics import (
    group_linf_lambda_max,
    group_linf_quadratic_kkt_diagnostic,
    group_linf_regression_kkt_diagnostic,
    group_linf_regression_lambda_max,
)
from imodels.tree.sparse_pruning.optimization.hicap import hicap_regression_path


def test_single_coordinate_kkt_and_lambda_max_are_analytic():
    group = [np.array([0])]
    result = group_linf_quadratic_kkt_diagnostic(
        np.array([[1.0]]), np.array([3.0]), group, 2.0, np.array([1.0])
    )

    assert result["lp_success"]
    assert result["certified"]
    assert result["kkt_residual"] == pytest.approx(0.0, abs=1e-12)
    assert group_linf_lambda_max(np.array([3.0]), group) == pytest.approx(3.0)


def test_tied_and_unique_nonzero_faces_use_the_true_linf_subgradient():
    group = [np.array([0, 1])]
    tied = group_linf_quadratic_kkt_diagnostic(
        np.eye(2),
        np.array([1.5, -2.5]),
        group,
        2.0,
        np.array([1.0, -1.0]),
        face_tolerance=0.0,
    )
    unique = group_linf_quadratic_kkt_diagnostic(
        np.eye(2),
        np.array([2.0, 2.0]),
        group,
        1.0,
        np.array([2.0, 1.0]),
        face_tolerance=0.0,
    )

    assert tied["certified"]
    assert tied["n_active_subgradient_variables"] == 2
    assert tied["kkt_residual"] == pytest.approx(0.0, abs=1e-12)
    assert not unique["certified"]
    assert unique["kkt_residual"] == pytest.approx(1.0)


def test_relaxed_near_tie_is_never_reported_as_a_strict_certificate():
    # beta[1] is close to, but is not, the unique maximum beta[0].  Enlarging
    # the face admits e_1 and can make the relaxed stationarity residual zero;
    # the true Linf subgradient contains only e_0, whose residual is one.
    beta = np.array([1.0, 0.995])
    result = group_linf_quadratic_kkt_diagnostic(
        np.ones(2),
        np.array([1.0, 1.995]),
        [np.array([0, 1])],
        1.0,
        beta,
        tolerance=1e-8,
        face_tolerance=1e-2,
    )

    assert not result["certified"]
    assert result["kkt_residual"] == pytest.approx(1.0)
    assert result["relative_kkt_residual"] > 0.5
    assert result["relaxed_accepted"]
    assert result["accepted"]
    assert result["acceptance_mode"] == "relaxed_face"
    assert result["relaxed_kkt_residual"] == pytest.approx(0.0, abs=1e-12)
    assert result["relaxed_face_residual"] == pytest.approx(0.005)


def test_nonzero_group_subgradient_has_unit_mass():
    result = group_linf_quadratic_kkt_diagnostic(
        np.eye(1),
        np.array([1.0]),
        [np.array([0])],
        1.0,
        np.array([1.0]),
        face_tolerance=0.0,
    )

    assert result["kkt_residual"] == pytest.approx(1.0)
    assert not result["certified"]


def test_zero_group_uses_l1_ball_and_lambda_zero_skips_lp():
    group = [np.array([0, 1])]
    zero_group = group_linf_quadratic_kkt_diagnostic(
        np.eye(2),
        np.array([0.2, -0.3]),
        group,
        0.4,
        np.zeros(2),
        tolerance=1e-10,
    )
    least_squares = group_linf_quadratic_kkt_diagnostic(
        np.diag([2.0, 4.0]),
        np.array([1.0, -2.0]),
        group,
        0.0,
        np.array([0.5, -0.5]),
    )

    assert zero_group["n_zero_groups"] == 1
    assert zero_group["kkt_residual"] == pytest.approx(0.05)
    assert not zero_group["certified"]
    assert least_squares["certified"]
    assert least_squares["lp_status"] is None
    assert "no subgradient LP" in least_squares["lp_message"]


@pytest.mark.parametrize(
    ("lam", "beta"),
    [
        (0.0, np.array([0.8, -0.2, 0.4])),
        (0.6, np.array([0.8, -0.8, 0.0])),
    ],
)
def test_diagonal_vector_kkt_matches_explicit_diagonal_matrix(lam, beta):
    diagonal = np.array([1.5, 0.75, 2.25])
    linear = np.array([1.1, -0.7, 0.35])
    groups = [np.array([0, 1, 2]), np.array([0]), np.array([1, 2])]

    compact = group_linf_quadratic_kkt_diagnostic(
        diagonal, linear, groups, lam, beta
    )
    explicit = group_linf_quadratic_kkt_diagnostic(
        np.diag(diagonal), linear, groups, lam, beta
    )

    assert compact.keys() == explicit.keys()
    for key, expected in explicit.items():
        if isinstance(expected, float):
            assert compact[key] == pytest.approx(expected)
        else:
            assert compact[key] == expected


@pytest.mark.parametrize(
    ("diagonal", "linear", "beta", "match"),
    [
        (np.array([]), np.array([]), np.array([]), "non-empty"),
        (np.ones(2), np.ones(1), np.zeros(1), "shape"),
        (np.array([1.0, np.nan]), np.ones(2), np.zeros(2), "finite"),
        (
            np.array([1.0, -1e-5]),
            np.ones(2),
            np.zeros(2),
            "positive semidefinite",
        ),
    ],
)
def test_invalid_gram_diagonals_raise(diagonal, linear, beta, match):
    with pytest.raises(ValueError, match=match):
        group_linf_quadratic_kkt_diagnostic(
            diagonal,
            linear,
            [np.array([0])],
            1.0,
            beta,
        )


@pytest.mark.parametrize(
    ("groups", "linear", "expected"),
    [
        ([np.array([0, 1])], np.array([2.0, -1.0]), 3.0),
        ([np.array([0]), np.array([1])], np.array([2.0, -1.0]), 2.0),
        (
            [np.array([0, 1]), np.array([0]), np.array([1])],
            np.array([1.0, 1.0]),
            2.0 / 3.0,
        ),
        (
            [np.array([0, 1]), np.array([0, 1])],
            np.array([2.0, 1.0]),
            1.5,
        ),
    ],
)
def test_lambda_max_known_group_geometries(groups, linear, expected):
    assert group_linf_lambda_max(linear, groups) == pytest.approx(expected)


def test_lambda_max_is_scale_homogeneous_and_handles_uncovered_coordinates():
    groups = [np.array([0, 1])]
    linear = np.array([2.0, -1.0])
    baseline = group_linf_lambda_max(linear, groups)

    for multiplier in (1e-14, 1e14):
        assert group_linf_lambda_max(multiplier * linear, groups) == pytest.approx(
            multiplier * baseline, rel=1e-9
        )
    assert np.isinf(
        group_linf_lambda_max(np.array([0.0, 1.0]), [np.array([0])])
    )


def test_lambda_max_and_zero_beta_diagnostic_agree_for_overlapping_groups():
    linear = np.array([1.0, 1.0])
    groups = [np.array([0, 1]), np.array([0]), np.array([1])]
    threshold = group_linf_lambda_max(linear, groups)

    at_threshold = group_linf_quadratic_kkt_diagnostic(
        np.eye(2), linear, groups, threshold, np.zeros(2)
    )
    above = group_linf_quadratic_kkt_diagnostic(
        np.eye(2), linear, groups, 1.1 * threshold, np.zeros(2)
    )
    below = group_linf_quadratic_kkt_diagnostic(
        np.eye(2), linear, groups, 0.9 * threshold, np.zeros(2)
    )

    assert at_threshold["certified"]
    assert above["certified"]
    assert not below["certified"]


def test_tiny_quadratic_problem_is_not_hidden_by_lp_absolute_tolerances():
    scale = 1e-14
    group = [np.array([0])]
    optimum = group_linf_quadratic_kkt_diagnostic(
        np.array([[scale]]),
        np.array([3.0 * scale]),
        group,
        2.0 * scale,
        np.array([1.0]),
    )
    wrong = group_linf_quadratic_kkt_diagnostic(
        np.array([[scale]]),
        np.array([3.0 * scale]),
        group,
        2.0 * scale,
        np.array([0.0]),
        tolerance=1e-8,
    )

    assert optimum["certified"]
    assert optimum["relative_kkt_residual"] < 1e-9
    assert not wrong["certified"]
    assert wrong["relative_kkt_residual"] > 0.1


def test_face_tolerance_is_relative_to_each_nonzero_group():
    beta = np.array([1e-12, 0.0])
    result = group_linf_quadratic_kkt_diagnostic(
        np.eye(2),
        np.array([1e-12, 1.0]),
        [np.array([0, 1])],
        1.0,
        beta,
        face_tolerance=1e-6,
    )

    # The zero second coordinate is not admitted as a maximum-magnitude face
    # merely because the whole group is small on an absolute scale.
    assert result["n_subgradient_variables"] == 1
    assert result["kkt_residual"] == pytest.approx(1.0)
    assert not result["certified"]


def test_numerical_faces_relaxedly_accept_all_exact_hicap_knots():
    rng = np.random.default_rng(20260902)
    X = rng.normal(size=(24, 3))
    X[:, 1] += 0.35 * X[:, 0]
    X[:, 2] -= 0.20 * X[:, 0]
    y = X @ np.array([1.4, -0.65, 0.35]) + rng.normal(scale=0.18, size=24)
    groups = [
        np.array([0, 1, 2]),
        np.array([0]),
        np.array([1, 2]),
        np.array([1]),
        np.array([2]),
    ]
    path = hicap_regression_path(X, y, groups, tolerance=1e-9)

    diagnostics_at_knots = [
        group_linf_regression_kkt_diagnostic(
            X,
            y,
            groups,
            float(lam),
            beta,
            fit_intercept=True,
            intercept=float(intercept),
            tolerance=1e-6,
            face_tolerance=1e-6,
        )
        for lam, beta, intercept in zip(
            path.lambdas, path.coefficients, path.intercepts
        )
    ]
    assert path.exact
    assert all(item["relaxed_accepted"] for item in diagnostics_at_knots)
    assert all(item["accepted"] for item in diagnostics_at_knots)
    assert all(
        item["acceptance_mode"] in {"strict", "relaxed_face"}
        for item in diagnostics_at_knots
    )


def test_weighted_regression_wrapper_matches_manual_profiled_quadratic():
    X = np.array([[0.0, 1.0], [1.0, -1.0], [2.0, 2.0], [3.0, 0.5]])
    y = np.array([1.0, 2.0, -1.0, 4.0])
    weight = np.array([1.0, 3.0, 0.0, 2.0])
    normalized = weight / weight.sum()
    x_mean = normalized @ X
    y_mean = float(normalized @ y)
    centered_X = X - x_mean
    centered_y = y - y_mean
    gram = centered_X.T @ (normalized[:, None] * centered_X)
    linear = centered_X.T @ (normalized * centered_y)
    beta = np.linalg.solve(gram, linear)
    intercept = y_mean - float(x_mean @ beta)
    groups = [np.array([0, 1]), np.array([1])]

    wrapped = group_linf_regression_kkt_diagnostic(
        X,
        y,
        groups,
        0.0,
        beta,
        sample_weight=weight,
        fit_intercept=True,
    )
    explicit = group_linf_quadratic_kkt_diagnostic(
        gram, linear, groups, 0.0, beta
    )

    assert wrapped["certified"]
    assert wrapped["intercept"] == pytest.approx(intercept)
    assert wrapped["intercept_stationarity_residual"] < 1e-12
    assert wrapped["kkt_residual"] == pytest.approx(explicit["kkt_residual"])
    assert group_linf_regression_lambda_max(
        X,
        y,
        groups,
        sample_weight=weight,
        fit_intercept=True,
    ) == pytest.approx(group_linf_lambda_max(linear, groups))


def test_profiled_intercept_diagnostic_is_stable_to_large_feature_offsets():
    X = np.array([[0.0], [1.0], [2.0], [4.0]])
    y = np.array([1.0, -1.0, 3.0, 2.0])
    weight = np.array([1.0, 2.0, 1.0, 3.0])
    groups = [np.array([0])]
    beta = np.array([0.25])
    kwargs = dict(
        groups=groups,
        lam=0.4,
        beta=beta,
        sample_weight=weight,
        fit_intercept=True,
    )

    baseline = group_linf_regression_kkt_diagnostic(X, y, **kwargs)
    shifted = group_linf_regression_kkt_diagnostic(X + 1e12, y, **kwargs)
    assert shifted["kkt_residual"] == pytest.approx(
        baseline["kkt_residual"], rel=1e-8, abs=1e-12
    )
    assert group_linf_regression_lambda_max(
        X + 1e12,
        y,
        groups,
        sample_weight=weight,
        fit_intercept=True,
    ) == pytest.approx(
        group_linf_regression_lambda_max(
            X,
            y,
            groups,
            sample_weight=weight,
            fit_intercept=True,
        ),
        rel=1e-8,
        abs=1e-12,
    )


def test_regression_diagnostics_are_invariant_to_weight_rescaling():
    X = np.array([[0.0], [1.0], [2.0]])
    y = np.array([0.0, 2.0, 1.0])
    groups = [np.array([0])]
    kwargs = dict(groups=groups, lam=0.2, beta=np.array([0.5]))
    first = group_linf_regression_kkt_diagnostic(
        X, y, sample_weight=np.array([1.0, 2.0, 3.0]), **kwargs
    )
    second = group_linf_regression_kkt_diagnostic(
        X, y, sample_weight=np.array([10.0, 20.0, 30.0]), **kwargs
    )

    assert first["kkt_residual"] == pytest.approx(second["kkt_residual"])
    assert first["relative_kkt_residual"] == pytest.approx(
        second["relative_kkt_residual"]
    )


def test_kkt_lp_failure_is_a_structured_noncertificate(monkeypatch):
    failure = SimpleNamespace(success=False, x=None, status=4, message="failed")
    monkeypatch.setattr(diagnostics, "linprog", lambda *args, **kwargs: failure)

    result = group_linf_quadratic_kkt_diagnostic(
        np.eye(1), np.array([1.0]), [np.array([0])], 1.0, np.zeros(1)
    )
    assert not result["lp_success"]
    assert not result["certified"]
    assert np.isinf(result["kkt_residual"])
    assert result["lp_status"] == 4
    with pytest.raises(RuntimeError, match="lambda_max LP failed"):
        group_linf_lambda_max(np.array([1.0]), [np.array([0])])


@pytest.mark.parametrize(
    "call",
    [
        lambda: group_linf_quadratic_kkt_diagnostic(
            np.array([[1.0, 1.0], [0.0, 1.0]]),
            np.ones(2),
            [np.array([0, 1])],
            1.0,
            np.zeros(2),
        ),
        lambda: group_linf_quadratic_kkt_diagnostic(
            np.diag([1.0, -1e-11]),
            np.ones(2),
            [np.array([0, 1])],
            1.0,
            np.zeros(2),
        ),
        lambda: group_linf_quadratic_kkt_diagnostic(
            np.eye(1), np.ones(1), [np.array([0])], -1.0, np.zeros(1)
        ),
        lambda: group_linf_lambda_max(np.ones(2), [np.array([2])]),
    ],
)
def test_invalid_quadratic_inputs_raise(call):
    with pytest.raises(ValueError):
        call()
