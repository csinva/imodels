"""Tests for the certified infinity-hiCAP regression homotopy."""
from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.optimize import LinearConstraint, minimize

from imodels.tree.sparse_pruning.optimization.hicap import hicap_regression_path


@pytest.mark.parametrize("field", ["X", "y", "sample_weight"])
def test_exact_hicap_rejects_complex_inputs_before_float_conversion(field):
    inputs = dict(
        X=np.array([[1.0], [2.0], [3.0]]),
        y=np.array([1.0, 2.0, 3.0]),
        groups=[np.array([0])],
        sample_weight=np.ones(3),
        fit_intercept=False,
    )
    inputs[field] = inputs[field].astype(complex) + 1j
    with pytest.raises(ValueError, match=f"{field} must contain real values"):
        hicap_regression_path(**inputs)


@pytest.mark.parametrize("parameter", ["tolerance", "tie_tolerance"])
def test_exact_hicap_rejects_complex_tolerances(parameter):
    with pytest.raises(ValueError, match=parameter):
        hicap_regression_path(
            np.ones((3, 1)),
            np.arange(3.0),
            [np.array([0])],
            **{parameter: np.complex128(1e-9 + 1j)},
        )


def _fixed_lambda_epigraph_oracle(
    X: np.ndarray,
    y: np.ndarray,
    groups: list[np.ndarray],
    lam: float,
    *,
    sample_weight: np.ndarray | None = None,
    fit_intercept: bool = True,
) -> tuple[np.ndarray, float, float]:
    """Independently solve one small epigraph QP with scipy's SLSQP."""

    n, p = X.shape
    weight = np.ones(n) if sample_weight is None else np.asarray(sample_weight)
    weight = weight / weight.sum()
    if fit_intercept:
        x_mean = weight @ X
        y_mean = float(weight @ y)
        X_work = X - x_mean
        y_work = y - y_mean
    else:
        x_mean = np.zeros(p)
        y_mean = 0.0
        X_work = X
        y_work = y
    H = X_work.T @ (weight[:, None] * X_work)
    h = X_work.T @ (weight * y_work)
    m = len(groups)
    rows = []
    for group_number, group in enumerate(groups):
        for feature in group:
            for sign in (1.0, -1.0):
                row = np.zeros(p + m)
                row[int(feature)] = sign
                row[p + group_number] = -1.0
                rows.append(row)
    A = np.asarray(rows)
    linear = np.r_[-h, lam * np.ones(m)]

    def objective(z: np.ndarray) -> float:
        return float(0.5 * z[:p] @ H @ z[:p] + linear @ z)

    def gradient(z: np.ndarray) -> np.ndarray:
        result = linear.copy()
        result[:p] += H @ z[:p]
        return result

    beta_ls = np.linalg.solve(H, h)
    t_ls = np.asarray([np.max(np.abs(beta_ls[group])) for group in groups])
    result = minimize(
        objective,
        np.r_[beta_ls, t_ls],
        jac=gradient,
        constraints=(LinearConstraint(A, -np.inf, 0.0),),
        method="SLSQP",
        options={"ftol": 1e-13, "maxiter": 4000},
    )
    assert result.success, result.message
    beta = result.x[:p]
    intercept = y_mean - float(x_mean @ beta)
    penalty = sum(float(np.max(np.abs(beta[group]))) for group in groups)
    residual = y - intercept - X @ beta
    full_objective = 0.5 * float(weight @ residual**2) + lam * penalty
    return beta, intercept, full_objective


def _objective(
    X: np.ndarray,
    y: np.ndarray,
    groups: list[np.ndarray],
    beta: np.ndarray,
    intercept: float,
    lam: float,
    sample_weight: np.ndarray | None = None,
) -> float:
    weight = np.ones(y.size) if sample_weight is None else sample_weight
    weight = weight / np.sum(weight)
    residual = y - intercept - X @ beta
    penalty = sum(float(np.max(np.abs(beta[group]))) for group in groups)
    return 0.5 * float(weight @ residual**2) + lam * penalty


def test_one_feature_path_is_the_analytic_soft_threshold_path() -> None:
    X = np.asarray([[-2.0], [-0.5], [1.0], [3.0]])
    y = np.asarray([-2.2, -0.1, 2.4, 6.2])
    weight = np.asarray([1.0, 2.0, 3.0, 4.0])
    groups = [np.asarray([0])]

    path = hicap_regression_path(
        X, y, groups, sample_weight=weight, fit_intercept=True
    )

    normalized_weight = weight / weight.sum()
    x_mean = normalized_weight @ X[:, 0]
    y_mean = normalized_weight @ y
    centered_x = X[:, 0] - x_mean
    centered_y = y - y_mean
    curvature = float(normalized_weight @ centered_x**2)
    score = float(normalized_weight @ (centered_x * centered_y))
    lambda_max = abs(score)
    beta_ls = score / curvature

    assert path.exact
    assert path.status == "complete"
    assert_allclose(path.lambdas, [lambda_max, 0.0], rtol=1e-10, atol=1e-12)
    assert_allclose(path.coefficients[:, 0], [0.0, beta_ls], atol=1e-11)
    assert_allclose(
        path.intercepts, y_mean - x_mean * path.coefficients[:, 0], atol=1e-11
    )
    lam = 0.37 * lambda_max
    beta, intercept = path.at(lam)
    expected_beta = np.sign(score) * (abs(score) - lam) / curvature
    assert_allclose(beta, [expected_beta], rtol=1e-10, atol=1e-11)
    assert_allclose(intercept, y_mean - x_mean * expected_beta, atol=1e-11)


def test_nested_path_matches_independent_qp_at_knots_and_midpoints() -> None:
    rng = np.random.default_rng(20260902)
    X = rng.normal(size=(24, 3))
    X[:, 1] += 0.35 * X[:, 0]
    X[:, 2] -= 0.20 * X[:, 0]
    y = X @ np.asarray([1.4, -0.65, 0.35]) + rng.normal(scale=0.18, size=24)
    groups = [
        np.asarray([0, 1, 2]),
        np.asarray([0]),
        np.asarray([1, 2]),
        np.asarray([1]),
        np.asarray([2]),
    ]

    path = hicap_regression_path(X, y, groups, tolerance=1e-9)

    assert path.exact, path.metadata.get("failure_message")
    assert path.status == "complete"
    assert path.lambdas[-1] == 0.0
    assert np.all(np.diff(path.lambdas) < 0)
    assert np.all(np.diff(path.penalties) >= -1e-9)
    assert len(path.events) == path.n_points
    assert all(diagnostic["certified"] for diagnostic in path.diagnostics)
    assert max(diagnostic["kkt_residual"] for diagnostic in path.diagnostics) < 1e-7

    evaluation_lambdas = list(path.lambdas)
    evaluation_lambdas.extend(
        0.5 * (left + right)
        for left, right in zip(path.lambdas[:-1], path.lambdas[1:])
    )
    for lam in evaluation_lambdas:
        beta, intercept = path.at(float(lam))
        oracle_beta, oracle_intercept, oracle_objective = (
            _fixed_lambda_epigraph_oracle(X, y, groups, float(lam))
        )
        assert_allclose(beta, oracle_beta, rtol=2e-6, atol=2e-7)
        assert_allclose(intercept, oracle_intercept, rtol=2e-6, atol=2e-7)
        assert_allclose(
            _objective(X, y, groups, beta, float(intercept), float(lam)),
            oracle_objective,
            rtol=1e-9,
            atol=2e-10,
        )


def test_path_interpolation_is_affine_within_every_segment() -> None:
    X = np.asarray(
        [
            [1.0, -0.2],
            [-1.0, 0.3],
            [0.2, 1.2],
            [0.7, -1.1],
            [-0.4, -0.8],
            [1.3, 0.5],
        ]
    )
    y = np.asarray([1.1, -0.7, -0.4, 1.4, 0.3, 0.8])
    groups = [np.asarray([0, 1]), np.asarray([0]), np.asarray([1])]
    path = hicap_regression_path(X, y, groups, fit_intercept=False)
    assert path.exact

    for index, (upper, lower) in enumerate(
        zip(path.lambdas[:-1], path.lambdas[1:])
    ):
        fraction = 0.31
        lam = fraction * upper + (1.0 - fraction) * lower
        beta, intercept = path.at(float(lam))
        expected_beta = (
            fraction * path.coefficients[index]
            + (1.0 - fraction) * path.coefficients[index + 1]
        )
        assert_allclose(beta, expected_beta, rtol=1e-10, atol=1e-11)
        assert intercept == pytest.approx(0.0)


def test_simultaneous_tied_faces_are_followed_without_perturbation() -> None:
    # The centered columns are orthonormal and have identical scores.  Both
    # coefficients therefore enter on the same infinity-norm face at the
    # first knot; no random jitter should be needed to choose a path.
    X = np.sqrt(2.0) * np.asarray(
        [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]
    )
    y = X @ np.asarray([1.0, 1.0])
    groups = [np.asarray([0, 1])]

    path = hicap_regression_path(
        X, y, groups, fit_intercept=True, tolerance=1e-10
    )

    assert path.exact
    assert path.status == "complete"
    assert_allclose(path.lambdas, [2.0, 0.0], atol=1e-11)
    assert_allclose(path.coefficients, [[0.0, 0.0], [1.0, 1.0]], atol=1e-11)
    beta, intercept = path.at(1.0)
    assert_allclose(beta, [0.5, 0.5], atol=1e-11)
    assert intercept == pytest.approx(0.0, abs=1e-12)


def test_integer_weights_equal_row_replication_and_fit_intercept() -> None:
    X = np.asarray(
        [[-1.0, 0.3], [0.1, 1.2], [0.8, -0.4], [1.7, 0.9], [-0.3, -1.0]]
    )
    y = np.asarray([-0.4, -0.7, 1.5, 1.2, 0.8])
    weight = np.asarray([1, 3, 2, 1, 2])
    groups = [np.asarray([0, 1]), np.asarray([0]), np.asarray([1])]
    weighted = hicap_regression_path(X, y, groups, sample_weight=weight)
    repeated = hicap_regression_path(
        np.repeat(X, weight, axis=0), np.repeat(y, weight), groups
    )

    assert weighted.exact and repeated.exact
    assert_allclose(weighted.lambdas, repeated.lambdas, rtol=2e-8, atol=2e-10)
    assert_allclose(weighted.coefficients, repeated.coefficients, atol=2e-8)
    assert_allclose(weighted.intercepts, repeated.intercepts, atol=2e-8)
    # Relative weights: uniform rescaling must not alter the path.
    rescaled = hicap_regression_path(X, y, groups, sample_weight=11.0 * weight)
    assert_allclose(weighted.lambdas, rescaled.lambdas, atol=2e-10)
    assert_allclose(weighted.coefficients, rescaled.coefficients, atol=2e-9)


def test_rank_deficiency_is_reported_as_nonexact_endpoint_path() -> None:
    x = np.linspace(-1.0, 1.0, 8)
    X = np.column_stack([x, x])
    y = 2.0 * x
    groups = [np.asarray([0, 1]), np.asarray([0]), np.asarray([1])]

    path = hicap_regression_path(X, y, groups, fit_intercept=False)

    assert not path.exact
    assert path.status == "nonunique_design"
    assert path.metadata["endpoint_only"]
    assert path.metadata["rank"] == 1


@pytest.mark.parametrize(
    "groups, match",
    [
        ([np.asarray([0]), np.asarray([1])], "root group"),
        (
            [np.asarray([0, 1, 2]), np.asarray([0, 1]), np.asarray([1, 2])],
            "laminar",
        ),
        (
            [np.asarray([0, 1, 2]), np.asarray([0]), np.asarray([0])],
            "duplicate groups",
        ),
    ],
)
def test_group_validation_rejects_non_tree_families(
    groups: list[np.ndarray], match: str
) -> None:
    X = np.arange(15.0).reshape(5, 3)
    y = np.arange(5.0)
    with pytest.raises(ValueError, match=match):
        hicap_regression_path(X, y, groups)
