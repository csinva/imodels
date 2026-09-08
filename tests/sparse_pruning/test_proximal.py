import numpy as np
import pytest

from imodels.tree.sparse_pruning.optimization import (
    hicap_regression_path,
    laminar_group_linf_regression,
    laminar_group_linf_regression_path,
)


def _diagonal_problem():
    diagonal = np.array([1.0, 0.7, 0.3, 0.15])
    linear = np.array([0.8, -0.5, 0.4, 0.2])
    n_features = diagonal.size
    X = np.diag(np.sqrt(n_features * diagonal))
    y = linear * np.sqrt(n_features) / np.sqrt(diagonal)
    groups = [
        np.arange(4),
        np.array([0, 1]),
        np.array([2, 3]),
        np.array([0]),
        np.array([3]),
    ]
    return X, y, groups


def test_diagonal_solver_compiles_group_hierarchy_only_once(monkeypatch):
    from imodels.tree.sparse_pruning.optimization import tree_prox
    from imodels.tree.sparse_pruning.optimization._quadratic import (
        _make_quadratic_regression_loss,
    )
    from imodels.tree.sparse_pruning.optimization.proximal import _LaminarQuadraticSolver

    calls = []
    original = tree_prox._laminar_leaf_to_root_order

    def record_compilation(*args):
        calls.append(1)
        return original(*args)

    monkeypatch.setattr(tree_prox, "_laminar_leaf_to_root_order", record_compilation)
    X, y, groups = _diagonal_problem()
    quadratic = _make_quadratic_regression_loss(X, y, np.ones(len(y)))
    solver = _LaminarQuadraticSolver(
        quadratic, groups, None, max_iter=1, tolerance=1e-10, restart=True
    )
    assert len(calls) == 1
    assert solver.diagonal_prox.groups is solver.euclidean_prox.groups
    assert solver.diagonal_prox.parents is solver.euclidean_prox.parents
    assert not solver.diagonal_prox.parents.flags.writeable
    assert all(not group.flags.writeable for group in solver.diagonal_prox.groups)
    np.testing.assert_array_equal(solver.euclidean_prox.coordinate_weights, np.ones(4))
    np.testing.assert_allclose(
        solver.diagonal_prox.coordinate_weights, 1.0 / np.sqrt(quadratic.diagonal)
    )
    assert not solver.diagonal_prox.coordinate_weights.flags.writeable


def test_diagonal_one_sweep_matches_complete_hicap_path():
    X, y, groups = _diagonal_problem()
    exact = hicap_regression_path(
        X, y, groups, fit_intercept=False, tolerance=1e-9
    )
    lambdas = np.unique(
        np.r_[exact.lambdas, 0.5 * (exact.lambdas[:-1] + exact.lambdas[1:])]
    )[::-1]

    sampled = laminar_group_linf_regression_path(
        X,
        y,
        groups,
        lambdas,
        fit_intercept=False,
        tol=1e-10,
    )
    expected = np.vstack([exact.at(lam)[0] for lam in lambdas])

    assert exact.exact
    assert sampled.status == "complete"
    assert not sampled.exact
    assert sampled.method == "laminar-diagonal-one-sweep"
    assert sampled.metadata["point_solutions_certified"] is True
    assert sampled.metadata["coefficient_knots_enumerated"] is False
    np.testing.assert_allclose(sampled.coefficients, expected, atol=2e-10)
    assert all(
        item["solver"] == "diagonal_one_sweep_laminar_prox"
        and item["n_iter"] == 1
        and item["relative_stationarity_residual"] <= 1e-10
        and abs(item["prox_raw_relative_duality_gap"]) <= 1e-10
        and item["prox_max_relative_dual_l1_violation"] <= 1e-10
        for item in sampled.diagnostics[:-1]
    )


def test_general_gram_fista_exact_prox_matches_hicap():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(80, 5))
    y = rng.normal(size=80)
    groups = [
        np.arange(5),
        np.array([0, 1]),
        np.array([2, 3, 4]),
        np.array([0]),
        np.array([3, 4]),
    ]
    exact = hicap_regression_path(
        X, y, groups, fit_intercept=False, tolerance=1e-9
    )
    lambdas = np.unique(
        np.r_[exact.lambdas, 0.5 * (exact.lambdas[:-1] + exact.lambdas[1:])]
    )[::-1]

    sampled = laminar_group_linf_regression_path(
        X,
        y,
        groups,
        lambdas,
        fit_intercept=False,
        tol=1e-9,
        max_iter=20_000,
    )
    expected = np.vstack([exact.at(lam)[0] for lam in lambdas])

    assert sampled.status == "complete"
    assert sampled.method == "laminar-exact-prox-fista"
    assert sampled.metadata["quadratic_backend"] == "gram"
    np.testing.assert_allclose(sampled.coefficients, expected, atol=2e-8)
    assert all(
        item.get("relative_stationarity_residual", 0.0) <= 1e-9
        for item in sampled.diagnostics
    )


def test_weighted_centered_point_solver_returns_profiled_intercept():
    X, y, groups = _diagonal_problem()
    X = np.column_stack((X[:, 0] + 4.0, X[:, 1:]))
    y = y + 7.5
    weights = np.array([1.0, 2.0, 3.0, 4.0])

    beta, info = laminar_group_linf_regression(
        X,
        y,
        groups,
        lam=0.07,
        sample_weight=weights,
        fit_intercept=True,
        return_info=True,
        tol=1e-9,
    )

    normalized = weights / weights.sum()
    residual = y - (info["intercept"] + X @ beta)
    assert normalized @ residual == pytest.approx(0.0, abs=2e-12)
    assert info["certified"] is True


def test_assume_diagonal_fast_path_is_reported():
    X, y, groups = _diagonal_problem()
    path = laminar_group_linf_regression_path(
        X,
        y,
        groups,
        [0.4, 0.1, 0.0],
        fit_intercept=False,
        assume_diagonal_gram=True,
    )

    assert path.metadata["assumed_diagonal_gram"] is True
    assert path.metadata["quadratic_backend"] == "diagonal"
    assert path.method == "laminar-diagonal-one-sweep"


def test_crossing_groups_are_rejected_by_regression_solver():
    X = np.eye(3)
    y = np.ones(3)
    with pytest.raises(ValueError, match="laminar"):
        laminar_group_linf_regression(
            X,
            y,
            [np.array([0, 1]), np.array([1, 2])],
            0.1,
            fit_intercept=False,
        )


def test_truncated_zero_lambda_lstsq_is_not_falsely_certified():
    X = np.column_stack(
        (np.array([1.0, 0.0, 0.0]), 1e-16 * np.array([1.0, 1.0, 0.0]))
    )
    y = np.array([0.0, 1.0, 0.0])

    _, info = laminar_group_linf_regression(
        X,
        y,
        [np.array([0, 1])],
        0.0,
        fit_intercept=False,
        tol=1e-12,
        return_info=True,
    )
    path = laminar_group_linf_regression_path(
        X,
        y,
        [np.array([0, 1])],
        [0.0],
        fit_intercept=False,
        tol=1e-12,
    )

    assert info["rank"] == 1
    assert info["certified"] is False
    assert info["relative_stationarity_residual"] > 0.9
    assert path.status == "partial"
    assert path.metadata["point_solutions_certified"] is False


def test_automatic_diagonal_surrogate_certifies_against_full_gram():
    n_features = 200
    correlation = 1e-12
    gram = np.full((n_features, n_features), correlation)
    np.fill_diagonal(gram, 1.0)
    factor = np.linalg.cholesky(gram)
    X = np.sqrt(n_features) * factor.T
    linear = np.linspace(-1.0, 1.0, n_features)
    y = n_features * np.linalg.solve(X.T, linear)

    _, info = laminar_group_linf_regression(
        X,
        y,
        [np.arange(n_features)],
        0.2,
        fit_intercept=False,
        tol=1e-13,
        return_info=True,
    )

    assert info["quadratic_backend"] == "diagonal"
    assert info["assumed_diagonal_gram"] is False
    assert info["certificate_uses_full_gram"] is True
    assert info["certified"] is False
    assert info["relative_stationarity_residual"] > 1e-13


def test_positive_lambda_zero_design_returns_certified_zero_solution():
    X = np.zeros((5, 3))
    y = np.arange(5.0)

    beta, info = laminar_group_linf_regression(
        X,
        y,
        [np.arange(3)],
        0.2,
        fit_intercept=False,
        return_info=True,
    )

    np.testing.assert_array_equal(beta, np.zeros(3))
    assert info["solver"] == "zero_quadratic"
    assert info["certified"] is True
    assert info["relative_stationarity_residual"] == 0.0


def test_diagonal_solver_certifies_extreme_finite_dynamic_range():
    scale = 1e-180
    sqrt_diagonal = np.array([1.0, 1e-17, 1e70])
    X = np.diag(np.sqrt(3.0) * sqrt_diagonal)
    y = np.sqrt(3.0) * np.array([scale, scale, 0.0])

    beta, info = laminar_group_linf_regression(
        X,
        y,
        [np.arange(3)],
        1e-200,
        fit_intercept=False,
        return_info=True,
    )

    expected = np.array([scale, 0.999 * scale / 1e-17, 0.0])
    np.testing.assert_allclose(beta, expected, rtol=2e-14, atol=0.0)
    assert info["quadratic_backend"] == "diagonal"
    assert info["certified"] is True
    assert info["relative_stationarity_residual"] <= 2e-14
    assert info["prox_relative_moreau_residual"] <= 2e-14
