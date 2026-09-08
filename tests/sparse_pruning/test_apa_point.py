import numpy as np
import pytest
from numpy.testing import assert_array_equal

from imodels.tree.sparse_pruning.optimization._quadratic import _make_quadratic_regression_loss
from imodels.tree.sparse_pruning.optimizations import (
    get_gcv_reg_param,
    get_reg_set,
    hiCAP_classification,
    hiCAP_regression,
)


def test_shared_input_validation_preserves_point_and_path_copy_contracts():
    from imodels.tree.sparse_pruning.optimization import _problem, apa_point

    X = np.arange(6.0).reshape(3, 2)
    y = np.arange(3.0)
    groups = [np.array([0, 1], dtype=np.intp)]
    beta_init = np.ones(2)
    weights = np.ones(3)
    path = _problem._prepare_problem(X, y, groups, beta_init, weights, np.inf)
    point = apa_point._validate_solver_inputs(
        X, y, groups, 0.5, beta_init, weights, 1.0, 1.0, 10, 1e-8, np.inf
    )

    assert not np.shares_memory(path[2][0], groups[0])
    assert not np.shares_memory(path[3], beta_init)
    assert not np.shares_memory(path[4], weights)
    assert np.shares_memory(point[2][0], groups[0])
    assert not np.shares_memory(point[4], beta_init)
    assert np.shares_memory(point[5], weights)
    assert path[-1] == point[-1] == "inf"
    assert apa_point._group_penalty is _problem._group_penalty


@pytest.mark.parametrize("solver", [hiCAP_regression, hiCAP_classification])
@pytest.mark.parametrize("field", ["X", "y", "beta_init", "sample_weight"])
def test_point_solvers_reject_complex_inputs_before_float_conversion(solver, field):
    inputs = dict(
        X=np.array([[1.0], [2.0], [3.0]]),
        y=np.array([0.0, 1.0, 1.0]),
        groups=[np.array([0])],
        beta_init=np.zeros(1),
        sample_weight=np.ones(3),
        lam=1.0,
    )
    inputs[field] = inputs[field].astype(complex) + 1j
    with pytest.raises(ValueError, match=f"{field} must contain real values"):
        solver(**inputs)


def test_diagonal_detection_is_scale_relative_for_tiny_correlated_column():
    first = np.array([-1.0, 1.0, -1.0, 1.0])
    X = np.column_stack((first, 1e-14 * first))
    loss = _make_quadratic_regression_loss(
        X, np.arange(4.0), np.ones(X.shape[0])
    )

    assert loss.backend == "gram"
    assert loss.max_off_diagonal < 1e-12
    assert loss.max_off_diagonal_correlation == pytest.approx(1.0)


@pytest.mark.parametrize("scales", [(1e100, 1e100), (1e-100, 1e-100),
                                   (1e150, 1e-150)])
def test_diagonal_detection_handles_extreme_finite_column_scales(scales):
    X = np.array([[1.0, 1.0], [2.0, 1.0], [3.0, 1.0]]) * scales
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        loss = _make_quadratic_regression_loss(X, np.arange(1.0, 4.0), np.ones(3))

    assert loss.backend == "gram"
    assert loss.max_off_diagonal_correlation == pytest.approx(np.sqrt(6.0 / 7.0))


def test_quadratic_cache_preserves_representable_large_diagonal():
    X = 1.5e154 * np.eye(2)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        loss = _make_quadratic_regression_loss(X, np.ones(2), np.ones(2))

    assert loss.backend == "diagonal"
    np.testing.assert_allclose(loss.diagonal, [1.125e308, 1.125e308])
    assert loss.max_off_diagonal_correlation == 0.0


@pytest.mark.parametrize("assume_diagonal", [False, True])
@pytest.mark.parametrize("field", ["X", "y"])
def test_quadratic_cache_rejects_unrepresentable_statistics(assume_diagonal, field):
    X, y = np.eye(2), np.ones(2)
    if field == "X":
        X *= 1e200
    else:
        y *= 1e200
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        with pytest.raises(ValueError, match="quadratic.*representable"):
            _make_quadratic_regression_loss(
                X, y, np.ones(2), assume_diagonal_gram=assume_diagonal
            )


def test_public_topology_path_rejects_large_correlated_design():
    from imodels.tree.sparse_pruning.optimization import laminar_group_linf_exact_topology_path

    X = 1e100 * np.array([[1.0, 1.0], [2.0, 1.0], [3.0, 1.0]])
    with pytest.raises(ValueError, match="diagonal"):
        laminar_group_linf_exact_topology_path(
            X, np.arange(1.0, 4.0), [[0, 1], [1]], fit_intercept=False
        )


def test_assumed_diagonal_quadratic_cache_does_not_store_full_gram():
    X = np.array(
        [
            [1.0, 2.0, -1.0],
            [3.0, -2.0, 0.5],
            [-4.0, 1.0, 2.0],
            [0.5, 3.0, -2.0],
        ]
    )
    y = np.array([2.0, -1.0, 4.0, 0.5])
    weights = np.array([1.0, 3.0, 2.0, 4.0])
    loss = _make_quadratic_regression_loss(
        X, y, weights, assume_diagonal_gram=True
    )
    normalized_weights = weights / weights.sum()

    assert loss.gram is None
    assert loss.backend == "diagonal"
    assert loss.assumed_diagonal_gram
    assert loss.max_off_diagonal is None
    assert loss.max_off_diagonal_correlation is None
    np.testing.assert_allclose(
        loss.diagonal,
        np.sum(normalized_weights[:, None] * X**2, axis=0),
    )
    np.testing.assert_allclose(loss.linear, X.T @ (normalized_weights * y))
    assert loss.response_squared == pytest.approx(normalized_weights @ (y**2))


def test_assumed_diagonal_quadratic_cache_flag_requires_boolean():
    with pytest.raises(ValueError, match="assume_diagonal_gram"):
        _make_quadratic_regression_loss(
            np.eye(2), np.ones(2), np.ones(2), assume_diagonal_gram="yes"
        )


def test_gcv_rejects_missing_tree():
    with pytest.raises((TypeError, ValueError), match="regression|Regressor|fitted"):
        get_gcv_reg_param(None, np.ones((2, 1)), np.ones(2))


def test_deprecated_reg_set_name_remains_functional():
    X = np.arange(12).reshape(6, 2)
    y = np.arange(6)

    with pytest.warns(DeprecationWarning, match="deprecated"):
        X_full, y_full = get_reg_set(
            "full",
            X,
            y,
            random_state=0,
            n_samples=6,
            n_samples_bootstrap=6,
        )

    assert_array_equal(X_full, X)
    assert_array_equal(y_full, y)


def test_nested_group_one_step_uses_full_vector_proximal_average_l2():
    X = np.zeros((2, 3))
    y = np.zeros(2)
    beta_init = np.array([3.0, 4.0, 3.0])
    groups = [np.array([1, 2]), np.array([2])]

    beta = hiCAP_regression(
        X,
        y,
        groups,
        lam=1.0,
        beta_init=beta_init,
        gamma1=1.0,
        max_iter=1,
        ord=2,
    )

    # Group 1 maps [4, 3] to [3.2, 2.4]; group 2 maps 3 to 2.
    # Each full-vector prox has weight 1/2, and index 0 is unchanged by both.
    np.testing.assert_allclose(beta, np.array([3.0, 3.6, 2.2]))


def test_nested_group_one_step_uses_full_vector_proximal_average_linf():
    X = np.zeros((2, 3))
    y = np.zeros(2)
    beta_init = np.array([5.0, 3.0, 1.0])
    groups = [np.array([1, 2]), np.array([2])]

    beta = hiCAP_regression(
        X,
        y,
        groups,
        lam=1.0,
        beta_init=beta_init,
        gamma1=1.0,
        max_iter=1,
        ord="inf",
    )

    # prox_{||.||_inf}([3, 1]) = [2, 1] and prox_{|.|}(1) = 0.
    np.testing.assert_allclose(beta, np.array([5.0, 2.5, 0.5]))


def test_multiple_groups_continue_the_adaptive_approximation_schedule():
    beta, info = hiCAP_regression(
        np.zeros((2, 3)),
        np.zeros(2),
        [np.array([1, 2]), np.array([2])],
        lam=1.0,
        beta_init=np.array([0.0, 3.0, 2.0]),
        gamma1=1.0,
        max_iter=5,
        tol=1.0,
        return_info=True,
    )

    assert np.all(np.isfinite(beta))
    assert info["n_iter"] == 5
    assert not info["converged"]
    assert info["approximation_parameter"] == pytest.approx(0.2)


def test_regression_single_group_matches_closed_form_solution_and_reports_info():
    X = np.eye(3)
    y = np.array([2.0, 3.0, 4.0])
    groups = [np.array([1, 2])]

    beta, info = hiCAP_regression(
        X,
        y,
        groups,
        lam=2.0 / 3.0,
        gamma1=2.0,
        max_iter=100,
        tol=1e-10,
        ord=2,
        return_info=True,
    )

    expected = np.array([2.0, 1.8, 2.4])
    np.testing.assert_allclose(beta, expected, atol=1e-7)
    assert info["converged"]
    assert 0 < info["n_iter"] <= 100
    assert info["relative_step_norm"] <= 1e-10
    assert info["loss"] == pytest.approx(2.0 / 3.0)
    assert info["penalty"] == pytest.approx(2.0)
    assert info["objective"] == pytest.approx(8.0 / 3.0)
    assert info["weight_sum"] == pytest.approx(3.0)


@pytest.mark.parametrize(
    "solver,y",
    [
        (hiCAP_regression, np.array([1.0, -2.0, 0.5])),
        (hiCAP_classification, np.array([0.0, 1.0, 0.0])),
    ],
)
def test_integer_sample_weights_match_row_duplication(solver, y):
    X = np.array(
        [
            [1.0, -1.0, 0.5],
            [1.0, 0.5, -0.5],
            [1.0, 1.5, 1.0],
        ]
    )
    weights = np.array([1, 3, 2])
    groups = [np.array([1, 2]), np.array([2])]
    repeated = np.repeat(np.arange(len(y)), weights)

    weighted_beta = solver(
        X,
        y,
        groups,
        lam=1.5,
        sample_weight=weights,
        max_iter=2000,
        tol=1e-9,
    )
    repeated_beta = solver(
        X[repeated],
        y[repeated],
        groups,
        lam=1.5,
        max_iter=2000,
        tol=1e-9,
    )

    np.testing.assert_allclose(weighted_beta, repeated_beta, atol=1e-7)


@pytest.mark.parametrize(
    "solver,y",
    [
        (hiCAP_regression, np.array([1.0, -2.0, 0.5])),
        (hiCAP_classification, np.array([0.0, 1.0, 0.0])),
    ],
)
def test_uniform_sample_weight_rescaling_does_not_change_solution(solver, y):
    X = np.array(
        [
            [1.0, -1.0, 0.5],
            [1.0, 0.5, -0.5],
            [1.0, 1.5, 1.0],
        ]
    )
    groups = [np.array([1, 2]), np.array([2])]

    beta = solver(
        X,
        y,
        groups,
        lam=0.25,
        sample_weight=np.ones(len(y)),
        max_iter=2000,
    )
    scaled_beta = solver(
        X,
        y,
        groups,
        lam=0.25,
        sample_weight=np.full(len(y), 100.0),
        max_iter=2000,
    )

    np.testing.assert_allclose(beta, scaled_beta, atol=1e-10)


def test_classification_expit_is_stable_for_extreme_linear_predictors():
    X = np.array([[1.0, 1e6], [1.0, -1e6]])
    y = np.array([1.0, 0.0])

    with np.errstate(over="raise", invalid="raise"):
        beta, info = hiCAP_classification(
            X,
            y,
            [np.array([1])],
            lam=1.0,
            beta_init=np.array([0.0, 1e6]),
            max_iter=1,
            return_info=True,
        )

    assert np.all(np.isfinite(beta))
    assert info["n_iter"] == 1
    assert np.isfinite(info["objective"])


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"lam": 0.0}, "lam"),
        ({"lam": np.complex128(1.0 + 2.0j)}, "lam"),
        ({"groups": []}, "groups"),
        ({"ord": 1}, "ord"),
        ({"groups": [np.array([3])]}, "outside"),
        ({"groups": [np.array([1, 1])]}, "duplicate"),
        ({"beta_init": np.zeros(2)}, "beta_init"),
        ({"sample_weight": np.ones(2)}, "sample_weight"),
        (
            {"sample_weight": np.full(3, np.finfo(float).max)},
            "total weight",
        ),
        ({"gamma1": "1"}, "gamma1"),
        ({"gamma1": np.complex128(1.0 + 2.0j)}, "gamma1"),
        ({"a": np.array([1.0])}, "a"),
        ({"a": np.complex128(1.0 + 2.0j)}, "a"),
        ({"tol": True}, "tol"),
        ({"tol": np.complex128(1.0 + 2.0j)}, "tol"),
        ({"max_iter": 1.5}, "max_iter"),
    ],
)
def test_solver_input_validation(kwargs, match):
    params = {
        "X": np.ones((3, 3)),
        "y": np.arange(3.0),
        "groups": [np.array([1, 2])],
        "lam": 1.0,
    }
    params.update(kwargs)

    with pytest.raises(ValueError, match=match):
        hiCAP_regression(**params)


def test_classification_rejects_nonbinary_targets():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        hiCAP_classification(
            np.ones((3, 2)),
            np.array([0.0, 1.0, 2.0]),
            [np.array([1])],
            lam=1.0,
        )
