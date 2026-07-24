import numpy as np
import pytest
from numpy.testing import assert_array_equal

from imodels.tree.sparse_pruning.optimizations import (
    get_gcv_reg_param,
    get_reg_set,
    hiCAP_classification,
    hiCAP_regression,
)


def test_deprecated_gcv_name_remains_importable_but_disabled():
    with pytest.warns(DeprecationWarning, match="GCV"):
        with pytest.raises(NotImplementedError, match="disabled"):
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
        ({"a": np.array([1.0])}, "a"),
        ({"tol": True}, "tol"),
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
