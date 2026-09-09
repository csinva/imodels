import numpy as np
import pytest

from imodels.tree.sparse_pruning.optimization import (
    RegularizationPath,
    apa_apg_classification_path as public_classification_path,
    apa_apg_regression_path as public_regression_path,
    group_linf_regression_lambda_max,
    hicap_regression_path,
)
from imodels.tree.sparse_pruning.optimization.apa import (
    apa_apg_classification_path,
    apa_apg_regression_path,
)


def test_path_solvers_are_public_optimization_exports():
    assert public_regression_path is apa_apg_regression_path
    assert public_classification_path is apa_apg_classification_path
    assert callable(group_linf_regression_lambda_max)
    assert callable(hicap_regression_path)
    assert RegularizationPath.__name__ == "RegularizationPath"


def test_sparse_pruning_exports_only_estimators_and_fitted_tree_helpers():
    from imodels.tree import sparse_pruning
    from imodels.tree.sparse_pruning import optimization

    expected = {
        "SHSTreeClassifier", "SHSTreeClassifierCV",
        "SHSTreeRegressor", "SHSTreeRegressorCV",
        "SPTreeClassifier", "SPTreeClassifierCV",
        "SPTreeRegressor", "SPTreeRegressorCV",
        "fitted_tree_linf_exact_coefficient_path",
        "fitted_tree_linf_exact_topology_path",
        "fitted_tree_linf_classification",
        "fitted_tree_linf_classification_path",
        "materialize_fitted_tree_topology",
    }
    assert set(sparse_pruning.__all__) == expected
    assert all(callable(getattr(sparse_pruning, name)) for name in expected)
    assert all(not hasattr(sparse_pruning, name) for name in optimization.__all__)


def test_historical_optimization_imports_reexport_canonical_implementations():
    from imodels.tree.sparse_pruning import optimization, optimizations

    for name in ["hiCAP_regression", "hiCAP_classification", "proj_l1_ball"]:
        assert getattr(optimizations, name) is getattr(optimization, name)


def test_experimental_paths_are_not_runtime_exports():
    from imodels.tree import sparse_pruning
    from imodels.tree.sparse_pruning import optimization

    for name in ["apa_apg_adaptive_regression_path", "laminar_group_linf_topology_path"]:
        assert not hasattr(sparse_pruning, name)
        assert not hasattr(optimization, name)


@pytest.mark.parametrize(
    "solver", [apa_apg_regression_path, apa_apg_classification_path]
)
@pytest.mark.parametrize("field", ["X", "y", "beta_init", "sample_weight"])
def test_sampled_paths_reject_complex_inputs_before_float_conversion(solver, field):
    inputs = dict(
        X=np.array([[1.0], [2.0], [3.0]]),
        y=np.array([0.0, 1.0, 1.0]),
        groups=[np.array([0])],
        beta_init=np.zeros(1),
        sample_weight=np.ones(3),
        lambdas=[0.0],
    )
    inputs[field] = inputs[field].astype(complex) + 1j
    with pytest.raises(ValueError, match=f"{field} must contain real values"):
        solver(**inputs)


def test_regression_path_sorts_deduplicates_and_warm_starts_point_solver():
    calls = []

    def recording_solver(**kwargs):
        calls.append(
            {
                "lambda": kwargs["lam"],
                "beta_init": kwargs["beta_init"].copy(),
                "groups": [group.copy() for group in kwargs["groups"]],
                "ord": kwargs["ord"],
                "sample_weight": kwargs["sample_weight"].copy(),
            }
        )
        beta = kwargs["beta_init"] + kwargs["lam"]
        return beta, {"converged": True, "n_iter": 1}

    groups = [np.array([0]), np.array([0, 1])]
    original_groups = [group.copy() for group in groups]
    path = apa_apg_regression_path(
        np.eye(2),
        np.array([1.0, 2.0]),
        groups,
        lambdas=[1.0, 3.0, 2.0, 3.0],
        beta_init=np.array([10.0, 20.0]),
        sample_weight=np.array([2.0, 1.0]),
        ord=np.inf,
        point_solver=recording_solver,
    )

    np.testing.assert_array_equal(path.lambdas, np.array([3.0, 2.0, 1.0]))
    np.testing.assert_allclose(
        path.coefficients,
        np.array([[13.0, 23.0], [15.0, 25.0], [16.0, 26.0]]),
    )
    assert [call["lambda"] for call in calls] == [3.0, 2.0, 1.0]
    np.testing.assert_array_equal(calls[0]["beta_init"], [10.0, 20.0])
    np.testing.assert_array_equal(calls[1]["beta_init"], [13.0, 23.0])
    np.testing.assert_array_equal(calls[2]["beta_init"], [15.0, 25.0])
    assert all(call["ord"] == "inf" for call in calls)
    assert all(
        np.array_equal(group, original)
        for group, original in zip(groups, original_groups)
    )
    assert not path.exact
    assert path.method == "apa-apg2-warm-start"
    assert path.metadata["duplicates_removed"] == 1
    assert not path.metadata["cached_sufficient_statistics"]
    assert path.metadata["requested_lambdas"] == (1.0, 3.0, 2.0, 3.0)
    assert [info["warm_started_from_lambda"] for info in path.diagnostics] == [
        None,
        3.0,
        2.0,
    ]
    np.testing.assert_allclose(path.penalties, [36.0, 40.0, 42.0])


@pytest.mark.parametrize("ord", [2, "inf"])
def test_cached_quadratic_path_matches_direct_design_backend(ord):
    rng = np.random.default_rng(31)
    X = rng.normal(size=(18, 5))
    y = rng.normal(size=18)
    weights = rng.uniform(0.25, 2.0, size=18)
    groups = [np.arange(5), np.array([1, 2, 3]), np.array([3, 4])]
    kwargs = {
        "X": X,
        "y": y,
        "groups": groups,
        "lambdas": [1.0, 0.3, 0.08, 0.0],
        "sample_weight": weights,
        "ord": ord,
        "max_iter": 40,
    }

    cached = apa_apg_regression_path(**kwargs, cache_quadratic=True)
    direct = apa_apg_regression_path(**kwargs, cache_quadratic=False)

    np.testing.assert_allclose(cached.coefficients, direct.coefficients, atol=2e-12)
    np.testing.assert_allclose(cached.penalties, direct.penalties, atol=2e-12)
    assert cached.metadata["cached_sufficient_statistics"]
    assert cached.metadata["quadratic_backend"] == "gram"
    assert not direct.metadata["cached_sufficient_statistics"]


def test_cached_path_detects_diagonal_gram_and_prepares_it_once(monkeypatch):
    import imodels.tree.sparse_pruning.optimization.apa as apa_module

    calls = 0
    original = apa_module._make_quadratic_regression_loss

    def recording_prepare(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(
        apa_module, "_make_quadratic_regression_loss", recording_prepare
    )
    X = np.array(
        [
            [1.0, 1.0],
            [1.0, -1.0],
            [-1.0, 1.0],
            [-1.0, -1.0],
        ]
    )
    path = apa_apg_regression_path(
        X,
        np.array([2.0, 1.0, -1.0, -2.0]),
        [np.array([0, 1]), np.array([1])],
        lambdas=[1.0, 0.5, 0.1],
        ord="inf",
        max_iter=5,
    )

    assert calls == 1
    assert path.metadata["quadratic_backend"] == "diagonal"
    assert path.metadata["gram_max_off_diagonal"] == pytest.approx(0.0)
    assert all(
        diagnostic["quadratic_backend"] == "diagonal"
        for diagnostic in path.diagnostics
    )


def test_assumed_diagonal_path_matches_verified_diagonal_without_dense_gram():
    X = np.diag([1.0, 2.0, 3.0, 4.0])
    y = np.array([2.0, -1.0, 3.0, 0.5])
    weights = np.array([1.0, 2.0, 3.0, 4.0])
    groups = [np.arange(4), np.array([1, 2, 3])]
    kwargs = {
        "X": X,
        "y": y,
        "groups": groups,
        "lambdas": [1.0, 0.2],
        "sample_weight": weights,
        "ord": "inf",
        "max_iter": 25,
    }

    verified = apa_apg_regression_path(**kwargs)
    assumed = apa_apg_regression_path(**kwargs, assume_diagonal_gram=True)

    np.testing.assert_allclose(assumed.coefficients, verified.coefficients)
    np.testing.assert_allclose(assumed.penalties, verified.penalties)
    assert assumed.metadata["quadratic_backend"] == "diagonal"
    assert assumed.metadata["assumed_diagonal_gram"] is True
    assert assumed.metadata["gram_max_off_diagonal"] is None
    assert assumed.metadata["gram_max_off_diagonal_correlation"] is None
    assert verified.metadata["assumed_diagonal_gram"] is False
    assert all(
        diagnostic["assumed_diagonal_gram"] is True
        for diagnostic in assumed.diagnostics
    )


@pytest.mark.parametrize(
    "extra_kwargs",
    [
        {"cache_quadratic": False},
        {
            "point_solver": lambda **kwargs: (
                kwargs["beta_init"],
                {"converged": True, "n_iter": 1},
            )
        },
    ],
)
def test_assumed_diagonal_path_rejects_configuration_that_bypasses_cache(
    extra_kwargs,
):
    with pytest.raises(ValueError, match="requires cache_quadratic=True"):
        apa_apg_regression_path(
            np.eye(2),
            np.ones(2),
            [np.array([0, 1])],
            lambdas=[1.0],
            assume_diagonal_gram=True,
            **extra_kwargs,
        )


def test_regression_zero_lambda_is_weighted_least_squares_not_apa_call():
    X = np.column_stack([np.ones(4), np.arange(4.0)])
    y = np.array([1.0, 2.0, 2.0, 5.0])
    weights = np.array([1.0, 3.0, 2.0, 4.0])

    def point_solver_must_not_run(**kwargs):
        raise AssertionError("the positive-lambda point solver received lambda zero")

    path = apa_apg_regression_path(
        X,
        y,
        [np.array([1])],
        lambdas=[0.0],
        beta_init=np.array([7.0, -3.0]),
        sample_weight=weights,
        ord="inf",
        point_solver=point_solver_must_not_run,
    )

    sqrt_weights = np.sqrt(weights / weights.sum())
    expected = np.linalg.lstsq(
        X * sqrt_weights[:, None], y * sqrt_weights, rcond=None
    )[0]
    np.testing.assert_allclose(path.coefficients[0], expected)
    assert path.lambdas.tolist() == [0.0]
    assert path.penalties[0] == pytest.approx(abs(expected[1]))
    assert path.diagnostics[0]["solver"] == "weighted_lstsq"
    assert path.diagnostics[0]["penalty"] == 0.0
    assert path.status == "complete"


def test_cached_diagonal_zero_lambda_avoids_lstsq(monkeypatch):
    import imodels.tree.sparse_pruning.optimization.apa as apa_module

    def lstsq_must_not_run(*args, **kwargs):
        raise AssertionError("a diagonal cached endpoint must not call lstsq")

    monkeypatch.setattr(apa_module.np.linalg, "lstsq", lstsq_must_not_run)
    X = np.array(
        [
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.0, 2.0],
            [0.0, -2.0],
        ]
    )
    y = np.array([2.0, -1.0, 3.0, -2.0])
    weights = np.array([1.0, 2.0, 3.0, 4.0])
    path = apa_apg_regression_path(
        X,
        y,
        [np.array([0, 1])],
        lambdas=[0.0],
        beta_init=np.array([10.0, -5.0]),
        sample_weight=weights,
        ord="inf",
    )

    normalized_weights = weights / weights.sum()
    diagonal = np.sum(normalized_weights[:, None] * X**2, axis=0)
    linear = X.T @ (normalized_weights * y)
    expected = linear / diagonal
    np.testing.assert_allclose(path.coefficients[0], expected)
    diagnostic = path.diagnostics[0]
    assert diagnostic["solver"] == "cached_diagonal_lstsq"
    assert diagnostic["rank"] == 2
    np.testing.assert_allclose(
        diagnostic["singular_values"], np.sort(np.sqrt(diagonal))[::-1]
    )
    assert diagnostic["quadratic_backend"] == "diagonal"
    assert path.metadata["zero_lambda_solver"] == "cached_diagonal_lstsq"


def test_cached_diagonal_zero_lambda_preserves_exact_null_space_component():
    X = np.array(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, -2.0, 0.0],
        ]
    )
    y = np.array([2.0, -1.0, 3.0, -2.0])
    beta_start = np.array([10.0, -5.0, 7.25])
    path = apa_apg_regression_path(
        X,
        y,
        [np.array([0, 1, 2])],
        lambdas=[0.0],
        beta_init=beta_start,
        ord="inf",
    )

    expected_identified = np.linalg.lstsq(X[:, :2], y, rcond=None)[0]
    np.testing.assert_allclose(path.coefficients[0, :2], expected_identified)
    assert path.coefficients[0, 2] == beta_start[2]
    assert path.diagnostics[0]["rank"] == 2
    np.testing.assert_allclose(
        path.diagnostics[0]["singular_values"][-1], 0.0
    )


def test_regression_path_runs_existing_linf_apa_apg2_solver():
    X = np.eye(3)
    y = np.array([2.0, 3.0, 4.0])
    path = apa_apg_regression_path(
        X,
        y,
        [np.array([1, 2])],
        lambdas=[2.0 / 3.0, 1.0 / 3.0],
        ord="inf",
        gamma1=2.0,
        max_iter=500,
        tol=1e-10,
    )

    # Index 0 is outside the group and is therefore unpenalized. For the
    # infinity norm, prox thresholds y[1:] through an l1-ball projection.
    np.testing.assert_allclose(path.coefficients[0], [2.0, 2.5, 2.5], atol=1e-7)
    np.testing.assert_allclose(path.coefficients[1], [2.0, 3.0, 3.0], atol=1e-7)
    np.testing.assert_allclose(path.penalties, [2.5, 3.0], atol=1e-7)
    assert all(info["converged"] for info in path.diagnostics)


def test_nonconverged_positive_point_marks_sampled_path_partial():
    def iteration_limited_solver(**kwargs):
        return kwargs["beta_init"], {"converged": False, "n_iter": 3}

    path = apa_apg_regression_path(
        np.eye(2),
        np.ones(2),
        [np.array([0, 1])],
        lambdas=[1.0, 0.0],
        point_solver=iteration_limited_solver,
    )

    assert path.status == "partial"
    assert path.diagnostics[0]["converged"] is False
    assert path.diagnostics[1]["converged"] is True


def test_classification_zero_lambda_solves_weighted_logistic_endpoint():
    X = np.ones((4, 1))
    y = np.array([0.0, 1.0, 0.0, 1.0])
    weights = np.array([1.0, 3.0, 1.0, 1.0])
    path = apa_apg_classification_path(
        X,
        y,
        [np.array([0])],
        lambdas=[0.0],
        sample_weight=weights,
        tol=1e-12,
    )

    expected = np.log(2.0)  # weighted positive fraction is 2/3
    np.testing.assert_allclose(path.coefficients[0], [expected], atol=1e-7)
    assert path.diagnostics[0]["solver"] == "weighted_logistic_lbfgs"
    assert path.diagnostics[0]["converged"]
    assert path.status == "complete"
    assert path.penalties[0] == pytest.approx(expected)


@pytest.mark.parametrize(
    "bad_lambdas,match",
    [
        ([], "non-empty"),
        (np.ones((2, 1)), "one-dimensional"),
        ([np.nan], "finite"),
        ([np.inf], "finite"),
        ([-0.1], "nonnegative"),
        ([True], "booleans"),
        (["1.0"], "numeric"),
        ([1.0 + 2.0j], "numeric"),
    ],
)
def test_path_rejects_invalid_lambdas(bad_lambdas, match):
    with pytest.raises(ValueError, match=match):
        apa_apg_regression_path(
            np.ones((2, 1)),
            np.ones(2),
            [np.array([0])],
            lambdas=bad_lambdas,
        )


def test_path_accepts_a_lambda_generator():
    path = apa_apg_regression_path(
        np.ones((2, 1)),
        np.ones(2),
        [np.array([0])],
        lambdas=(lam for lam in [2.0, 1.0]),
        max_iter=10,
    )

    np.testing.assert_array_equal(path.lambdas, [2.0, 1.0])


def test_path_point_solver_must_return_coefficients_and_diagnostics():
    def invalid_solver(**kwargs):
        return kwargs["beta_init"]

    with pytest.raises(TypeError, match="diagnostics"):
        apa_apg_classification_path(
            np.ones((2, 1)),
            np.array([0.0, 1.0]),
            [np.array([0])],
            lambdas=[1.0],
            point_solver=invalid_solver,
        )
