"""Dispatch and fitted-statistic solves used by pruning estimators."""
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from imodels.importance.local_stumps import make_stumps, tree_feature_transform
from imodels.tree.sparse_pruning import _solver
from imodels.tree.sparse_pruning import SPTreeRegressor
from imodels.tree.sparse_pruning.fitted_tree import (
    fitted_tree_linf_exact_topology_path,
    materialize_fitted_tree_topology,
)
from imodels.tree.sparse_pruning._solver import (
    iter_fitted_tree_solutions,
    resolve_solver,
    solve_fitted_tree,
    validate_solver,
)
from imodels.tree.sparse_pruning.optimization import laminar_group_linf_regression_path


@pytest.fixture
def fitted_problem():
    rng = np.random.default_rng(849)
    X = rng.normal(size=(64, 3))
    y = 3 + X[:, 0] - X[:, 1] ** 2 + 0.1 * rng.normal(size=64)
    weight = rng.uniform(0.3, 2, 64)
    tree = DecisionTreeRegressor(max_leaf_nodes=7, random_state=0).fit(
        X, y, sample_weight=weight
    )
    return tree, X, y, weight


def _resolve(tree, name="auto", **kwargs):
    options = dict(estimator=tree, ord=np.inf, matched_training=True, support_tol=None)
    options.update(kwargs)
    return resolve_solver(name, **options)


@pytest.mark.parametrize("name", ["auto", "topology", "proximal", "coefficient_path",
                                 "hicap", "apa_apg2"])
def test_solver_names_are_explicit(name):
    assert validate_solver(name) == name


@pytest.mark.parametrize("name", [None, "green", "APA-APG2", 0, [], np.nan])
def test_invalid_solver_names_are_rejected(name):
    with pytest.raises(ValueError, match="solver"):
        validate_solver(name)


def test_auto_chooses_structure_only_for_matched_infinity_regression(fitted_problem):
    tree, _, _, _ = fitted_problem
    assert _resolve(tree) == "topology"
    assert _resolve(tree, support_tol=0) == "topology"
    assert _resolve(tree, ord="inf") == "topology"
    assert _resolve(tree, matched_training=False) == "proximal"
    assert _resolve(tree, support_tol=1e-8) == "proximal"
    assert _resolve(tree, ord=2) == "apa_apg2"
    assert _resolve(DecisionTreeClassifier()) == "apa_apg2"
    assert _resolve(RandomForestRegressor()) == "proximal"


def test_auto_does_not_assume_ineligible_fitted_statistics(fitted_problem):
    tree, X, y, _ = fitted_problem
    assert _resolve(DecisionTreeRegressor()) == "proximal"
    absolute = DecisionTreeRegressor(criterion="absolute_error").fit(X, y)
    assert _resolve(absolute) == "proximal"
    multioutput = DecisionTreeRegressor().fit(X, np.column_stack((y, y)))
    assert _resolve(multioutput) == "proximal"
    tree.monotonic_cst = np.ones(X.shape[1], dtype=int)
    assert _resolve(tree) == "proximal"
    tree.monotonic_cst = np.zeros(X.shape[1], dtype=int)
    assert _resolve(tree) == "topology"


def test_cv_eligibility_inspects_template_without_fitting():
    tree = DecisionTreeRegressor()
    assert not _solver._native_tree_eligible(tree)
    assert _solver._native_tree_eligible(tree, require_fitted=False)
    assert not hasattr(tree, "tree_")
    assert _solver._native_tree_eligible(
        DecisionTreeClassifier(), require_fitted=False
    )
    assert not _solver._native_tree_eligible(
        DecisionTreeRegressor(criterion="absolute_error"), require_fitted=False
    )


@pytest.mark.parametrize("name", ["topology", "coefficient_path"])
@pytest.mark.parametrize("options", [dict(matched_training=False), dict(ord=2),
                                     dict(support_tol=1e-8)])
def test_explicit_native_solvers_reject_incompatible_objectives(fitted_problem, name, options):
    with pytest.raises(ValueError):
        _resolve(fitted_problem[0], name, **options)


def test_generic_exact_hicap_remains_regression_only():
    with pytest.raises(ValueError, match="regression"):
        _resolve(DecisionTreeClassifier(), "hicap")


@pytest.mark.parametrize("n_classes", [2, 3])
def test_classifier_dispatch_requires_matched_unconstrained_tree(n_classes):
    X = np.arange(24.).reshape(-1, 1)
    y = np.arange(len(X)) % n_classes
    tree = DecisionTreeClassifier(max_leaf_nodes=3, random_state=0).fit(X, y)
    assert _resolve(tree) == "topology"
    assert _resolve(tree, "apa_apg2") == "apa_apg2"
    assert _resolve(tree, ord=2) == "apa_apg2"
    for name in ("topology", "proximal", "coefficient_path"):
        assert _resolve(tree, name) == name
        for options in (dict(matched_training=False), dict(ord=2), dict(support_tol=.01)):
            with pytest.raises(ValueError):
                _resolve(tree, name, **options)
    tree.monotonic_cst = np.ones(X.shape[1], dtype=int)
    assert _resolve(tree) == "apa_apg2"


@pytest.mark.parametrize("solver", ["topology", "proximal", "coefficient_path"])
def test_classification_native_solutions_preserve_tree_and_zero_endpoint(solver):
    X = np.repeat([[-1.], [1.]], 10, axis=0)
    y = np.array([0] * 8 + [1] * 2 + [0] * 2 + [1] * 8)
    tree = DecisionTreeClassifier(max_depth=1, random_state=0).fit(X, y)
    before = deepcopy(tree.tree_.__getstate__())
    zero, interior, above = list(iter_fitted_tree_solutions(
        tree, [0., .1, .4], solver, tol=1e-8, max_iter=2000,
    ))
    assert zero.coefficients is zero.intercept is None
    assert zero.info["certified"] is True
    assert zero.info["certificate_scope"] == "structure"
    assert not zero.info["coefficients_available"]
    np.testing.assert_array_equal(zero.retained_node_ids, [0])
    np.testing.assert_array_equal(interior.retained_node_ids, [0])
    assert len(above.retained_node_ids) == 0
    if solver != "topology":
        assert zero.info["coefficients_unavailable_reason"] == "zero_penalty_may_have_infinite_logits"
        for result in (interior, above):
            assert result.info["certified"] is True
            assert result.info["certificate_scope"] == "structure_and_coefficients"
            assert result.info["coefficients_available"]
        np.testing.assert_allclose(interior.coefficients, [np.log(.7 / .3)], atol=1e-6)
        np.testing.assert_allclose(interior.intercept, 0., atol=1e-6)
        np.testing.assert_allclose(above.coefficients, [0.], atol=1e-7)
    else:
        for result in (interior, above):
            assert result.info["certified"] is True
            assert result.info["certificate_scope"] == "structure"
            assert not result.info["coefficients_available"]
    if solver == "coefficient_path":
        path = interior.coefficient_path
        assert path is zero.coefficient_path is above.coefficient_path
        assert not path.exact and path.status == "complete"
        assert np.all(path.lambdas > 0)
        assert .1 in path.lambdas and .4 in path.lambdas
        np.testing.assert_allclose(path.at(.1)[0], interior.coefficients)
    np.testing.assert_array_equal(tree.tree_.__getstate__()["nodes"], before["nodes"])
    np.testing.assert_array_equal(tree.tree_.value, before["values"])


@pytest.mark.parametrize("name", ["proximal", "hicap"])
def test_explicit_coefficient_solvers_require_infinity_norm(fitted_problem, name):
    with pytest.raises(ValueError, match="ord=inf"):
        _resolve(fitted_problem[0], name, ord=2)


def test_custom_point_solver_keeps_legacy_dispatch(fitted_problem):
    tree = fitted_problem[0]
    for name in ["auto", "apa_apg2"]:
        assert _resolve(tree, name, custom_solver=True) == "apa_apg2"
    for name in ["topology", "proximal", "coefficient_path", "hicap"]:
        with pytest.raises(ValueError, match="custom"):
            _resolve(tree, name, custom_solver=True)


@pytest.mark.parametrize("options", [dict(ord=1), dict(ord=2j), dict(ord=True),
                                     dict(support_tol=-1), dict(support_tol=np.inf),
                                     dict(matched_training="yes")])
def test_dispatch_rejects_invalid_controls(fitted_problem, options):
    with pytest.raises(ValueError):
        _resolve(fitted_problem[0], **options)


@pytest.mark.parametrize("solver", ["topology", "proximal", "coefficient_path"])
def test_native_solutions_match_design_and_do_not_mutate_tree(fitted_problem, solver):
    tree, X, y, weight = fitted_problem
    before = deepcopy(tree.tree_.__getstate__())
    structural = solve_fitted_tree(tree, 0, "topology", tol=1e-9, max_iter=1000)
    alpha = float(structural.topology_path.lambdas[0]) * 0.31
    result = solve_fitted_tree(tree, alpha, solver, tol=1e-9, max_iter=1000)

    np.testing.assert_array_equal(result.node_ids, structural.node_ids)
    np.testing.assert_array_equal(result.retained_node_ids,
                                  structural.topology_path.tree_nodes_at(alpha))
    assert result.info["converged"] is True
    assert result.info["solver"] == solver
    assert result.info["status"] == "complete"
    assert isinstance(result.info["n_iter"], int)
    assert result.intercept == tree.tree_.value[0, 0, 0]
    assert not result.node_ids.flags.writeable
    assert not result.retained_node_ids.flags.writeable
    if solver == "topology":
        assert result.coefficients is None
        assert result.coefficient_path is None
    else:
        assert not result.coefficients.flags.writeable
        assert result.info["certified"] is True
        parents = np.asarray(structural.topology_path.metadata["parent_indices"])
        groups = [[node] for node in range(parents.size)]
        for child in range(parents.size - 1, -1, -1):
            if parents[child] >= 0:
                groups[parents[child]].extend(groups[child])
        Z = tree_feature_transform(make_stumps(tree.tree_), X)
        reference = laminar_group_linf_regression_path(
            Z, y, groups, [alpha], sample_weight=weight, fit_intercept=True, tol=1e-10
        )
        assert reference.metadata["point_solutions_certified"]
        np.testing.assert_allclose(result.coefficients, reference.coefficients[0],
                                   atol=1e-9, rtol=1e-8)
        assert (result.coefficient_path is not None) == (solver == "coefficient_path")
    after = tree.tree_.__getstate__()
    np.testing.assert_array_equal(before["nodes"], after["nodes"])
    np.testing.assert_array_equal(before["values"], after["values"])


@pytest.mark.parametrize("solver", ["topology", "proximal", "coefficient_path"])
def test_zero_alpha_preserves_original_splits_including_zero_scores(fitted_problem, solver):
    tree = fitted_problem[0]
    # A mean-consistent, zero-score tree can retain redundant backing splits.
    tree.tree_.value[:] = 2.0
    result = solve_fitted_tree(tree, 0, solver, tol=1e-9, max_iter=1000)
    np.testing.assert_array_equal(result.retained_node_ids, result.node_ids)
    assert result.topology_path.tree_nodes_at(0) == ()
    if solver != "topology":
        np.testing.assert_array_equal(result.coefficients, np.zeros(result.node_ids.size))


@pytest.mark.parametrize("solver", ["topology", "proximal", "coefficient_path"])
def test_native_root_only_tree_returns_empty_results(solver):
    tree = DecisionTreeRegressor().fit(np.zeros((8, 1)), np.arange(8.0))
    for alpha in (0, 1):
        result = solve_fitted_tree(tree, alpha, solver, tol=1e-9, max_iter=1000)
        assert result.node_ids.size == result.retained_node_ids.size == 0
        assert result.intercept == 3.5
        if solver == "topology":
            assert result.coefficients is None
        else:
            assert result.coefficients.shape == (0,)


def test_coefficient_path_queries_above_lambda_max_are_zero(fitted_problem):
    tree = fitted_problem[0]
    result = solve_fitted_tree(tree, 1e6, "coefficient_path", tol=1e-9, max_iter=1000)
    np.testing.assert_array_equal(result.coefficients, np.zeros(result.node_ids.size))
    assert result.retained_node_ids.size == 0


def test_partial_coefficient_path_is_rejected_before_interpolation(fitted_problem, monkeypatch):
    tree = fitted_problem[0]
    complete = _solver.fitted_tree_linf_exact_coefficient_path(tree)
    partial = replace(complete, exact=False, status="event_limit")
    monkeypatch.setattr(_solver, "fitted_tree_linf_exact_coefficient_path",
                        lambda *args, **kwargs: partial)
    with pytest.raises(RuntimeError, match="complete certified coefficient path"):
        solve_fitted_tree(tree, 0, "coefficient_path", tol=1e-9, max_iter=1000)


def test_numerical_topology_failure_is_not_silently_replaced(fitted_problem, monkeypatch):
    def fail(*args, **kwargs):
        raise ValueError("deliberate numerical failure")

    monkeypatch.setattr(_solver, "fitted_tree_linf_exact_topology_path", fail)
    with pytest.raises(ValueError, match="deliberate numerical failure"):
        solve_fitted_tree(fitted_problem[0], 0.1, "topology", tol=1e-9, max_iter=1000)


def test_uncertified_proximal_solution_is_rejected(fitted_problem, monkeypatch):
    class BadCertificate:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, value, alpha, *, return_info):
            return value, {"raw_relative_duality_gap": 1.0,
                           "max_relative_dual_l1_violation": 0.0,
                           "relative_moreau_residual": 0.0}

    monkeypatch.setattr(_solver, "LaminarGroupLinfProx", BadCertificate)
    with pytest.raises(RuntimeError, match="could not be certified"):
        solve_fitted_tree(fitted_problem[0], 0.1, "proximal", tol=1e-9, max_iter=1000)


@pytest.mark.parametrize("options", [dict(alpha=-1), dict(alpha=np.inf),
                                     dict(alpha=1j), dict(tol=0),
                                     dict(max_iter=0), dict(max_iter=True),
                                     dict(solver="hicap")])
def test_native_helper_rejects_invalid_controls(fitted_problem, options):
    arguments = dict(alpha=0.1, solver="topology", tol=1e-9, max_iter=1000)
    arguments.update(options)
    with pytest.raises(ValueError):
        solve_fitted_tree(fitted_problem[0], **arguments)


def test_proximal_endpoints_do_not_compile_groups(fitted_problem, monkeypatch):
    tree = fitted_problem[0]

    def unexpected_groups(*args, **kwargs):
        pytest.fail("An analytic endpoint must not compile descendant groups")

    monkeypatch.setattr(_solver, "LaminarGroupLinfProx", unexpected_groups)
    zero, above = list(iter_fitted_tree_solutions(
        tree, [0, 1e6], "proximal", tol=1e-9, max_iter=1000
    ))
    scores, diagonal, *_ = _solver._fitted_tree_statistics(tree)
    np.testing.assert_allclose(zero.coefficients, scores / diagonal)
    np.testing.assert_array_equal(above.coefficients, 0)
    assert zero.info["n_iter"] == above.info["n_iter"] == 0


def test_native_grid_prepares_statistics_and_groups_once(fitted_problem, monkeypatch):
    tree = fitted_problem[0]
    maximum = fitted_tree_linf_exact_topology_path(tree).lambdas[0]
    alphas = [0, maximum * .2, maximum * .6, maximum * 2, maximum * .4]
    expected = [solve_fitted_tree(tree, a, "proximal", tol=1e-9, max_iter=1000)
                for a in alphas]
    counts = {"statistics": 0, "groups": 0}
    original_statistics = _solver._fitted_tree_statistics
    original_operator = _solver.LaminarGroupLinfProx

    def statistics(*args, **kwargs):
        counts["statistics"] += 1
        return original_statistics(*args, **kwargs)

    def operator(*args, **kwargs):
        counts["groups"] += 1
        return original_operator(*args, **kwargs)

    monkeypatch.setattr(_solver, "_fitted_tree_statistics", statistics)
    monkeypatch.setattr(_solver, "LaminarGroupLinfProx", operator)
    actual = list(iter_fitted_tree_solutions(
        tree, alphas, "proximal", tol=1e-9, max_iter=1000
    ))
    assert counts == {"statistics": 1, "groups": 1}
    for result, reference in zip(actual, expected):
        np.testing.assert_allclose(result.coefficients, reference.coefficients)
        np.testing.assert_array_equal(result.retained_node_ids, reference.retained_node_ids)
        assert result.topology_path is actual[0].topology_path


@pytest.mark.parametrize("solver", ["topology", "proximal", "coefficient_path", "hicap"])
@pytest.mark.parametrize("side", ["at", "below", "above"])
def test_wrapper_solvers_agree_at_and_adjacent_to_structural_knots(solver, side):
    # Generic hiCAP can leave tiny nonzero affine-roundoff coefficients at a
    # true knot. Matched-tree pruning must use the exact structural event, not
    # an equality-to-zero test on those coefficient rows.
    rng = np.random.default_rng(629)
    X = rng.normal(size=(40, 2))
    y = X[:, 0] + X[:, 1] ** 2 + 0.1 * rng.normal(size=40)
    template = DecisionTreeRegressor(max_leaf_nodes=5, random_state=0)
    base = deepcopy(template).fit(X, y)
    path = fitted_tree_linf_exact_topology_path(base)
    for knot in path.lambdas[path.lambdas > 0]:
        alpha = float(knot)
        if side != "at":
            alpha = float(np.nextafter(alpha, 0 if side == "below" else np.inf))
        model = SPTreeRegressor(
            estimator_=template, solver=solver, sp_alpha=alpha,
            support_tol=0, tol=1e-9, max_iter=1000, random_state=0,
        ).fit(X, y)
        expected = materialize_fitted_tree_topology(base, path, alpha)
        assert model.complexity_ == len(path.tree_nodes_at(alpha))
        np.testing.assert_allclose(model.predict(X), expected.predict(X), atol=1e-12)
        assert model.pruning_path_ is not None


@pytest.mark.parametrize("solver", ["proximal", "hicap"])
@pytest.mark.parametrize("alpha", [0, 0.1])
def test_prefit_root_only_coefficient_attributes_use_pruning_response_mean(solver, alpha):
    X = np.zeros((8, 1))
    y_fit = np.arange(8.0)
    y_prune = y_fit + 10
    weights = np.arange(1.0, 9.0)
    base = DecisionTreeRegressor().fit(X, y_fit)
    model = SPTreeRegressor(
        estimator_=base, prefit=True, solver=solver, sp_alpha=alpha
    ).fit(X, y_prune, sample_weight=weights)
    assert model.coef_.shape == model.coef_node_ids_.shape == (0,)
    assert model.intercept_ == pytest.approx(np.average(y_prune, weights=weights))
    np.testing.assert_allclose(model.beta_stars_[0], [model.intercept_])
    if solver == "hicap":
        assert model.coefficient_path_.exact
        coefficients, intercept = model.coefficient_path_.at(0)
        assert coefficients.shape == (0,)
        assert intercept == model.intercept_
    # Pruning coefficients are diagnostics, not the CART prediction values.
    np.testing.assert_allclose(model.predict(X), base.predict(X))
