"""Compact public-API correctness tests for sparse pruning and shrinkage.

The analytic example is an independent oracle; the remaining checks exercise
estimator behavior, solver agreement, and model selection on small local data.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.base import clone
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from imodels import (
    HSTreeRegressor, SHSTreeClassifier, SHSTreeClassifierCV,
    SHSTreeRegressor, SHSTreeRegressorCV, SPTreeClassifier,
    SPTreeClassifierCV, SPTreeRegressor, SPTreeRegressorCV,
)
from imodels.tree.sparse_pruning.optimization import (
    hicap_regression_path, laminar_group_linf_regression_path,
    tree_group_linf_exact_coefficient_path, tree_group_linf_exact_topology_path,
)


def _data():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(48, 3))
    y = X[:, 0] + 2 * (X[:, 1] > 0) + .2 * rng.normal(size=len(X))
    return X, y


@pytest.mark.parametrize("wrapper", [SPTreeRegressor, SHSTreeRegressor])
def test_regression_endpoints_clone_and_refit(wrapper):
    X, y = _data()
    source = DecisionTreeRegressor(max_leaf_nodes=5, random_state=0)
    model = wrapper(estimator_=source, sp_alpha=0, reg_param=0)
    assert model.fit(X, y) is model
    expected = clone(source).fit(X, y)
    assert_allclose(model.predict(X), expected.predict(X))
    assert model.solver_ == "topology"
    assert model.coef_ is model.coefficient_path_ is None
    assert not hasattr(source, "tree_")  # The constructor tree stays unfitted.
    assert_allclose(clone(model).fit(X, y).predict(X), model.predict(X))
    model.set_params(sp_alpha=1e6).fit(X, y)
    assert model.estimator_.get_n_leaves() == 1
    assert_allclose(model.predict(X), y.mean())
    model.set_params(sp_alpha=0).fit(X, y + 3)
    assert_allclose(model.predict(X), expected.predict(X) + 3)


@pytest.mark.parametrize("wrapper", [
    SPTreeClassifier, SHSTreeClassifier, SPTreeClassifierCV, SHSTreeClassifierCV,
])
def test_binary_classification_probabilities_and_numeric_cv(wrapper):
    X, target = _data()
    y = np.where(target > np.median(target), "high", "low")
    is_cv = wrapper in (SPTreeClassifierCV, SHSTreeClassifierCV)
    choices = (dict(sp_alpha_list=[0, .1], reg_param_list=[0, 2], cv=2)
               if is_cv else dict(sp_alpha=0, reg_param=0))
    model = wrapper(max_leaf_nodes=3, random_state=0, tol=1e-5, **choices)
    assert model.fit(X, y) is model
    probabilities = model.predict_proba(X)
    assert probabilities.shape == (len(X), 2)
    assert np.all((probabilities >= 0) & (probabilities <= 1))
    assert_allclose(probabilities.sum(axis=1), 1)
    assert_array_equal(model.predict(X), model.classes_[probabilities.argmax(axis=1)])
    assert_allclose(clone(model).fit(X, y).predict_proba(X), probabilities)
    if is_cv:
        assert model.cv_path_mode_ == "grid"
        assert model.cv_scores_.shape == (4, 2)
        assert model.solver_ == "apa_apg2"
    else:
        cart = DecisionTreeClassifier(max_leaf_nodes=3, random_state=0).fit(X, y)
        assert_allclose(probabilities, cart.predict_proba(X))


def test_analytic_coefficient_knots_and_interpolation():
    # Minimize .5 * ||beta - [1, 2]||^2
    #          + lambda * (max(|beta_0|, |beta_1|) + |beta_1|).
    # Above .5 both coefficients move together; below .5 beta_0 stays at 1.
    path = tree_group_linf_exact_coefficient_path([1, 2], [1, 1], [-1, 0])
    assert path.exact and path.status == "complete"
    assert_allclose(path.lambdas, [1.5, .5, 0], atol=1e-10)
    assert_allclose(path.coefficients, [[0, 0], [1, 1], [1, 2]], atol=1e-10)
    queries = [1.5, 1, .5, .25, 0]
    expected = [[0, 0], [.5, .5], [1, 1], [1, 1.5], [1, 2]]
    assert_allclose([path.at(a)[0] for a in queries], expected, atol=1e-10)
    # The normalized squared loss on this design has the same quadratic.
    X, y = np.sqrt(2) * np.eye(2), np.sqrt(2) * np.array([1, 2])
    reference = hicap_regression_path(X, y, [[0, 1], [1]], fit_intercept=False)
    proximal = laminar_group_linf_regression_path(
        X, y, [[0, 1], [1]], queries, fit_intercept=False,
    )
    assert reference.exact and reference.status == "complete"
    assert proximal.metadata["point_solutions_certified"]
    assert_allclose([reference.at(a)[0] for a in queries], expected, atol=1e-8)
    assert_allclose(proximal.coefficients, expected, atol=1e-8)
    topology = tree_group_linf_exact_topology_path([1, 2], [-1, 0])
    assert topology.topology_at(.75) == topology.topology_at(.25)
    assert not np.any(np.isclose(topology.lambdas, .5))


def test_classification_coefficient_path_and_checked_in_between_point():
    from imodels.tree.sparse_pruning import (
        fitted_tree_linf_classification, fitted_tree_linf_classification_path,
        fitted_tree_linf_exact_topology_path,
    )

    # Equal-mass leaves with P(class 1)=.2 and .8 have beta=logit(.8-lambda)
    # for 0<lambda<.3, and beta=0 above .3. This is not a linear path.
    X = np.repeat([[-1.], [1.]], 10, axis=0)
    y = np.array([0] * 8 + [1] * 2 + [0] * 2 + [1] * 8)
    tree = DecisionTreeClassifier(max_depth=1, random_state=0).fit(X, y)
    original = tree.tree_.value.copy()
    topology = fitted_tree_linf_exact_topology_path(tree)
    assert_allclose(topology.lambdas, [.3, 0.], atol=1e-14)
    path = fitted_tree_linf_classification_path(tree, [.4, .2, .1], tol=1e-9)
    assert path.status == "complete" and not path.exact
    assert_allclose(path.coefficients[:, 0], [0., np.log(.6 / .4), np.log(.7 / .3)], atol=1e-7)
    beta, info = fitted_tree_linf_classification(tree, .15, tol=1e-9, return_info=True)
    assert info["certified"]
    assert_allclose(beta, [np.log(.65 / .35)], atol=1e-7)
    assert_allclose(info["leaf_probabilities"][:, 1], [.35, .65], atol=1e-7)
    assert not np.isclose(path.at(.15)[0][0], beta[0])
    assert_array_equal(tree.tree_.value, original)


def test_zero_coefficient_ancestor_is_retained_for_active_descendant():
    path = tree_group_linf_exact_coefficient_path([0, 2], [3, 1], [-1, 0])
    assert_allclose(path.at(.5)[0], [0, 1], atol=1e-10)
    topology = tree_group_linf_exact_topology_path([0, 2], [-1, 0])
    assert topology.topology_at(.5) == (0, 1)
    assert topology.topology_at(1) == ()


@pytest.mark.parametrize("solver", ["topology", "proximal", "coefficient_path", "hicap"])
def test_weighted_solvers_match_repeated_observations(solver):
    X, y = _data()
    weight = 1 + np.arange(len(y)) % 3
    options = dict(sp_alpha=.1, max_leaf_nodes=5, random_state=0, solver=solver)
    model = SPTreeRegressor(**options).fit(X, y, sample_weight=weight)
    repeated = SPTreeRegressor(**options).fit(
        np.repeat(X, weight, axis=0), np.repeat(y, weight),
    )
    structural = SPTreeRegressor(**dict(options, solver="topology")).fit(
        X, y, sample_weight=7 * weight,
    )
    assert_allclose(model.predict(X), repeated.predict(X), atol=1e-10)
    assert_allclose(model.predict(X), structural.predict(X), atol=1e-10)
    if model.coef_ is not None:
        assert_allclose(model.coef_, repeated.coef_, atol=1e-8)


@pytest.mark.parametrize("solver", ["coefficient_path", "hicap"])
def test_non_cv_path_does_not_change_the_requested_penalty_model(solver):
    X, y = _data()
    options = dict(sp_alpha=.1, max_leaf_nodes=5, random_state=0, reg_param=2)
    model = SHSTreeRegressor(solver=solver, **options).fit(X, y)
    point = SHSTreeRegressor(solver="proximal", **options).fit(X, y)
    assert model.sp_alpha == .1
    assert_allclose(model.predict(X), point.predict(X), atol=1e-10)
    assert_allclose(model.coef_, point.coef_, atol=1e-8)
    path = model.coefficient_path_
    assert path.exact and path.status == "complete"
    assert_allclose(path.at(.1)[0], model.coef_, atol=1e-8)
    assert len(model.coef_node_ids_) == len(model.coef_)
    predictions, coefficients = model.predict(X).copy(), model.coef_.copy()
    elsewhere, _ = path.at(0)
    assert not np.allclose(elsewhere, coefficients)
    elsewhere[:] = 0  # Queries return independent arrays, not live model state.
    assert_array_equal(model.predict(X), predictions)
    assert_array_equal(model.coef_, coefficients)


@pytest.mark.parametrize("wrapper", [SPTreeRegressorCV, SHSTreeRegressorCV])
def test_default_structural_cv_matches_independent_fold_scores(wrapper):
    X, y = _data()
    model = wrapper(max_leaf_nodes=4, reg_param_list=[0], cv=3, random_state=0).fit(X, y)
    assert model.cv_path_mode_ == "structural" and model.solver_ == "topology"
    assert model.selection_rule == "one_se"
    scores = np.empty_like(model.cv_scores_)
    for fold, (train, test) in enumerate(KFold(3, shuffle=True, random_state=0).split(X)):
        for candidate, params in enumerate(model.cv_params_):
            fresh = SPTreeRegressor(
                max_leaf_nodes=4, random_state=0, sp_alpha=params["sp_alpha"],
            ).fit(X[train], y[train])
            scores[candidate, fold] = -np.mean((fresh.predict(X[test]) - y[test]) ** 2)
    assert_allclose(model.cv_scores_, scores, atol=1e-12)
    means = scores.mean(axis=1)
    best = means.argmax()
    eligible = means >= means[best] - scores[best].std(ddof=1) / np.sqrt(3)
    complexity = model.cv_complexities_.mean(axis=1)
    assert eligible[model.selected_index_]
    assert complexity[model.selected_index_] == complexity[eligible].min()
    final = SPTreeRegressor(
        max_leaf_nodes=4, random_state=0, sp_alpha=model.sp_alpha_,
    ).fit(X, y)
    assert_allclose(model.predict(X), final.predict(X))


@pytest.mark.parametrize("hs_choices", [[0, 2], "gcv"])
def test_shrinkage_cv_refits_the_selected_numeric_strength(hs_choices):
    X, y = _data()
    model = SHSTreeRegressorCV(
        max_leaf_nodes=4, sp_alpha_list=[0, .1], reg_param_list=hs_choices,
        cv=3, random_state=0, selection_rule="best",
    ).fit(X, y)
    expected = SHSTreeRegressor(
        max_leaf_nodes=4, sp_alpha=model.sp_alpha_, reg_param=model.reg_param_,
        random_state=0,
    ).fit(X, y)
    assert_allclose(model.predict(X), expected.predict(X), atol=1e-12)
    assert model.cv_scores_.shape[1] == 3  # GCV for HS still uses held-out pruning CV.
    assert (model.gcv_results_ is not None) == (hs_choices == "gcv")


@pytest.mark.parametrize("wrapper", [HSTreeRegressor, SHSTreeRegressor])
def test_auto_gcv_is_independent_across_instances_and_refits(wrapper):
    X, y = _data()
    model = wrapper(max_leaf_nodes=4, reg_param="gcv", random_state=0).fit(X, y)
    other = wrapper(max_leaf_nodes=4, reg_param="gcv", random_state=0).fit(X, y)
    assert_allclose(model.predict(X), other.predict(X))
    numeric = clone(model).set_params(reg_param=model.reg_param_).fit(X, y)
    assert_allclose(model.predict(X), numeric.predict(X), atol=1e-12)
    expected = clone(model).fit(X, y + 3)
    assert model.fit(X, y + 3) is model
    assert_allclose(model.predict(X), expected.predict(X), atol=1e-12)
    assert model.gcv_results_["conditional_on_tree"]


@pytest.mark.parametrize("options", [
    dict(solver="unknown"), dict(solver="topology", ord=2),
    dict(solver="coefficient_path", support_tol=.01), dict(sp_alpha=-1),
])
def test_incompatible_options_raise_clear_errors(options):
    X, y = _data()
    with pytest.raises(ValueError):
        SPTreeRegressor(max_leaf_nodes=4, **options).fit(X, y)
    with pytest.raises(ValueError, match="regression"):
        SPTreeClassifier(solver="coefficient_path").fit(X, y > np.median(y))


@pytest.mark.parametrize("solver", ["coefficient_path", "hicap"])
def test_incomplete_paths_are_not_used_as_exact_solutions(solver):
    X, y = _data()
    with pytest.raises(RuntimeError, match="complete"):
        SPTreeRegressor(
            solver=solver, max_leaf_nodes=5, max_iter=1, sp_alpha=.1, random_state=0,
        ).fit(X, y)


def test_root_only_tree_has_a_complete_empty_coefficient_path():
    X, y = np.zeros((8, 1)), np.arange(8.)
    model = SPTreeRegressor(solver="coefficient_path").fit(X, y)
    assert model.coefficient_path_.exact
    assert model.coef_.shape == (0,)
    assert_allclose(model.coefficient_path_.lambdas, [0])
    assert_allclose(model.predict(X), y.mean())
