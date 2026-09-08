"""Independent fixed-tree checks for automatic regression HS via GCV."""

from copy import deepcopy

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from imodels import (
    HSTreeClassifier, HSTreeClassifierCV, HSTreeRegressor,
    HSTreeRegressorCV, SHSTreeRegressor,
)
from imodels.tree._hs_gcv import select_hs_reg_param
from imodels.tree.sparse_pruning.optimizations import get_gcv_reg_param


def _data(seed=0, n=32):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    y = 1.2 * X[:, 0] + (X[:, 1] > 0) + rng.normal(size=n)
    return X, y


def _hs_matrix(estimator, X, reg_param, *, derivative=False):
    """Build the smoother from observation memberships, not impurity formulas."""
    memberships = estimator.decision_path(X).toarray().astype(float)
    tree = estimator.tree_
    matrix = np.zeros((len(X), len(X))) if derivative else (
        np.ones((len(X), len(X))) / len(X)
    )
    stack = [0]
    while stack:
        parent = stack.pop()
        left, right = tree.children_left[parent], tree.children_right[parent]
        if left == right:
            continue
        parent_members = memberships[:, parent]
        parent_count = parent_members.sum()
        factor = 0.0 if np.isinf(reg_param) else parent_count / (
            parent_count + reg_param
        )
        if derivative and not np.isinf(reg_param):
            factor /= -(parent_count + reg_param)
        for child in (left, right):
            child_members = memberships[:, child]
            contrast = child_members / child_members.sum() - (
                parent_members / parent_count
            )
            matrix += factor * np.outer(child_members, contrast)
            stack.append(child)
    return matrix


def _assert_scores_match_explicit_matrices(estimator, X, y, result):
    chosen, info = result
    candidates = np.asarray(info["reg_params"])
    scores = np.asarray(info["gcv_scores"])
    assert candidates.ndim == 1
    assert scores.shape == candidates.shape
    assert np.all(candidates >= 0)
    assert not np.isnan(scores).any()
    assert chosen == candidates[info["selected_index"]]
    assert info["gcv_score"] == scores[info["selected_index"]]
    assert info["gcv_score"] <= np.min(scores) + 1e-12
    assert info["n_samples"] == len(y)
    assert info["conditional_on_tree"] is True
    assert isinstance(info["search_converged"], (bool, np.bool_))
    for index, (lam, score) in enumerate(zip(candidates, scores)):
        smoother = _hs_matrix(estimator, X, lam)
        df = np.trace(smoother)
        rss = np.sum((y - smoother @ y) ** 2)
        assert_allclose(info["effective_dfs"][index], df, atol=1e-10)
        assert_allclose(info["rss"][index], rss, rtol=2e-8, atol=2e-10)
        if np.isclose(df, len(y), rtol=0, atol=1e-12):
            # A separate analytic right-hand limit may be provided at the
            # interpolating endpoint, where the raw formula is 0 / 0.
            continue
        expected = (rss / len(y)) / (1 - df / len(y)) ** 2
        assert_allclose(score, expected, rtol=2e-8, atol=2e-10)
    selected_matrix = _hs_matrix(estimator, X, chosen)
    assert_allclose(info["effective_df"], np.trace(selected_matrix), atol=1e-10)


@pytest.mark.parametrize("seed", [0, 4, 19])
def test_gcv_scores_and_selected_df_match_explicit_fixed_tree_smoother(seed):
    X, y = _data(seed)
    estimator = DecisionTreeRegressor(max_leaf_nodes=6, random_state=0).fit(X, y)
    before = estimator.tree_.value.copy()
    result = select_hs_reg_param(estimator)

    _assert_scores_match_explicit_matrices(estimator, X, y, result)
    assert_array_equal(estimator.tree_.value, before)
    # A fresh dense scan is independent of the selector's candidate grid.
    scan = [0.0, *np.logspace(-5, 7, 401), np.inf]
    reference = []
    for lam in scan:
        smoother = _hs_matrix(estimator, X, lam)
        reference.append(
            np.mean((y - smoother @ y) ** 2)
            / (1 - np.trace(smoother) / len(y)) ** 2
        )
    assert result[1]["gcv_score"] <= min(reference) * (1 + 1e-7) + 1e-10


@pytest.mark.parametrize("offset", [0.0, 2.0])
def test_zero_node_means_and_old_negative_penalty_stump(offset):
    X = np.arange(8.0)[:, None]
    y = np.r_[-np.ones(4), np.ones(4)] + offset
    estimator = DecisionTreeRegressor(max_depth=1, random_state=0).fit(X, y)

    result = select_hs_reg_param(estimator)

    _assert_scores_match_explicit_matrices(estimator, X, y, result)
    assert result[0] == 0
    assert_allclose(result[1]["gcv_score"], 0, atol=1e-14)


@pytest.mark.parametrize("constant_target", [False, True])
def test_one_node_tree_keeps_intercept_and_has_one_degree_of_freedom(constant_target):
    X = np.arange(8.0)[:, None]
    y = np.full(8, 2.0) if constant_target else np.arange(8.0)
    estimator = DecisionTreeRegressor(min_samples_split=9).fit(X, y)

    result = select_hs_reg_param(estimator)

    _assert_scores_match_explicit_matrices(estimator, X, y, result)
    assert result[1]["effective_df"] == 1
    assert_allclose(
        result[1]["gcv_score"], np.var(y) / (1 - 1 / len(y)) ** 2
    )


def test_interpolating_zero_endpoint_does_not_produce_nan_or_fake_zero_gcv():
    X, y = _data(7, n=12)
    estimator = DecisionTreeRegressor(random_state=0).fit(X, y)
    assert estimator.get_n_leaves() == len(y)

    chosen, info = select_hs_reg_param(estimator)

    _assert_scores_match_explicit_matrices(estimator, X, y, (chosen, info))
    assert np.isfinite(info["gcv_score"])
    assert info["gcv_score"] > 0
    zero_index = np.flatnonzero(np.asarray(info["reg_params"]) == 0)
    if len(zero_index):
        derivative = _hs_matrix(estimator, X, 0, derivative=True)
        expected_limit = len(y) * np.sum((derivative @ y) ** 2) / (
            np.trace(derivative) ** 2
        )
        assert info["zero_score_is_limit"] is True
        assert_allclose(info["gcv_scores"][zero_index[0]], expected_limit, rtol=1e-10)


def test_pruned_tree_ignores_unreachable_storage_nodes():
    X = np.arange(16.0)[:, None]
    y = np.repeat([0.0, 1.0, 10.0, 11.0], 4)
    estimator = DecisionTreeRegressor(max_depth=2, random_state=0).fit(X, y)
    tree = estimator.tree_
    pruned = tree.children_left[0]
    assert tree.children_left[pruned] != tree.children_right[pruned]
    tree.children_left[pruned] = tree.children_right[pruned] = -1
    # Deliberately retain unreachable storage, as old pruning used to do.
    assert tree.node_count == 7

    result = select_hs_reg_param(estimator)

    _assert_scores_match_explicit_matrices(estimator, X, y, result)
    assert np.trace(_hs_matrix(estimator, X, 0)) == 3


@pytest.mark.parametrize("wrapper", [HSTreeRegressor, SHSTreeRegressor])
def test_auto_wrapper_keeps_constructor_parameter_and_matches_numeric_hs(wrapper):
    X, y = _data(5)
    kwargs = {"sp_alpha": 0} if wrapper is SHSTreeRegressor else {}
    model = wrapper(
        estimator_=DecisionTreeRegressor(max_leaf_nodes=5, random_state=0),
        reg_param="gcv",
        **kwargs,
    ).fit(X, y)
    original = DecisionTreeRegressor(max_leaf_nodes=5, random_state=0).fit(X, y)
    expected_param, _ = select_hs_reg_param(original)

    assert model.reg_param == "gcv"
    assert model.get_params()["reg_param"] == "gcv"
    assert model.reg_param_ == expected_param
    assert model.gcv_results_["conditional_on_tree"] is True
    assert_allclose(
        model.predict(X), _hs_matrix(original, X, expected_param) @ y, atol=1e-12
    )


def test_sparse_none_alias_and_gcv_string_are_equivalent():
    X, y = _data(6)
    models = [
        SHSTreeRegressor(
            estimator_=DecisionTreeRegressor(max_leaf_nodes=5, random_state=0),
            sp_alpha=0,
            reg_param=reg_param,
        ).fit(X, y)
        for reg_param in (None, "gcv")
    ]
    assert models[0].reg_param is None
    assert models[0].reg_param_ == models[1].reg_param_
    assert_allclose(models[0].predict(X), models[1].predict(X))


def test_sparse_gcv_is_selected_after_pruning_and_before_shrinkage():
    X, y = _data(8)
    model = SHSTreeRegressor(
        estimator_=DecisionTreeRegressor(max_leaf_nodes=5, random_state=0),
        sp_alpha=1,
        reg_param="gcv",
    )
    model.hiCAP = lambda **kwargs: np.zeros(kwargs["X"].shape[1])

    model.fit(X, y)

    assert model.estimator_.tree_.node_count == 1
    assert model.gcv_results_["effective_df"] == 1
    assert_allclose(model.predict(X), y.mean())
    assert_allclose(
        model.gcv_results_["gcv_score"], np.var(y) / (1 - 1 / len(y)) ** 2
    )


@pytest.mark.parametrize(
    "estimator",
    [
        DecisionTreeClassifier(max_depth=2),
        RandomForestRegressor(n_estimators=2, random_state=0),
        DecisionTreeRegressor(criterion="absolute_error", max_depth=2),
        DecisionTreeRegressor(monotonic_cst=[1, 0, 0], max_depth=2),
    ],
)
def test_core_rejects_unsupported_estimator_regimes(estimator):
    X, y = _data()
    target = y > np.median(y) if isinstance(estimator, DecisionTreeClassifier) else y
    estimator.fit(X, target)

    with pytest.raises((ValueError, TypeError)):
        select_hs_reg_param(estimator)


def test_core_rejects_nonuniform_weights_even_if_not_resupplied():
    X, y = _data()
    weights = np.linspace(0.5, 2.0, len(y))
    estimator = DecisionTreeRegressor(max_depth=2).fit(X, y, sample_weight=weights)

    with pytest.raises((ValueError, TypeError)):
        select_hs_reg_param(estimator, sample_weight=weights)
    with pytest.raises((ValueError, TypeError)):
        select_hs_reg_param(estimator)


def test_uniform_weights_only_rescale_pseudocount_and_not_the_gcv_score():
    X, y = _data(22)
    plain = DecisionTreeRegressor(max_leaf_nodes=5, random_state=0).fit(X, y)
    weight = 4.0
    weighted = DecisionTreeRegressor(max_leaf_nodes=5, random_state=0).fit(
        X, y, sample_weight=np.full(len(y), weight)
    )

    plain_param, plain_info = select_hs_reg_param(plain)
    weighted_param, weighted_info = select_hs_reg_param(
        weighted, sample_weight=np.full(len(y), weight)
    )

    assert_allclose(weighted_param, weight * plain_param, rtol=1e-6, atol=1e-8)
    assert_allclose(weighted_info["gcv_score"], plain_info["gcv_score"], rtol=1e-10)
    assert_allclose(weighted_info["effective_df"], plain_info["effective_df"], rtol=1e-6)


@pytest.mark.parametrize("wrapper", [HSTreeRegressor, SHSTreeRegressor])
@pytest.mark.parametrize("scheme", ["leaf_based", "constant"])
def test_auto_wrapper_rejects_other_shrinkage_schemes(wrapper, scheme):
    X, y = _data()
    kwargs = {"sp_alpha": 0} if wrapper is SHSTreeRegressor else {}
    model = wrapper(
        estimator_=DecisionTreeRegressor(max_leaf_nodes=4, random_state=0),
        reg_param="gcv",
        **kwargs,
    )
    model.shrinkage_scheme_ = scheme

    with pytest.raises((ValueError, TypeError), match="node_based"):
        model.fit(X, y)


def test_prefit_standalone_auto_uses_original_tree_statistics():
    X, y = _data(12)
    original = DecisionTreeRegressor(max_leaf_nodes=4, random_state=0).fit(X, y)
    chosen, _ = select_hs_reg_param(original)

    model = HSTreeRegressor(estimator_=deepcopy(original), reg_param="gcv")

    assert model.reg_param_ == chosen
    assert_allclose(model.predict(X), _hs_matrix(original, X, chosen) @ y, atol=1e-12)


@pytest.mark.parametrize("wrapper", [HSTreeRegressor, SHSTreeRegressor])
def test_repeated_fit_recomputes_auto_selection_and_numeric_fit_clears_diagnostics(wrapper):
    X, y_first = _data(3)
    _, y_second = _data(4)
    kwargs = {"sp_alpha": 0} if wrapper is SHSTreeRegressor else {}
    model = wrapper(
        estimator_=DecisionTreeRegressor(max_leaf_nodes=5, random_state=0),
        reg_param="gcv",
        **kwargs,
    ).fit(X, y_first)

    model.fit(X, y_second)

    original = DecisionTreeRegressor(max_leaf_nodes=5, random_state=0).fit(X, y_second)
    expected, _ = select_hs_reg_param(original)
    assert model.reg_param_ == expected
    assert_allclose(model.predict(X), _hs_matrix(original, X, expected) @ y_second)

    model.set_params(reg_param=2.0).fit(X, y_second)

    assert model.reg_param_ == 2.0
    assert model.gcv_results_ is None
    assert_allclose(model.predict(X), _hs_matrix(original, X, 2.0) @ y_second)


@pytest.mark.parametrize("wrapper", [HSTreeRegressor, SHSTreeRegressor])
def test_auto_hs_supports_trees_deeper_than_python_recursion_limit(wrapper):
    X = np.arange(1100.0)[:, None]
    y = np.arange(1100) % 2
    kwargs = {"sp_alpha": 0} if wrapper is SHSTreeRegressor else {}
    model = wrapper(
        estimator_=DecisionTreeRegressor(random_state=0),
        reg_param="gcv",
        **kwargs,
    ).fit(X, y)

    assert model.estimator_.get_depth() > 1000
    assert np.isfinite(model.gcv_results_["gcv_score"])
    assert np.isfinite(model.predict(X)).all()


@pytest.mark.parametrize("raw_tree", [False, True])
@pytest.mark.parametrize("sparse_input", [False, True])
def test_legacy_gcv_helper_checks_fitting_data_and_matches_shared_selector(raw_tree, sparse_input):
    from scipy.sparse import csr_matrix

    X, y = _data(9)
    fitted = DecisionTreeRegressor(max_leaf_nodes=6, random_state=0).fit(X, y)
    expected, expected_info = select_hs_reg_param(fitted)
    tree = fitted.tree_ if raw_tree else fitted
    data = csr_matrix(X) if sparse_input else X

    actual, info = get_gcv_reg_param(tree, data, y, return_info=True)

    assert actual == expected
    assert info["gcv_score"] == expected_info["gcv_score"]
    with pytest.raises(ValueError, match="match|unchanged"):
        get_gcv_reg_param(tree, data, y + 1)


def test_raw_gcv_helper_requires_data_and_rejects_non_quadratic_statistics():
    X = np.arange(3.)[:, None]
    y = np.array([0., 1., 100.])
    mean_tree = DecisionTreeRegressor(min_samples_split=4).fit(X, y)
    median_tree = DecisionTreeRegressor(
        criterion="absolute_error", min_samples_split=4
    ).fit(X, y)

    with pytest.raises(ValueError, match="fitting X and y"):
        get_gcv_reg_param(mean_tree.tree_)
    with pytest.raises(ValueError, match="mean|impurit|variance"):
        get_gcv_reg_param(median_tree.tree_, X, y)


def test_sparse_prefit_gcv_requires_the_matching_fitting_observations():
    X, y = _data(2)
    fitted = DecisionTreeRegressor(max_leaf_nodes=5, random_state=0).fit(X, y)
    model = SHSTreeRegressor(
        estimator_=fitted, prefit=True, sp_alpha=0, reg_param="gcv"
    ).fit(X, y)

    assert model.gcv_results_["conditional_on_tree"] is True
    with pytest.raises(ValueError, match="match|unchanged"):
        model.fit(X, y + 1)


def test_gcv_one_observation_reports_undefined_score_without_nan():
    fitted = DecisionTreeRegressor().fit([[0.]], [2.])

    selected, info = select_hs_reg_param(fitted)

    assert selected == 0
    assert info["gcv_defined"] is False
    assert info["gcv_score"] == np.inf


def test_sparse_auto_rejects_complex_weights_before_conversion():
    X, y = _data()
    model = SHSTreeRegressor(sp_alpha=0, reg_param="gcv")

    with pytest.raises(ValueError, match="real"):
        model.fit(X, y, sample_weight=np.full(len(y), 1 + 2j))


def test_gcv_checks_target_variance_and_handles_constant_response_roundoff():
    X, y = _data()
    fitted = DecisionTreeRegressor(max_leaf_nodes=5, random_state=0).fit(X, y)

    with pytest.raises(ValueError, match="variance does not match"):
        select_hs_reg_param(fitted, y=2 * y)
    constant = np.full(len(y), .1)
    constant_tree = DecisionTreeRegressor(min_samples_split=len(y) + 1).fit(X, constant)
    selected, info = select_hs_reg_param(constant_tree, y=constant)
    assert selected == 0
    assert info["gcv_score"] == 0


def test_failed_refinement_cannot_return_an_invalid_penalty(monkeypatch):
    from types import SimpleNamespace
    from imodels.tree import _hs_gcv

    X, y = _data()
    fitted = DecisionTreeRegressor(max_leaf_nodes=5, random_state=0).fit(X, y)
    monkeypatch.setattr(
        _hs_gcv, "minimize_scalar",
        lambda *args, **kwargs: SimpleNamespace(success=False, fun=-np.inf, x=-4.),
    )

    selected, info = select_hs_reg_param(fitted)

    assert selected >= 0
    assert info["search_converged"] is False
    assert "refined" not in info["candidate_kinds"]
    assert np.isfinite(info["gcv_score"])


def test_supplied_constant_target_cannot_erase_real_root_variance():
    X, y = _data()
    root_only = DecisionTreeRegressor(min_samples_split=len(y) + 1).fit(X, y)

    with pytest.raises(ValueError, match="variance does not match"):
        select_hs_reg_param(root_only, y=np.full(len(y), root_only.tree_.value[0, 0, 0]))


@pytest.mark.parametrize("reg_param", [3., "gcv"])
def test_default_hs_instances_are_independent_before_and_after_fit(reg_param):
    X, y = _data(0, n=60)
    first = HSTreeRegressor(
        reg_param=reg_param, max_leaf_nodes=5, random_state=0,
    ).fit(X, y)
    before = first.predict(X).copy()
    second = HSTreeRegressor(
        reg_param=reg_param, max_leaf_nodes=5, random_state=0,
    )
    assert first.estimator_ is not second.estimator_
    assert not hasattr(second.estimator_, "tree_")
    assert second.get_params()["estimator_"] is None
    second.fit(X, -y)
    assert_array_equal(first.predict(X), before)


@pytest.mark.parametrize("reg_param", [3., "gcv"])
def test_two_hs_wrappers_can_share_one_prefitted_tree_without_mutating_it(reg_param):
    X, y = _data(17)
    source = DecisionTreeRegressor(max_leaf_nodes=5, random_state=0).fit(X, y)
    values = source.tree_.value.copy()
    predictions = source.predict(X).copy()
    first = HSTreeRegressor(estimator_=source, reg_param=reg_param)
    second = HSTreeRegressor(estimator_=source, reg_param=reg_param)
    assert first.estimator_ is not second.estimator_
    assert first.estimator_ is not source
    assert first.get_params(deep=False)["estimator_"] is source
    assert_array_equal(source.tree_.value, values)
    assert_array_equal(source.predict(X), predictions)
    assert_allclose(first.predict(X), second.predict(X), rtol=0, atol=0)


@pytest.mark.parametrize("reg_param", [3., "gcv"])
@pytest.mark.parametrize("prefitted", [False, True])
def test_hs_fit_and_clone_preserve_supplied_estimator_and_constructor_parameters(reg_param, prefitted):
    X, y = _data(18)
    source = DecisionTreeRegressor(max_leaf_nodes=7, random_state=12)
    if prefitted:
        source.fit(X, y)
    before = deepcopy(source)
    model = HSTreeRegressor(
        estimator_=source, reg_param=reg_param,
        max_leaf_nodes=4, random_state=0,
    ).fit(X, y)
    assert source.get_params() == before.get_params()
    if prefitted:
        assert_array_equal(source.tree_.value, before.tree_.value)
    else:
        assert not hasattr(source, "tree_")
    assert model.estimator_.max_leaf_nodes == 4
    assert model.estimator_.random_state == 0
    prediction = model.predict(X).copy()
    copied = clone(model)
    assert copied.reg_param == reg_param
    assert copied.max_leaf_nodes == 4
    assert copied.random_state == 0
    assert not hasattr(copied.estimator_, "tree_")
    copied.fit(X, y)
    assert_allclose(copied.predict(X), prediction, rtol=0, atol=0)
    assert_array_equal(model.predict(X), prediction)


@pytest.mark.parametrize("reg_param", [3., "gcv"])
def test_hs_repeated_fit_starts_from_fresh_tree_means(reg_param):
    X, y = _data(21)
    model = HSTreeRegressor(
        reg_param=reg_param, max_leaf_nodes=5, random_state=0,
    ).fit(X, y)
    before = model.predict(X).copy()
    model.fit(X, y)
    assert_allclose(model.predict(X), before, rtol=0, atol=0)
    model.fit(X, -y)
    independent = HSTreeRegressor(
        reg_param=reg_param, max_leaf_nodes=5, random_state=0,
    ).fit(X, -y)
    assert_allclose(model.predict(X), independent.predict(X), rtol=0, atol=0)


def test_hs_nested_and_replaced_estimator_parameters_apply_to_next_fit_and_clone():
    X, y = _data(23)
    model = HSTreeRegressor(
        estimator_=DecisionTreeRegressor(random_state=0), reg_param="gcv",
    ).fit(X, y)
    before = model.predict(X).copy()
    model.set_params(estimator___max_leaf_nodes=3)
    assert model.get_params()["estimator___max_leaf_nodes"] == 3
    assert_array_equal(model.predict(X), before)
    model.fit(X, y)
    assert model.estimator_.get_n_leaves() == 3
    replacement = DecisionTreeRegressor(max_depth=1, random_state=0)
    model.set_params(estimator_=replacement)
    assert model.get_params(deep=False)["estimator_"] is replacement
    copied = clone(model).fit(X, y)
    assert copied.estimator_.get_depth() == 1
    assert not hasattr(replacement, "tree_")


@pytest.mark.parametrize("wrapper", [HSTreeRegressorCV, HSTreeClassifierCV])
@pytest.mark.parametrize("explicit_estimator", [False, True])
def test_ordinary_hs_cv_clone_preserves_grid_and_legacy_explicit_leaf_cap(wrapper, explicit_estimator):
    X, y = _data(25, n=45)
    tree_class = DecisionTreeRegressor
    if wrapper is HSTreeClassifierCV:
        tree_class = DecisionTreeClassifier
        y = (y > np.median(y)).astype(int)
    source = tree_class(max_leaf_nodes=7, random_state=0) if explicit_estimator else None
    model = wrapper(
        estimator_=source, reg_param_list=(0., 3.), max_leaf_nodes=3, cv=3,
    )
    copied = clone(model)
    assert copied.reg_param_list == (0., 3.)
    assert copied.max_leaf_nodes == 3
    assert copied.fit(X, y) is copied
    expected_cap = 7 if explicit_estimator else 3
    assert copied.estimator_.max_leaf_nodes == expected_cap
    assert clone(copied).fit(X, y).estimator_.max_leaf_nodes == expected_cap
    if explicit_estimator:
        assert not hasattr(source, "tree_")


def test_prefitted_hs_classifier_preserves_labels_and_correct_probability_scale():
    X = np.arange(8.)[:, None]
    y = np.array(["left"] * 4 + ["right"] * 4)
    source = DecisionTreeClassifier(max_depth=1, random_state=0).fit(X, y)
    before = source.tree_.value.copy()
    model = HSTreeClassifier(estimator_=source, reg_param=8.)
    assert_array_equal(source.tree_.value, before)
    assert_array_equal(model.predict(X), y)
    expected = np.array([[.75, .25]] * 4 + [[.25, .75]] * 4)
    assert_allclose(model.predict_proba(X), expected, rtol=0, atol=1e-15)


def test_prefitted_warm_start_forest_is_not_mutated_or_shrunk_twice_on_refit():
    X, y = _data(26)
    source = RandomForestRegressor(
        n_estimators=3, max_leaf_nodes=5, random_state=0, warm_start=True,
    ).fit(X, y)
    before = source.predict(X).copy()
    model = HSTreeRegressor(estimator_=source, reg_param=3.)
    model.fit(X, y)
    first = model.predict(X).copy()
    model.fit(X, y)
    assert_allclose(model.predict(X), first, rtol=0, atol=0)
    assert_array_equal(source.predict(X), before)


def test_copying_a_prefitted_tree_does_not_bypass_gcv_statistic_validation():
    X, y = _data(27)
    shrunk = HSTreeRegressor(
        estimator_=DecisionTreeRegressor(max_leaf_nodes=5), reg_param=3.,
    ).fit(X, y)
    with pytest.raises(ValueError, match="unmodified"):
        HSTreeRegressor(estimator_=shrunk.estimator_, reg_param="gcv")


@pytest.mark.parametrize("classification", [False, True])
@pytest.mark.parametrize("prefitted", [False, True])
def test_ccp_hs_consumer_retains_selected_pruning_without_mutating_source(classification, prefitted):
    from imodels.tree.cart_ccp import (
        DecisionTreeCCPClassifier, DecisionTreeCCPRegressor,
        HSDecisionTreeCCPClassifierCV, HSDecisionTreeCCPRegressorCV,
    )
    from imodels.util.tree import compute_tree_complexity

    X, y = _data(29, n=40)
    if classification:
        y = (y > np.median(y)).astype(int)
        source = DecisionTreeClassifier(random_state=0)
        ccp_class, hs_class = DecisionTreeCCPClassifier, HSDecisionTreeCCPClassifierCV
    else:
        source = DecisionTreeRegressor(random_state=0)
        ccp_class, hs_class = DecisionTreeCCPRegressor, HSDecisionTreeCCPRegressorCV
    if prefitted:
        source.fit(X, y)
    before = deepcopy(source)
    reference = ccp_class(clone(source), desired_complexity=2)
    reference.fit(X, y)
    model = hs_class(source, desired_complexity=2, reg_param_list=[0., 1.], cv=2)
    model.fit(X, y)
    assert model.estimator_.ccp_alpha == reference.estimator_.ccp_alpha
    assert model.complexity_ == compute_tree_complexity(reference.estimator_.tree_)
    assert source.get_params() == before.get_params()
    if prefitted:
        assert_array_equal(source.tree_.value, before.tree_.value)
    else:
        assert not hasattr(source, "tree_")
