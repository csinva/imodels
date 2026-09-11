"""Focused behavioral tests for sparse-pruning tree wrappers."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy import sparse
from sklearn.base import clone, is_classifier, is_regressor
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.exceptions import ConvergenceWarning, NotFittedError
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils.validation import check_is_fitted

from imodels import (
    SHSTreeClassifier,
    SHSTreeClassifierCV,
    SHSTreeRegressor,
    SHSTreeRegressorCV,
    SPTreeClassifier,
    SPTreeClassifierCV,
    SPTreeRegressor,
    SPTreeRegressorCV,
)
from imodels.tree.sparse_pruning.sparse_hierarchical_shrinkage import (
    _collect_internal_node_ids,
)
from imodels.util.tree import compute_tree_complexity


def _regression_tree_data():
    """Data whose depth-two CART has a root and two internal children."""
    X = np.arange(8, dtype=float).reshape(-1, 1)
    y = np.array([0, 0, 1, 1, 10, 10, 11, 11], dtype=float)
    return X, y


def _classification_data(labels=(0, 1)):
    X = np.arange(8, dtype=float).reshape(-1, 1)
    y01 = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y = np.where(y01 == 0, labels[0], labels[1])
    return X, y


@pytest.mark.parametrize(
    "estimator_cls",
    [
        SHSTreeRegressor,
        SHSTreeClassifier,
        SPTreeRegressor,
        SPTreeClassifier,
    ],
)
def test_default_instances_do_not_share_base_estimator(estimator_cls):
    first = estimator_cls()
    second = estimator_cls()

    both_deferred = first.estimator_ is None and second.estimator_ is None
    both_distinct = (
        first.estimator_ is not None
        and second.estimator_ is not None
        and first.estimator_ is not second.estimator_
    )
    assert both_deferred or both_distinct
    assert first.prune_set == second.prune_set == "auto"


def test_defaults_select_infinity_structural_path_and_normalized_hs():
    model = SPTreeRegressorCV()

    assert model.max_iter == 2000
    assert model.sp_alpha_list == "auto"
    assert model.solver == "auto"
    assert model.ord == np.inf
    assert model.selection_rule == "one_se"
    assert model.reg_param_mode == "normalized"


@pytest.mark.parametrize("wrapper", [
    SPTreeRegressorCV, SHSTreeRegressorCV,
    SPTreeClassifierCV, SHSTreeClassifierCV,
])
@pytest.mark.parametrize("template_cap, cap_options, expected_cap", [
    (6, {}, 6),
    (6, {"max_leaf_nodes": None}, 6),
    (6, {"max_leaf_nodes": 3}, 3),
    (None, {}, None),
    (None, {"max_leaf_nodes": None}, None),
    (None, {"max_leaf_nodes": 3}, 3),
])
def test_cv_leaf_cap_respects_template_unless_overridden(
    wrapper, template_cap, cap_options, expected_cap,
):
    rng = np.random.default_rng(12)
    X = rng.normal(size=(256, 4))
    classification = is_classifier(wrapper())
    y = np.tile([0, 1], 128) if classification else rng.normal(size=len(X))
    tree_cls = DecisionTreeClassifier if classification else DecisionTreeRegressor
    options = dict(cap_options)
    if template_cap is not None:
        options["estimator_"] = tree_cls(max_leaf_nodes=template_cap)
    model = wrapper(cv=2, reg_param_list=[0], random_state=0, **options)

    assert model.max_leaf_nodes == cap_options.get("max_leaf_nodes")
    assert clone(model).max_leaf_nodes == model.max_leaf_nodes
    model.fit(X, y)

    assert model.estimator_.max_leaf_nodes == expected_cap
    for path in [model.pruning_path_, *model.cv_path_results_]:
        if expected_cap is None:
            assert len(path.node_ids) > 19
        else:
            assert len(path.node_ids) == expected_cap - 1
    if template_cap is not None:
        assert options["estimator_"].max_leaf_nodes == template_cap
        assert not hasattr(options["estimator_"], "tree_")


@pytest.mark.parametrize("wrapper", [
    SPTreeRegressor, SHSTreeRegressor, SPTreeClassifier, SHSTreeClassifier,
])
@pytest.mark.parametrize("cap_options", [
    {}, {"max_leaf_nodes": None}, {"max_leaf_nodes": 7},
])
def test_non_cv_default_tree_has_no_hidden_leaf_cap(wrapper, cap_options):
    X = np.arange(64, dtype=float).reshape(-1, 1)
    y = np.tile([0, 1], 32)
    model = wrapper(sp_alpha=0, reg_param=0, random_state=0, **cap_options)
    model.fit(X, y)

    cap = cap_options.get("max_leaf_nodes")
    assert model.estimator_.max_leaf_nodes == cap
    assert model.estimator_.get_n_leaves() == (64 if cap is None else cap)


@pytest.mark.parametrize("estimator_cls", [
    SHSTreeRegressor, SHSTreeClassifier, SHSTreeRegressorCV, SHSTreeClassifierCV,
    SPTreeRegressor, SPTreeClassifier, SPTreeRegressorCV, SPTreeClassifierCV,
])
def test_public_estimators_have_their_own_api_documentation(estimator_cls):
    import inspect

    assert estimator_cls.__doc__
    doc = inspect.getdoc(estimator_cls)
    assert "Mixin class" not in doc
    assert "solver" in doc or "SHSTreeClassifierCV" in doc


def test_group_collection_handles_deep_trees_without_recursion():
    from types import SimpleNamespace
    from imodels.tree.sparse_pruning.sparse_hierarchical_shrinkage import (
        _collect_internal_node_ids, _find_subtrees,
    )

    p = 1100
    feature = np.full(2 * p + 1, -2)
    left = np.full(2 * p + 1, -1)
    right = left.copy()
    ids = np.arange(0, 2 * p, 2)
    feature[ids], left[ids], right[ids] = 0, ids + 1, ids + 2
    tree = SimpleNamespace(feature=feature, children_left=left, children_right=right)
    assert_array_equal(_collect_internal_node_ids(tree), ids)
    groups = _find_subtrees(tree, 0, ids)
    assert len(groups) == p
    assert_array_equal(groups[0], [p])
    assert_array_equal(groups[-1], np.arange(1, p + 1))


def test_gcv_cv_accepts_a_single_string_or_singleton_grid_on_refit():
    X, y = _regression_tree_data()
    common = dict(cv=2, max_leaf_nodes=3, random_state=0)
    expected = SHSTreeRegressorCV(reg_param_list="gcv", **common).fit(X, y)
    model = SHSTreeRegressorCV(reg_param_list=["gcv"], **common)
    for _ in range(2):
        model.fit(X, y)
        assert_allclose(model.cv_scores_, expected.cv_scores_)
        assert_allclose(model.predict(X), expected.predict(X))
        assert model.reg_param_ == expected.reg_param_


def test_repeated_fit_rebuilds_tree_from_new_data():
    X, y_first = _regression_tree_data()
    y_second = -y_first
    tree_spec = DecisionTreeRegressor(max_depth=2, random_state=0)
    expected = clone(tree_spec).fit(X, y_second)
    model = SPTreeRegressor(
        estimator_=clone(tree_spec),
        sp_alpha=0,
        reg_param=0,
        random_state=0,
    )

    model.fit(X, y_first)
    first_predictions = model.predict(X).copy()
    model.fit(X, y_second)

    assert not np.allclose(first_predictions, model.predict(X))
    assert_allclose(model.predict(X), expected.predict(X))
    assert_array_equal(
        model.estimator_.tree_.children_left, expected.tree_.children_left
    )


@pytest.mark.parametrize("task", ["regression", "classification"])
def test_alpha_zero_preserves_fresh_base_tree(task):
    if task == "regression":
        X, y = _regression_tree_data()
        tree_spec = DecisionTreeRegressor(max_depth=2, random_state=0)
        model = SPTreeRegressor(
            estimator_=clone(tree_spec),
            sp_alpha=0,
            reg_param=0,
            random_state=0,
        )
    else:
        X, y = _classification_data()
        tree_spec = DecisionTreeClassifier(max_depth=2, random_state=0)
        model = SPTreeClassifier(
            estimator_=clone(tree_spec),
            sp_alpha=0,
            reg_param=0,
            random_state=0,
        )

    expected = clone(tree_spec).fit(X, y)
    model.fit(X, y)

    assert_array_equal(
        model.estimator_.tree_.children_left, expected.tree_.children_left
    )
    assert_array_equal(
        model.estimator_.tree_.children_right, expected.tree_.children_right
    )
    assert_allclose(model.estimator_.tree_.value, expected.tree_.value)
    assert_array_equal(model.predict(X), expected.predict(X))
    if task == "classification":
        assert_allclose(model.predict_proba(X), expected.predict_proba(X))


def test_pruning_keeps_ancestors_of_active_descendant():
    X, y = _regression_tree_data()
    model = SPTreeRegressor(
        estimator_=DecisionTreeRegressor(max_depth=2, random_state=0),
        sp_alpha=1,
        reg_param=0,
        random_state=0,
    )

    def root_zero_descendant_active(**kwargs):
        # Columns are [intercept, root, left descendant, right descendant].
        beta = np.zeros(kwargs["X"].shape[1])
        beta[2] = 1.0
        return beta

    model.hiCAP = root_zero_descendant_active
    model.fit(X, y)

    reachable_internal_nodes = _collect_internal_node_ids(model.estimator_.tree_)
    assert_array_equal(reachable_internal_nodes, np.array([0, 1]))
    assert model.complexity_ == 2


def test_classifier_preserves_arbitrary_binary_labels():
    X, y = _classification_data(labels=(-3, 7))
    model = SPTreeClassifier(
        estimator_=DecisionTreeClassifier(max_depth=1, random_state=0),
        sp_alpha=0,
        reg_param=0,
        random_state=0,
    ).fit(X, y)

    assert_array_equal(model.classes_, np.array([-3, 7]))
    assert_array_equal(model.predict(X), y)
    predicted_from_proba = model.classes_[np.argmax(model.predict_proba(X), axis=1)]
    assert_array_equal(predicted_from_proba, model.predict(X))
    assert model.__sklearn_tags__().classifier_tags.multi_class


@pytest.mark.parametrize("scheme, pure_class_probability", [
    ("node_based", .75), ("constant", .5 + .5 / 9), ("leaf_based", .5 + .5 / 3),
])
def test_classifier_shrinkage_uses_node_probabilities_not_sample_counts(scheme, pure_class_probability):
    X, y = _classification_data()
    model = SHSTreeClassifier(
        estimator_=DecisionTreeClassifier(max_depth=1, random_state=0),
        sp_alpha=0,
        reg_param=8,
        random_state=0,
    )
    model.shrinkage_scheme_ = scheme
    model.fit(X, y)

    tree_values = model.estimator_.tree_.value[:, 0, :]
    assert_allclose(tree_values[0], [0.5, 0.5])
    assert_allclose(tree_values[1], [pure_class_probability, 1 - pure_class_probability])
    assert_allclose(tree_values[2], [1 - pure_class_probability, pure_class_probability])
    assert_allclose(model.predict_proba(X).sum(axis=1), 1)


@pytest.mark.parametrize("scheme", ["node_based", "constant", "leaf_based"])
def test_classifier_shrinkage_handles_deep_retained_tree_without_recursion(scheme):
    # Alternating labels yield a depth-1099 chain, beyond Python's usual
    # recursion limit. Do not change that limit to accommodate the estimator.
    X = np.arange(1100., dtype=float).reshape(-1, 1)
    y = np.where(np.arange(len(X)) % 2, "odd", "even")
    model = SHSTreeClassifier(
        estimator_=DecisionTreeClassifier(random_state=0),
        sp_alpha=0., reg_param=1., random_state=0,
    )
    model.shrinkage_scheme_ = scheme
    model.fit(X, y)
    assert model.estimator_.get_depth() == len(X) - 1
    assert model.complexity_ == len(X) - 1
    probability = model.predict_proba(X)
    assert np.isfinite(probability).all()
    assert np.all((probability >= 0.) & (probability <= 1.))
    assert_allclose(probability.sum(axis=1), 1., atol=1e-12)
    assert_array_equal(model.predict(X), model.classes_[probability.argmax(axis=1)])


def test_legacy_apa_classifier_rejects_multiclass_targets():
    X = np.arange(9, dtype=float).reshape(-1, 1)
    y = np.tile(np.arange(3), 3)
    model = SPTreeClassifier(
        estimator_=DecisionTreeClassifier(max_depth=2, random_state=0),
        sp_alpha=0,
        reg_param=0,
        random_state=0,
        solver="apa_apg2",
    )

    with pytest.raises(ValueError):
        model.fit(X, y)


@pytest.mark.parametrize("wrapper", [SPTreeClassifier, SHSTreeClassifier])
@pytest.mark.parametrize("solver", ["topology", "proximal", "coefficient_path"])
def test_multiclass_range_penalty_endpoints_and_coefficient_metadata(wrapper, solver):
    # The class-range threshold is 4/11. An incorrect sum-zero 2*linf
    # penalty instead prunes at 3/11 and fails this interior support check.
    X = np.repeat([[-1.], [1.]], 11, axis=0)
    labels = np.array(["oak", "maple", "pine"])
    y = np.r_[np.repeat(labels, [1, 5, 5]), np.repeat(labels, [9, 1, 1])]
    base = DecisionTreeClassifier(max_depth=1, random_state=0).fit(X, y)
    model = wrapper(
        max_leaf_nodes=2, solver=solver, sp_alpha=7 / 22, reg_param=0,
        random_state=0, tol=1e-8,
    ).fit(X, y)
    assert model.complexity_ == 1
    assert_allclose(model.pruning_path_.lambdas, [4 / 11, 0.], atol=1e-14)
    assert_array_equal(model.classes_, base.classes_)
    assert_allclose(model.predict_proba(X), base.predict_proba(X))
    if solver != "topology":
        assert model.coef_.shape == (1, 3) and model.intercept_.shape == (3,)
        assert_allclose(model.coef_.sum(axis=1), 0., atol=1e-12)
        assert_allclose(model.intercept_.sum(), 0., atol=1e-12)
        assert np.ptp(model.coef_[0]) > .1
        assert model.optimization_results_[0]["certificate_scope"] == (
            "structure_and_coefficients"
        )
    if solver == "coefficient_path":
        assert not model.coefficient_path_.exact
        assert 7 / 22 in model.coefficient_path_.lambdas
        assert_allclose(model.coefficient_path_.at(7 / 22)[0], model.coef_)
    model.set_params(sp_alpha=1.).fit(X, y)
    assert model.complexity_ == 0
    assert_allclose(model.predict_proba(X), np.tile(base.tree_.value[0, 0] /
                    base.tree_.value[0, 0].sum(), (len(X), 1)))
    model.set_params(sp_alpha=0).fit(X, y)
    assert model.coef_ is model.intercept_ is None
    assert_allclose(model.predict_proba(X), base.predict_proba(X))
    info = model.optimization_results_[0]
    assert info["certified"] is True
    assert info["certificate_scope"] == "structure"
    assert info["coefficients_available"] is False
    assert model.optimization_certified_
    assert model.complexity_ == 1
    if solver == "coefficient_path":
        assert model.coefficient_path_ is not None
        assert np.all(model.coefficient_path_.lambdas > 0)


@pytest.mark.parametrize("class_weight", ["balanced", {"oak": 2., "elm": 1., "pine": 3.}])
def test_native_multiclass_class_weights_match_effective_sample_weights(class_weight):
    from sklearn.utils.class_weight import compute_sample_weight

    X = np.arange(36., dtype=float).reshape(-1, 1)
    y = np.repeat(["oak", "elm", "pine"], [8, 12, 16])
    weights = 1. + np.arange(len(y)) % 3
    common = dict(sp_alpha=.03, reg_param=5., random_state=0)
    model = SHSTreeClassifier(
        estimator_=DecisionTreeClassifier(max_leaf_nodes=4, class_weight=class_weight),
        **common,
    ).fit(X, y, sample_weight=weights)
    effective = weights * compute_sample_weight(class_weight, y)
    reference = SHSTreeClassifier(
        estimator_=DecisionTreeClassifier(max_leaf_nodes=4), **common,
    ).fit(X, y, sample_weight=effective)
    assert model.solver_ == reference.solver_ == "topology"
    assert_allclose(model.pruning_path_.activation_lambdas,
                    reference.pruning_path_.activation_lambdas)
    assert_allclose(model.predict_proba(X), reference.predict_proba(X), atol=1e-12)
    assert_allclose(clone(model).fit(X, y, sample_weight=weights).predict_proba(X),
                    model.predict_proba(X), atol=1e-12)


@pytest.mark.parametrize("options", [dict(ord=2), dict(solver="apa_apg2"),
                                     dict(solver="hicap"), dict(support_tol=.01)])
def test_multiclass_rejects_unsupported_solver_objectives(options):
    X = np.arange(18.).reshape(-1, 1)
    y = np.arange(len(X)) % 3
    with pytest.raises(ValueError):
        SPTreeClassifier(max_leaf_nodes=3, sp_alpha=.1, **options).fit(X, y)


@pytest.mark.parametrize("reg_param", ["invalid", -1.0])
def test_unvalidated_or_negative_shrinkage_is_rejected(reg_param):
    X, y = _regression_tree_data()
    model = SHSTreeRegressor(
        estimator_=DecisionTreeRegressor(max_depth=2, random_state=0),
        sp_alpha=0,
        reg_param=reg_param,
        random_state=0,
    )

    with pytest.raises(ValueError, match="reg_param|GCV"):
        model.fit(X, y)


@pytest.mark.parametrize("task", ["regression", "classification"])
def test_cv_routes_sample_weight_with_fold_compatible_shapes(task):
    X = np.arange(12, dtype=float).reshape(-1, 1)
    sample_weight = np.linspace(1.0, 2.0, len(X))
    if task == "regression":
        y = np.sin(X[:, 0])
        model = SPTreeRegressorCV(
            estimator_=DecisionTreeRegressor(max_depth=2, random_state=0),
            sp_alpha_list=(0.1,),
            reg_param_list=(0,),
            cv=2,
            random_state=0,
            max_iter=25,
        )
    else:
        y = np.tile([0, 1], len(X) // 2)
        model = SPTreeClassifierCV(
            estimator_=DecisionTreeClassifier(max_depth=2, random_state=0),
            sp_alpha_list=(0.1,),
            reg_param_list=(0,),
            cv=2,
            random_state=0,
            max_iter=25,
        )

    model.fit(X, y, sample_weight=sample_weight)

    assert model.predict(X).shape == y.shape


@pytest.mark.parametrize(
    ("estimator_cls", "predicate"),
    [
        (SHSTreeRegressor, is_regressor),
        (SHSTreeRegressorCV, is_regressor),
        (SPTreeRegressor, is_regressor),
        (SPTreeRegressorCV, is_regressor),
        (SHSTreeClassifier, is_classifier),
        (SHSTreeClassifierCV, is_classifier),
        (SPTreeClassifier, is_classifier),
        (SPTreeClassifierCV, is_classifier),
    ],
)
def test_sklearn_clone_and_estimator_type_tags(estimator_cls, predicate):
    estimator = estimator_cls()
    cloned = clone(estimator)

    assert type(cloned) is estimator_cls
    assert predicate(estimator)
    assert predicate(cloned)


@pytest.mark.parametrize("task", ["regression", "classification"])
def test_fitted_cv_estimator_remains_cloneable(task):
    if task == "regression":
        X, y = _regression_tree_data()
        model = SPTreeRegressorCV(
            sp_alpha_list=(0,),
            reg_param_list=(0,),
            cv=2,
            random_state=0,
        )
    else:
        X, y = _classification_data()
        model = SPTreeClassifierCV(
            sp_alpha_list=(0,),
            reg_param_list=(0,),
            cv=2,
            random_state=0,
        )

    model.fit(X, y)
    cloned = clone(model)

    assert type(cloned) is type(model)
    assert not hasattr(cloned, "sp_alpha_")
    assert not hasattr(cloned, "reg_param_")


def test_cv_repr_uses_constructor_parameters():
    model = SPTreeRegressorCV(
        sp_alpha_list=(0,),
        reg_param_list=(0,),
        cv=2,
        selection_rule="best",
        reg_param_mode="raw",
    )

    representation = repr(model)
    reconstructed = eval(
        representation, {"SPTreeRegressorCV": SPTreeRegressorCV}
    )

    assert reconstructed.sp_alpha_list == (0,)
    assert reconstructed.reg_param_list == (0,)
    assert reconstructed.cv == 2
    assert reconstructed.selection_rule == "best"
    assert reconstructed.reg_param_mode == "raw"


def test_cv_normalizes_fold_shrinkage_and_refits_full_parameter():
    X, y = _regression_tree_data()
    sample_weight = np.arange(1, len(y) + 1, dtype=float)
    observed_reg_params = []

    def record_reg_param(estimator, X, y):
        observed_reg_params.append(estimator.reg_param)
        return 0.0

    model = SHSTreeRegressorCV(
        estimator_=DecisionTreeRegressor(max_depth=2, random_state=0),
        sp_alpha_list=(0,),
        reg_param_list=(8,),
        cv=2,
        scoring=record_reg_param,
        random_state=0,
    ).fit(X, y, sample_weight=sample_weight)

    splits = KFold(
        n_splits=2, shuffle=True, random_state=0
    ).split(X)
    expected_fractions = np.array(
        [
            sample_weight[train].sum() / sample_weight.sum()
            for train, _ in splits
        ]
    )
    assert_allclose(model.cv_weight_fractions_, expected_fractions)
    assert_allclose(observed_reg_params, 8 * expected_fractions)
    assert_allclose(model.cv_reg_params_[0], 8 * expected_fractions)
    assert model.reg_param_ == model.reg_param == 8

    direct = SHSTreeRegressor(
        estimator_=DecisionTreeRegressor(max_depth=2, random_state=0),
        sp_alpha=0,
        reg_param=8,
        random_state=0,
    ).fit(X, y, sample_weight=sample_weight)
    assert_allclose(model.predict(X), direct.predict(X))


def test_classifier_cv_normalization_uses_folded_class_weights():
    X, y = _classification_data()
    sample_weight = np.arange(1, len(y) + 1, dtype=float)
    class_weight = {0: 1, 1: 3}
    effective_weight = sample_weight * np.where(y == 0, 1, 3)
    observed_reg_params = []

    def record_reg_param(estimator, X, y):
        observed_reg_params.append(estimator.reg_param)
        return 0.0

    model = SHSTreeClassifierCV(
        estimator_=DecisionTreeClassifier(
            max_depth=2,
            class_weight=class_weight,
            random_state=0,
        ),
        sp_alpha_list=(0,),
        reg_param_list=(8,),
        cv=2,
        scoring=record_reg_param,
        random_state=0,
    ).fit(X, y, sample_weight=sample_weight)

    splits = StratifiedKFold(
        n_splits=2, shuffle=True, random_state=0
    ).split(X, y)
    expected_fractions = np.array(
        [
            effective_weight[train].sum() / effective_weight.sum()
            for train, _ in splits
        ]
    )
    assert_allclose(model.cv_weight_fractions_, expected_fractions)
    assert_allclose(observed_reg_params, 8 * expected_fractions)


def test_cv_raw_and_constant_shrinkage_are_not_fold_scaled():
    X, y = _regression_tree_data()
    sample_weight = np.arange(1, len(y) + 1, dtype=float)

    for mode, scheme in (("raw", "node_based"), ("normalized", "constant")):
        observed_reg_params = []

        def record_reg_param(estimator, X, y):
            observed_reg_params.append(estimator.reg_param)
            return 0.0

        model = SHSTreeRegressorCV(
            sp_alpha_list=(0,),
            reg_param_list=(8,),
            cv=2,
            scoring=record_reg_param,
            reg_param_mode=mode,
            random_state=0,
        )
        model.shrinkage_scheme_ = scheme
        model.fit(X, y, sample_weight=sample_weight)

        assert_allclose(observed_reg_params, [8, 8])
        assert model.shrinkage_scheme_ == scheme


@pytest.mark.parametrize("task", ["regression", "classification"])
def test_cv_default_one_se_selects_simplest_competitive_tree(task):
    if task == "regression":
        X, y = _regression_tree_data()
        estimator = DecisionTreeRegressor(max_depth=2, random_state=0)
        cv_class = SPTreeRegressorCV
    else:
        X, y = _classification_data()
        estimator = DecisionTreeClassifier(max_depth=2, random_state=0)
        cv_class = SPTreeClassifierCV

    def zero_solver(**kwargs):
        return np.zeros(kwargs["X"].shape[1])

    def competitive_score(estimator, X, y):
        if estimator.sp_alpha == 0:
            return 1.2 if np.mean(X) > 3.5 else 0.8
        return 0.9

    common = {
        "estimator_": estimator,
        "sp_alpha_list": (0, 1),
        "reg_param_list": (0,),
        "cv": 2,
        "scoring": competitive_score,
        "random_state": 0,
    }
    one_se = cv_class(**common)
    one_se.hiCAP = zero_solver
    one_se.fit(X, y)

    assert one_se.best_index_ == 0
    assert one_se.selected_index_ == 1
    assert one_se.sp_alpha_ == 1
    assert one_se.mean_complexity_[1] == 0
    assert one_se.mean_complexity_[0] > 0
    assert_allclose(one_se.cv_score_se_[0], 0.2)
    assert_allclose(one_se.selection_threshold_, 0.8)
    assert_array_equal(one_se.one_se_candidate_mask_, [True, True])
    assert one_se.complexity_ == 0

    best = cv_class(**common, selection_rule="best")
    best.hiCAP = zero_solver
    best.fit(X, y)

    assert best.selected_index_ == best.best_index_ == 0
    assert best.sp_alpha_ == 0
    assert best.complexity_ > 0


def test_cv_one_se_tie_prefers_stronger_shrinkage():
    X, y = _regression_tree_data()

    def tied_score(estimator, X, y):
        return 0.0

    model = SHSTreeRegressorCV(
        sp_alpha_list=(0,),
        reg_param_list=(0, 1, 10),
        cv=2,
        scoring=tied_score,
        random_state=0,
    ).fit(X, y)

    assert model.reg_param_ == 10
    assert model.selected_index_ == 2


def test_cv_excludes_candidates_with_nonfinite_aggregate_scores():
    X, y = _regression_tree_data()

    def zero_solver(**kwargs):
        return np.zeros(kwargs["X"].shape[1])

    def overflowing_score(estimator, X, y):
        if estimator.sp_alpha == 0:
            return np.finfo(float).max
        return 0.0

    model = SPTreeRegressorCV(
        sp_alpha_list=(0, 1),
        reg_param_list=(0,),
        cv=2,
        scoring=overflowing_score,
        selection_rule="best",
        random_state=0,
    )
    model.hiCAP = zero_solver
    model.fit(X, y)

    assert np.isinf(model.scores_[0])
    assert model.best_index_ == model.selected_index_ == 1
    assert model.sp_alpha_ == 1


def test_cv_caches_pruning_and_retains_candidate_fold_results():
    X, y = _regression_tree_data()
    solver_calls = []

    def active_solver(**kwargs):
        solver_calls.append((len(kwargs["y"]), kwargs["lam"]))
        return np.ones(kwargs["X"].shape[1])

    def prefer_alpha_one_reg_ten(estimator, X, y):
        return -abs(estimator.sp_alpha - 1) - abs(
            estimator.reg_param - 10
        )

    model = SPTreeRegressorCV(
        estimator_=DecisionTreeRegressor(max_depth=2, random_state=0),
        sp_alpha_list=(1, 2),
        reg_param_list=(0, 1, 10),
        cv=2,
        scoring=prefer_alpha_one_reg_ten,
        selection_rule="best",
        reg_param_mode="raw",
        random_state=0,
    )
    model.hiCAP = active_solver
    model.fit(X, y)

    assert sorted(solver_calls) == [
        (4, 1),
        (4, 1),
        (4, 2),
        (4, 2),
        (8, 1),
    ]
    assert model.cv_scores_.shape == (6, 2)
    assert model.cv_complexities_.shape == (6, 2)
    assert model.cv_reg_params_.shape == (6, 2)
    assert_allclose(model.scores_, model.cv_scores_.mean(axis=1))
    assert model.cv_params_ == [
        {"sp_alpha": 1, "reg_param": 0},
        {"sp_alpha": 1, "reg_param": 1},
        {"sp_alpha": 1, "reg_param": 10},
        {"sp_alpha": 2, "reg_param": 0},
        {"sp_alpha": 2, "reg_param": 1},
        {"sp_alpha": 2, "reg_param": 10},
    ]
    assert_allclose(
        model.cv_reg_params_,
        [[0, 0], [1, 1], [10, 10], [0, 0], [1, 1], [10, 10]],
    )
    assert all(
        len(candidate_results) == 2
        for candidate_results in model.cv_optimization_results_
    )
    assert model.sp_alpha_ == 1
    assert model.reg_param_ == 10


def test_cv_uses_sklearn_higher_is_better_scorer_convention():
    class PreferLargerAlpha:
        _sign = 1

        def __call__(self, estimator, X, y, **kwargs):
            return float(estimator.sp_alpha)

    X, y = _regression_tree_data()
    model = SPTreeRegressorCV(
        sp_alpha_list=(0, 1),
        reg_param_list=(0,),
        cv=2,
        scoring=PreferLargerAlpha(),
        random_state=0,
        max_iter=10,
    ).fit(X, y)

    assert model.sp_alpha_ == 1


def test_cv_accepts_plain_estimator_scorer_callable():
    def prefer_larger_alpha(estimator, X, y):
        return float(estimator.sp_alpha)

    X, y = _regression_tree_data()
    model = SPTreeRegressorCV(
        sp_alpha_list=(0, 1),
        reg_param_list=(0,),
        cv=2,
        scoring=prefer_larger_alpha,
        random_state=0,
        max_iter=10,
    ).fit(X, y, sample_weight=np.ones(len(y)))

    assert model.sp_alpha_ == 1


def test_cv_preserves_two_argument_loss_metric_api():
    def mean_absolute_loss(y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred))

    X, y = _regression_tree_data()
    model = SPTreeRegressorCV(
        sp_alpha_list=(0,),
        reg_param_list=(0,),
        cv=2,
        scoring=mean_absolute_loss,
        random_state=0,
    ).fit(X, y, sample_weight=np.ones(len(y)))

    assert model.scores_.shape == (1,)
    assert model.scores_[0] <= 0


def test_cv_candidates_are_fully_finalized_before_estimator_scoring():
    observed_metadata = []

    def complexity_penalized_score(estimator, X, y):
        observed_metadata.append(
            (
                hasattr(estimator, "complexity_"),
                hasattr(estimator, "prune_set_"),
                hasattr(estimator, "optimization_stable_"),
            )
        )
        return estimator.score(X, y) - 0.01 * estimator.complexity_

    X, y = _regression_tree_data()
    SPTreeRegressorCV(
        sp_alpha_list=(0,),
        reg_param_list=(0,),
        cv=2,
        scoring=complexity_penalized_score,
        random_state=0,
    ).fit(X, y)

    assert observed_metadata == [(True, True, True), (True, True, True)]


def test_cv_uses_custom_solver_for_folds_and_final_refit():
    X, y = _regression_tree_data()
    sample_counts = []

    def zero_solver(**kwargs):
        sample_counts.append(len(kwargs["y"]))
        return np.zeros(kwargs["X"].shape[1])

    model = SPTreeRegressorCV(
        sp_alpha_list=(1,),
        reg_param_list=(0,),
        cv=2,
        random_state=0,
    )
    model.hiCAP = zero_solver
    model.fit(X, y)

    assert sorted(sample_counts) == [4, 4, 8]
    assert not model.cv_optimization_certified_
    assert not model.optimization_certified_


def test_cv_retains_diagnostics_for_unselected_unstable_candidates():
    class PreferNoSparsePruning:
        _sign = 1

        def __call__(self, estimator, X, y, **kwargs):
            return -float(estimator.sp_alpha)

    X, y = _regression_tree_data()
    with pytest.warns(ConvergenceWarning, match="CV candidate"):
        model = SPTreeRegressorCV(
            solver="apa_apg2",
            sp_alpha_list=(0, 1),
            reg_param_list=(0,),
            cv=2,
            scoring=PreferNoSparsePruning(),
            random_state=0,
            max_iter=1,
        ).fit(X, y)

    assert model.sp_alpha_ == 0
    assert len(model.cv_optimization_results_) == 2
    assert all(
        len(candidate_results) == 2
        for candidate_results in model.cv_optimization_results_
    )
    assert not model.cv_optimization_certified_
    assert not model.cv_optimization_stable_
    assert model.optimization_certified_
    assert model.optimization_stable_


@pytest.mark.parametrize("task", ["regression", "classification"])
def test_gradient_boosting_is_rejected_clearly(task):
    if task == "regression":
        X, y = _regression_tree_data()
        base_estimator = GradientBoostingRegressor(
            n_estimators=2, max_depth=1, random_state=0
        )
        wrapper_cls = SPTreeRegressor
    else:
        X, y = _classification_data()
        base_estimator = GradientBoostingClassifier(
            n_estimators=2, max_depth=1, random_state=0
        )
        wrapper_cls = SPTreeClassifier

    with pytest.raises(
        (ValueError, NotImplementedError),
        match=r"(?i)(gradient.?boost|not supported|unsupported)",
    ):
        wrapper_cls(
            estimator_=base_estimator,
            sp_alpha=1,
            reg_param=0,
            random_state=0,
        ).fit(X, y)


def test_complexity_counts_only_reachable_splits_after_pruning():
    X, y = _regression_tree_data()
    unpruned = DecisionTreeRegressor(max_depth=2, random_state=0).fit(X, y)
    unpruned_complexity = compute_tree_complexity(unpruned.tree_)
    model = SPTreeRegressor(
        estimator_=DecisionTreeRegressor(max_depth=2, random_state=0),
        sp_alpha=1,
        reg_param=0,
        random_state=0,
    )

    def all_splits_zero(**kwargs):
        return np.zeros(kwargs["X"].shape[1])

    model.hiCAP = all_splits_zero
    model.fit(X, y)

    reachable_complexity = compute_tree_complexity(model.estimator_.tree_)
    assert unpruned_complexity == 3
    assert reachable_complexity == 0
    assert model.complexity_ == reachable_complexity
    assert model.complexity_ < unpruned_complexity
    assert model.estimator_.tree_.node_count == 1
    assert model.estimator_.get_n_leaves() == 1
    assert model.estimator_.get_depth() == 0
    assert_allclose(model.estimator_.feature_importances_, np.zeros(X.shape[1]))


def test_default_estimator_is_lazy_and_fitted_state_is_correct():
    X, y = _regression_tree_data()
    model = SPTreeRegressor(sp_alpha=0, reg_param=0, random_state=0)

    assert model.estimator_ is None
    assert model.get_params(deep=False)["estimator_"] is None
    with pytest.raises(NotFittedError):
        check_is_fitted(model)
    with pytest.raises(NotFittedError):
        model.predict(X)

    model.fit(X, y)

    check_is_fitted(model)
    assert isinstance(model.estimator_, DecisionTreeRegressor)
    assert model.get_params(deep=False)["estimator_"] is None
    assert not hasattr(model, "predict_proba")
    assert model.n_iter_ == 1
    assert model.solver_ == "topology"
    assert model.optimization_results_[0]["converged"]
    assert model.coef_ is None
    assert model.beta_stars_ == []
    assert model.prune_set_ == "full"


def test_nested_base_estimator_parameters_round_trip():
    model = SPTreeRegressor(
        estimator_=DecisionTreeRegressor(max_depth=3, random_state=0),
        sp_alpha=0,
        reg_param=0,
    )

    assert model.get_params(deep=True)["estimator___max_depth"] == 3
    model.set_params(estimator___max_depth=1)

    assert model.get_params(deep=True)["estimator___max_depth"] == 1
    assert clone(model).get_params(deep=True)["estimator___max_depth"] == 1


def test_clone_preserves_custom_solver_and_shrinkage_scheme():
    def custom_solver(**kwargs):
        return np.zeros(kwargs["X"].shape[1])

    model = SPTreeRegressor(sp_alpha=1, reg_param=0)
    model.hiCAP = custom_solver
    model.shrinkage_scheme_ = "constant"
    cloned = clone(model)

    assert cloned.hiCAP is custom_solver
    assert cloned.shrinkage_scheme_ == "constant"


def test_prefit_wrapper_rejects_nested_parameter_changes():
    X, y = _regression_tree_data()
    base_estimator = DecisionTreeRegressor(
        max_depth=2, random_state=0
    ).fit(X, y)
    model = SPTreeRegressor(
        estimator_=base_estimator,
        sp_alpha=0,
        reg_param=0,
        prefit=True,
    )

    with pytest.raises(ValueError, match="inconsistent"):
        model.set_params(estimator___max_depth=1)
    assert model.get_params(deep=True)["estimator___max_depth"] == 2


def test_set_params_invalidates_fitted_sparse_pruning_state():
    X, y = _regression_tree_data()
    model = SPTreeRegressor(sp_alpha=0, reg_param=0).fit(X, y)

    assert model.set_params() is model
    check_is_fitted(model)
    model.set_params(sp_alpha=0.1)

    with pytest.raises(NotFittedError):
        check_is_fitted(model)


def test_invalid_set_params_is_transactional_for_fitted_model():
    X, y = _regression_tree_data()
    model = SPTreeRegressor(
        estimator_=DecisionTreeRegressor(max_depth=2, random_state=0),
        sp_alpha=0,
        reg_param=0,
    ).fit(X, y)
    fitted_estimator = model.estimator_
    predictions = model.predict(X)

    with pytest.raises(ValueError, match="unknown"):
        model.set_params(unknown=1)
    check_is_fitted(model)
    assert model.estimator_ is fitted_estimator
    assert_allclose(model.predict(X), predictions)

    with pytest.raises(ValueError, match="not_a_tree_parameter"):
        model.set_params(estimator___not_a_tree_parameter=1)
    check_is_fitted(model)
    assert model.estimator_ is fitted_estimator
    assert_allclose(model.predict(X), predictions)


def test_failed_refit_invalidates_wrapper_fitted_state():
    X, y = _classification_data()
    model = SPTreeClassifier(sp_alpha=0, reg_param=0).fit(X, y)
    continuous_y = np.linspace(0.1, 0.9, len(y))

    with pytest.raises(ValueError):
        model.fit(X, continuous_y)
    with pytest.raises(NotFittedError):
        check_is_fitted(model)


def test_prefit_wrapper_is_not_fitted_until_pruning_runs():
    X, y = _regression_tree_data()
    base_estimator = DecisionTreeRegressor(max_depth=2, random_state=0).fit(X, y)
    original_children = base_estimator.tree_.children_left.copy()
    model = SPTreeRegressor(
        estimator_=base_estimator,
        sp_alpha=0,
        reg_param=0,
        prefit=True,
    )

    with pytest.raises(NotFittedError):
        check_is_fitted(model)
    model.fit(X, y)

    check_is_fitted(model)
    assert model.estimator_ is not base_estimator
    assert_array_equal(base_estimator.tree_.children_left, original_children)


def test_prefit_alpha_zero_auto_matches_resolved_proximal_coefficients():
    X, y = _regression_tree_data()
    base = DecisionTreeRegressor(max_depth=2, random_state=0).fit(X, y)
    original_children = base.tree_.children_left.copy()
    models = [SPTreeRegressor(
        estimator_=base, sp_alpha=0, reg_param=0, prefit=True, solver=solver,
    ).fit(X, y) for solver in ("auto", "proximal")]

    for model in models:
        assert model.solver_ == "proximal"
        assert model.coef_ is not None
        assert model.intercept_ is not None
        assert_allclose(model.predict(X), base.predict(X))
        assert_array_equal(model.estimator_.tree_.children_left, original_children)
    assert_allclose(models[0].coef_, models[1].coef_)
    assert_allclose(models[0].intercept_, models[1].intercept_)
    assert_array_equal(base.tree_.children_left, original_children)


def test_prefit_wrapper_rejects_multioutput_estimator():
    X, y = _regression_tree_data()
    multioutput_estimator = DecisionTreeRegressor(random_state=0).fit(
        X, np.column_stack([y, y + 1])
    )
    model = SPTreeRegressor(
        estimator_=multioutput_estimator,
        sp_alpha=0,
        reg_param=0,
        prefit=True,
    )

    with pytest.raises(ValueError, match="single-output"):
        model.fit(X, y)


def test_prefit_wrapper_rejects_clone_based_model_selection():
    X, y = _regression_tree_data()
    base_estimator = DecisionTreeRegressor(max_depth=2, random_state=0).fit(X, y)
    model = SPTreeRegressor(
        estimator_=base_estimator,
        sp_alpha=0,
        reg_param=0,
        prefit=True,
    )

    with pytest.raises(TypeError, match="cannot be cloned safely"):
        clone(model)


def test_wrapper_exposes_solver_diagnostics():
    X, y = _regression_tree_data()
    with pytest.warns(ConvergenceWarning, match="max_iter"):
        model = SPTreeRegressor(
            solver="apa_apg2",
            estimator_=DecisionTreeRegressor(max_depth=2, random_state=0),
            sp_alpha=0.1,
            reg_param=0,
            max_iter=7,
            random_state=0,
        ).fit(X, y)

    assert model.n_iter_ == 7
    assert len(model.optimization_results_) == 1
    assert model.optimization_results_[0]["n_iter"] == 7
    assert not model.optimization_results_[0]["converged"]
    assert not model.optimization_certified_
    assert not model.optimization_stable_


def test_small_apa_step_does_not_claim_overlapping_support_convergence():
    X, y = _regression_tree_data()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = SPTreeRegressor(
            solver="apa_apg2",
            estimator_=DecisionTreeRegressor(max_depth=2, random_state=0),
            sp_alpha=0.1,
            reg_param=0,
            max_iter=5,
            tol=1e3,
            random_state=0,
        ).fit(X, y)

    assert not any(
        issubclass(warning.category, ConvergenceWarning)
        for warning in caught
    )
    assert model.optimization_results_[0]["relative_step_norm"] <= model.tol
    assert not model.optimization_results_[0]["converged"]
    assert not model.optimization_certified_
    assert model.optimization_stable_ is None


def test_custom_solver_without_diagnostics_is_not_claimed_as_certified():
    X, y = _regression_tree_data()
    model = SPTreeRegressor(
        estimator_=DecisionTreeRegressor(max_depth=2, random_state=0),
        sp_alpha=0.1,
        reg_param=0,
        random_state=0,
    )
    model.hiCAP = lambda **kwargs: np.zeros(kwargs["X"].shape[1])
    model.fit(X, y)

    assert model.optimization_results_ == [None]
    assert not model.optimization_certified_
    assert model.optimization_stable_ is None


def test_nondefault_decimals_warns_and_points_to_support_tol():
    X, y = _regression_tree_data()
    with pytest.warns(FutureWarning, match="support_tol"):
        SPTreeRegressor(
            sp_alpha=0,
            reg_param=0,
        ).fit(X, y, decimals=3)


def test_legacy_fixed_signature_custom_solver_remains_supported():
    X, y = _regression_tree_data()

    def legacy_solver(
        X,
        y,
        groups,
        lam,
        beta_init=None,
        gamma1=1.0,
        a=1.0,
        max_iter=500,
        tol=1e-6,
        ord=2,
        verbose=False,
    ):
        return np.zeros(X.shape[1])

    model = SPTreeRegressor(
        estimator_=DecisionTreeRegressor(max_depth=2, random_state=0),
        sp_alpha=0.1,
        reg_param=0,
    )
    model.hiCAP = legacy_solver
    model.fit(X, y)

    assert model.complexity_ == 0


@pytest.mark.parametrize(
    ("parameter", "value"),
    [
        ("sp_alpha", -1),
        ("gamma1", 0),
        ("a", np.inf),
        ("max_iter", 0),
        ("tol", 0),
        ("support_tol", -1),
        ("prune_set", "invalid"),
        ("ord", 1),
    ],
)
def test_invalid_sparse_pruning_hyperparameters_are_rejected(parameter, value):
    X, y = _regression_tree_data()
    model = SPTreeRegressor(sp_alpha=0, reg_param=0)
    model.set_params(**{parameter: value})

    with pytest.raises(ValueError, match=parameter):
        model.fit(X, y)


def test_classifier_rejects_continuous_targets_before_encoding():
    X = np.arange(8, dtype=float).reshape(-1, 1)
    y = np.linspace(0.1, 0.8, len(X))

    with pytest.raises(ValueError, match="continuous"):
        SPTreeClassifier(sp_alpha=0, reg_param=0).fit(X, y)


@pytest.mark.parametrize(
    ("grid_name", "grid"),
    [
        ("sp_alpha_list", ()),
        ("sp_alpha_list", (-1,)),
        ("reg_param_list", None),
        ("reg_param_list", (None,)),
        ("reg_param_list", (-1,)),
    ],
)
def test_cv_rejects_invalid_parameter_grids(grid_name, grid):
    X, y = _regression_tree_data()
    kwargs = {
        "sp_alpha_list": (0,),
        "reg_param_list": (0,),
        "cv": 2,
        grid_name: grid,
    }

    with pytest.raises(ValueError, match=grid_name):
        SPTreeRegressorCV(**kwargs).fit(X, y)


@pytest.mark.parametrize(
    ("parameter", "value"),
    [
        ("selection_rule", "smallest"),
        ("reg_param_mode", "relative"),
    ],
)
def test_cv_rejects_invalid_selection_policy(parameter, value):
    X, y = _regression_tree_data()
    model = SPTreeRegressorCV(
        sp_alpha_list=(0,),
        reg_param_list=(0,),
        cv=2,
        **{parameter: value},
    )

    with pytest.raises(ValueError, match=parameter):
        model.fit(X, y)


def test_cv_rejects_zero_total_weight_fold_clearly():
    X, y = _regression_tree_data()
    sample_weight = np.zeros(len(y))
    sample_weight[0] = 1

    with pytest.raises(ValueError, match="CV .* fold"):
        SPTreeRegressorCV(
            sp_alpha_list=(0,),
            reg_param_list=(0,),
            cv=2,
            random_state=0,
        ).fit(X, y, sample_weight=sample_weight)


def test_classifier_cv_rejects_too_few_minority_samples():
    X = np.arange(40, dtype=float).reshape(20, 2)
    y = np.zeros(len(X), dtype=int)
    y[0] = 1

    with pytest.raises(ValueError, match="at least cv"):
        SPTreeClassifierCV(
            sp_alpha_list=(0,),
            reg_param_list=(0,),
            cv=3,
            scoring="roc_auc",
            random_state=0,
        ).fit(X, y)


def test_cv_rejects_non_tree_estimator_before_candidate_search():
    X, y = _regression_tree_data()

    with pytest.raises(ValueError, match="decision tree or forest"):
        SPTreeRegressorCV(
            estimator_=LinearRegression(),
            sp_alpha_list=(0, 1),
            reg_param_list=(0,),
            cv=2,
            random_state=0,
        ).fit(X, y)


def test_cv_accepts_dataframe_and_sparse_inputs():
    X, y = _regression_tree_data()
    frame = pd.DataFrame(X, columns=["signal"])
    regression = SPTreeRegressorCV(
        sp_alpha_list=(0,),
        reg_param_list=(0,),
        cv=2,
        random_state=0,
    ).fit(frame, y)

    assert list(regression.feature_names_) == ["signal"]
    assert_array_equal(regression.feature_names_in_, ["signal"])
    assert regression.predict(frame).shape == y.shape

    X_class, y_class = _classification_data()
    classification = SPTreeClassifierCV(
        sp_alpha_list=(0,),
        reg_param_list=(0,),
        cv=2,
        random_state=0,
    ).fit(sparse.csr_matrix(X_class), y_class)

    assert classification.predict(sparse.csr_matrix(X_class)).shape == y_class.shape


def test_dataframe_column_order_is_enforced_at_prediction():
    X = pd.DataFrame(
        {
            "first": np.arange(20, dtype=float),
            "second": np.arange(20, dtype=float)[::-1],
        }
    )
    y = X["first"] - X["second"]
    model = SPTreeRegressor(sp_alpha=0, reg_param=0).fit(X, y)

    assert_array_equal(model.feature_names_in_, ["first", "second"])
    with pytest.raises(ValueError, match="names and order"):
        model.predict(X[["second", "first"]])


def test_prefit_dataframe_column_order_is_enforced_before_pruning():
    X = pd.DataFrame(
        {
            "first": np.arange(20, dtype=float),
            "second": np.arange(20, dtype=float)[::-1],
        }
    )
    y = X["first"] - X["second"]
    base_estimator = DecisionTreeRegressor(random_state=0).fit(X, y)
    model = SPTreeRegressor(
        estimator_=base_estimator,
        sp_alpha=0,
        reg_param=0,
        prefit=True,
    )

    with pytest.raises(ValueError, match="prefit estimator"):
        model.fit(X[["second", "first"]], y)


def test_prefit_dataframe_prediction_preserves_names_for_inner_tree():
    X = pd.DataFrame(
        {
            "first": np.arange(20, dtype=float),
            "second": np.arange(20, dtype=float)[::-1],
        }
    )
    y = X["first"] - X["second"]
    base_estimator = DecisionTreeRegressor(random_state=0).fit(X, y)
    model = SPTreeRegressor(
        estimator_=base_estimator,
        sp_alpha=0,
        reg_param=0,
        prefit=True,
    ).fit(X, y)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        predictions = model.predict(X)

    assert_allclose(predictions, base_estimator.predict(X))
    assert not any("feature names" in str(w.message) for w in caught)


def test_prediction_feature_count_uses_sklearn_error_contract():
    X, y = _regression_tree_data()
    model = SPTreeRegressor(sp_alpha=0, reg_param=0).fit(X, y)

    with pytest.raises(
        ValueError,
        match=(
            r"X has 2 features, but SPTreeRegressor is expecting "
            r"1 features as input"
        ),
    ):
        model.predict(np.column_stack([X, X]))


def test_bootstrap_forest_uses_tree_specific_oob_pruning_sets():
    X = np.arange(80, dtype=float).reshape(40, 2)
    y = np.sin(X[:, 0] / 5)
    model = SPTreeRegressor(
        estimator_=RandomForestRegressor(
            n_estimators=3,
            max_depth=3,
            bootstrap=True,
            random_state=0,
        ),
        sp_alpha=0.1,
        reg_param=0,
        prune_set="oob",
        max_iter=2000,
        random_state=0,
    ).fit(X, y)

    assert len(model.beta_stars_) == 3
    assert len(model.optimization_results_) == 3
    assert np.all(np.isfinite(model.predict(X)))
    for estimator in model.estimator_.estimators_:
        tree = estimator.tree_
        assert tree.node_count == len(tree.children_left)
        assert np.all(
            (tree.children_left == -1)
            | (tree.children_left < tree.node_count)
        )


def test_nonbootstrap_forest_rejects_oob_pruning():
    X = np.arange(40, dtype=float).reshape(20, 2)
    y = np.sin(X[:, 0])
    model = SPTreeRegressor(
        estimator_=RandomForestRegressor(
            n_estimators=2,
            max_depth=2,
            bootstrap=False,
            random_state=0,
        ),
        sp_alpha=0.1,
        reg_param=0,
        prune_set="oob",
        max_iter=5,
        random_state=0,
    )

    with pytest.raises(ValueError, match="bootstrap=True"):
        model.fit(X, y)


def test_auto_prune_set_uses_full_data_for_nonbootstrap_forest():
    X = np.arange(40, dtype=float).reshape(20, 2)
    y = np.sin(X[:, 0])
    model = SPTreeRegressor(
        estimator_=RandomForestRegressor(
            n_estimators=2,
            max_depth=2,
            bootstrap=False,
            random_state=0,
        ),
        sp_alpha=0,
        reg_param=0,
    ).fit(X, y)

    assert model.prune_set_ == "full"
    assert np.all(np.isfinite(model.predict(X)))


def test_modified_forest_invalidates_stale_oob_outputs():
    X = np.arange(80, dtype=float).reshape(40, 2)
    y = np.sin(X[:, 0])
    model = SHSTreeRegressor(
        estimator_=RandomForestRegressor(
            n_estimators=10,
            max_depth=3,
            bootstrap=True,
            oob_score=True,
            random_state=0,
        ),
        sp_alpha=0,
        reg_param=10,
        random_state=0,
    ).fit(X, y)

    assert set(model.oob_attributes_invalidated_) == {
        "oob_prediction_",
        "oob_score_",
    }
    assert model.prune_set_ == "oob"
    assert not hasattr(model.estimator_, "oob_prediction_")
    assert not hasattr(model.estimator_, "oob_score_")


def test_unmodified_forest_retains_valid_oob_outputs():
    X = np.arange(80, dtype=float).reshape(40, 2)
    y = np.sin(X[:, 0])
    model = SPTreeRegressor(
        estimator_=RandomForestRegressor(
            n_estimators=10,
            max_depth=3,
            bootstrap=True,
            oob_score=True,
            random_state=0,
        ),
        sp_alpha=0,
        reg_param=0,
        random_state=0,
    ).fit(X, y)

    assert model.oob_attributes_invalidated_ == ()
    assert hasattr(model.estimator_, "oob_prediction_")
    assert hasattr(model.estimator_, "oob_score_")


def test_prefit_forest_requires_full_pruning_set():
    X = np.arange(80, dtype=float).reshape(40, 2)
    y = np.sin(X[:, 0])
    forest = RandomForestRegressor(
        n_estimators=2,
        max_depth=2,
        bootstrap=True,
        random_state=0,
    ).fit(X, y)
    model = SPTreeRegressor(
        estimator_=forest,
        sp_alpha=0.1,
        reg_param=0,
        prune_set="oob",
        max_iter=5,
        prefit=True,
    )

    with pytest.raises(ValueError, match="prefit forest"):
        model.fit(X, y)


@pytest.mark.parametrize("solver", ["auto", "proximal", "hicap"])
def test_prefit_bootstrap_forest_skips_unverifiable_oob_solve_at_zero(solver):
    X = np.arange(80, dtype=float).reshape(40, 2)
    y = np.sin(X[:, 0])
    forest = RandomForestRegressor(
        n_estimators=2,
        max_depth=2,
        bootstrap=True,
        random_state=0,
    ).fit(X, y)
    expected = forest.predict(X)
    model = SPTreeRegressor(
        estimator_=forest,
        sp_alpha=0,
        reg_param=0,
        prefit=True,
        solver=solver,
    ).fit(X, y)

    assert_allclose(model.predict(X), expected)
    assert model.optimization_results_ == []
    assert model.beta_stars_ == []


def test_zero_weight_oob_subset_skips_only_affected_trees():
    X = np.arange(80, dtype=float).reshape(40, 2)
    y = np.sin(X[:, 0])
    sample_weight = np.zeros(len(y))
    sample_weight[:2] = 1
    model = SPTreeRegressor(
        estimator_=RandomForestRegressor(
            n_estimators=20,
            max_depth=3,
            bootstrap=True,
            random_state=0,
        ),
        sp_alpha=0.1,
        reg_param=0,
        prune_set="oob",
        max_iter=5,
        random_state=0,
    ).fit(X, y, sample_weight=sample_weight)

    statuses = [
        result.get("status")
        for result in model.optimization_results_
        if result is not None
    ]
    assert "zero_weight_pruning_subset" in statuses
    assert model.predict(X).shape == y.shape


def test_empty_oob_subset_skips_only_affected_trees():
    X = np.arange(3, dtype=float).reshape(-1, 1)
    y = np.array([0.0, 1.0, 2.0])
    model = SPTreeRegressor(
        estimator_=RandomForestRegressor(
            n_estimators=20,
            max_depth=2,
            bootstrap=True,
            random_state=0,
        ),
        sp_alpha=0.1,
        reg_param=0,
        max_iter=5,
        random_state=0,
    ).fit(X, y)

    statuses = [
        result.get("status")
        for result in model.optimization_results_
        if result is not None
    ]
    assert "empty_pruning_subset" in statuses
    assert len(model.optimization_results_) == 20
    assert model.predict(X).shape == y.shape


def test_all_skipped_oob_subsets_are_reported_as_uncertified():
    X = np.array([[0.0]])
    y = np.array([1.0])
    model = SPTreeRegressor(
        estimator_=RandomForestRegressor(
            n_estimators=3,
            bootstrap=True,
            random_state=0,
        ),
        sp_alpha=0.1,
        reg_param=0,
        random_state=0,
    ).fit(X, y)

    assert {
        result["status"] for result in model.optimization_results_
    } == {"empty_pruning_subset"}
    assert not model.optimization_certified_
    assert model.optimization_stable_ is None


@pytest.mark.parametrize("task", ["regression", "classification"])
def test_degenerate_tree_diagnostics_remain_aligned(task):
    if task == "regression":
        X = np.arange(8, dtype=float).reshape(-1, 1)
        y = np.ones(len(X))
        model = SPTreeRegressor(
            estimator_=DecisionTreeRegressor(random_state=0),
            sp_alpha=1,
            reg_param=0,
        )
    else:
        X, y = _classification_data()
        model = SPTreeClassifier(
            estimator_=DecisionTreeClassifier(
                min_samples_split=100,
                random_state=0,
            ),
            sp_alpha=1,
            reg_param=0,
        )

    model.fit(X, y)

    assert len(model.beta_stars_) == 0
    assert len(model.support_thresholds_) == 1
    assert len(model.optimization_results_) == 1
    assert model.optimization_results_[0]["status"] == "complete"


def test_single_tree_rejects_bootstrap_only_pruning_sets():
    X, y = _regression_tree_data()

    with pytest.raises(ValueError, match="bootstrap forest"):
        SPTreeRegressor(
            estimator_=DecisionTreeRegressor(max_depth=2, random_state=0),
            sp_alpha=0.1,
            reg_param=0,
            prune_set="oob",
            max_iter=5,
        ).fit(X, y)


def test_apa_classifier_class_weight_matches_explicit_sample_weight():
    X = np.arange(40, dtype=float).reshape(20, 2)
    y = np.array([0] * 14 + [1] * 6)
    class_weight = {0: 1, 1: 5}
    common = {
        "solver": "apa_apg2",
        "sp_alpha": 10,
        "reg_param": 0,
        "max_iter": 100,
        "random_state": 0,
    }
    weighted_class = SPTreeClassifier(
        estimator_=DecisionTreeClassifier(
            max_depth=3,
            class_weight=class_weight,
            random_state=0,
        ),
        **common,
    ).fit(X, y)
    weighted_samples = SPTreeClassifier(
        estimator_=DecisionTreeClassifier(max_depth=3, random_state=0),
        **common,
    ).fit(X, y, sample_weight=np.where(y == 1, 5.0, 1.0))

    assert_allclose(weighted_class.beta_stars_[0], weighted_samples.beta_stars_[0])
    assert_array_equal(weighted_class.predict(X), weighted_samples.predict(X))
    assert weighted_class.estimator_.class_weight is None


def test_balanced_subsample_is_allowed_when_sparse_pruning_is_disabled():
    X, y = _classification_data()
    model = SPTreeClassifier(
        estimator_=RandomForestClassifier(
            n_estimators=3,
            max_depth=2,
            class_weight="balanced_subsample",
            random_state=0,
        ),
        sp_alpha=0,
        reg_param=0,
    ).fit(X, y)

    assert model.estimator_.class_weight == "balanced_subsample"
    assert model.predict(X).shape == y.shape


def test_classifier_cv_balanced_subsample_requires_only_zero_alpha_grid():
    X, y = _classification_data()
    base_estimator = RandomForestClassifier(
        n_estimators=3,
        max_depth=2,
        class_weight="balanced_subsample",
        random_state=0,
    )
    model = SPTreeClassifierCV(
        estimator_=base_estimator,
        sp_alpha_list=(0,),
        reg_param_list=(0,),
        cv=2,
        random_state=0,
    ).fit(X, y)

    assert model.sp_alpha_ == 0
    with pytest.raises(ValueError, match="balanced_subsample"):
        SPTreeClassifierCV(
            estimator_=base_estimator,
            sp_alpha_list=(0, 0.1),
            reg_param_list=(0,),
            cv=2,
            random_state=0,
        ).fit(X, y)


def test_sparse_pruning_is_invariant_to_uniform_weight_rescaling():
    X, y = _regression_tree_data()
    common = {
        "estimator_": DecisionTreeRegressor(
            max_depth=2, random_state=0
        ),
        "sp_alpha": 0.1,
        "reg_param": 0,
        "max_iter": 100,
        "random_state": 0,
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        unit_weights = SPTreeRegressor(**common).fit(
            X, y, sample_weight=np.ones(len(y))
        )
        scaled_weights = SPTreeRegressor(**common).fit(
            X, y, sample_weight=np.full(len(y), 100.0)
        )

    assert_allclose(
        unit_weights.pruning_path_.activation_lambdas,
        scaled_weights.pruning_path_.activation_lambdas,
    )
    assert_array_equal(
        unit_weights.estimator_.tree_.children_left,
        scaled_weights.estimator_.tree_.children_left,
    )
    assert unit_weights.complexity_ == scaled_weights.complexity_


def test_prefit_classifier_class_weight_is_allowed_without_sparse_pruning():
    X, y = _classification_data()
    base_estimator = DecisionTreeClassifier(
        max_depth=2,
        class_weight={0: 1.0, 1: 5.0},
        random_state=0,
    ).fit(X, y)
    expected = base_estimator.predict_proba(X)
    model = SPTreeClassifier(
        estimator_=base_estimator,
        sp_alpha=0,
        reg_param=0,
        prefit=True,
    ).fit(X, y)

    assert_allclose(model.predict_proba(X), expected)


@pytest.mark.parametrize(
    ("wrapper", "base_estimator", "data_factory"),
    [
        (
            SPTreeRegressor,
            DecisionTreeClassifier(random_state=0),
            _regression_tree_data,
        ),
        (
            SPTreeClassifier,
            DecisionTreeRegressor(random_state=0),
            _classification_data,
        ),
    ],
)
def test_wrapper_rejects_wrong_base_estimator_type(
    wrapper, base_estimator, data_factory
):
    X, y = data_factory()

    with pytest.raises(ValueError, match="requires a"):
        wrapper(
            estimator_=base_estimator,
            sp_alpha=0,
            reg_param=0,
        ).fit(X, y)
