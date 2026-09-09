"""Independent fold-by-fold checks of structural pruning-path selection."""
from copy import deepcopy

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.metrics import log_loss, mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.base import clone
from sklearn.utils.class_weight import compute_sample_weight

from imodels import (
    HSTreeRegressor, SHSTreeRegressor, SHSTreeRegressorCV,
    SHSTreeClassifier, SHSTreeClassifierCV, SPTreeClassifierCV,
    SPTreeRegressor, SPTreeRegressorCV,
)
from imodels.tree._hs_gcv import select_hs_reg_param
from imodels.tree.sparse_pruning._cv import (
    evaluate_structural_cv, _local_penalties, _score_fold_states,
)
from imodels.tree.sparse_pruning.fitted_tree import fitted_tree_linf_exact_topology_path
from imodels.tree.sparse_pruning.sparse_hierarchical_shrinkage import (
    _compact_tree, _finalize_cv_selection,
)


def _data():
    rng = np.random.default_rng(14)
    X = rng.normal(size=(53, 3))
    y = X[:, 0] + 2 * (X[:, 1] > 0) + rng.normal(size=len(X))
    return X, y


@pytest.mark.parametrize("solver", ["coefficient_path", "hicap"])
def test_explicit_grid_compiles_and_shares_one_coefficient_path_per_fold(monkeypatch, solver):
    from imodels.tree.sparse_pruning import _solver
    from imodels.tree.sparse_pruning.optimization import hicap

    X, y = _data()
    module, name = (
        (_solver, "fitted_tree_linf_exact_coefficient_path")
        if solver == "coefficient_path" else (hicap, "hicap_regression_path")
    )
    original = getattr(module, name)
    paths = []
    scored_paths = []

    def counted_path(*args, **kwargs):
        path = original(*args, **kwargs)
        paths.append(path)
        return path

    def scorer(model, X, y):
        scored_paths.append(model.coefficient_path_)
        return -mean_squared_error(y, model.predict(X))

    monkeypatch.setattr(module, name, counted_path)
    model = SHSTreeRegressorCV(
        max_leaf_nodes=5, cv=3, random_state=9, solver=solver,
        sp_alpha_list=[0, .1, 10], reg_param_list=[0, 3], scoring=scorer,
    ).fit(X, y)
    assert len(paths) == 4  # Three training folds and the final full-data fit.
    assert len(scored_paths) == 3 * 3 * 2
    assert {id(path) for path in scored_paths} == {id(path) for path in paths[:3]}
    assert model.coefficient_path_ is paths[-1]


def _scorer(estimator, X, y, sample_weight=None):
    # A real wrapper and consistent compact sklearn metadata are part of the
    # estimator-scoring contract, not just a prediction vector.
    assert isinstance(estimator, SHSTreeRegressor)
    assert estimator.coef_ is None
    assert estimator.beta_stars_ == []
    assert estimator.coefficient_path_ is None
    assert estimator.complexity_ == estimator.estimator_.get_n_leaves() - 1
    assert estimator.estimator_.tree_.node_count == 2 * estimator.complexity_ + 1
    return -mean_squared_error(y, estimator.predict(X), sample_weight=sample_weight)


def _model(**kwargs):
    return SHSTreeRegressorCV(
        estimator_=DecisionTreeRegressor(max_leaf_nodes=7),
        max_leaf_nodes=7, ord=np.inf, random_state=9, cv=3, **kwargs,
    )


def _brute_tree(base, path, alpha):
    tree = deepcopy(base)
    if alpha > 0:
        keep = set(path.tree_nodes_at(alpha))
        for node in path.node_ids:
            if node not in keep:
                tree.tree_.children_left[node] = -1
                tree.tree_.children_right[node] = -1
                tree.tree_.feature[node] = -2
                tree.tree_.threshold[node] = -2
    _compact_tree(tree.tree_)
    return tree


@pytest.mark.parametrize("reg_param_mode", ["normalized", "raw"])
@pytest.mark.parametrize("weighted", [False, True])
def test_structural_cv_matches_independent_every_candidate_every_fold(reg_param_mode, weighted):
    X, y = _data()
    weights = np.linspace(.2, 2., len(y)) if weighted else None
    model = _model(reg_param_mode=reg_param_mode)
    regs = [0., 3., 30.]
    params = evaluate_structural_cv(
        model, X=X, y=y, sample_weight=weights, reg_param_list=regs,
        scorer=_scorer, n_splits=3,
    )
    scores, strengths, complexities = [], [], []
    paths = []
    for train, test in KFold(n_splits=3, shuffle=True, random_state=9).split(X):
        weight_in = None if weights is None else weights[train]
        weight_out = None if weights is None else weights[test]
        base = DecisionTreeRegressor(max_leaf_nodes=7, random_state=9).fit(
            X[train], y[train], sample_weight=weight_in
        )
        path = fitted_tree_linf_exact_topology_path(base)
        paths.append(path)
        fraction = len(train) / len(y) if weights is None else weight_in.sum() / weights.sum()
        fold_scores, fold_strengths, fold_complexities = [], [], []
        for alpha, reg in params:
            tree = _brute_tree(base, path, alpha)
            complexity = tree.get_n_leaves() - 1
            rho = reg * fraction if reg_param_mode == "normalized" else reg
            hs = HSTreeRegressor(estimator_=tree, reg_param=rho)
            fold_scores.append(-mean_squared_error(y[test], hs.predict(X[test]), sample_weight=weight_out))
            fold_strengths.append(rho)
            fold_complexities.append(complexity)
        scores.append(fold_scores)
        strengths.append(fold_strengths)
        complexities.append(fold_complexities)
    assert_allclose(model.cv_scores_, np.asarray(scores).T, rtol=1e-12, atol=1e-12)
    assert_allclose(model.cv_reg_params_, np.asarray(strengths).T)
    assert_array_equal(model.cv_complexities_, np.asarray(complexities).T)
    assert_array_equal(model.cv_sp_alphas_, np.unique(np.concatenate([_local_penalties(p) for p in paths])))
    assert_array_equal(model.cv_n_pruning_states_, [len(_local_penalties(p)) for p in paths])
    assert model.cv_params_ == [{"sp_alpha": a, "reg_param": r} for a, r in params]
    assert len(model.cv_optimization_results_) == len(params)
    assert all(result[0]["converged"] for candidate in model.cv_optimization_results_ for result in candidate)


@pytest.mark.parametrize("mode", ["gcv", ["gcv"]])
def test_gcv_is_reselected_on_each_training_fold_state(mode):
    X, y = _data()
    model = _model()
    params = evaluate_structural_cv(
        model, X=X, y=y, sample_weight=None, reg_param_list=mode,
        scorer=_scorer, n_splits=3,
    )
    expected = []
    for train, _ in KFold(n_splits=3, shuffle=True, random_state=9).split(X):
        base = DecisionTreeRegressor(max_leaf_nodes=7, random_state=9).fit(X[train], y[train])
        path = fitted_tree_linf_exact_topology_path(base)
        expected.append([
            select_hs_reg_param(_brute_tree(base, path, alpha), y=y[train])[0]
            for alpha, _ in params
        ])
    assert all(reg == "gcv" for _, reg in params)
    assert_allclose(model.cv_reg_params_, np.asarray(expected).T, rtol=1e-10)
    _finalize_cv_selection(model, params)
    assert model.reg_param_ == "gcv"


def test_each_fold_fits_once_and_scores_only_its_local_states(monkeypatch):
    X, y = _data()
    model = _model()
    fits = []
    original_fit = DecisionTreeRegressor.fit

    def counting_fit(self, X, y, *args, **kwargs):
        fits.append(len(y))
        return original_fit(self, X, y, *args, **kwargs)

    monkeypatch.setattr(DecisionTreeRegressor, "fit", counting_fit)
    scores = []

    def counting_score(estimator, X, y):
        scores.append(estimator.complexity_)
        return _scorer(estimator, X, y)

    evaluate_structural_cv(
        model, X=X, y=y, sample_weight=None, reg_param_list=[0, 4],
        scorer=counting_score, n_splits=3,
    )
    assert len(fits) == 3
    assert len(scores) == 2 * sum(model.cv_n_pruning_states_)
    assert len(scores) <= 2 * len(model.cv_sp_alphas_) * 3


def test_zero_activation_has_separate_unpruned_zero_baseline():
    from types import SimpleNamespace

    path = SimpleNamespace(lambdas=np.array([.5, 0.]), activation_lambdas=np.array([.5, 0.]))
    assert_array_equal(_local_penalties(path), [0, np.nextafter(0., 1.), .5])
    root = SimpleNamespace(lambdas=np.array([0.]), activation_lambdas=np.array([]))
    assert_array_equal(_local_penalties(root), [0.])


def test_zero_gain_cart_stump_is_preserved_only_at_exact_zero():
    X = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
    y = np.array([0., 1., 1., 0.])
    base = DecisionTreeRegressor(max_depth=1, random_state=0).fit(X, y)
    assert base.get_n_leaves() == 2
    path = fitted_tree_linf_exact_topology_path(base)
    assert_array_equal(path.activation_lambdas, [0.])
    penalties = _local_penalties(path)
    scores, complexities, _, _ = _score_fold_states(
        _model(), base, path, penalties, X, y, X, y, None, None,
        [0.], _scorer, 1.,
    )
    assert_array_equal(penalties, [0., np.nextafter(0., 1.)])
    assert_array_equal(complexities[:, 0], [1, 0])
    assert_allclose(scores[:, 0], [-.25, -.25])


def test_one_se_selection_agrees_with_direct_score_and_complexity_rule():
    X, y = _data()
    model = _model()
    params = evaluate_structural_cv(
        model, X=X, y=y, sample_weight=None, reg_param_list=[0, 3, 30],
        scorer=_scorer, n_splits=3,
    )
    scores = np.asarray(model.cv_scores_)
    means = scores.mean(axis=1)
    best = means.argmax()
    threshold = means[best] - scores[best].std(ddof=1) / np.sqrt(3)
    complexities = np.mean(model.cv_complexities_, axis=1)
    eligible = np.flatnonzero(means >= threshold)
    simplest = eligible[complexities[eligible] == complexities[eligible].min()]
    top = simplest[np.isclose(means[simplest], means[simplest].max(), rtol=0, atol=1e-12)]
    selected = max(top, key=lambda i: (params[i][1], params[i][0], -i))
    _finalize_cv_selection(model, params)
    assert model.best_index_ == best
    assert model.selected_index_ == selected
    assert (model.sp_alpha_, model.reg_param_) == params[selected]


def test_root_only_folds_have_one_state_and_finite_gcv_scores():
    X = np.zeros((15, 2))
    y = np.arange(15.)
    model = _model()
    params = evaluate_structural_cv(
        model, X=X, y=y, sample_weight=None, reg_param_list="gcv",
        scorer=_scorer, n_splits=3,
    )
    assert params == [(0., "gcv")]
    assert_array_equal(model.cv_n_pruning_states_, [1, 1, 1])
    assert_array_equal(model.cv_complexities_, [[0, 0, 0]])
    assert_array_equal(model.cv_reg_params_, [[0, 0, 0]])
    assert np.isfinite(model.cv_scores_).all()


def test_structural_cv_rejects_zero_weight_training_or_validation_fold():
    X, y = _data()
    weights = np.zeros(len(y))
    weights[0] = 1.
    with pytest.raises(ValueError, match="zero total weight"):
        evaluate_structural_cv(
            _model(), X=X, y=y, sample_weight=weights, reg_param_list=[0],
            scorer=_scorer, n_splits=3,
        )


def test_default_public_structural_cv_skips_apa_and_stump_matrix(monkeypatch):
    import imodels.tree.sparse_pruning.sparse_hierarchical_shrinkage as wrappers

    X, y = _data()

    def forbidden(*args, **kwargs):
        raise AssertionError("structural CV must not solve APA or materialize local stumps")

    monkeypatch.setattr(wrappers, "make_stumps", forbidden)
    monkeypatch.setattr(wrappers, "tree_feature_transform", forbidden)
    model = _model(reg_param_list=[0, 3]).fit(X, y)
    assert model.cv_solver_ == "topology"
    assert model.cv_path_mode_ == "structural"
    assert model.solver_ == "topology"
    assert model.cv_scores_.shape == (2 * len(model.cv_sp_alphas_), 3)
    assert model.sp_alpha_ in model.cv_sp_alphas_
    full = DecisionTreeRegressor(max_leaf_nodes=7, random_state=9).fit(X, y)
    path = fitted_tree_linf_exact_topology_path(full)
    expected = full.get_n_leaves() - 1 if model.sp_alpha_ == 0 else len(
        path.tree_nodes_at(model.sp_alpha_)
    )
    assert model.complexity_ == expected
    assert np.isfinite(model.predict(X)).all()


@pytest.mark.parametrize("solver", ["topology", "proximal", "coefficient_path", "hicap"])
@pytest.mark.parametrize("reg_params", [[0, 3], "gcv"])
def test_public_structural_cv_honors_final_solver_and_reselects_full_fit_gcv(solver, reg_params):
    X, y = _data()
    model = _model(solver=solver, reg_param_list=reg_params).fit(X, y)
    reference = _model(solver="auto", reg_param_list=reg_params).fit(X, y)
    assert model.cv_solver_ == "topology"
    assert model.solver_ == solver
    assert (model.coef_ is None) == (solver == "topology")
    assert model.sp_alpha_ == reference.sp_alpha_
    assert_allclose(model.cv_scores_, reference.cv_scores_, rtol=0, atol=0)
    assert_allclose(model.predict(X), reference.predict(X), rtol=0, atol=1e-12)
    if reg_params == "gcv":
        full = DecisionTreeRegressor(max_leaf_nodes=7, random_state=9).fit(X, y)
        path = fitted_tree_linf_exact_topology_path(full)
        unshrunk = _brute_tree(full, path, model.sp_alpha_)
        expected, _ = select_hs_reg_param(unshrunk, y=y)
        assert model.reg_param == "gcv"
        assert_allclose(model.reg_param_, expected, rtol=1e-10)


@pytest.mark.parametrize("change_method", ["set_params", "direct"])
def test_structural_to_numeric_grid_refit_clears_stale_path_metadata(change_method):
    X, y = _data()
    model = _model(reg_param_list=[0]).fit(X, y)
    assert model.cv_path_mode_ == "structural"
    assert hasattr(model, "cv_path_results_")
    assert hasattr(model, "cv_n_pruning_states_")
    alphas = [0., .17, .41]
    if change_method == "set_params":
        model.set_params(sp_alpha_list=alphas)
        for name in (
            "cv_solver_", "cv_path_mode_", "cv_sp_alphas_",
            "cv_path_results_", "cv_n_pruning_states_",
        ):
            assert not hasattr(model, name)
    else:
        model.sp_alpha_list = alphas
    model.fit(X, y)
    assert model.cv_path_mode_ == "grid"
    assert model.cv_solver_ == "topology"
    assert_array_equal(model.cv_sp_alphas_, alphas)
    assert not hasattr(model, "cv_path_results_")
    assert not hasattr(model, "cv_n_pruning_states_")
    assert model.cv_scores_.shape == (len(alphas), 3)


@pytest.mark.parametrize("mode", ["classification_apa", "ord2", "custom", "positive_support"])
@pytest.mark.filterwarnings("ignore:APA-APG:sklearn.exceptions.ConvergenceWarning")
def test_ineligible_automatic_cv_uses_numeric_grid_without_structural_claim(mode, monkeypatch):
    from imodels.tree.sparse_pruning import _cv
    from imodels.tree.sparse_pruning.sparse_hierarchical_shrinkage import _DEFAULT_SP_ALPHA_GRID

    X, y = _data()

    def forbidden(*args, **kwargs):
        raise AssertionError("ineligible problem entered structural CV")

    monkeypatch.setattr(_cv, "evaluate_structural_cv", forbidden)
    if mode == "classification_apa":
        y = (y > np.median(y)).astype(int)
        model = SHSTreeClassifierCV(
            estimator_=DecisionTreeClassifier(max_leaf_nodes=4),
            max_leaf_nodes=4, reg_param_list=[0], random_state=9,
            cv=3, max_iter=100, tol=1e-4, solver="apa_apg2",
        )
    else:
        model = _model(reg_param_list=[0], max_iter=100, tol=1e-4)
        if mode == "ord2":
            model.set_params(ord=2)
        elif mode == "positive_support":
            model.set_params(support_tol=.1)
        else:
            model.hiCAP = lambda **kwargs: np.zeros(kwargs["X"].shape[1])
    model.fit(X, y)
    assert model.cv_path_mode_ == "grid"
    expected_solver = "proximal" if mode == "positive_support" else "apa_apg2"
    assert model.cv_solver_ == expected_solver
    assert model.solver_ == expected_solver
    assert_array_equal(model.cv_sp_alphas_, _DEFAULT_SP_ALPHA_GRID)
    assert not hasattr(model, "cv_path_results_")
    assert not hasattr(model, "cv_n_pruning_states_")


@pytest.mark.parametrize("wrapper", [SHSTreeRegressor, SHSTreeRegressorCV, SPTreeRegressor, SPTreeRegressorCV])
@pytest.mark.parametrize("solver", ["auto", "topology", "proximal", "coefficient_path", "hicap", "apa_apg2"])
def test_solver_and_infinity_norm_survive_sklearn_clone(wrapper, solver):
    original = wrapper(solver=solver)
    copied = clone(original)
    assert copied.solver == solver
    assert copied.get_params(deep=False)["solver"] == solver
    assert copied.ord == np.inf


def test_positive_support_tolerance_uses_point_grid_and_matches_exact_hicap():
    X, y = _data()
    alphas = [0., .05, .25, .8]
    common = dict(
        reg_param_list=[0., 4.], sp_alpha_list=alphas,
        support_tol=.1, tol=1e-9, max_iter=2000,
    )
    models = {
        solver: _model(solver=solver, **common).fit(X, y)
        for solver in ("auto", "proximal", "hicap")
    }
    reference = models["hicap"]
    for solver, model in models.items():
        assert model.cv_path_mode_ == "grid"
        assert model.cv_solver_ == ("proximal" if solver == "auto" else solver)
        assert not hasattr(model, "cv_path_results_")
        assert_allclose(model.cv_scores_, reference.cv_scores_, rtol=0, atol=1e-10)
        assert_array_equal(model.cv_complexities_, reference.cv_complexities_)
        assert model.sp_alpha_ == reference.sp_alpha_
        assert_allclose(model.predict(X), reference.predict(X), atol=1e-10)

    # Independent fits check candidate/fold alignment, not merely agreement
    # between two point solvers passing through the same cached CV code.
    for fold, (train, test) in enumerate(
        KFold(n_splits=3, shuffle=True, random_state=9).split(X)
    ):
        for index, params in enumerate(reference.cv_params_):
            rho = params["reg_param"] * len(train) / len(y)
            independent = SHSTreeRegressor(
                estimator_=DecisionTreeRegressor(max_leaf_nodes=7),
                max_leaf_nodes=7, random_state=9, solver="proximal",
                sp_alpha=params["sp_alpha"], reg_param=rho,
                support_tol=.1, tol=1e-9,
            ).fit(X[train], y[train])
            expected = -mean_squared_error(y[test], independent.predict(X[test]))
            assert_allclose(reference.cv_scores_[index, fold], expected, atol=1e-10)
            assert reference.cv_complexities_[index, fold] == independent.complexity_
            assert_allclose(reference.cv_reg_params_[index, fold], rho)


@pytest.mark.parametrize("solver", ["topology", "coefficient_path"])
def test_native_exact_zero_paths_reject_positive_support_threshold(solver):
    X, y = _data()
    with pytest.raises(ValueError, match="support_tol"):
        _model(solver=solver, support_tol=.1, reg_param_list=[0]).fit(X, y)


@pytest.mark.parametrize("n_classes", [2, 3])
@pytest.mark.parametrize("class_weight", [None, "balanced"])
def test_classifier_structural_cv_matches_fold_local_weighted_hs_oracle(n_classes, class_weight):
    rng = np.random.default_rng(85)
    X = rng.normal(size=(69, 3))
    signal = X[:, 0] + .4 * X[:, 1]
    cuts = np.quantile(signal, np.linspace(.25, .7, n_classes - 1))
    y = np.array(["oak", "elm", "pine"])[np.digitize(signal, cuts)]
    weights = 1. + np.arange(len(y)) % 3

    def scorer(candidate, X_out, y_out, sample_weight=None):
        assert isinstance(candidate, SHSTreeClassifier)
        assert candidate.coef_ is candidate.intercept_ is None
        assert candidate.estimator_.tree_.node_count == 2 * candidate.complexity_ + 1
        probability = candidate.predict_proba(X_out)
        assert_allclose(probability.sum(axis=1), 1., atol=1e-14)
        assert_array_equal(candidate.predict(X_out), candidate.classes_[probability.argmax(axis=1)])
        return -log_loss(y_out, probability, labels=candidate.classes_, sample_weight=sample_weight)

    base = DecisionTreeClassifier(max_leaf_nodes=5, class_weight=class_weight, random_state=9)
    model = SHSTreeClassifierCV(
        estimator_=base, max_leaf_nodes=5, reg_param_list=[0., 7.],
        cv=3, scoring=scorer, random_state=9,
    ).fit(X, y, sample_weight=weights)
    assert model.cv_path_mode_ == "structural" and model.solver_ == "topology"
    assert not hasattr(base, "tree_")
    full_weight = weights * compute_sample_weight(class_weight, y)
    expected_scores = np.empty_like(model.cv_scores_)
    expected_complexity = np.empty_like(model.cv_complexities_)
    fractions = []
    for fold, (train, test) in enumerate(StratifiedKFold(3, shuffle=True, random_state=9).split(X, y)):
        tree = clone(base).fit(X[train], y[train], sample_weight=weights[train])
        path = fitted_tree_linf_exact_topology_path(tree)
        effective = weights[train] * compute_sample_weight(class_weight, y[train])
        fraction = effective.sum() / full_weight.sum()
        fractions.append(fraction)
        for index, params in enumerate(model.cv_params_):
            pruned = _brute_tree(tree, path, params["sp_alpha"])
            structure = pruned.tree_
            original = structure.value[:, 0].copy()
            original /= original.sum(axis=1, keepdims=True)
            rho = params["reg_param"] * fraction
            # Independent vector HS: each parent-to-child probability increment
            # is shrunk according to the parent's effective fitting mass.
            pending = [(0, original[0])]
            while pending:
                node, probability = pending.pop()
                structure.value[node, 0] = probability
                if structure.children_left[node] >= 0:
                    for child in (structure.children_left[node], structure.children_right[node]):
                        increment = (original[child] - original[node]) / (
                            1. + rho / structure.weighted_n_node_samples[node]
                        )
                        pending.append((child, probability + increment))
            expected_scores[index, fold] = -log_loss(
                y[test], pruned.predict_proba(X[test]), labels=pruned.classes_,
                sample_weight=weights[test],
            )
            expected_complexity[index, fold] = pruned.get_n_leaves() - 1
            assert_allclose(model.cv_reg_params_[index, fold], rho)
    assert_allclose(model.cv_scores_, expected_scores, atol=1e-12)
    assert_array_equal(model.cv_complexities_, expected_complexity)
    assert_allclose(model.cv_weight_fractions_, fractions)
    final = SHSTreeClassifier(
        estimator_=clone(base), max_leaf_nodes=5, sp_alpha=model.sp_alpha_,
        reg_param=model.reg_param_, random_state=9,
    ).fit(X, y, sample_weight=weights)
    assert_allclose(model.predict_proba(X), final.predict_proba(X), atol=1e-12)


@pytest.mark.parametrize("wrapper", [SPTreeClassifierCV, SHSTreeClassifierCV])
def test_classifier_default_structural_cv_never_builds_stump_matrix(wrapper, monkeypatch):
    import imodels.tree.sparse_pruning.sparse_hierarchical_shrinkage as wrappers

    def forbidden(*args, **kwargs):
        pytest.fail("structural classification must not build a design or run APA")

    monkeypatch.setattr(wrappers, "make_stumps", forbidden)
    monkeypatch.setattr(wrappers, "tree_feature_transform", forbidden)
    X, signal = _data()
    y = np.digitize(signal, np.quantile(signal, [1 / 3, 2 / 3]))
    model = wrapper(max_leaf_nodes=4, reg_param_list=[0., 2.], cv=3, random_state=9).fit(X, y)
    assert model.cv_path_mode_ == "structural" and model.cv_solver_ == "topology"
    assert model.cv_scores_.shape == (2 * len(model.cv_sp_alphas_), 3)
    assert_allclose(model.predict_proba(X).sum(axis=1), 1.)


@pytest.mark.parametrize("n_classes", [2, 3])
def test_classifier_numeric_grid_shares_one_nonlinear_path_per_fold(n_classes, monkeypatch):
    from imodels.tree.sparse_pruning import _solver

    X, signal = _data()
    y = np.digitize(signal, np.quantile(signal, np.arange(1, n_classes) / n_classes))
    original = _solver.fitted_tree_linf_classification_path
    paths, scored_paths = [], []

    def counted_path(*args, **kwargs):
        path = original(*args, **kwargs)
        paths.append(path)
        return path

    def scorer(candidate, X_out, y_out):
        scored_paths.append(candidate.coefficient_path_)
        return np.mean(candidate.predict(X_out) == y_out)

    monkeypatch.setattr(_solver, "fitted_tree_linf_classification_path", counted_path)
    common = dict(max_leaf_nodes=3, sp_alpha_list=[0., .1, 1.],
                  reg_param_list=[0., 3.], cv=3, random_state=9, tol=1e-7)
    model = SHSTreeClassifierCV(solver="coefficient_path", scoring=scorer, **common).fit(X, y)
    reference = SHSTreeClassifierCV(solver="topology", **common).fit(X, y)
    assert model.cv_path_mode_ == "grid" and model.cv_solver_ == "coefficient_path"
    assert len(paths) == 4  # Three folds plus the final fitted tree.
    assert len(scored_paths) == 3 * 3 * 2
    assert {id(path) for path in scored_paths} == {id(path) for path in paths[:3]}
    assert model.coefficient_path_ is paths[-1]
    assert all(not path.exact and np.all(path.lambdas > 0) for path in paths)
    assert_allclose(model.cv_scores_, reference.cv_scores_, atol=0.)
    assert model.sp_alpha_ == reference.sp_alpha_
    assert_allclose(model.predict_proba(X), reference.predict_proba(X))
