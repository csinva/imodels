"""Fold-local structural-path validation for sparse regression trees.

Only training-fold trees determine candidate penalties. Each fold is fitted
once, and every distinct local topology is scored once per HS strength. The
union of fold knots is assembled afterwards, reusing those scores exactly.
"""
from __future__ import annotations

from copy import deepcopy
from itertools import product

import numpy as np
from sklearn.model_selection import KFold
from sklearn.tree._tree import TREE_LEAF, TREE_UNDEFINED

from imodels.tree._hs_gcv import apply_node_based_hs, select_hs_reg_param
from .fitted_tree import fitted_tree_linf_exact_topology_path


def _local_penalties(path):
    """Keep alpha=0's explicit unpruned baseline, including zero-score splits."""
    penalties = [0.0, *path.lambdas[path.lambdas > 0]]
    if np.any(path.activation_lambdas == 0):
        penalties.append(np.nextafter(0.0, 1.0))
    return np.unique(penalties)


def _score_fold_states(
    estimator, base, path, penalties, X_in, y_in, X_out, y_out,
    weight_in, weight_out, reg_params, scorer, weight_fraction,
):
    # Import at call time: the public estimator dispatches into this module.
    from .sparse_hierarchical_shrinkage import (
        SHSTreeRegressor, _compact_tree, _fold_reg_param,
        _score_with_optional_sample_weight,
    )

    # Cut source nodes incrementally, retaining original IDs. A single compact
    # copy per state keeps sklearn node_count/depth/leaves accurate for custom
    # scorers. Copying remains O(original nodes * states); no estimator is
    # copied again for the HS grid, and there are no per-state fingerprints.
    source = deepcopy(base)
    order = np.argsort(path.activation_lambdas, kind="stable")
    next_event = 0
    shape = (len(penalties), len(reg_params))
    scores, complexities, strengths = (np.empty(shape) for _ in range(3))
    diagnostics = []
    candidate = SHSTreeRegressor(
        sp_alpha=0, reg_param=0, ord=np.inf,
        random_state=estimator.random_state,
    )
    candidate.shrinkage_scheme_ = estimator.shrinkage_scheme_
    candidate.n_features_in_ = X_in.shape[1]
    candidate.prune_set_ = "full"
    candidate.solver_ = "topology"
    candidate.beta_stars_ = []
    candidate.coef_ = None
    candidate.intercept_ = float(base.tree_.value[0, 0, 0])
    candidate.coef_node_ids_ = path.node_ids
    candidate.pruning_path_ = path
    candidate.coefficient_path_ = None
    candidate.support_thresholds_ = [0.0]
    candidate.n_iter_ = 1

    for state_index, alpha in enumerate(penalties):
        if alpha > 0:
            while (next_event < len(order)
                   and path.activation_lambdas[order[next_event]] <= alpha):
                node = int(path.node_ids[order[next_event]])
                source.tree_.children_left[node] = TREE_LEAF
                source.tree_.children_right[node] = TREE_LEAF
                source.tree_.feature[node] = TREE_UNDEFINED
                source.tree_.threshold[node] = TREE_UNDEFINED
                next_event += 1
        candidate.estimator_ = deepcopy(source)
        _compact_tree(candidate.estimator_.tree_)
        original_values = candidate.estimator_.tree_.value.copy()
        candidate.sp_alpha = float(alpha)
        result = {
            "solver": "topology", "status": "structural_path",
            "converged": True, "n_iter": 0, "step_norm": 0.0,
            "relative_step_norm": 0.0, "approximation_parameter": 0.0,
        }
        candidate.optimization_results_ = [result]
        candidate._update_optimization_diagnostics(warn=False)
        diagnostics.append(result)
        for reg_index, reg_param in enumerate(reg_params):
            candidate.estimator_.tree_.value[:] = original_values
            candidate.reg_param = reg_param
            if reg_param == "gcv":
                candidate.reg_param_, candidate.gcv_results_ = select_hs_reg_param(
                    candidate.estimator_, sample_weight=weight_in, y=y_in
                )
                apply_node_based_hs(candidate.estimator_.tree_, candidate.reg_param_)
            else:
                effective = _fold_reg_param(estimator, reg_param, weight_fraction)
                candidate.reg_param = effective
                candidate.reg_param_ = effective
                candidate.gcv_results_ = None
                if candidate.shrinkage_scheme_ == "node_based":
                    apply_node_based_hs(candidate.estimator_.tree_, effective)
                else:
                    candidate._shrink(X_in, y_in, sample_weight=weight_in)
            candidate._update_estimator_metadata()
            scores[state_index, reg_index] = _score_with_optional_sample_weight(
                scorer, candidate, X_out, y_out, weight_out
            )
            complexities[state_index, reg_index] = candidate.complexity_
            strengths[state_index, reg_index] = candidate.reg_param_
    return scores, complexities, strengths, diagnostics


def evaluate_structural_cv(
    estimator, *, X, y, sample_weight, reg_param_list, scorer, n_splits,
    fit_args=(), fit_kwargs=None,
):
    """Populate CV diagnostics and return ``[(alpha, reference_HS), ...]``.

    The caller validates eligibility (a fresh single regression tree with the
    exact infinity-norm structural objective), inputs, and numeric HS values.
    ``reg_param_list='gcv'`` or ``['gcv']`` instead selects HS using only each
    state's training-fold statistics. The returned parameter remains ``'gcv'``
    so the caller reselects HS after its final full-data fit.

    Scores follow sklearn's higher-is-better convention. Candidate penalties
    are absolute normalized-loss penalties, not fractions of a full-data knot.
    The selected penalty must be used literally in the final fit, not snapped
    to a different full-tree knot.
    """
    from .sparse_hierarchical_shrinkage import (
        _effective_weight_sum, _initialize_cv_tracking,
    )

    reg_params = (
        [reg_param_list] if isinstance(reg_param_list, str)
        else list(reg_param_list)
    )
    contains_string = any(isinstance(value, str) for value in reg_params)
    if not reg_params or (contains_string and reg_params != ["gcv"]):
        raise ValueError(
            "structural CV requires numeric HS values or the sole value 'gcv'"
        )
    fit_kwargs = {} if fit_kwargs is None else dict(fit_kwargs)
    full_weight = _effective_weight_sum(sample_weight, len(y))
    folds = []
    paths = []
    fractions = []
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=estimator.random_state)
    for train, test in kfold.split(X):
        X_in, y_in, X_out, y_out = X[train], y[train], X[test], y[test]
        weight_in = None if sample_weight is None else sample_weight[train]
        weight_out = None if sample_weight is None else sample_weight[test]
        for weights, label in ((weight_in, "training"), (weight_out, "validation")):
            if weights is not None and not np.any(weights > 0):
                raise ValueError(f"sample_weight has zero total weight in a CV {label} fold")
        # _fresh_estimator preserves the public template/max-leaf/random-state
        # override semantics and returns a fresh clone when prefit=False.
        base = estimator._fresh_estimator().fit(
            X_in, y_in, *fit_args, sample_weight=weight_in, **fit_kwargs
        )
        path = fitted_tree_linf_exact_topology_path(base)
        penalties = _local_penalties(path)
        fraction = _effective_weight_sum(weight_in, len(y_in)) / full_weight
        fold_scores = _score_fold_states(
            estimator, base, path, penalties, X_in, y_in, X_out, y_out,
            weight_in, weight_out, reg_params, scorer, fraction,
        )
        folds.append((penalties, *fold_scores))
        paths.append(path)
        fractions.append(fraction)

    alphas = np.unique(np.concatenate([fold[0] for fold in folds]))
    param_list = list(product(alphas.tolist(), reg_params))
    _initialize_cv_tracking(estimator, param_list)
    for penalties, scores, complexities, strengths, diagnostics in folds:
        # At a knot, the strict topology is the state AFTER the event.
        states = np.searchsorted(penalties, alphas, side="right") - 1
        for candidate_index, (state, reg_index) in enumerate(product(states, range(len(reg_params)))):
            estimator.cv_scores_[candidate_index].append(float(scores[state, reg_index]))
            estimator.cv_complexities_[candidate_index].append(float(complexities[state, reg_index]))
            estimator.cv_reg_params_[candidate_index].append(float(strengths[state, reg_index]))
            estimator.cv_optimization_results_[candidate_index].append([dict(diagnostics[state])])
    estimator.cv_weight_fractions_ = fractions
    estimator.cv_sp_alphas_ = alphas
    estimator.cv_path_results_ = paths
    estimator.cv_n_pruning_states_ = np.asarray([len(fold[0]) for fold in folds], dtype=int)
    estimator.cv_solver_ = "topology"
    estimator.cv_path_mode_ = "structural"
    return param_list


__all__ = ["evaluate_structural_cv"]
