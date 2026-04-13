"""Sparse hierarchical shrinkage trees."""
from __future__ import annotations

import itertools
from copy import deepcopy
from typing import Sequence

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import log_loss, mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.tree._tree import TREE_LEAF

from imodels.util import checks
from imodels.util.arguments import check_fit_arguments
from imodels.util.tree import compute_tree_complexity

from imodels.importance.local_stumps import make_stumps, tree_feature_transform
from .optimizations import hiCAP_classification, hiCAP_regression, get_gcv_reg_param, get_reg_set

try:
    from sklearn.ensemble._forest import _get_n_samples_bootstrap
    from sklearn.utils.validation import check_random_state
except ImportError:  # pragma: no cover
    _get_n_samples_bootstrap = None
    check_random_state = None


def _find_subtrees(tree, idx: int, ids: np.ndarray) -> list[np.ndarray]:
    id_lookup = {node_id: pos for pos, node_id in enumerate(ids, start=1)}

    def explore(current_idx: int) -> list[int]:
        if tree.feature[current_idx] == -2:
            return []
        left_subtrees = explore(tree.children_left[current_idx])
        right_subtrees = explore(tree.children_right[current_idx])
        current = [current_idx] + left_subtrees + right_subtrees
        positions = [id_lookup[node_id] for node_id in current if node_id in id_lookup]
        if positions:
            all_groups.append(np.array(positions, dtype=int))
        return current

    all_groups: list[np.ndarray] = []
    explore(idx)
    return all_groups


def _collect_internal_node_ids(tree) -> np.ndarray:
    node_ids: list[int] = []

    def traverse(node_idx: int) -> None:
        if tree.feature[node_idx] == -2:
            return
        node_ids.append(node_idx)
        left = tree.children_left[node_idx]
        right = tree.children_right[node_idx]
        if left != -1:
            traverse(left)
        if right != -1:
            traverse(right)

    traverse(0)
    return np.array(node_ids, dtype=int)


class SHSTree(BaseEstimator):
    """Sparse hierarchical shrinkage tree."""

    def __init__(
        self,
        estimator_: BaseEstimator | None = DecisionTreeRegressor(max_leaf_nodes=20),
        sp_alpha: float = 1,
        reg_param: float | None = None,
        gamma1: float = 1.0,
        a: float = 1.0,
        max_iter: int = 500,
        tol: float = 1e-6,
        ord: int | str = 2,
        prune_set: str = "oob",
        max_leaf_nodes: int | None = None,
        random_state: int | None = None,
    ) -> None:
        super().__init__()
        assert ord in [2, "inf"], "ord must be 2 or 'inf'"
        if estimator_ is None:
            estimator_ = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes)
        self.sp_alpha = sp_alpha
        self.reg_param = reg_param
        self.estimator_ = estimator_
        self.gamma1 = gamma1
        self.a = a
        self.max_iter = max_iter
        self.tol = tol
        self.ord = ord
        self.prune_set = prune_set
        self.max_leaf_nodes = max_leaf_nodes
        self.random_state = random_state
        self.shrinkage_scheme_ = "node_based"
        self.hiCAP = hiCAP_regression
        if max_leaf_nodes is not None and hasattr(self.estimator_, "max_leaf_nodes"):
            self.estimator_.max_leaf_nodes = max_leaf_nodes
        if hasattr(self.estimator_, "random_state"):
            self.estimator_.random_state = random_state

    def get_params(self, deep: bool = True) -> dict:
        params = {
            "sp_alpha": self.sp_alpha,
            "reg_param": self.reg_param,
            "estimator_": self.estimator_,
            "prune_set": self.prune_set,
            "max_leaf_nodes": getattr(self.estimator_, "max_leaf_nodes", None),
            "gamma1": self.gamma1,
            "a": self.a,
            "max_iter": self.max_iter,
            "tol": self.tol,
            "ord": self.ord,
            "random_state": self.random_state,
        }
        return deepcopy(params) if deep else params

    def fit(self, X, y, sample_weight=None, decimals: int = 0, verbose: bool = False, *args, **kwargs):
        feature_names = kwargs.pop("feature_names", None)
        X, y, feature_names = check_fit_arguments(self, X, y, feature_names)
        if feature_names is not None:
            self.feature_names = feature_names

        if not checks.check_is_fitted(self.estimator_):
            self.estimator_ = self.estimator_.fit(
                X, y, *args, sample_weight=sample_weight, **kwargs
            )

        self.beta_stars_ = []
        self._prune(X, y, decimals=decimals, verbose=verbose)
        self._shrink(X, y)

        if hasattr(self.estimator_, "tree_"):
            self.complexity_ = compute_tree_complexity(self.estimator_.tree_)
        elif hasattr(self.estimator_, "estimators_"):
            self.complexity_ = 0
            for est in self.estimator_.estimators_:
                t = est
                if isinstance(t, np.ndarray):
                    assert t.size == 1
                    t = t[0]
                self.complexity_ += compute_tree_complexity(t.tree_)
        return self

    def _prune_tree(
        self,
        tree,
        X,
        y,
        sp_alpha: float,
        decimals: int = 0,
        beta_init=None,
        verbose: bool = False,
    ):
        if sp_alpha is None or sp_alpha <= 0:
            return tree

        ids = _collect_internal_node_ids(tree)
        if ids.size == 0:
            # Degenerate tree with a single leaf: nothing to prune.
            self.beta_stars_.append(np.array([0.0]))
            return tree

        tree_stumps = make_stumps(tree)
        X_tree = tree_feature_transform(tree_stumps, X)
        X_opt = np.concatenate((np.ones(len(y)).reshape(-1, 1), X_tree), axis=1)

        groups = _find_subtrees(tree, 0, ids)
        beta_star = self.hiCAP(
            X=X_opt,
            y=y,
            groups=groups,
            lam=sp_alpha,
            beta_init=beta_init,
            gamma1=self.gamma1,
            a=self.a,
            max_iter=self.max_iter,
            tol=self.tol,
            ord=self.ord,
            verbose=verbose,
        )

        self.beta_stars_.append(beta_star)

        zero_mask = np.where(np.round(beta_star[1:], decimals) == 0)[0]
        pruned_ids = ids[zero_mask]

        for nid in np.sort(pruned_ids)[::-1]:
            tree.value[tree.children_left[nid], 0, 0] = 0
            tree.value[tree.children_right[nid], 0, 0] = 0
            tree.children_left[nid] = TREE_LEAF
            tree.children_right[nid] = TREE_LEAF
            tree.feature[nid] = -2
            tree.threshold[nid] = -2
        return tree

    def _prune(self, X, y, decimals: int = 0, verbose: bool = False, beta_init=None):
        self.beta_stars_ = []
        if self.sp_alpha <= 0:
            return
        if hasattr(self.estimator_, "tree_"):
            self._prune_tree(self.estimator_.tree_, X, y, self.sp_alpha, decimals=decimals, beta_init=beta_init, verbose=verbose)
        elif hasattr(self.estimator_, "estimators_"):
            for est in self.estimator_.estimators_:
                t = est
                if isinstance(t, np.ndarray):
                    assert t.size == 1
                    t = t[0]

                # Forest estimators can support ib/oob/full pruning splits.
                if hasattr(self.estimator_, "max_samples"):
                    if _get_n_samples_bootstrap is None:
                        raise ImportError("sklearn >= 1.3 is required for sparse forestry pruning")
                    n_samples = len(X)
                    n_samples_bootstrap = _get_n_samples_bootstrap(n_samples, self.estimator_.max_samples)
                    rng = check_random_state(self.random_state)
                    rs = getattr(t, "random_state", rng)
                    X_prune, y_prune = get_reg_set(
                        self.prune_set,
                        X,
                        y,
                        rs,
                        n_samples,
                        n_samples_bootstrap,
                    )
                else:
                    # Generic ensembles (e.g., gradient boosting): no bootstrap metadata.
                    X_prune, y_prune = X, y
                self._prune_tree(t.tree_, X_prune, y_prune, self.sp_alpha, decimals=decimals, beta_init=beta_init, verbose=verbose)

    def _shrink_tree(self, tree, reg_param, i: int = 0, parent_val=None, parent_num=None, cum_sum=0):
        if reg_param is None:
            reg_param = 1.0
        left = tree.children_left[i]
        right = tree.children_right[i]
        is_leaf = left == right
        n_samples = tree.weighted_n_node_samples[i]
        if isinstance(self, RegressorMixin) or isinstance(self.estimator_, GradientBoostingClassifier):
            val = deepcopy(tree.value[i, :, :])
        else:
            val = tree.value[i, :, :] / n_samples

        if parent_val is None and parent_num is None:
            cum_sum = val
        else:
            if self.shrinkage_scheme_ == "node_based":
                val_new = (val - parent_val) / (1 + reg_param / parent_num)
            elif self.shrinkage_scheme_ == "constant":
                val_new = (val - parent_val) / (1 + reg_param)
            else:
                val_new = 0
            cum_sum += val_new

        if self.shrinkage_scheme_ in ["node_based", "constant"]:
            tree.value[i, :, :] = cum_sum
        else:
            if is_leaf:
                root_val = tree.value[0, :, :]
                tree.value[i, :, :] = root_val + (val - root_val) / (1 + reg_param / n_samples)
            else:
                tree.value[i, :, :] = val

        if not is_leaf:
            self._shrink_tree(
                tree,
                reg_param,
                left,
                parent_val=val,
                parent_num=n_samples,
                cum_sum=deepcopy(cum_sum),
            )
            self._shrink_tree(
                tree,
                reg_param,
                right,
                parent_val=val,
                parent_num=n_samples,
                cum_sum=deepcopy(cum_sum),
            )
        return tree

    def _shrink(self, X, y):
        if self.reg_param is not None and self.reg_param <= 0:
            return
        if hasattr(self.estimator_, "tree_"):
            if self.reg_param is None:
                self.reg_param = get_gcv_reg_param(self.estimator_.tree_, X, y)
            self._shrink_tree(self.estimator_.tree_, self.reg_param)
        elif hasattr(self.estimator_, "estimators_"):
            if self.reg_param is None:
                self.reg_params = []
            for est in self.estimator_.estimators_:
                t = est
                if isinstance(t, np.ndarray):
                    assert t.size == 1
                    t = t[0]
                if self.reg_param is None:
                    if hasattr(self.estimator_, "max_samples"):
                        if _get_n_samples_bootstrap is None:
                            raise ImportError("sklearn >= 1.3 is required for sparse forestry shrinkage")
                        n_samples = len(X)
                        n_samples_bootstrap = _get_n_samples_bootstrap(n_samples, self.estimator_.max_samples)
                        rng = check_random_state(self.random_state)
                        rs = getattr(t, "random_state", rng)
                        X_shrink, y_shrink = get_reg_set(
                            self.prune_set,
                            X,
                            y,
                            rs,
                            n_samples,
                            n_samples_bootstrap,
                        )
                    else:
                        X_shrink, y_shrink = X, y
                    reg_param = get_gcv_reg_param(t.tree_, X_shrink, y_shrink)
                    self.reg_params.append(reg_param)
                    self._shrink_tree(t.tree_, reg_param)
                else:
                    self._shrink_tree(t.tree_, self.reg_param)

    def predict(self, X, *args, **kwargs):
        return self.estimator_.predict(X, *args, **kwargs)

    def predict_proba(self, X, *args, **kwargs):
        if hasattr(self.estimator_, "predict_proba"):
            return self.estimator_.predict_proba(X, *args, **kwargs)
        return NotImplemented

    def score(self, X, y, *args, **kwargs):
        if hasattr(self.estimator_, "score"):
            return self.estimator_.score(X, y, *args, **kwargs)
        return NotImplemented

    def __repr__(self) -> str:
        attrs = [
            "estimator_",
            "sp_alpha",
            "reg_param",
            "prune_set",
            "max_leaf_nodes",
            "gamma1",
            "a",
            "max_iter",
            "tol",
            "ord",
            "random_state",
        ]
        params = ", ".join(f"{attr}={getattr(self, attr)!r}" for attr in attrs)
        return f"{self.__class__.__name__}({params})"


class SHSTreeRegressor(SHSTree, RegressorMixin):
    def __init__(
        self,
        estimator_: BaseEstimator | None = DecisionTreeRegressor(max_leaf_nodes=20),
        sp_alpha: float = 1,
        reg_param: float | None = None,
        prune_set: str = "oob",
        gamma1: float = 1.0,
        a: float = 1.0,
        max_iter: int = 500,
        tol: float = 1e-6,
        ord: int | str = 2,
        max_leaf_nodes: int | None = None,
        random_state: int | None = None,
    ) -> None:
        super().__init__(
            estimator_=estimator_,
            sp_alpha=sp_alpha,
            reg_param=reg_param,
            prune_set=prune_set,
            gamma1=gamma1,
            a=a,
            max_iter=max_iter,
            tol=tol,
            ord=ord,
            max_leaf_nodes=max_leaf_nodes,
            random_state=random_state,
        )
        self.hiCAP = hiCAP_regression


class SHSTreeClassifier(SHSTree, ClassifierMixin):
    def __init__(
        self,
        estimator_: BaseEstimator | None = DecisionTreeClassifier(max_leaf_nodes=20),
        sp_alpha: float = 1,
        reg_param: float | None = 1,
        prune_set: str = "oob",
        gamma1: float = 1.0,
        a: float = 1.0,
        max_iter: int = 500,
        tol: float = 1e-6,
        ord: int | str = 2,
        max_leaf_nodes: int | None = None,
        random_state: int | None = None,
    ) -> None:
        super().__init__(
            estimator_=estimator_,
            sp_alpha=sp_alpha,
            reg_param=reg_param,
            prune_set=prune_set,
            gamma1=gamma1,
            a=a,
            max_iter=max_iter,
            tol=tol,
            ord=ord,
            max_leaf_nodes=max_leaf_nodes,
            random_state=random_state,
        )
        self.hiCAP = hiCAP_classification

    def predict_proba(self, X, *args, **kwargs):
        proba = super().predict_proba(X, *args, **kwargs)
        if proba is NotImplemented:
            return proba
        proba = np.clip(proba, 0.0, None)
        row_sums = proba.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return proba / row_sums


def _get_cv_criterion(scorer):
    # Explicitly map known scorers; avoids invalid probing for metrics like accuracy.
    if scorer is accuracy_score:
        return np.argmax
    if scorer in {log_loss, mean_squared_error}:
        return np.argmin
    return np.argmin


class SHSTreeClassifierCV(SHSTreeClassifier):
    def __init__(
        self,
        estimator_: BaseEstimator | None = None,
        sp_alpha_list: Sequence[float] = (0, 0.1, 1, 10, 50, 100, 500),
        reg_param_list: Sequence[float | None] | None = (0, 0.1, 1, 10, 50, 100, 500),
        max_leaf_nodes: int = 20,
        cv: int = 3,
        scoring=None,
        prune_set: str = "oob",
        random_state: int | None = None,
        *args,
        **kwargs,
    ) -> None:
        if estimator_ is None:
            estimator_ = DecisionTreeClassifier(max_leaf_nodes=max_leaf_nodes)
        super().__init__(
            estimator_=estimator_,
            sp_alpha=None,
            reg_param=None,
            prune_set=prune_set,
            random_state=random_state,
            gamma1=kwargs.get("gamma1", 1.0),
            a=kwargs.get("a", 1.0),
            max_iter=kwargs.get("max_iter", 500),
            tol=kwargs.get("tol", 1e-6),
            ord=kwargs.get("ord", 2),
            max_leaf_nodes=max_leaf_nodes,
        )
        self.sp_alpha_list = np.array(list(sp_alpha_list))
        if reg_param_list is None:
            reg_param_list = [None]
        self.reg_param_list = list(reg_param_list)
        self.cv = cv
        self.scoring = scoring

    def fit(self, X, y, decimals: int = 0, *args, **kwargs):
        param_list = list(itertools.product(self.sp_alpha_list, self.reg_param_list))
        self.scores_ = [[] for _ in param_list]
        scorer = kwargs.pop("scoring", self.scoring)
        if scorer is None:
            scorer = accuracy_score
        kf = StratifiedKFold(
            n_splits=self.cv, shuffle=True, random_state=self.random_state
        )
        for train_index, test_index in kf.split(X, y):
            X_out, y_out = X[test_index, :], y[test_index]
            X_in, y_in = X[train_index, :], y[train_index]
            base_est = deepcopy(self.estimator_)
            base_est = base_est.fit(X_in, y_in, *args, **kwargs)
            for i, (sp_alpha, reg_param) in enumerate(param_list):
                est_shs = SHSTreeClassifier(
                    estimator_=deepcopy(base_est),
                    sp_alpha=sp_alpha,
                    reg_param=reg_param,
                    prune_set=self.prune_set,
                    random_state=self.random_state,
                    gamma1=self.gamma1,
                    a=self.a,
                    max_iter=self.max_iter,
                    tol=self.tol,
                    ord=self.ord,
                )
                est_shs._prune(
                    X=X_in, y=y_in, decimals=decimals, beta_init=None, *args, **kwargs
                )
                est_shs._shrink(X=X_in, y=y_in)
                if scorer is log_loss:
                    pred = est_shs.predict_proba(X_out)
                else:
                    pred = est_shs.predict(X_out)
                self.scores_[i].append(scorer(y_out, pred))
        self.scores_ = [np.mean(s) for s in self.scores_]
        cv_criterion = _get_cv_criterion(scorer)
        self.sp_alpha, self.reg_param = param_list[cv_criterion(self.scores_)]
        return super().fit(X=X, y=y, decimals=decimals, *args, **kwargs)

    def get_params(self, deep: bool = True):
        params = super().get_params(deep=deep)
        params.update(
            {
                "sp_alpha_list": deepcopy(self.sp_alpha_list) if deep else self.sp_alpha_list,
                "reg_param_list": deepcopy(self.reg_param_list) if deep else self.reg_param_list,
                "cv": self.cv,
                "scoring": self.scoring,
            }
        )
        return params


class SHSTreeRegressorCV(SHSTreeRegressor):
    def __init__(
        self,
        estimator_: BaseEstimator | None = None,
        sp_alpha_list: Sequence[float] = (0, 0.1, 1, 10, 50, 100, 500),
        reg_param_list: Sequence[float | None] | None = None,
        max_leaf_nodes: int = 20,
        cv: int = 3,
        scoring=None,
        prune_set: str = "oob",
        random_state: int | None = None,
        *args,
        **kwargs,
    ) -> None:
        if estimator_ is None:
            estimator_ = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes)
        super().__init__(
            estimator_=estimator_,
            sp_alpha=None,
            reg_param=None,
            prune_set=prune_set,
            random_state=random_state,
            gamma1=kwargs.get("gamma1", 1.0),
            a=kwargs.get("a", 1.0),
            max_iter=kwargs.get("max_iter", 500),
            tol=kwargs.get("tol", 1e-6),
            ord=kwargs.get("ord", 2),
            max_leaf_nodes=max_leaf_nodes,
        )
        self.sp_alpha_list = list(sp_alpha_list)
        if reg_param_list is None:
            reg_param_list = [None]
        self.reg_param_list = list(reg_param_list)
        self.cv = cv
        self.scoring = scoring

    def fit(self, X, y, decimals: int = 0, *args, **kwargs):
        param_list = list(itertools.product(self.sp_alpha_list, self.reg_param_list))
        self.scores_ = [[] for _ in param_list]
        scorer = kwargs.pop("scoring", self.scoring)
        if scorer is None:
            scorer = mean_squared_error
        kf = KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
        for train_index, test_index in kf.split(X):
            X_out, y_out = X[test_index, :], y[test_index]
            X_in, y_in = X[train_index, :], y[train_index]
            base_est = deepcopy(self.estimator_)
            base_est = base_est.fit(X_in, y_in, *args, **kwargs)
            for i, (sp_alpha, reg_param) in enumerate(param_list):
                est_shs = SHSTreeRegressor(
                    estimator_=deepcopy(base_est),
                    sp_alpha=sp_alpha,
                    reg_param=reg_param,
                    prune_set=self.prune_set,
                    random_state=self.random_state,
                    gamma1=self.gamma1,
                    a=self.a,
                    max_iter=self.max_iter,
                    tol=self.tol,
                    ord=self.ord,
                )
                est_shs._prune(
                    X=X_in, y=y_in, decimals=decimals, beta_init=None, *args, **kwargs
                )
                est_shs._shrink(X=X_in, y=y_in)
                self.scores_[i].append(scorer(y_out, est_shs.predict(X_out)))
        self.scores_ = [np.mean(s) for s in self.scores_]
        cv_criterion = _get_cv_criterion(scorer)
        self.sp_alpha, self.reg_param = param_list[cv_criterion(self.scores_)]
        return super().fit(X=X, y=y, decimals=decimals, *args, **kwargs)

    def get_params(self, deep: bool = True):
        params = super().get_params(deep=deep)
        params.update(
            {
                "sp_alpha_list": deepcopy(self.sp_alpha_list) if deep else self.sp_alpha_list,
                "reg_param_list": deepcopy(self.reg_param_list) if deep else self.reg_param_list,
                "cv": self.cv,
                "scoring": self.scoring,
            }
        )
        return params


class SPTreeRegressor(SHSTreeRegressor):
    def __init__(self, *args, reg_param: float = 0, **kwargs) -> None:
        super().__init__(*args, reg_param=reg_param, **kwargs)


class SPTreeClassifier(SHSTreeClassifier):
    def __init__(self, *args, reg_param: float = 0, **kwargs) -> None:
        super().__init__(*args, reg_param=reg_param, **kwargs)


class SPTreeRegressorCV(SHSTreeRegressorCV):
    def __init__(self, *args, reg_param_list: Sequence[float | None] = (0,), **kwargs) -> None:
        super().__init__(*args, reg_param_list=reg_param_list, **kwargs)


class SPTreeClassifierCV(SHSTreeClassifierCV):
    def __init__(self, *args, reg_param_list: Sequence[float | None] = (0,), **kwargs) -> None:
        super().__init__(*args, reg_param_list=reg_param_list, **kwargs)
