"""Sparse hierarchical shrinkage trees."""
from __future__ import annotations

import inspect
import itertools
import warnings
from copy import deepcopy
from typing import Sequence

import numpy as np
from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
    RegressorMixin,
    clone,
    is_classifier,
    is_regressor,
)
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    accuracy_score,
    get_scorer,
    log_loss,
    make_scorer,
    mean_squared_error,
)
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.tree._tree import TREE_LEAF
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted

from imodels.util import checks
from imodels.util.arguments import check_fit_arguments, check_fit_X
from imodels.util.tree import compute_tree_complexity

from imodels.importance.local_stumps import make_stumps, tree_feature_transform
from .optimizations import (
    hiCAP_classification,
    hiCAP_regression,
)

_DEFAULT_SP_ALPHA_GRID = (
    0,
    0.01,
    0.03,
    0.1,
    0.3,
    1,
    3,
    10,
    30,
    50,
    100,
    300,
    500,
)

try:
    from sklearn.ensemble._forest import (
        BaseForest,
        _generate_sample_indices,
        _generate_unsampled_indices,
        _get_n_samples_bootstrap,
    )
    from sklearn.utils.validation import check_random_state
except ImportError:  # pragma: no cover
    BaseForest = ()
    _generate_sample_indices = None
    _generate_unsampled_indices = None
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


def _compact_tree(tree):
    """Remove nodes made unreachable by structural pruning.

    sklearn stores tree metadata such as ``node_count`` and ``max_depth`` in
    the underlying ``Tree`` state. Updating child pointers alone leaves those
    values, feature importances, and serialized size inconsistent.
    """
    state = tree.__getstate__()
    reachable: list[int] = []
    max_depth = 0
    stack = [(0, 0)]
    while stack:
        node_id, depth = stack.pop()
        reachable.append(node_id)
        max_depth = max(max_depth, depth)
        left = int(tree.children_left[node_id])
        right = int(tree.children_right[node_id])
        if left != TREE_LEAF:
            stack.append((right, depth + 1))
            stack.append((left, depth + 1))

    if len(reachable) == tree.node_count:
        return tree

    old_to_new = {old: new for new, old in enumerate(reachable)}
    nodes = state["nodes"][reachable].copy()
    values = state["values"][reachable].copy()
    for new_id, old_id in enumerate(reachable):
        left = int(tree.children_left[old_id])
        right = int(tree.children_right[old_id])
        nodes["left_child"][new_id] = (
            TREE_LEAF if left == TREE_LEAF else old_to_new[left]
        )
        nodes["right_child"][new_id] = (
            TREE_LEAF if right == TREE_LEAF else old_to_new[right]
        )

    state["nodes"] = np.ascontiguousarray(nodes)
    state["values"] = np.ascontiguousarray(values)
    state["node_count"] = len(reachable)
    state["max_depth"] = max_depth
    tree.__setstate__(state)
    return tree


def _summarize_optimization_state(results, tol):
    """Summarize certified convergence and observed final-iterate drift."""
    attempted_results = [
        result
        for result in results
        if result is None
        or result.get("status") != "no_internal_nodes"
    ]
    uncertified = [
        result
        for result in attempted_results
        if result is None or result.get("converged") is not True
    ]
    changing = [
        result
        for result in uncertified
        if result is not None
        and result.get("relative_step_norm", 0.0) > tol
    ]
    certified = not uncertified
    if changing:
        stable = False
    elif uncertified:
        # APA-APG with overlapping groups has no finite-iteration support
        # certificate here. A small final step is encouraging but can still
        # precede a support change as gamma continues toward zero.
        stable = None
    else:
        stable = True
    return certified, stable, changing


def _callable_accepts_keyword(function, keyword):
    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        return False
    keyword_parameter = signature.parameters.get(keyword)
    return (
        keyword_parameter is not None
        and keyword_parameter.kind != inspect.Parameter.POSITIONAL_ONLY
    ) or any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )


class SHSTree(BaseEstimator):
    """Tree pruned by hierarchical-LASSO support, with optional shrinkage.

    APA-APG coefficients are used only to select a hierarchy-safe tree
    topology. Predictions continue to use the fitted CART node values on that
    topology, optionally transformed by hierarchical shrinkage; they are not
    ``X_opt @ beta_stars_``. The sparse penalty ``sp_alpha`` is relative to
    mean weighted loss, while ``reg_param`` retains HST's pseudo-count
    interpretation through weighted node sample counts.
    """

    def __init__(
        self,
        estimator_: BaseEstimator | None = None,
        sp_alpha: float = 1,
        reg_param: float | None = 0,
        gamma1: float = 1.0,
        a: float = 1.0,
        max_iter: int = 2000,
        tol: float = 1e-6,
        ord: int | str = 2,
        prune_set: str = "auto",
        max_leaf_nodes: int | None = None,
        random_state: int | None = None,
        support_tol: float | None = None,
        prefit: bool = False,
    ) -> None:
        super().__init__()
        self.sp_alpha = sp_alpha
        self.reg_param = reg_param
        self.estimator_ = estimator_
        # ``estimator_`` is retained as the familiar fitted-estimator alias.
        # This private template is never fitted or structurally modified.
        self._estimator_template = estimator_
        self.gamma1 = gamma1
        self.a = a
        self.max_iter = max_iter
        self.tol = tol
        self.ord = ord
        self.prune_set = prune_set
        self.max_leaf_nodes = max_leaf_nodes
        self.random_state = random_state
        self.support_tol = support_tol
        self.prefit = prefit
        self.shrinkage_scheme_ = "node_based"
        self.hiCAP = hiCAP_regression

    @property
    def hiCAP(self):
        """Optimization routine, exposed for backwards-compatible overrides."""
        return self._hicap_solver

    @hiCAP.setter
    def hiCAP(self, solver):
        self._hicap_solver = solver

    @property
    def shrinkage_scheme_(self):
        return self._shrinkage_scheme

    @shrinkage_scheme_.setter
    def shrinkage_scheme_(self, scheme):
        self._shrinkage_scheme = scheme

    def __sklearn_is_fitted__(self) -> bool:
        return getattr(self, "_wrapper_is_fitted", False)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.sparse = True
        return tags

    def __sklearn_clone__(self):
        if self.prefit:
            raise TypeError(
                "Sparse-pruning wrappers with prefit=True cannot be cloned "
                "safely: carrying the fitted tree into cross-validation would "
                "leak validation data. Use prefit=False for clone-based model "
                "selection."
            )
        params = {}
        for name, value in self.get_params(deep=False).items():
            params[name] = clone(value, safe=False)
        cloned = self.__class__(**params)
        cloned.hiCAP = self.hiCAP
        cloned.shrinkage_scheme_ = self.shrinkage_scheme_
        return cloned

    @staticmethod
    def _validate_finite_scalar(
        value,
        name: str,
        *,
        positive: bool = False,
        nonnegative: bool = False,
    ) -> float:
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, float, np.number))
        ):
            qualifier = "positive" if positive else "nonnegative"
            raise ValueError(f"{name} must be a {qualifier} finite scalar")
        numeric_value = float(value)
        if not np.isfinite(numeric_value):
            qualifier = "positive" if positive else "nonnegative"
            raise ValueError(f"{name} must be a {qualifier} finite scalar")
        if positive and numeric_value <= 0:
            raise ValueError(f"{name} must be a positive finite scalar")
        if nonnegative and numeric_value < 0:
            raise ValueError(f"{name} must be a nonnegative finite scalar")
        return numeric_value

    def _validate_hyperparameters(self, *, allow_unset: bool = False) -> None:
        if (
            isinstance(self.ord, (bool, np.bool_))
            or not np.isscalar(self.ord)
            or (
                self.ord != 2
                and self.ord != "inf"
                and self.ord != np.inf
            )
        ):
            raise ValueError("ord must be 2, 'inf', or np.inf")
        if self.prune_set not in {"auto", "ib", "oob", "full"}:
            raise ValueError(
                "prune_set must be one of {'auto', 'ib', 'oob', 'full'}"
            )
        if not isinstance(self.prefit, (bool, np.bool_)):
            raise ValueError("prefit must be a boolean")

        self._validate_finite_scalar(self.gamma1, "gamma1", positive=True)
        self._validate_finite_scalar(self.a, "a", positive=True)
        self._validate_finite_scalar(self.tol, "tol", positive=True)
        if (
            isinstance(self.max_iter, (bool, np.bool_))
            or not isinstance(self.max_iter, (int, np.integer))
            or self.max_iter <= 0
        ):
            raise ValueError("max_iter must be a positive integer")
        if self.support_tol is not None:
            self._validate_finite_scalar(
                self.support_tol, "support_tol", nonnegative=True
            )

        if self.sp_alpha is None:
            if not allow_unset:
                raise ValueError("sp_alpha must be a nonnegative finite scalar")
        else:
            self._validate_finite_scalar(
                self.sp_alpha, "sp_alpha", nonnegative=True
            )

        if self.reg_param is None:
            if not allow_unset:
                raise ValueError(
                    "Automatic GCV shrinkage is disabled because it is not "
                    "validated for sparse-pruned trees; provide an explicit "
                    "nonnegative reg_param."
                )
        else:
            self._validate_finite_scalar(
                self.reg_param, "reg_param", nonnegative=True
            )

    def get_params(self, deep: bool = True) -> dict:
        params = {
            "sp_alpha": self.sp_alpha,
            "reg_param": self.reg_param,
            "estimator_": self._estimator_template,
            "prune_set": self.prune_set,
            "max_leaf_nodes": self.max_leaf_nodes,
            "gamma1": self.gamma1,
            "a": self.a,
            "max_iter": self.max_iter,
            "tol": self.tol,
            "ord": self.ord,
            "random_state": self.random_state,
            "support_tol": self.support_tol,
            "prefit": self.prefit,
        }
        if deep:
            for name, value in list(params.items()):
                if hasattr(value, "get_params") and not isinstance(value, type):
                    for key, nested_value in value.get_params(deep=True).items():
                        params[f"{name}__{key}"] = nested_value
        return params

    def set_params(self, **params):
        if not params:
            return self
        nested_prefix = "estimator___"
        estimator_params = {
            key[len(nested_prefix):]: value
            for key, value in params.items()
            if key.startswith(nested_prefix)
        }
        params = {
            key: value
            for key, value in params.items()
            if not key.startswith(nested_prefix)
        }

        valid_params = self.get_params(deep=False)
        unknown = sorted(set(params) - set(valid_params))
        if unknown:
            raise ValueError(
                f"Invalid parameter {unknown[0]!r} for estimator {self}."
            )
        effective_prefit = params.get("prefit", self.prefit)
        if estimator_params and effective_prefit:
            raise ValueError(
                "Nested estimator_ parameters cannot be changed while "
                "prefit=True because that would make the fitted tree state "
                "inconsistent. Supply a newly fitted estimator_ instead."
            )

        estimator_template = params.get(
            "estimator_", self._estimator_template
        )
        if estimator_params:
            if estimator_template is None:
                raise ValueError(
                    "Nested estimator_ parameters require an explicit estimator_"
                )
            # Validate and apply nested updates on a copy so a failed update
            # cannot corrupt either the fitted estimator or its template.
            estimator_template = deepcopy(estimator_template)
            estimator_template.set_params(**estimator_params)

        for name, value in params.items():
            if name != "estimator_":
                setattr(self, name, value)
        self._estimator_template = estimator_template
        self.estimator_ = estimator_template
        for fitted_attribute in (
            "beta_stars_",
            "classes_",
            "complexity_",
            "cv_optimization_certified_",
            "cv_optimization_results_",
            "cv_optimization_stable_",
            "feature_names_in_",
            "feature_names_",
            "n_features_in_",
            "n_iter_",
            "oob_attributes_invalidated_",
            "optimization_certified_",
            "optimization_results_",
            "optimization_stable_",
            "prune_set_",
            "reg_param_",
            "scores_",
            "sp_alpha_",
            "support_thresholds_",
        ):
            if hasattr(self, fitted_attribute):
                delattr(self, fitted_attribute)
        self._wrapper_is_fitted = False
        return self

    def _fresh_estimator(self):
        template = self._base_estimator_template()
        if isinstance(self, ClassifierMixin) and not is_classifier(template):
            raise ValueError(
                "SHSTreeClassifier requires a classifier as estimator_"
            )
        if isinstance(self, RegressorMixin) and not is_regressor(template):
            raise ValueError(
                "SHSTreeRegressor requires a regressor as estimator_"
            )
        if isinstance(
            template, (GradientBoostingClassifier, GradientBoostingRegressor)
        ):
            raise NotImplementedError(
                "Gradient boosting is not supported: pruning each stage "
                "against the raw response ignores its fitted offset and "
                "pseudo-residual target."
            )

        if self.prefit:
            if not checks.check_is_fitted(template):
                raise ValueError(
                    "prefit=True requires an already-fitted estimator_"
                )
            return deepcopy(template)

        estimator = clone(template)
        overrides = {}
        estimator_params = estimator.get_params(deep=False)
        if (
            self.max_leaf_nodes is not None
            and "max_leaf_nodes" in estimator_params
        ):
            overrides["max_leaf_nodes"] = self.max_leaf_nodes
        if self.random_state is not None and "random_state" in estimator_params:
            overrides["random_state"] = self.random_state
        if overrides:
            estimator.set_params(**overrides)
        return estimator

    @staticmethod
    def _validate_fitted_tree_estimator(estimator) -> None:
        if not (
            hasattr(estimator, "tree_")
            or isinstance(estimator, BaseForest)
        ):
            raise ValueError(
                "estimator_ must be a fitted sklearn decision tree or forest"
            )

    def _base_estimator_template(self):
        if self._estimator_template is not None:
            return self._estimator_template
        max_leaf_nodes = (
            20 if self.max_leaf_nodes is None else self.max_leaf_nodes
        )
        if isinstance(self, ClassifierMixin):
            return DecisionTreeClassifier(
                max_leaf_nodes=max_leaf_nodes,
                random_state=self.random_state,
            )
        return DecisionTreeRegressor(
            max_leaf_nodes=max_leaf_nodes,
            random_state=self.random_state,
        )

    def _resolved_prune_set(self, estimator=None) -> str:
        if self.prune_set != "auto":
            return self.prune_set
        estimator = self.estimator_ if estimator is None else estimator
        if isinstance(estimator, BaseForest) and getattr(
            estimator, "bootstrap", False
        ):
            return "oob"
        return "full"

    @staticmethod
    def _validated_sample_weight(sample_weight, n_samples):
        if sample_weight is None:
            return None
        sample_weight = np.asarray(sample_weight, dtype=float)
        if sample_weight.ndim != 1 or sample_weight.shape[0] != n_samples:
            raise ValueError(
                "sample_weight must be one-dimensional with one value per row"
            )
        if not np.all(np.isfinite(sample_weight)):
            raise ValueError("sample_weight must contain only finite values")
        with np.errstate(over="ignore", invalid="ignore"):
            weight_sum = float(sample_weight.sum())
        if (
            np.any(sample_weight < 0)
            or not np.isfinite(weight_sum)
            or weight_sum <= 0
        ):
            raise ValueError(
                "sample_weight must be nonnegative with positive finite total "
                "weight"
            )
        return sample_weight

    def _fold_class_weight_into_sample_weight(
        self,
        estimator,
        y,
        sample_weight,
        *,
        prefit: bool = False,
        sparse_pruning_requested: bool | None = None,
    ):
        if not isinstance(self, ClassifierMixin):
            return estimator, sample_weight
        estimator_params = estimator.get_params(deep=False)
        class_weight = estimator_params.get("class_weight")
        if class_weight is None:
            return estimator, sample_weight
        if sparse_pruning_requested is None:
            sparse_pruning_requested = (
                self.sp_alpha is not None and self.sp_alpha > 0
            )
        if prefit:
            if sparse_pruning_requested:
                raise ValueError(
                    "prefit classifiers with class_weight are unsupported "
                    "during sparse pruning because the original effective "
                    "training weights cannot be verified"
                )
            # With sparse pruning disabled, the fitted tree already encodes
            # class weighting in its node values/counts. Do not apply it again.
            return estimator, sample_weight
        if class_weight == "balanced_subsample":
            if sparse_pruning_requested:
                raise ValueError(
                    "class_weight='balanced_subsample' is unsupported by "
                    "sparse pruning; use explicit sample_weight instead"
                )
            return estimator, sample_weight

        class_sample_weight = compute_sample_weight(class_weight, y)
        if sample_weight is None:
            effective_weight = class_sample_weight
        else:
            effective_weight = sample_weight * class_sample_weight
        estimator.set_params(class_weight=None)
        effective_weight = self._validated_sample_weight(
            effective_weight, len(y)
        )
        return estimator, effective_weight

    def fit(
        self,
        X,
        y,
        sample_weight=None,
        decimals: int = 0,
        verbose: bool = False,
        *args,
        **kwargs,
    ):
        self._wrapper_is_fitted = False
        if decimals != 0:
            warnings.warn(
                "decimals is deprecated and no longer controls sparse support; "
                "use support_tol for an explicit response-scaled threshold.",
                FutureWarning,
                stacklevel=2,
            )
        feature_names = kwargs.pop("feature_names", None)
        has_named_features = feature_names is not None or hasattr(X, "columns")
        if feature_names is None and hasattr(X, "columns"):
            feature_names = list(X.columns)
        X = check_fit_X(X)
        if y is None:
            raise ValueError(
                "This estimator requires y to be passed, but the target y is None"
            )
        y_for_estimator = np.asarray(y)
        if isinstance(self, ClassifierMixin):
            check_classification_targets(y_for_estimator)
        self._validate_hyperparameters()
        X, y, feature_names = check_fit_arguments(
            self, X, y_for_estimator, feature_names
        )
        if has_named_features and all(
            isinstance(name, str) for name in feature_names
        ):
            self.feature_names_in_ = np.asarray(feature_names, dtype=object)

        sample_weight = self._validated_sample_weight(sample_weight, len(y))
        is_classifier = isinstance(self, ClassifierMixin)
        if is_classifier and len(self.classes_) != 2:
            raise ValueError(
                "Only binary classification is supported. Sparse hierarchical "
                f"pruning got {len(self.classes_)} classes."
            )

        estimator = self._fresh_estimator()
        estimator, sample_weight = self._fold_class_weight_into_sample_weight(
            estimator,
            y_for_estimator,
            sample_weight,
            prefit=self.prefit,
        )
        if self.prefit:
            if getattr(estimator, "n_outputs_", 1) != 1:
                raise ValueError(
                    "Sparse-pruning wrappers require a single-output prefit "
                    "estimator_"
                )
            estimator_feature_names = getattr(
                estimator, "feature_names_in_", None
            )
            if (
                estimator_feature_names is not None
                and has_named_features
                and not np.array_equal(
                    np.asarray(estimator_feature_names, dtype=object),
                    np.asarray(feature_names, dtype=object),
                )
            ):
                raise ValueError(
                    "DataFrame feature names and order must match those used "
                    "to fit the prefit estimator_"
                )
            if getattr(estimator, "n_features_in_", X.shape[1]) != X.shape[1]:
                raise ValueError(
                    "The prefit estimator_ was fitted with a different number "
                    "of input features."
                )
            if is_classifier and not np.array_equal(
                np.asarray(estimator.classes_), np.asarray(self.classes_)
            ):
                raise ValueError(
                    "The prefit classifier classes do not match the labels "
                    "provided to fit."
                )
            if (
                self.sp_alpha is not None
                and self.sp_alpha > 0
                and isinstance(estimator, BaseForest)
                and self._resolved_prune_set(estimator) != "full"
            ):
                raise ValueError(
                    "prefit forest ib/oob membership cannot be verified from "
                    "X and y. Use prune_set='full' with prefit=True."
                )
        else:
            fit_y = y_for_estimator if is_classifier else y
            estimator = estimator.fit(
                X, fit_y, *args, sample_weight=sample_weight, **kwargs
            )
        self.estimator_ = estimator
        self.prune_set_ = self._resolved_prune_set()
        self._validate_fitted_tree_estimator(self.estimator_)

        self.beta_stars_ = []
        self._prune(
            X,
            y,
            sample_weight=sample_weight,
            decimals=decimals,
            verbose=verbose,
        )
        self._update_optimization_diagnostics(warn=True)
        self._shrink(X, y)
        self._update_estimator_metadata()
        return self

    def _update_optimization_diagnostics(self, *, warn):
        (
            self.optimization_certified_,
            self.optimization_stable_,
            changing_optimizations,
        ) = _summarize_optimization_state(
            self.optimization_results_, self.tol
        )
        if warn and changing_optimizations:
            worst_relative_step = max(
                result["relative_step_norm"]
                for result in changing_optimizations
            )
            warnings.warn(
                "APA-APG continuation reached max_iter without a convergence "
                "certificate (largest final relative step="
                f"{worst_relative_step:.3g}). With overlapping groups, a "
                "small step alone does not certify the support; the pruned "
                "support can depend on max_iter. Increase max_iter and inspect "
                "optimization_results_.",
                ConvergenceWarning,
                stacklevel=3,
            )
        # sklearn's iterative-estimator contract expects a positive n_iter_
        # after fit. A value of one denotes a completed fit in which pruning
        # required no APA-APG iterations (for example, sp_alpha=0).
        self.n_iter_ = max(1, self.n_iter_)

    def _update_estimator_metadata(self):
        self._invalidate_stale_oob_attributes()

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
        self._wrapper_is_fitted = True

    def _invalidate_stale_oob_attributes(self) -> None:
        self.oob_attributes_invalidated_ = ()
        if not isinstance(self.estimator_, BaseForest):
            return
        model_changed = (
            self.sp_alpha is not None and self.sp_alpha > 0
        ) or (
            self.reg_param is not None and self.reg_param > 0
        )
        if not model_changed:
            return
        invalidated = []
        for attribute in (
            "oob_decision_function_",
            "oob_prediction_",
            "oob_score_",
        ):
            if hasattr(self.estimator_, attribute):
                delattr(self.estimator_, attribute)
                invalidated.append(attribute)
        self.oob_attributes_invalidated_ = tuple(invalidated)

    def _prune_tree(
        self,
        tree,
        X,
        y,
        sp_alpha: float,
        sample_weight=None,
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
            self.support_thresholds_.append(np.nan)
            self.optimization_results_.append(
                {
                    "converged": True,
                    "n_iter": 0,
                    "step_norm": 0.0,
                    "relative_step_norm": 0.0,
                    "approximation_parameter": 0.0,
                    "status": "no_internal_nodes",
                }
            )
            return tree

        if sample_weight is not None and not np.any(
            np.asarray(sample_weight) > 0
        ):
            self.beta_stars_.append(
                np.full(ids.size + 1, np.nan, dtype=float)
            )
            self.support_thresholds_.append(np.nan)
            self.optimization_results_.append(
                {
                    "converged": False,
                    "n_iter": 0,
                    "step_norm": np.nan,
                    "relative_step_norm": np.nan,
                    "approximation_parameter": np.nan,
                    "status": "zero_weight_pruning_subset",
                }
            )
            return tree
        sample_weight = self._validated_sample_weight(sample_weight, len(y))
        if isinstance(self, ClassifierMixin):
            positive_weight = (
                np.ones(len(y), dtype=bool)
                if sample_weight is None
                else sample_weight > 0
            )
            if np.unique(y[positive_weight]).size < 2:
                # A binary logistic intercept has no finite optimum on a
                # one-class OOB sample. Safely leave this tree unchanged.
                self.beta_stars_.append(
                    np.full(ids.size + 1, np.nan, dtype=float)
                )
                self.support_thresholds_.append(np.nan)
                self.optimization_results_.append(
                    {
                        "converged": False,
                        "n_iter": 0,
                        "step_norm": np.nan,
                        "relative_step_norm": np.nan,
                        "approximation_parameter": np.nan,
                        "status": "one_class_pruning_subset",
                    }
                )
                return tree

        tree_stumps = make_stumps(tree)
        X_tree = tree_feature_transform(tree_stumps, X)
        X_opt = np.concatenate((np.ones(len(y)).reshape(-1, 1), X_tree), axis=1)

        groups = _find_subtrees(tree, 0, ids)
        solver_kwargs = {
            "X": X_opt,
            "y": y,
            "groups": groups,
            "lam": sp_alpha,
            "beta_init": beta_init,
            "gamma1": self.gamma1,
            "a": self.a,
            "max_iter": self.max_iter,
            "tol": self.tol,
            "ord": self.ord,
            "verbose": verbose,
            "sample_weight": sample_weight,
        }
        if self.hiCAP in (hiCAP_regression, hiCAP_classification):
            solver_result = self.hiCAP(return_info=True, **solver_kwargs)
        else:
            custom_solver_kwargs = solver_kwargs.copy()
            if sample_weight is None:
                # Preserve the original fixed-signature custom-solver API.
                custom_solver_kwargs.pop("sample_weight")
            elif not _callable_accepts_keyword(
                self.hiCAP, "sample_weight"
            ):
                raise TypeError(
                    "A custom hiCAP solver used with sample_weight must accept "
                    "a sample_weight keyword argument"
                )
            solver_result = self.hiCAP(**custom_solver_kwargs)
        if (
            isinstance(solver_result, tuple)
            and len(solver_result) == 2
            and isinstance(solver_result[1], dict)
        ):
            beta_star, solver_info = solver_result
        else:
            beta_star, solver_info = solver_result, None
        beta_star = np.asarray(beta_star, dtype=float)
        if beta_star.shape != (X_opt.shape[1],):
            raise ValueError(
                "hiCAP solver returned coefficients with shape "
                f"{beta_star.shape}; expected ({X_opt.shape[1]},)"
            )
        if not np.all(np.isfinite(beta_star)):
            raise FloatingPointError(
                "hiCAP solver returned non-finite coefficients"
            )

        self.beta_stars_.append(beta_star)
        self.optimization_results_.append(solver_info)
        if solver_info is not None:
            self.n_iter_ = max(self.n_iter_, int(solver_info["n_iter"]))

        # ``decimals`` is retained for call compatibility, but support is now
        # determined by a response-scale-aware numerical tolerance.
        _ = decimals
        if isinstance(self, ClassifierMixin):
            response_scale = 1.0
        elif sample_weight is None:
            response_scale = float(np.std(y))
        else:
            response_mean = float(np.average(y, weights=sample_weight))
            response_scale = float(
                np.sqrt(
                    np.average(
                        (y - response_mean) ** 2, weights=sample_weight
                    )
                )
            )
        response_scale = max(response_scale, np.finfo(float).tiny)
        relative_tol = (
            max(10 * self.tol, np.sqrt(np.finfo(float).eps))
            if self.support_tol is None
            else self.support_tol
        )
        support_threshold = relative_tol * response_scale
        self.support_thresholds_.append(support_threshold)

        inactive = np.abs(beta_star[1:]) <= support_threshold
        candidate_ids = {
            int(ids[int(group[0]) - 1])
            for group in groups
            if np.all(inactive[group - 1])
        }

        # Prune only maximal inactive subtrees. An ancestor is retained if any
        # descendant coefficient is active, even when its own coefficient is
        # numerically zero.
        parent = {}
        for node_id in ids:
            left = tree.children_left[node_id]
            right = tree.children_right[node_id]
            if left != TREE_LEAF:
                parent[int(left)] = int(node_id)
            if right != TREE_LEAF:
                parent[int(right)] = int(node_id)

        pruned_ids = []
        for node_id in sorted(candidate_ids):
            ancestor = parent.get(node_id)
            has_pruned_ancestor = False
            while ancestor is not None:
                if ancestor in candidate_ids:
                    has_pruned_ancestor = True
                    break
                ancestor = parent.get(ancestor)
            if not has_pruned_ancestor:
                pruned_ids.append(node_id)

        for nid in pruned_ids:
            tree.children_left[nid] = TREE_LEAF
            tree.children_right[nid] = TREE_LEAF
            tree.feature[nid] = -2
            tree.threshold[nid] = -2
        if pruned_ids:
            _compact_tree(tree)
        return tree

    def _forest_sample_indices(self, tree, n_samples):
        prune_set = self._resolved_prune_set()
        if prune_set == "full":
            return np.arange(n_samples, dtype=int)
        if not isinstance(self.estimator_, BaseForest):
            raise NotImplementedError(
                "Ensemble sparse pruning currently supports sklearn forest "
                "estimators only."
            )
        if not getattr(self.estimator_, "bootstrap", False):
            raise ValueError(
                f"prune_set={prune_set!r} requires bootstrap=True; "
                "use prune_set='full' for a non-bootstrap forest."
            )
        if (
            _get_n_samples_bootstrap is None
            or _generate_sample_indices is None
            or _generate_unsampled_indices is None
        ):
            raise ImportError(
                "sklearn >= 1.3 is required for sparse forest pruning"
            )

        n_samples_bootstrap = _get_n_samples_bootstrap(
            n_samples, self.estimator_.max_samples
        )
        random_state = getattr(tree, "random_state", None)
        if random_state is None:
            random_state = check_random_state(self.random_state)
        if prune_set == "ib":
            indices = _generate_sample_indices(
                random_state, n_samples, n_samples_bootstrap
            )
        else:
            indices = _generate_unsampled_indices(
                random_state, n_samples, n_samples_bootstrap
            )
        return indices

    def _prune(
        self,
        X,
        y,
        sample_weight=None,
        decimals: int = 0,
        verbose: bool = False,
        beta_init=None,
    ):
        self.beta_stars_ = []
        self.support_thresholds_ = []
        self.optimization_results_ = []
        self.n_iter_ = 0
        if self.sp_alpha is None or self.sp_alpha <= 0:
            return
        if hasattr(self.estimator_, "tree_"):
            if self._resolved_prune_set() != "full":
                raise ValueError(
                    "prune_set='ib' and prune_set='oob' require a bootstrap "
                    "forest; use prune_set='full' for a single tree."
                )
            self._prune_tree(
                self.estimator_.tree_,
                X,
                y,
                self.sp_alpha,
                sample_weight=sample_weight,
                decimals=decimals,
                beta_init=beta_init,
                verbose=verbose,
            )
        elif hasattr(self.estimator_, "estimators_"):
            if not isinstance(self.estimator_, BaseForest):
                raise NotImplementedError(
                    "Ensemble sparse pruning currently supports sklearn "
                    "forest estimators only."
                )
            for est in self.estimator_.estimators_:
                t = est
                if isinstance(t, np.ndarray):
                    if t.size != 1:
                        raise NotImplementedError(
                            "Multi-tree boosting stages are not supported"
                        )
                    t = t[0]
                indices = self._forest_sample_indices(t, len(X))
                if len(indices) == 0:
                    ids = _collect_internal_node_ids(t.tree_)
                    self.beta_stars_.append(
                        np.full(ids.size + 1, np.nan, dtype=float)
                    )
                    self.support_thresholds_.append(np.nan)
                    self.optimization_results_.append(
                        {
                            "converged": False,
                            "n_iter": 0,
                            "step_norm": np.nan,
                            "relative_step_norm": np.nan,
                            "approximation_parameter": np.nan,
                            "status": "empty_pruning_subset",
                        }
                    )
                    continue
                weight_prune = (
                    None
                    if sample_weight is None
                    else np.asarray(sample_weight)[indices]
                )
                self._prune_tree(
                    t.tree_,
                    X[indices],
                    y[indices],
                    self.sp_alpha,
                    sample_weight=weight_prune,
                    decimals=decimals,
                    beta_init=beta_init,
                    verbose=verbose,
                )

    def _shrink_tree(
        self,
        tree,
        reg_param,
        i: int = 0,
        parent_val=None,
        parent_num=None,
        cum_sum=0,
    ):
        left = tree.children_left[i]
        right = tree.children_right[i]
        is_leaf = left == right
        n_samples = tree.weighted_n_node_samples[i]
        val = deepcopy(tree.value[i, :, :])
        if isinstance(self, ClassifierMixin):
            # sklearn exposes normalized class distributions in current
            # versions, while older serialized trees may contain weighted
            # class counts. Normalize by the class total, never by the raw
            # node sample count, so every node remains on the same scale.
            class_totals = val.sum(axis=1, keepdims=True)
            val = np.divide(
                val,
                class_totals,
                out=np.zeros_like(val),
                where=class_totals != 0,
            )

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
        if self.reg_param is None:
            raise ValueError(
                "Automatic GCV shrinkage is disabled because it is not "
                "validated for sparse-pruned trees; provide an explicit "
                "nonnegative reg_param."
            )
        if not np.isscalar(self.reg_param) or not np.isfinite(self.reg_param):
            raise ValueError("reg_param must be a nonnegative finite scalar")
        if self.reg_param < 0:
            raise ValueError("reg_param must be nonnegative")
        if self.reg_param == 0:
            return
        if hasattr(self.estimator_, "tree_"):
            self._shrink_tree(self.estimator_.tree_, self.reg_param)
        elif hasattr(self.estimator_, "estimators_"):
            if not isinstance(self.estimator_, BaseForest):
                raise NotImplementedError(
                    "Ensemble shrinkage currently supports sklearn forest "
                    "estimators only in sparse-pruning wrappers."
                )
            for est in self.estimator_.estimators_:
                t = est
                if isinstance(t, np.ndarray):
                    if t.size != 1:
                        raise NotImplementedError(
                            "Multi-tree boosting stages are not supported"
                        )
                    t = t[0]
                self._shrink_tree(t.tree_, self.reg_param)

    def _validated_prediction_X(self, X):
        original_X = X
        if hasattr(self, "feature_names_in_") and hasattr(X, "columns"):
            incoming_names = np.asarray(list(X.columns), dtype=object)
            if not np.array_equal(incoming_names, self.feature_names_in_):
                raise ValueError(
                    "DataFrame feature names and order must match those seen "
                    "during fit"
                )
        validated_X = check_fit_X(X)
        if validated_X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {validated_X.shape[1]} features, but "
                f"{self.__class__.__name__} is expecting "
                f"{self.n_features_in_} features as input"
            )
        if (
            hasattr(original_X, "columns")
            and hasattr(self.estimator_, "feature_names_in_")
        ):
            return original_X
        return validated_X

    def predict(self, X, *args, **kwargs):
        check_is_fitted(self)
        X = self._validated_prediction_X(X)
        return self.estimator_.predict(X, *args, **kwargs)

    def score(self, X, y, *args, **kwargs):
        check_is_fitted(self)
        X = self._validated_prediction_X(X)
        return self.estimator_.score(X, y, *args, **kwargs)

class SHSTreeRegressor(RegressorMixin, SHSTree):
    def __init__(
        self,
        estimator_: BaseEstimator | None = None,
        sp_alpha: float = 1,
        reg_param: float | None = 1,
        prune_set: str = "auto",
        gamma1: float = 1.0,
        a: float = 1.0,
        max_iter: int = 2000,
        tol: float = 1e-6,
        ord: int | str = 2,
        max_leaf_nodes: int | None = None,
        random_state: int | None = None,
        support_tol: float | None = None,
        prefit: bool = False,
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
            support_tol=support_tol,
            prefit=prefit,
        )
        self.hiCAP = hiCAP_regression


class SHSTreeClassifier(ClassifierMixin, SHSTree):
    def __init__(
        self,
        estimator_: BaseEstimator | None = None,
        sp_alpha: float = 1,
        reg_param: float | None = 1,
        prune_set: str = "auto",
        gamma1: float = 1.0,
        a: float = 1.0,
        max_iter: int = 2000,
        tol: float = 1e-6,
        ord: int | str = 2,
        max_leaf_nodes: int | None = None,
        random_state: int | None = None,
        support_tol: float | None = None,
        prefit: bool = False,
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
            support_tol=support_tol,
            prefit=prefit,
        )
        self.hiCAP = hiCAP_classification

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        if getattr(tags, "classifier_tags", None) is not None:
            tags.classifier_tags.multi_class = False
        return tags

    def predict_proba(self, X, *args, **kwargs):
        check_is_fitted(self)
        X = self._validated_prediction_X(X)
        proba = self.estimator_.predict_proba(X, *args, **kwargs)
        proba = np.clip(proba, 0.0, None)
        row_sums = proba.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return proba / row_sums


def _resolve_cv_scorer(scoring, *, classification: bool):
    """Return a sklearn scorer, whose convention is always higher-is-better."""
    if scoring is None:
        return (
            make_scorer(accuracy_score)
            if classification
            else make_scorer(mean_squared_error, greater_is_better=False)
        )
    if isinstance(scoring, str):
        return get_scorer(scoring)
    if scoring is accuracy_score:
        return make_scorer(accuracy_score)
    if scoring is mean_squared_error:
        return make_scorer(mean_squared_error, greater_is_better=False)
    if scoring is log_loss:
        return make_scorer(
            log_loss,
            greater_is_better=False,
            response_method="predict_proba",
        )
    if callable(scoring):
        try:
            signature = inspect.signature(scoring)
        except (TypeError, ValueError):
            signature = None
        if signature is not None:
            positional_parameters = [
                parameter
                for parameter in signature.parameters.values()
                if parameter.kind
                in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                )
            ]
            has_var_positional = any(
                parameter.kind == inspect.Parameter.VAR_POSITIONAL
                for parameter in signature.parameters.values()
            )
            if len(positional_parameters) <= 2 and not has_var_positional:
                # Preserve the legacy two-argument metric API. Unknown raw
                # metrics were historically minimized, so wrap them as losses.
                metric_scorer = make_scorer(
                    scoring, greater_is_better=False
                )
                metric_scorer._sparse_pruning_accepts_sample_weight = (
                    _callable_accepts_keyword(scoring, "sample_weight")
                )
                return metric_scorer
        # sklearn's public scoring API also accepts estimator scorers with
        # signature ``scorer(estimator, X, y)``. Their convention is already
        # higher-is-better and they need not expose private scorer attributes.
        return scoring
    raise ValueError(
        "scoring must be a scorer name or a callable with signature "
        "scorer(estimator, X, y)"
    )


def _score_with_optional_sample_weight(
    scorer, estimator, X, y, sample_weight
):
    if sample_weight is None:
        return scorer(estimator, X, y)
    accepts_sample_weight = getattr(
        scorer, "_sparse_pruning_accepts_sample_weight", None
    )
    if accepts_sample_weight is None:
        accepts_sample_weight = _callable_accepts_keyword(
            scorer, "sample_weight"
        )
    if accepts_sample_weight:
        return scorer(
            estimator, X, y, sample_weight=sample_weight
        )
    return scorer(estimator, X, y)


def _validated_nonnegative_grid(values, name: str) -> list[float]:
    if values is None:
        raise ValueError(f"{name} must be a non-empty sequence")
    try:
        candidates = list(values)
    except TypeError as exc:
        raise ValueError(f"{name} must be a non-empty sequence") from exc
    if not candidates:
        raise ValueError(f"{name} must be a non-empty sequence")
    for value in candidates:
        if value is None:
            raise ValueError(
                f"{name} cannot contain None because automatic GCV is disabled"
            )
        SHSTree._validate_finite_scalar(
            value, name, nonnegative=True
        )
    return candidates


def _validated_n_splits(cv) -> int:
    if (
        isinstance(cv, (bool, np.bool_))
        or not isinstance(cv, (int, np.integer))
        or cv < 2
    ):
        raise ValueError("cv must be an integer greater than or equal to 2")
    return int(cv)


def _prepare_cv_data(X, y, feature_names=None):
    if feature_names is None and hasattr(X, "columns"):
        feature_names = list(X.columns)
    X = check_fit_X(X)
    y = np.asarray(y)
    if y.ndim != 1:
        raise ValueError(f"y must be one-dimensional; got shape {y.shape}")
    if len(y) != len(X):
        raise ValueError(
            "X and y have inconsistent sample counts: "
            f"{len(X)} != {len(y)}"
        )
    return X, y, feature_names


class SHSTreeClassifierCV(SHSTreeClassifier):
    def __init__(
        self,
        estimator_: BaseEstimator | None = None,
        sp_alpha_list: Sequence[float] = _DEFAULT_SP_ALPHA_GRID,
        reg_param_list: Sequence[float | None] | None = (0, 0.1, 1, 10, 50, 100, 500),
        max_leaf_nodes: int = 20,
        cv: int = 3,
        scoring=None,
        prune_set: str = "auto",
        random_state: int | None = None,
        gamma1: float = 1.0,
        a: float = 1.0,
        max_iter: int = 2000,
        tol: float = 1e-6,
        ord: int | str = 2,
        support_tol: float | None = None,
        prefit: bool = False,
    ) -> None:
        super().__init__(
            estimator_=estimator_,
            sp_alpha=None,
            reg_param=None,
            prune_set=prune_set,
            random_state=random_state,
            gamma1=gamma1,
            a=a,
            max_iter=max_iter,
            tol=tol,
            ord=ord,
            max_leaf_nodes=max_leaf_nodes,
            support_tol=support_tol,
            prefit=prefit,
        )
        self.sp_alpha_list = sp_alpha_list
        self.reg_param_list = reg_param_list
        self.cv = cv
        self.scoring = scoring

    def fit(
        self,
        X,
        y,
        sample_weight=None,
        decimals: int = 0,
        *args,
        **kwargs,
    ):
        self._wrapper_is_fitted = False
        if self.prefit:
            raise ValueError(
                "prefit=True is incompatible with cross-validation tuning"
            )
        self._validate_hyperparameters(allow_unset=True)
        self._fresh_estimator()
        feature_names = kwargs.pop("feature_names", None)
        X, y, feature_names = _prepare_cv_data(X, y, feature_names)
        check_classification_targets(y)
        classes, y_encoded = np.unique(y, return_inverse=True)
        if len(classes) != 2:
            raise ValueError(
                "Only binary classification is supported. Sparse hierarchical "
                f"pruning got {len(classes)} classes."
            )
        sample_weight = self._validated_sample_weight(sample_weight, len(y))
        n_splits = _validated_n_splits(self.cv)
        positive_weight = (
            np.ones(len(y), dtype=bool)
            if sample_weight is None
            else sample_weight > 0
        )
        effective_class_counts = np.bincount(
            y_encoded[positive_weight], minlength=2
        )
        if np.any(effective_class_counts < n_splits):
            raise ValueError(
                "Each class must have at least cv positive-weight samples for "
                "stratified sparse-pruning CV"
            )
        sp_alpha_list = _validated_nonnegative_grid(
            self.sp_alpha_list, "sp_alpha_list"
        )
        reg_param_list = _validated_nonnegative_grid(
            self.reg_param_list, "reg_param_list"
        )
        param_list = list(itertools.product(sp_alpha_list, reg_param_list))
        self.scores_ = [[] for _ in param_list]
        # Indexed as [candidate][fold][tree]. This retains diagnostics for
        # every candidate, including candidates not selected for the final fit.
        self.cv_optimization_results_ = [[] for _ in param_list]
        scorer = _resolve_cv_scorer(
            kwargs.pop("scoring", self.scoring), classification=True
        )
        kf = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=self.random_state
        )
        for train_index, test_index in kf.split(X, y):
            X_out, y_out = X[test_index, :], y[test_index]
            X_in, y_in = X[train_index, :], y[train_index]
            y_in_encoded = y_encoded[train_index].astype(float)
            weight_in = (
                None
                if sample_weight is None
                else sample_weight[train_index]
            )
            weight_out = (
                None
                if sample_weight is None
                else sample_weight[test_index]
            )
            if weight_in is not None and not np.any(weight_in > 0):
                raise ValueError(
                    "sample_weight has zero total weight in a CV training fold"
                )
            if weight_out is not None and not np.any(weight_out > 0):
                raise ValueError(
                    "sample_weight has zero total weight in a CV validation fold"
                )
            train_positive = (
                np.ones(len(y_in), dtype=bool)
                if weight_in is None
                else weight_in > 0
            )
            test_positive = (
                np.ones(len(y_out), dtype=bool)
                if weight_out is None
                else weight_out > 0
            )
            if (
                np.unique(y_in[train_positive]).size < 2
                or np.unique(y_out[test_positive]).size < 2
            ):
                raise ValueError(
                    "Every classifier CV train and validation fold must "
                    "contain both classes with positive weight"
                )
            base_est = clone(self._base_estimator_template())
            estimator_params = base_est.get_params(deep=False)
            overrides = {}
            if (
                self.max_leaf_nodes is not None
                and "max_leaf_nodes" in estimator_params
            ):
                overrides["max_leaf_nodes"] = self.max_leaf_nodes
            if (
                self.random_state is not None
                and "random_state" in estimator_params
            ):
                overrides["random_state"] = self.random_state
            if overrides:
                base_est.set_params(**overrides)
            base_est, weight_in = self._fold_class_weight_into_sample_weight(
                base_est,
                y_in,
                weight_in,
                sparse_pruning_requested=any(
                    sp_alpha > 0 for sp_alpha in sp_alpha_list
                ),
            )
            if (
                weight_in is not None
                and np.unique(y_in[weight_in > 0]).size < 2
            ):
                raise ValueError(
                    "class_weight and sample_weight must leave both classes "
                    "with positive weight in every CV training fold"
                )
            if isinstance(
                base_est,
                (GradientBoostingClassifier, GradientBoostingRegressor),
            ):
                raise NotImplementedError(
                    "Gradient boosting is not supported by sparse pruning"
                )
            base_est = base_est.fit(
                X_in,
                y_in,
                *args,
                sample_weight=weight_in,
                **kwargs,
            )
            self._validate_fitted_tree_estimator(base_est)
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
                    support_tol=self.support_tol,
                    prefit=True,
                )
                est_shs.hiCAP = self.hiCAP
                est_shs.classes_ = classes
                est_shs.n_features_in_ = X.shape[1]
                est_shs._prune(
                    X=X_in,
                    y=y_in_encoded,
                    sample_weight=weight_in,
                    decimals=decimals,
                    beta_init=None,
                )
                est_shs.prune_set_ = est_shs._resolved_prune_set()
                est_shs._update_optimization_diagnostics(warn=False)
                est_shs._shrink(X=X_in, y=y_in)
                est_shs._update_estimator_metadata()
                self.cv_optimization_results_[i].append(
                    deepcopy(est_shs.optimization_results_)
                )
                self.scores_[i].append(
                    _score_with_optional_sample_weight(
                        scorer, est_shs, X_out, y_out, weight_out
                    )
                )
        flat_cv_optimization_results = [
            result
            for candidate_results in self.cv_optimization_results_
            for fold_results in candidate_results
            for result in fold_results
        ]
        (
            self.cv_optimization_certified_,
            self.cv_optimization_stable_,
            changing_cv_optimizations,
        ) = _summarize_optimization_state(
            flat_cv_optimization_results, self.tol
        )
        if changing_cv_optimizations:
            worst_relative_step = max(
                result["relative_step_norm"]
                for result in changing_cv_optimizations
            )
            warnings.warn(
                "APA-APG coefficients were still changing in at least one CV "
                "candidate when max_iter was reached (largest final relative "
                f"step={worst_relative_step:.3g}). Candidate rankings and "
                "selected hyperparameters can depend on max_iter; inspect "
                "cv_optimization_results_.",
                ConvergenceWarning,
                stacklevel=2,
            )
        self.scores_ = np.asarray([np.mean(s) for s in self.scores_])
        if not np.any(np.isfinite(self.scores_)):
            raise ValueError("Cross-validation produced no finite scores")
        self.sp_alpha_, self.reg_param_ = param_list[
            int(np.nanargmax(self.scores_))
        ]
        # Operational values used by the final parent fit. They are selected
        # results, not constructor parameters for the CV estimator.
        self.sp_alpha = self.sp_alpha_
        self.reg_param = self.reg_param_
        return super().fit(
            X=X,
            y=y,
            sample_weight=sample_weight,
            decimals=decimals,
            *args,
            feature_names=feature_names,
            **kwargs,
        )

    def get_params(self, deep: bool = True):
        params = super().get_params(deep=deep)
        params.pop("sp_alpha", None)
        params.pop("reg_param", None)
        params.update(
            {
                "sp_alpha_list": self.sp_alpha_list,
                "reg_param_list": self.reg_param_list,
                "cv": self.cv,
                "scoring": self.scoring,
            }
        )
        return params


class SHSTreeRegressorCV(SHSTreeRegressor):
    def __init__(
        self,
        estimator_: BaseEstimator | None = None,
        sp_alpha_list: Sequence[float] = _DEFAULT_SP_ALPHA_GRID,
        reg_param_list: Sequence[float | None] | None = (
            0,
            0.1,
            1,
            10,
            50,
            100,
            500,
        ),
        max_leaf_nodes: int = 20,
        cv: int = 3,
        scoring=None,
        prune_set: str = "auto",
        random_state: int | None = None,
        gamma1: float = 1.0,
        a: float = 1.0,
        max_iter: int = 2000,
        tol: float = 1e-6,
        ord: int | str = 2,
        support_tol: float | None = None,
        prefit: bool = False,
    ) -> None:
        super().__init__(
            estimator_=estimator_,
            sp_alpha=None,
            reg_param=None,
            prune_set=prune_set,
            random_state=random_state,
            gamma1=gamma1,
            a=a,
            max_iter=max_iter,
            tol=tol,
            ord=ord,
            max_leaf_nodes=max_leaf_nodes,
            support_tol=support_tol,
            prefit=prefit,
        )
        self.sp_alpha_list = sp_alpha_list
        self.reg_param_list = reg_param_list
        self.cv = cv
        self.scoring = scoring

    def fit(
        self,
        X,
        y,
        sample_weight=None,
        decimals: int = 0,
        *args,
        **kwargs,
    ):
        self._wrapper_is_fitted = False
        if self.prefit:
            raise ValueError(
                "prefit=True is incompatible with cross-validation tuning"
            )
        self._validate_hyperparameters(allow_unset=True)
        self._fresh_estimator()
        feature_names = kwargs.pop("feature_names", None)
        X, y, feature_names = _prepare_cv_data(X, y, feature_names)
        sample_weight = self._validated_sample_weight(sample_weight, len(y))
        n_splits = _validated_n_splits(self.cv)
        sp_alpha_list = _validated_nonnegative_grid(
            self.sp_alpha_list, "sp_alpha_list"
        )
        reg_param_list = _validated_nonnegative_grid(
            self.reg_param_list, "reg_param_list"
        )
        param_list = list(itertools.product(sp_alpha_list, reg_param_list))
        self.scores_ = [[] for _ in param_list]
        # Indexed as [candidate][fold][tree]. This retains diagnostics for
        # every candidate, including candidates not selected for the final fit.
        self.cv_optimization_results_ = [[] for _ in param_list]
        scorer = _resolve_cv_scorer(
            kwargs.pop("scoring", self.scoring), classification=False
        )
        kf = KFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=self.random_state,
        )
        for train_index, test_index in kf.split(X):
            X_out, y_out = X[test_index, :], y[test_index]
            X_in, y_in = X[train_index, :], y[train_index]
            weight_in = (
                None
                if sample_weight is None
                else sample_weight[train_index]
            )
            weight_out = (
                None
                if sample_weight is None
                else sample_weight[test_index]
            )
            if weight_in is not None and not np.any(weight_in > 0):
                raise ValueError(
                    "sample_weight has zero total weight in a CV training fold"
                )
            if weight_out is not None and not np.any(weight_out > 0):
                raise ValueError(
                    "sample_weight has zero total weight in a CV validation fold"
                )
            base_est = clone(self._base_estimator_template())
            estimator_params = base_est.get_params(deep=False)
            overrides = {}
            if (
                self.max_leaf_nodes is not None
                and "max_leaf_nodes" in estimator_params
            ):
                overrides["max_leaf_nodes"] = self.max_leaf_nodes
            if (
                self.random_state is not None
                and "random_state" in estimator_params
            ):
                overrides["random_state"] = self.random_state
            if overrides:
                base_est.set_params(**overrides)
            if isinstance(
                base_est,
                (GradientBoostingClassifier, GradientBoostingRegressor),
            ):
                raise NotImplementedError(
                    "Gradient boosting is not supported by sparse pruning"
                )
            base_est = base_est.fit(
                X_in,
                y_in,
                *args,
                sample_weight=weight_in,
                **kwargs,
            )
            self._validate_fitted_tree_estimator(base_est)
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
                    support_tol=self.support_tol,
                    prefit=True,
                )
                est_shs.hiCAP = self.hiCAP
                est_shs.n_features_in_ = X.shape[1]
                est_shs._prune(
                    X=X_in,
                    y=y_in,
                    sample_weight=weight_in,
                    decimals=decimals,
                    beta_init=None,
                )
                est_shs.prune_set_ = est_shs._resolved_prune_set()
                est_shs._update_optimization_diagnostics(warn=False)
                est_shs._shrink(X=X_in, y=y_in)
                est_shs._update_estimator_metadata()
                self.cv_optimization_results_[i].append(
                    deepcopy(est_shs.optimization_results_)
                )
                self.scores_[i].append(
                    _score_with_optional_sample_weight(
                        scorer, est_shs, X_out, y_out, weight_out
                    )
                )
        flat_cv_optimization_results = [
            result
            for candidate_results in self.cv_optimization_results_
            for fold_results in candidate_results
            for result in fold_results
        ]
        (
            self.cv_optimization_certified_,
            self.cv_optimization_stable_,
            changing_cv_optimizations,
        ) = _summarize_optimization_state(
            flat_cv_optimization_results, self.tol
        )
        if changing_cv_optimizations:
            worst_relative_step = max(
                result["relative_step_norm"]
                for result in changing_cv_optimizations
            )
            warnings.warn(
                "APA-APG coefficients were still changing in at least one CV "
                "candidate when max_iter was reached (largest final relative "
                f"step={worst_relative_step:.3g}). Candidate rankings and "
                "selected hyperparameters can depend on max_iter; inspect "
                "cv_optimization_results_.",
                ConvergenceWarning,
                stacklevel=2,
            )
        self.scores_ = np.asarray([np.mean(s) for s in self.scores_])
        if not np.any(np.isfinite(self.scores_)):
            raise ValueError("Cross-validation produced no finite scores")
        self.sp_alpha_, self.reg_param_ = param_list[
            int(np.nanargmax(self.scores_))
        ]
        self.sp_alpha = self.sp_alpha_
        self.reg_param = self.reg_param_
        return super().fit(
            X=X,
            y=y,
            sample_weight=sample_weight,
            decimals=decimals,
            *args,
            feature_names=feature_names,
            **kwargs,
        )

    def get_params(self, deep: bool = True):
        params = super().get_params(deep=deep)
        params.pop("sp_alpha", None)
        params.pop("reg_param", None)
        params.update(
            {
                "sp_alpha_list": self.sp_alpha_list,
                "reg_param_list": self.reg_param_list,
                "cv": self.cv,
                "scoring": self.scoring,
            }
        )
        return params


class SPTreeRegressor(SHSTreeRegressor):
    def __init__(
        self,
        estimator_: BaseEstimator | None = None,
        sp_alpha: float = 1,
        reg_param: float = 0,
        prune_set: str = "auto",
        gamma1: float = 1.0,
        a: float = 1.0,
        max_iter: int = 2000,
        tol: float = 1e-6,
        ord: int | str = 2,
        max_leaf_nodes: int | None = None,
        random_state: int | None = None,
        support_tol: float | None = None,
        prefit: bool = False,
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
            support_tol=support_tol,
            prefit=prefit,
        )


class SPTreeClassifier(SHSTreeClassifier):
    def __init__(
        self,
        estimator_: BaseEstimator | None = None,
        sp_alpha: float = 1,
        reg_param: float = 0,
        prune_set: str = "auto",
        gamma1: float = 1.0,
        a: float = 1.0,
        max_iter: int = 2000,
        tol: float = 1e-6,
        ord: int | str = 2,
        max_leaf_nodes: int | None = None,
        random_state: int | None = None,
        support_tol: float | None = None,
        prefit: bool = False,
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
            support_tol=support_tol,
            prefit=prefit,
        )


class SPTreeRegressorCV(SHSTreeRegressorCV):
    def __init__(
        self,
        estimator_: BaseEstimator | None = None,
        sp_alpha_list: Sequence[float] = _DEFAULT_SP_ALPHA_GRID,
        reg_param_list: Sequence[float] | None = (0,),
        max_leaf_nodes: int = 20,
        cv: int = 3,
        scoring=None,
        prune_set: str = "auto",
        random_state: int | None = None,
        gamma1: float = 1.0,
        a: float = 1.0,
        max_iter: int = 2000,
        tol: float = 1e-6,
        ord: int | str = 2,
        support_tol: float | None = None,
        prefit: bool = False,
    ) -> None:
        super().__init__(
            estimator_=estimator_,
            sp_alpha_list=sp_alpha_list,
            reg_param_list=reg_param_list,
            max_leaf_nodes=max_leaf_nodes,
            cv=cv,
            scoring=scoring,
            prune_set=prune_set,
            random_state=random_state,
            gamma1=gamma1,
            a=a,
            max_iter=max_iter,
            tol=tol,
            ord=ord,
            support_tol=support_tol,
            prefit=prefit,
        )


class SPTreeClassifierCV(SHSTreeClassifierCV):
    def __init__(
        self,
        estimator_: BaseEstimator | None = None,
        sp_alpha_list: Sequence[float] = _DEFAULT_SP_ALPHA_GRID,
        reg_param_list: Sequence[float] | None = (0,),
        max_leaf_nodes: int = 20,
        cv: int = 3,
        scoring=None,
        prune_set: str = "auto",
        random_state: int | None = None,
        gamma1: float = 1.0,
        a: float = 1.0,
        max_iter: int = 2000,
        tol: float = 1e-6,
        ord: int | str = 2,
        support_tol: float | None = None,
        prefit: bool = False,
    ) -> None:
        super().__init__(
            estimator_=estimator_,
            sp_alpha_list=sp_alpha_list,
            reg_param_list=reg_param_list,
            max_leaf_nodes=max_leaf_nodes,
            cv=cv,
            scoring=scoring,
            prune_set=prune_set,
            random_state=random_state,
            gamma1=gamma1,
            a=a,
            max_iter=max_iter,
            tol=tol,
            ord=ord,
            support_tol=support_tol,
            prefit=prefit,
        )
