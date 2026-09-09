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
from imodels.tree._hs_gcv import apply_node_based_hs, select_hs_reg_param

from imodels.importance.local_stumps import make_stumps, tree_feature_transform
from .optimizations import (
    get_gcv_reg_param,
    hiCAP_classification,
    hiCAP_regression,
)
from ._solver import (
    _callable_accepts_keyword, _native_tree_eligible, coefficient_path_solution,
    iter_fitted_tree_solutions, resolve_solver,
    solve_design, solve_fitted_tree, validate_solver,
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
    all_groups: list[np.ndarray] = []
    descendants = {}
    stack = [(idx, False)]
    while stack:
        node, visited = stack.pop()
        if tree.feature[node] == -2:
            continue
        left, right = tree.children_left[node], tree.children_right[node]
        if not visited:
            stack.extend(((node, True), (right, False), (left, False)))
            continue
        positions = ([id_lookup[node]] if node in id_lookup else [])
        positions += descendants.pop(left, []) + descendants.pop(right, [])
        descendants[node] = positions
        if positions:
            all_groups.append(np.asarray(positions, dtype=int))
    return all_groups


def _collect_internal_node_ids(tree) -> np.ndarray:
    node_ids: list[int] = []
    stack = [0]
    while stack:
        node_idx = stack.pop()
        if tree.feature[node_idx] == -2:
            continue
        node_ids.append(node_idx)
        stack.extend((tree.children_right[node_idx], tree.children_left[node_idx]))
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
        or result.get("certified", True) is not True
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


class SHSTree(BaseEstimator):
    """Tree pruned by hierarchical-LASSO support, with optional shrinkage.

    Penalized coefficients or exact structural knots select a hierarchy-safe tree
    topology. Predictions continue to use the fitted CART node values on that
    topology, optionally transformed by hierarchical shrinkage; they are not
    ``X_opt @ beta_stars_``. The sparse penalty ``sp_alpha`` is relative to
    mean weighted loss, while ``reg_param`` retains HST's pseudo-count
    interpretation through weighted node sample counts.

    For a single squared-error regression tree, ``reg_param="gcv"`` (or
    ``None``) selects node-based HS after pruning. This is conditional-on-the-
    tree GCV, not cross-validation of tree construction or pruning. Only
    uniform observation weights are supported. The selected strength is in
    ``reg_param_`` (possibly infinity for root-mean predictions), and
    ``gcv_results_`` records the search without changing ``reg_param``.

    ``solver='auto'`` uses exact structural knots for eligible single
    regression or classification trees under their fitting measure and an
    infinity penalty. Other infinity-regression problems use ``'proximal'``;
    other supported binary-classification/two-norm problems use ``'apa_apg2'``.
    ``'coefficient_path'`` exposes a full regression coefficient path or
    positive-penalty classification samples. Structural fits leave
    ``coef_`` as None; coefficient solvers expose original-node coefficients
    in ``coef_``, ``coef_node_ids_`` and ``intercept_`` for single trees.

    Parameters
    ----------
    estimator_ : sklearn tree or forest, optional
        Tree template. Each fit uses a fresh clone; ``prefit=True`` instead
        prunes a copy of an already fitted estimator. With no template, use
        a decision tree with 20 leaves unless ``max_leaf_nodes`` overrides it.
    sp_alpha : float, default=1
        Nonnegative hiCAP penalty relative to mean weighted loss. Zero keeps
        every original split, including splits with zero coefficient.
    reg_param : float, "gcv", or None, default=0
        HS pseudo-count; zero disables HS. Regression-only ``"gcv"`` or None
        selects conditional fixed-tree GCV after pruning, requiring uniform
        positive weights. Subclasses may use a different numeric default.
    solver : {"auto", "topology", "proximal", "coefficient_path", "hicap", "apa_apg2"}, default="auto"
        ``topology`` computes exact structural events only. ``proximal``
        computes coefficients at the selected penalty. ``coefficient_path``
        computes a complete regression path, or nonlinear classification
        samples at positive structural knots and the requested penalty.
        ``hicap`` is the regression-only reference homotopy. ``apa_apg2`` is
        the approximate legacy solver (binary classification only). Native
        solvers require an unconstrained single-output CART tree, its fitting
        rows/weights, ord=inf, and zero support tolerance. Regression node
        values must be means. Classification uses logistic/softmax loss with
        the class-range descendant-group penalty.
    ord : {2, numpy.inf, "inf"}, default=numpy.inf
        Norm within each descendant group. Exact path solvers require infinity.
    max_iter : int, default=2000
        APA/FISTA iteration budget, per sampled classification penalty. For
        regression ``coefficient_path`` and ``hicap``, this instead caps path
        events. Native topology and regression proximal sweeps are noniterative.
    tol : float, default=1e-6
        Positive numerical convergence/certificate tolerance. Exact paths are
        certified in floating-point arithmetic, not symbolic arithmetic.
    support_tol : float or None, default=None
        Relative coefficient-zero threshold, scaled by response standard
        deviation (one for classification). Native paths use structural
        events when None or zero. General solvers default to a numerical
        threshold based on ``tol``; literal zero there is roundoff-sensitive.
        A positive threshold disables native structural-path selection.
    prune_set : {"auto", "full", "ib", "oob"}, default="auto"
        Pruning observations: full data for a single tree; auto uses OOB
        observations for a bootstrap forest. IB/OOB membership is reconstructed
        per tree and cannot be verified for externally supplied prefit forests.
    max_leaf_nodes : int or None, default=None
        Optional leaf cap overriding the unfitted tree template.
    random_state : int or None, default=None
        Seed for tree fitting and, in CV subclasses, fold construction.
    gamma1, a : float, default=1
        Positive legacy APA smoothing-schedule controls; unused by exact solvers.
    prefit : bool, default=False
        Whether the supplied estimator is already fitted. Native geometry is
        not assumed for external prefit data. Incompatible with CV tuning.

    Attributes
    ----------
    estimator_ : fitted sklearn estimator
        Independently owned, compact pruned tree, after any HS transformation.
    solver_ : str
        Resolved final-fit backend, distinct from the constructor's "auto".
    coef_, coef_node_ids_, intercept_ : arrays, scalar, or None
        Optional penalized coefficients of unnormalized original-tree stumps,
        original split IDs, and intercept, before HS. Not prediction weights
        of the compact pruned tree. Multiclass coefficients have shape
        (n_splits, n_classes), with a vector intercept and class order in
        ``classes_``. Topology-only fits have ``coef_=None``; classifier
        topology or zero-penalty fits also leave ``intercept_=None`` because
        no finite optimized logits are computed at zero penalty.
    pruning_path_ : TreeTopologyPath or None
        Exact structural events, when eligible fitted geometry is available.
    coefficient_path_ : RegularizationPath or None
        Complete regression homotopy path, or nonlinear classification samples
        with ``exact=False``. Classification interpolation is approximate;
        samples exclude zero and can be absent for a root-only zero-penalty fit.
    beta_stars_ : list of arrays
        Legacy intercept-first per-tree coefficients; empty for topology fits.
    optimization_results_ : list
        Per-tree numerical diagnostics. Inspect ``optimization_certified_``
        as well as ``optimization_stable_`` for approximate solves.
    reg_param_ : float
        Effective HS strength, possibly infinity for a GCV root-only limit.
    gcv_results_ : dict or None
        Conditional GCV candidates, scores, and effective degrees of freedom.
    complexity_ : int
        Number of retained split nodes (summed over a forest).
    """

    def __init__(
        self,
        estimator_: BaseEstimator | None = None,
        sp_alpha: float = 1,
        reg_param: float | str | None = 0,
        gamma1: float = 1.0,
        a: float = 1.0,
        max_iter: int = 2000,
        tol: float = 1e-6,
        ord: int | str = np.inf,
        prune_set: str = "auto",
        max_leaf_nodes: int | None = None,
        random_state: int | None = None,
        support_tol: float | None = None,
        prefit: bool = False,
        solver: str = "auto",
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
        self.solver = solver
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
        validate_solver(self.solver)
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

        if self.reg_param is None or (
            isinstance(self.reg_param, str) and self.reg_param == "gcv"
        ):
            if not (allow_unset and self.reg_param is None):
                if not isinstance(self, RegressorMixin):
                    raise ValueError("Automatic GCV HS supports regression only")
                if self.shrinkage_scheme_ != "node_based":
                    raise ValueError("Automatic GCV HS requires node_based shrinkage")
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
            "solver": self.solver,
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
            "coef_",
            "coef_node_ids_",
            "intercept_",
            "coefficient_path_",
            "pruning_path_",
            "solver_",
            "cv_solver_",
            "cv_path_mode_",
            "cv_sp_alphas_",
            "cv_path_results_",
            "cv_n_pruning_states_",
            "classes_",
            "complexity_",
            "cv_complexities_",
            "cv_optimization_certified_",
            "cv_optimization_results_",
            "cv_optimization_stable_",
            "cv_params_",
            "cv_reg_params_",
            "cv_score_se_",
            "cv_score_std_",
            "cv_scores_",
            "cv_weight_fractions_",
            "feature_names_in_",
            "feature_names_",
            "gcv_results_",
            "best_index_",
            "best_score_",
            "best_sp_alpha_",
            "best_reg_param_",
            "mean_complexity_",
            "n_features_in_",
            "n_iter_",
            "oob_attributes_invalidated_",
            "one_se_candidate_mask_",
            "optimization_certified_",
            "optimization_results_",
            "optimization_stable_",
            "prune_set_",
            "reg_param_",
            "scores_",
            "selected_index_",
            "selection_rule_",
            "selection_threshold_",
            "sp_alpha_",
            "support_thresholds_",
        ):
            if hasattr(self, fitted_attribute):
                delattr(self, fitted_attribute)
        self._wrapper_is_fitted = False
        return self

    def _resolved_solver(self):
        return resolve_solver(
            self.solver, estimator=self.estimator_, ord=self.ord,
            matched_training=(
                not self.prefit
                or getattr(self, "_pruning_training_measure_matched", False)
            ),
            support_tol=self.support_tol,
            custom_solver=self.hiCAP not in (hiCAP_regression, hiCAP_classification),
        )

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
        sample_weight = np.asarray(sample_weight)
        if np.iscomplexobj(sample_weight):
            raise ValueError("sample_weight must contain real values")
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
        self.gcv_results_ = None
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
        elif hasattr(self, "feature_names_in_"):
            del self.feature_names_in_

        sample_weight = self._validated_sample_weight(sample_weight, len(y))
        is_classifier = isinstance(self, ClassifierMixin)
        if is_classifier and len(self.classes_) < 2:
            raise ValueError(
                "Sparse classification requires at least two classes; got one class."
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
        self._shrink(X, y, sample_weight=sample_weight)
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
            getattr(self, "reg_param_", self.reg_param) is not None
            and getattr(self, "reg_param_", self.reg_param) > 0
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
        coefficient_path=None,
    ):
        if sp_alpha is None or (
            sp_alpha <= 0 and self.solver_ not in {"proximal", "hicap"}
        ):
            return tree

        ids = _collect_internal_node_ids(tree)
        if ids.size == 0:
            # Degenerate tree with a single leaf: nothing to prune.
            intercept = float(tree.value[0, 0, 0]) if isinstance(self, RegressorMixin) else 0.0
            if isinstance(self, RegressorMixin) and (
                sample_weight is None or np.any(np.asarray(sample_weight) > 0)
            ):
                intercept = float(np.average(y, weights=sample_weight))
            self.beta_stars_.append(np.array([intercept]))
            if isinstance(self, RegressorMixin) and hasattr(self.estimator_, "tree_"):
                self.coef_ = np.empty(0)
                self.coef_node_ids_ = ids
                self.intercept_ = intercept
                if self.solver_ == "hicap":
                    from .optimization import RegularizationPath

                    self.coefficient_path_ = RegularizationPath(
                        lambdas=np.array([0.0]), coefficients=np.empty((1, 0)),
                        intercepts=np.array([intercept]), exact=True,
                        method="hicap", metadata={"tree_node_ids": ()},
                    )
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

        if coefficient_path is None:
            X_tree = tree_feature_transform(make_stumps(tree), X)
            groups = _find_subtrees(tree, 0, ids)
            beta_star, solver_info, path = solve_design(
                X_tree, y, [group - 1 for group in groups], sp_alpha, self.solver_,
                point_solver=self.hiCAP, sample_weight=sample_weight,
                beta_init=beta_init, gamma1=self.gamma1, a=self.a,
                max_iter=self.max_iter, tol=self.tol, ord=self.ord, verbose=verbose,
            )
        else:
            beta_star, solver_info, path = coefficient_path_solution(coefficient_path, sp_alpha)
        if hasattr(self.estimator_, "tree_"):
            self.coefficient_path_ = path

        self.beta_stars_.append(beta_star)
        if hasattr(self.estimator_, "tree_"):
            self.coef_ = beta_star[1:].copy()
            self.intercept_ = float(beta_star[0])
            self.coef_node_ids_ = ids.copy()
        self.optimization_results_.append(solver_info)
        if solver_info is not None:
            self.n_iter_ = max(self.n_iter_, int(solver_info["n_iter"]))

        # Matched native geometry supplies exact structural support. General
        # coefficient solves use a response-scaled numerical threshold unless
        # the caller requests a specific tolerance (including literal zero).
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
        relative_tol = self.support_tol
        if relative_tol is None:
            relative_tol = (
                0.0 if self.pruning_path_ is not None
                else max(10 * self.tol, np.sqrt(np.finfo(float).eps))
            )
        support_threshold = relative_tol * response_scale
        self.support_thresholds_.append(support_threshold)

        if sp_alpha == 0:
            return tree

        if self.pruning_path_ is not None:
            # A generic homotopy's affine coefficients can contain roundoff at
            # an exact knot. Use the fitted-tree structural events to choose
            # its topology, just as CV does, without thresholding away a small
            # genuinely active coefficient immediately before that knot.
            retained = set(self.pruning_path_.tree_nodes_at(sp_alpha))
            candidate_ids = set(ids) - retained
        else:
            # Bottom-up support propagation avoids materializing descendant
            # groups when reading a cached coefficient path.
            active = dict(zip(ids, np.abs(beta_star[1:]) > support_threshold))
            for node in reversed(ids):
                active[node] = (
                    active[node] or active.get(tree.children_left[node], False)
                    or active.get(tree.children_right[node], False)
                )
            candidate_ids = {node for node in ids if not active[node]}

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

        pruned_ids = [node for node in candidate_ids if parent.get(node) not in candidate_ids]

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
        fitted_solution=None,
        coefficient_path=None,
    ):
        self.beta_stars_ = []
        self.support_thresholds_ = []
        self.optimization_results_ = []
        self.n_iter_ = 0
        self.coef_ = self.coef_node_ids_ = self.intercept_ = None
        self.pruning_path_ = self.coefficient_path_ = None
        self.solver_ = self._resolved_solver()
        if (isinstance(self, ClassifierMixin) and self.solver_ == "apa_apg2"
                and len(self.classes_) != 2):
            raise ValueError(
                "APA-APG2 supports only binary classification. Multiclass "
                "pruning requires an eligible fitted CART tree, ord=inf, "
                "support_tol=None or 0, and a native solver."
            )
        if (
            hasattr(self.estimator_, "tree_")
            and self._resolved_prune_set() != "full"
        ):
            raise ValueError(
                "prune_set='ib' and prune_set='oob' require a bootstrap "
                "forest; use prune_set='full' for a single tree."
            )
        native = self.solver_ in {"topology", "coefficient_path"}
        if self.solver_ in {"proximal", "hicap"}:
            fitted_geometry = resolve_solver(
                "auto", estimator=self.estimator_, ord=self.ord,
                matched_training=(
                    not self.prefit
                    or getattr(self, "_pruning_training_measure_matched", False)
                ),
                support_tol=self.support_tol,
            ) == "topology"
            native = self.solver_ == "proximal" and fitted_geometry
            if self.solver_ == "hicap" and fitted_geometry:
                from .fitted_tree import fitted_tree_linf_exact_topology_path

                self.pruning_path_ = (
                    fitted_solution.topology_path if fitted_solution is not None
                    else fitted_tree_linf_exact_topology_path(self.estimator_)
                )
        if native:
            result = fitted_solution if fitted_solution is not None else solve_fitted_tree(
                self.estimator_, self.sp_alpha, self.solver_,
                tol=self.tol, max_iter=self.max_iter,
            )
            self.pruning_path_ = result.topology_path
            self.coefficient_path_ = result.coefficient_path
            self.coef_ = result.coefficients
            self.coef_node_ids_ = result.node_ids
            self.intercept_ = result.intercept
            if result.coefficients is not None:
                self.beta_stars_.append(
                    np.vstack((result.intercept, result.coefficients))
                    if result.coefficients.ndim == 2
                    else np.r_[result.intercept, result.coefficients]
                )
            self.support_thresholds_.append(0.0)
            self.optimization_results_.append(result.info)
            self.n_iter_ = int(result.info["n_iter"])
            retained = set(result.retained_node_ids)
            tree = self.estimator_.tree_
            for node in result.node_ids:
                if node not in retained:
                    tree.children_left[node] = tree.children_right[node] = TREE_LEAF
                    tree.feature[node] = -2
                    tree.threshold[node] = -2
            if len(retained) != len(result.node_ids):
                _compact_tree(tree)
            return
        if self.sp_alpha is None or (
            self.sp_alpha <= 0 and self.solver not in {"proximal", "hicap"}
        ):
            return
        if hasattr(self.estimator_, "tree_"):
            self._prune_tree(
                self.estimator_.tree_,
                X,
                y,
                self.sp_alpha,
                sample_weight=sample_weight,
                decimals=decimals,
                beta_init=beta_init,
                verbose=verbose,
                coefficient_path=coefficient_path,
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
        """Shrink in preorder without a recursion limit on deep retained trees."""
        classification = isinstance(self, ClassifierMixin)
        stack = [(i, parent_val, parent_num, cum_sum)]
        while stack:
            node, parent_value, parent_mass, cumulative = stack.pop()
            left, right = tree.children_left[node], tree.children_right[node]
            mass = tree.weighted_n_node_samples[node]
            value = tree.value[node].copy()
            if classification:
                # Old sklearn trees store counts, newer trees probabilities.
                totals = value.sum(axis=1, keepdims=True)
                value = np.divide(value, totals, out=np.zeros_like(value), where=totals != 0)
            if parent_value is None:
                prediction = value
            elif self.shrinkage_scheme_ == "node_based":
                prediction = cumulative + (value - parent_value) / (1 + reg_param / parent_mass)
            elif self.shrinkage_scheme_ == "constant":
                prediction = cumulative + (value - parent_value) / (1 + reg_param)
            else:
                prediction = cumulative
            if self.shrinkage_scheme_ in ("node_based", "constant"):
                tree.value[node] = prediction
            elif left == TREE_LEAF:
                root_value = value if node == 0 else tree.value[0]
                tree.value[node] = root_value + (value - root_value) / (1 + reg_param / mass)
            else:
                tree.value[node] = value
            if left != TREE_LEAF:
                # Values carried to siblings are read-only; each child creates
                # its own prediction rather than mutating the parent's array.
                stack.append((right, value, mass, prediction))
                stack.append((left, value, mass, prediction))
        return tree

    def _shrink(self, X, y, sample_weight=None):
        self.gcv_results_ = None
        automatic = self.reg_param is None or (
            isinstance(self.reg_param, str) and self.reg_param == "gcv"
        )
        if automatic:
            if not isinstance(self, RegressorMixin):
                raise ValueError("Automatic GCV HS supports regression only")
            if self.shrinkage_scheme_ != "node_based":
                raise ValueError("Automatic GCV HS requires node_based shrinkage")
            if self.prefit:
                # Verify that the supplied observations really describe the
                # stored fitting statistics; held-out data are not a GCV fit.
                self.reg_param_, self.gcv_results_ = get_gcv_reg_param(
                    self.estimator_, X, y, sample_weight=sample_weight,
                    return_info=True,
                )
            else:
                self.reg_param_, self.gcv_results_ = select_hs_reg_param(
                    self.estimator_, sample_weight=sample_weight, y=y
                )
            apply_node_based_hs(self.estimator_.tree_, self.reg_param_)
            return
        self.reg_param_ = self._validate_finite_scalar(
            self.reg_param, "reg_param", nonnegative=True
        )
        if self.reg_param_ == 0:
            return
        if (isinstance(self.estimator_, DecisionTreeRegressor)
                and self.shrinkage_scheme_ == "node_based"):
            apply_node_based_hs(self.estimator_.tree_, self.reg_param_)
        elif hasattr(self.estimator_, "tree_"):
            self._shrink_tree(self.estimator_.tree_, self.reg_param_)
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
                self._shrink_tree(t.tree_, self.reg_param_)

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
    """Sparse-pruned regression tree with hierarchical shrinkage.

    Parameters
    ----------
    sp_alpha : float, default=1
        HiCAP pruning penalty relative to mean weighted squared loss.
    reg_param : float, "gcv", or None, default=1
        HS pseudo-count; zero disables HS. "gcv" or None chooses conditional
        fixed-tree GCV after pruning; it does not tune tree construction.
    solver : str, default="auto"
        Defaults to exact structural pruning for eligible infinity-penalty
        trees. "proximal" exposes point coefficients; "coefficient_path"
        and "hicap" expose complete coefficient paths.

    Other parameters and fitted attributes are documented in :class:`SHSTree`.
    In particular, ``max_iter`` caps path events for homotopy solvers and
    ``coef_`` describes original-tree stumps, not the final tree's predictions.

    Examples
    --------
    >>> model = SHSTreeRegressor(sp_alpha=0.1, reg_param="gcv", random_state=0)
    >>> model.get_params()["solver"]
    'auto'

    See Also
    --------
    SHSTree : Shared parameters, solver restrictions, and fitted attributes.
    SHSTreeRegressorCV : Select pruning and HS strengths using held-out scores.
    """
    def __init__(
        self,
        estimator_: BaseEstimator | None = None,
        sp_alpha: float = 1,
        reg_param: float | str | None = 1,
        prune_set: str = "auto",
        gamma1: float = 1.0,
        a: float = 1.0,
        max_iter: int = 2000,
        tol: float = 1e-6,
        ord: int | str = np.inf,
        max_leaf_nodes: int | None = None,
        random_state: int | None = None,
        support_tol: float | None = None,
        prefit: bool = False,
        solver: str = "auto",
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
            solver=solver,
        )
        self.hiCAP = hiCAP_regression


class SHSTreeClassifier(ClassifierMixin, SHSTree):
    """Sparse-pruned binary/multiclass tree with hierarchical shrinkage.

    ``solver="auto"`` uses exact structural knots for eligible fitted CART
    trees with the infinity penalty. ``proximal`` exposes positive-penalty
    logistic/softmax coefficients; ``coefficient_path`` adds warm-started
    samples at positive structural knots and the supplied penalty, not a
    piecewise-linear path. ``apa_apg2`` remains available for binary problems.
    ``reg_param`` is a numeric HS pseudo-count (default 1); GCV and the
    regression-only ``hicap`` reference solver are unsupported.

    See :class:`SHSTree` for parameters and fitted attributes. Predictions use
    retained node class probabilities after HS, not penalized logistic weights.
    """
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
        ord: int | str = np.inf,
        max_leaf_nodes: int | None = None,
        random_state: int | None = None,
        support_tol: float | None = None,
        prefit: bool = False,
        solver: str = "auto",
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
            solver=solver,
        )
        self.hiCAP = hiCAP_classification

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        if getattr(tags, "classifier_tags", None) is not None:
            tags.classifier_tags.multi_class = bool(
                self.solver in {"auto", "topology", "proximal", "coefficient_path"}
                and self.ord in ("inf", np.inf)
                and self.support_tol in (None, 0)
                and not self.prefit
                and self.hiCAP is hiCAP_classification
                and _native_tree_eligible(
                    self._base_estimator_template(), require_fitted=False
                )
            )
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
                f"{name} must contain numeric values; use reg_param='gcv' "
                "or reg_param_list='gcv' for automatic HS"
            )
        SHSTree._validate_finite_scalar(
            value, name, nonnegative=True
        )
    return candidates


def _auto_alpha_grid(values) -> bool:
    if isinstance(values, str):
        if values != "auto":
            raise ValueError("sp_alpha_list must be 'auto' or a numeric sequence")
        return True
    return False


def _alpha_grid_or_legacy(values):
    return _DEFAULT_SP_ALPHA_GRID if _auto_alpha_grid(values) else values


def _validated_hs_grid(values, *, classification):
    if not isinstance(values, str) and values is not None:
        try:
            values = list(values)
        except TypeError as exc:
            raise ValueError("reg_param_list must be 'gcv' or a numeric sequence") from exc
    automatic = (isinstance(values, str) and values == "gcv") or (
        isinstance(values, list) and len(values) == 1
        and isinstance(values[0], str) and values[0] == "gcv"
    )
    if automatic:
        if classification:
            raise ValueError("Automatic GCV HS supports regression only")
        return ["gcv"]
    return _validated_nonnegative_grid(values, "reg_param_list")


def _uses_structural_cv(estimator) -> bool:
    return bool(
        _auto_alpha_grid(estimator.sp_alpha_list)
        and estimator.solver != "apa_apg2"
        and estimator.hiCAP in (hiCAP_regression, hiCAP_classification)
        and not (isinstance(estimator, ClassifierMixin) and estimator.solver == "hicap")
        and estimator.ord in ("inf", np.inf)
        and estimator.support_tol in (None, 0)
        and estimator.prune_set in ("auto", "full")
        and _native_tree_eligible(
            estimator._base_estimator_template(), require_fitted=False
        )
    )


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


def _validate_cv_policy(selection_rule, reg_param_mode) -> None:
    if selection_rule not in {"one_se", "best"}:
        raise ValueError("selection_rule must be one of {'one_se', 'best'}")
    if reg_param_mode not in {"normalized", "raw"}:
        raise ValueError(
            "reg_param_mode must be one of {'normalized', 'raw'}"
        )


def _effective_weight_sum(sample_weight, n_samples) -> float:
    if sample_weight is None:
        return float(n_samples)
    return float(np.sum(sample_weight))


def _fold_reg_param(
    estimator,
    reg_param: float,
    fold_weight_fraction: float,
) -> float:
    """Return the shrinkage parameter used inside one CV training fold.

    In normalized mode, grid values are full-data reference values. Scaling
    them by the training-fold fraction keeps count-based HST shrinkage at
    approximately the same relative strength when tree root masses scale with
    total training mass. This is dataset-mass normalization, not per-tree root
    normalization; notably, a forest with fixed integer ``max_samples`` need
    not have scaling root masses. Constant shrinkage is already independent of
    node sample counts.
    """
    if reg_param == "gcv":
        return "gcv"
    if (
        estimator.reg_param_mode == "normalized"
        and estimator.shrinkage_scheme_ != "constant"
    ):
        return float(reg_param) * fold_weight_fraction
    return float(reg_param)


def _initialize_cv_tracking(estimator, param_list) -> None:
    n_candidates = len(param_list)
    estimator.cv_params_ = [
        {"sp_alpha": sp_alpha, "reg_param": reg_param}
        for sp_alpha, reg_param in param_list
    ]
    estimator.cv_scores_ = [[] for _ in param_list]
    estimator.cv_complexities_ = [[] for _ in param_list]
    estimator.cv_reg_params_ = [[] for _ in param_list]
    # Indexed as [candidate][fold][tree]. Cached pruning diagnostics are
    # duplicated across shrinkage values to preserve the public indexing.
    estimator.cv_optimization_results_ = [
        [] for _ in range(n_candidates)
    ]
    estimator.cv_weight_fractions_ = []


def _reset_cv_path_tracking(estimator) -> None:
    # A direct refit can switch from automatic structural CV to a grid.
    for name in (
        "cv_solver_", "cv_path_mode_", "cv_sp_alphas_",
        "cv_path_results_", "cv_n_pruning_states_",
    ):
        if hasattr(estimator, name):
            delattr(estimator, name)


def _select_structural_cv(estimator, X, y, sample_weight, n_splits, fit_args, fit_kwargs):
    """Share fold-local path scoring and final parameter selection across tasks."""
    from ._cv import evaluate_structural_cv

    classification = isinstance(estimator, ClassifierMixin)
    reg_params = _validated_hs_grid(estimator.reg_param_list, classification=classification)
    if reg_params == ["gcv"] and estimator.shrinkage_scheme_ != "node_based":
        raise ValueError("Automatic GCV HS requires node_based shrinkage")
    scorer = _resolve_cv_scorer(
        fit_kwargs.pop("scoring", estimator.scoring), classification=classification
    )
    param_list = evaluate_structural_cv(
        estimator, X=X, y=y, sample_weight=sample_weight,
        reg_param_list=reg_params, scorer=scorer, n_splits=n_splits,
        fit_args=fit_args, fit_kwargs=fit_kwargs,
    )
    estimator.cv_optimization_certified_ = True
    estimator.cv_optimization_stable_ = True
    _finalize_cv_selection(estimator, param_list)
    estimator.sp_alpha = estimator.sp_alpha_
    estimator.reg_param = estimator.reg_param_


def _evaluate_cached_cv_fold(
    estimator,
    wrapper_class,
    base_estimator,
    X_in,
    y_prune,
    y_shrink,
    X_out,
    y_out,
    weight_in,
    weight_out,
    sp_alpha_list,
    reg_param_list,
    scorer,
    fold_weight_fraction,
    decimals,
    classes=None,
) -> None:
    """Prune once per alpha, then score fresh shrinkage copies."""
    n_reg_params = len(reg_param_list)
    resolved_solver = resolve_solver(
        estimator.solver, estimator=base_estimator, ord=estimator.ord,
        matched_training=isinstance(base_estimator, (DecisionTreeRegressor, DecisionTreeClassifier)),
        support_tol=estimator.support_tol,
        custom_solver=estimator.hiCAP not in (hiCAP_regression, hiCAP_classification),
    )
    native = resolved_solver in {"topology", "coefficient_path"} or (
        resolved_solver == "proximal"
        and _native_tree_eligible(base_estimator)
        and estimator.support_tol in (None, 0)
    )
    reference_native = (
        resolved_solver == "hicap" and _native_tree_eligible(base_estimator)
        and estimator.support_tol in (None, 0)
    )
    solutions = iter_fitted_tree_solutions(
        base_estimator, sp_alpha_list, "topology" if reference_native else resolved_solver,
        tol=estimator.tol, max_iter=estimator.max_iter,
    ) if native or reference_native else None
    reference_path = None
    for alpha_index, sp_alpha in enumerate(sp_alpha_list):
        pruned = wrapper_class(
            estimator_=deepcopy(base_estimator),
            sp_alpha=sp_alpha,
            reg_param=0,
            prune_set=estimator.prune_set,
            random_state=estimator.random_state,
            gamma1=estimator.gamma1,
            a=estimator.a,
            max_iter=estimator.max_iter,
            tol=estimator.tol,
            ord=estimator.ord,
            support_tol=estimator.support_tol,
            prefit=True,
            solver=estimator.solver,
        )
        pruned._pruning_training_measure_matched = isinstance(
            base_estimator, (DecisionTreeRegressor, DecisionTreeClassifier)
        )
        pruned.hiCAP = estimator.hiCAP
        pruned.shrinkage_scheme_ = estimator.shrinkage_scheme_
        if classes is not None:
            pruned.classes_ = classes
        pruned.n_features_in_ = X_in.shape[1]
        pruned._prune(
            X=X_in,
            y=y_prune,
            sample_weight=weight_in,
            decimals=decimals,
            beta_init=None,
            fitted_solution=None if solutions is None else next(solutions),
            coefficient_path=reference_path,
        )
        if resolved_solver == "hicap" and hasattr(base_estimator, "tree_"):
            reference_path = pruned.coefficient_path_
        pruned.prune_set_ = pruned._resolved_prune_set()
        pruned._update_optimization_diagnostics(warn=False)
        estimator.cv_solver_ = pruned.solver_
        pruned._update_estimator_metadata()
        pruning_results = deepcopy(pruned.optimization_results_)

        for reg_index, reg_param in enumerate(reg_param_list):
            candidate_index = alpha_index * n_reg_params + reg_index
            effective_reg_param = _fold_reg_param(
                estimator, reg_param, fold_weight_fraction
            )
            # Path arrays are immutable snapshots. Share those potentially
            # O(p*K) results; each scored candidate still owns its tree and
            # other mutable estimator state.
            path_memo = {
                id(path): path for path in (pruned.pruning_path_, pruned.coefficient_path_)
                if path is not None
            }
            candidate = deepcopy(pruned, path_memo)
            candidate.reg_param = effective_reg_param
            candidate._shrink(X=X_in, y=y_shrink, sample_weight=weight_in)
            candidate._update_estimator_metadata()

            score = _score_with_optional_sample_weight(
                scorer, candidate, X_out, y_out, weight_out
            )
            estimator.cv_scores_[candidate_index].append(float(score))
            estimator.cv_complexities_[candidate_index].append(
                float(candidate.complexity_)
            )
            estimator.cv_reg_params_[candidate_index].append(
                candidate.reg_param_
            )
            estimator.cv_optimization_results_[candidate_index].append(
                deepcopy(pruning_results)
            )


def _finalize_cv_selection(estimator, param_list) -> None:
    estimator.cv_scores_ = np.asarray(estimator.cv_scores_, dtype=float)
    estimator.cv_complexities_ = np.asarray(
        estimator.cv_complexities_, dtype=float
    )
    estimator.cv_reg_params_ = np.asarray(
        estimator.cv_reg_params_, dtype=float
    )
    estimator.cv_weight_fractions_ = np.asarray(
        estimator.cv_weight_fractions_, dtype=float
    )

    with np.errstate(over="ignore", invalid="ignore"):
        estimator.scores_ = np.mean(estimator.cv_scores_, axis=1)
        estimator.cv_score_std_ = np.std(
            estimator.cv_scores_, axis=1, ddof=1
        )
        estimator.cv_score_se_ = estimator.cv_score_std_ / np.sqrt(
            estimator.cv_scores_.shape[1]
        )
        estimator.mean_complexity_ = np.mean(
            estimator.cv_complexities_, axis=1
        )

    finite_candidates = np.all(
        np.isfinite(estimator.cv_scores_), axis=1
    ) & np.isfinite(
        estimator.scores_
    ) & np.isfinite(
        estimator.cv_score_se_
    ) & np.isfinite(
        estimator.mean_complexity_
    )
    if not np.any(finite_candidates):
        raise ValueError("Cross-validation produced no finite scores")
    finite_indices = np.flatnonzero(finite_candidates)
    estimator.best_index_ = int(
        finite_indices[
            np.argmax(estimator.scores_[finite_candidates])
        ]
    )
    estimator.best_score_ = float(
        estimator.scores_[estimator.best_index_]
    )
    estimator.best_sp_alpha_, estimator.best_reg_param_ = param_list[
        estimator.best_index_
    ]
    estimator.selection_threshold_ = (
        estimator.best_score_
        - estimator.cv_score_se_[estimator.best_index_]
    )
    estimator.one_se_candidate_mask_ = (
        finite_candidates
        & np.isfinite(estimator.mean_complexity_)
        & (estimator.scores_ >= estimator.selection_threshold_)
    )

    if estimator.selection_rule == "best":
        selected_index = estimator.best_index_
    else:
        eligible = np.flatnonzero(estimator.one_se_candidate_mask_)
        minimum_complexity = np.min(
            estimator.mean_complexity_[eligible]
        )
        simplest = eligible[
            np.isclose(
                estimator.mean_complexity_[eligible],
                minimum_complexity,
                rtol=0,
                atol=1e-12,
            )
        ]
        simplest_best_score = np.max(estimator.scores_[simplest])
        score_tied = simplest[
            np.isclose(
                estimator.scores_[simplest],
                simplest_best_score,
                rtol=0,
                atol=1e-12,
            )
        ]
        # Split count cannot distinguish shrinkage values on the same
        # topology. When their CV scores are numerically tied, prefer the
        # stronger shrinkage and then the stronger sparse penalty.
        selected_index = int(
            max(
                score_tied,
                key=lambda index: (
                    param_list[index][1],
                    param_list[index][0],
                    -index,
                ),
            )
        )

    estimator.selected_index_ = selected_index
    estimator.selection_rule_ = estimator.selection_rule
    estimator.sp_alpha_, estimator.reg_param_ = param_list[selected_index]


class SHSTreeClassifierCV(SHSTreeClassifier):
    """Stratified structural-path CV for binary/multiclass pruning and HS.

    Parameters follow :class:`SHSTreeRegressorCV`, but folds are stratified
    and HS strengths must be numeric (no GCV). Scoring defaults to accuracy.
    ``sp_alpha_list="auto"`` scores training-fold structural knots for
    eligible CART trees; numeric lists or explicit APA request grid CV.
    Class weights are recomputed on each training fold. Optional coefficient
    solvers run only for the final fit during automatic structural CV.
    ``cv_path_mode_``, ``cv_params_``, ``cv_scores_``, and
    ``cv_optimization_results_`` expose candidate/fold diagnostics.

    The default one-standard-error rule chooses the lowest observed mean split
    complexity whose score is within one standard error of the best candidate.
    Score and complexity ties prefer stronger shrinkage and sparse penalties.
    In normalized reg-parameter mode, grid values are full-data reference
    values and count-based shrinkage is scaled by each fold's effective
    training-weight fraction. This is dataset-mass, not per-tree root-mass,
    normalization. Set ``selection_rule='best'`` and
    ``reg_param_mode='raw'`` for the historical joint-argmax behavior.
    """

    def __init__(
        self,
        estimator_: BaseEstimator | None = None,
        sp_alpha_list: Sequence[float] | str = "auto",
        reg_param_list: Sequence[float] = (0, 0.1, 1, 10, 50, 100, 500),
        max_leaf_nodes: int = 20,
        cv: int = 3,
        scoring=None,
        selection_rule: str = "one_se",
        reg_param_mode: str = "normalized",
        prune_set: str = "auto",
        random_state: int | None = None,
        gamma1: float = 1.0,
        a: float = 1.0,
        max_iter: int = 2000,
        tol: float = 1e-6,
        ord: int | str = np.inf,
        support_tol: float | None = None,
        prefit: bool = False,
        solver: str = "auto",
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
            solver=solver,
        )
        self.sp_alpha_list = sp_alpha_list
        self.reg_param_list = reg_param_list
        self.cv = cv
        self.scoring = scoring
        self.selection_rule = selection_rule
        self.reg_param_mode = reg_param_mode

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
        _reset_cv_path_tracking(self)
        self._validate_hyperparameters(allow_unset=True)
        _validate_cv_policy(self.selection_rule, self.reg_param_mode)
        self._fresh_estimator()
        feature_names = kwargs.pop("feature_names", None)
        X, y, feature_names = _prepare_cv_data(X, y, feature_names)
        check_classification_targets(y)
        classes, y_encoded = np.unique(y, return_inverse=True)
        if len(classes) < 2:
            raise ValueError(
                "Sparse classification requires at least two classes; got one class."
            )
        sample_weight = self._validated_sample_weight(sample_weight, len(y))
        n_splits = _validated_n_splits(self.cv)
        positive_weight = (
            np.ones(len(y), dtype=bool)
            if sample_weight is None
            else sample_weight > 0
        )
        effective_class_counts = np.bincount(
            y_encoded[positive_weight], minlength=len(classes)
        )
        if np.any(effective_class_counts < n_splits):
            raise ValueError(
                "Each class must have at least cv positive-weight samples for "
                "stratified sparse-pruning CV"
            )
        if _uses_structural_cv(self):
            _select_structural_cv(self, X, y, sample_weight, n_splits, args, kwargs)
            return super().fit(
                X=X, y=y, sample_weight=sample_weight, decimals=decimals,
                *args, feature_names=feature_names, **kwargs,
            )
        sp_alpha_list = _validated_nonnegative_grid(
            _alpha_grid_or_legacy(self.sp_alpha_list), "sp_alpha_list"
        )
        self.cv_path_mode_ = "grid"
        self.cv_sp_alphas_ = np.asarray(sp_alpha_list)
        reg_param_list = _validated_hs_grid(
            self.reg_param_list, classification=isinstance(self, ClassifierMixin)
        )
        param_list = list(itertools.product(sp_alpha_list, reg_param_list))
        _initialize_cv_tracking(self, param_list)
        sparse_pruning_requested = any(
            sp_alpha > 0 for sp_alpha in sp_alpha_list
        )
        normalization_estimator = clone(self._base_estimator_template())
        (
            normalization_estimator,
            full_effective_weight,
        ) = self._fold_class_weight_into_sample_weight(
            normalization_estimator,
            y,
            sample_weight,
            sparse_pruning_requested=sparse_pruning_requested,
        )
        del normalization_estimator
        full_weight_sum = _effective_weight_sum(
            full_effective_weight, len(y)
        )
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
                np.unique(y_in[train_positive]).size != len(classes)
                or np.unique(y_out[test_positive]).size != len(classes)
            ):
                raise ValueError(
                    "Every classifier CV train and validation fold must "
                    "contain every class with positive weight"
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
                sparse_pruning_requested=sparse_pruning_requested,
            )
            if (
                weight_in is not None
                and np.unique(y_in[weight_in > 0]).size != len(classes)
            ):
                raise ValueError(
                    "class_weight and sample_weight must leave every class "
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
            fold_weight_fraction = (
                _effective_weight_sum(weight_in, len(y_in))
                / full_weight_sum
            )
            self.cv_weight_fractions_.append(fold_weight_fraction)
            _evaluate_cached_cv_fold(
                estimator=self,
                wrapper_class=SHSTreeClassifier,
                base_estimator=base_est,
                X_in=X_in,
                y_prune=y_in_encoded,
                y_shrink=y_in,
                X_out=X_out,
                y_out=y_out,
                weight_in=weight_in,
                weight_out=weight_out,
                sp_alpha_list=sp_alpha_list,
                reg_param_list=reg_param_list,
                scorer=scorer,
                fold_weight_fraction=fold_weight_fraction,
                decimals=decimals,
                classes=classes,
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
        _finalize_cv_selection(self, param_list)
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
                "selection_rule": self.selection_rule,
                "reg_param_mode": self.reg_param_mode,
            }
        )
        return params


class SHSTreeRegressorCV(SHSTreeRegressor):
    """Joint CV for sparse pruning and hierarchical shrinkage.

    With ``sp_alpha_list='auto'`` (the default), eligible infinity-penalty
    regression trees are evaluated at every distinct training-fold structural
    state using the exact topology solver. Scores are aligned at the union of
    fold knots; no full-data tree is used to generate CV candidates. Numeric
    alpha lists remain supported. Ineligible problems and ``solver='apa_apg2'``
    use the historical numeric alpha grid when ``sp_alpha_list='auto'``.
    Set ``reg_param_list='gcv'`` to select HS within each training state;
    the default still tunes a numeric HS grid by held-out scores.

    The default one-standard-error rule chooses the lowest observed mean split
    complexity whose score is within one standard error of the best candidate.
    Score and complexity ties prefer stronger shrinkage and sparse penalties.
    In normalized reg-parameter mode, grid values are full-data reference
    values and count-based shrinkage is scaled by each fold's effective
    training-weight fraction. This is dataset-mass, not per-tree root-mass,
    normalization. Set ``selection_rule='best'`` and
    ``reg_param_mode='raw'`` for the historical joint-argmax behavior.

    Parameters
    ----------
    sp_alpha_list : "auto" or sequence of float, default="auto"
        Training-fold structural events, or an explicit nonnegative penalty
        grid. Ineligible objectives use a fixed historical grid for "auto".
    reg_param_list : sequence of float or "gcv", default=(0, 0.1, 1, 10, 50, 100, 500)
        Reference HS strengths, or training-only GCV within each pruning state.
        A singleton ["gcv"] is equivalent to "gcv"; mixed numeric/GCV grids
        are unsupported. Full-data GCV is reselected after final pruning.
    cv : int, default=3
        Number of shuffled training/validation folds.
    scoring : str or callable, optional
        sklearn scoring name/callable; defaults to negative mean squared error.
        Structural score reuse requires dependence on predictions or tree
        structure, not the numerical penalty label or coefficient diagnostics.
    selection_rule : {"one_se", "best"}, default="one_se"
        Select the simplest competitive tree, or the highest mean CV score.
    reg_param_mode : {"normalized", "raw"}, default="normalized"
        Scale count-based HS by training-fold mass, or use literal grid values.
    max_leaf_nodes : int, default=20
        Leaf cap used for each fold tree and the final tree.
    solver : str, default="auto"
        Final-fit solver. Eligible automatic CV uses structural events even
        when the final fit requests a coefficient-producing exact solver.

    Other parameters follow :class:`SHSTree`; ``prefit=True`` is unsupported.

    Attributes
    ----------
    sp_alpha_, reg_param_ : float
        Selected pruning penalty and effective full-data HS strength.
    cv_path_mode_, cv_solver_ : str
        "structural" versus "grid" CV, and the backend used within CV.
    cv_params_, cv_scores_, cv_complexities_, cv_reg_params_ : list or array
        Candidate parameters and per-candidate/per-fold scores, split counts,
        and effective HS strengths. Scores use higher-is-better convention.
    cv_sp_alphas_ : ndarray
        Evaluated penalties, including the union of fold events in structural CV.
    cv_path_results_, cv_n_pruning_states_ : list, ndarray
        Fold structural paths and numbers of local states, only in structural CV.
    coefficient_path_, coef_ : object or None
        Optional final-fit coefficients; see :class:`SHSTree` for their meaning.

    Examples
    --------
    >>> model = SHSTreeRegressorCV(reg_param_list="gcv", random_state=0)
    >>> model.sp_alpha_list
    'auto'
    """

    def __init__(
        self,
        estimator_: BaseEstimator | None = None,
        sp_alpha_list: Sequence[float] | str = "auto",
        reg_param_list: Sequence[float] | str = (
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
        selection_rule: str = "one_se",
        reg_param_mode: str = "normalized",
        prune_set: str = "auto",
        random_state: int | None = None,
        gamma1: float = 1.0,
        a: float = 1.0,
        max_iter: int = 2000,
        tol: float = 1e-6,
        ord: int | str = np.inf,
        support_tol: float | None = None,
        prefit: bool = False,
        solver: str = "auto",
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
            solver=solver,
        )
        self.sp_alpha_list = sp_alpha_list
        self.reg_param_list = reg_param_list
        self.cv = cv
        self.scoring = scoring
        self.selection_rule = selection_rule
        self.reg_param_mode = reg_param_mode

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
        _reset_cv_path_tracking(self)
        self._validate_hyperparameters(allow_unset=True)
        _validate_cv_policy(self.selection_rule, self.reg_param_mode)
        self._fresh_estimator()
        feature_names = kwargs.pop("feature_names", None)
        X, y, feature_names = _prepare_cv_data(X, y, feature_names)
        sample_weight = self._validated_sample_weight(sample_weight, len(y))
        n_splits = _validated_n_splits(self.cv)
        if _uses_structural_cv(self):
            _select_structural_cv(self, X, y, sample_weight, n_splits, args, kwargs)
            return super().fit(
                X=X, y=y, sample_weight=sample_weight, decimals=decimals,
                *args, feature_names=feature_names, **kwargs,
            )
        sp_alpha_list = _validated_nonnegative_grid(
            _alpha_grid_or_legacy(self.sp_alpha_list), "sp_alpha_list"
        )
        self.cv_path_mode_ = "grid"
        self.cv_sp_alphas_ = np.asarray(sp_alpha_list)
        reg_param_list = _validated_hs_grid(
            self.reg_param_list, classification=isinstance(self, ClassifierMixin)
        )
        param_list = list(itertools.product(sp_alpha_list, reg_param_list))
        _initialize_cv_tracking(self, param_list)
        full_weight_sum = _effective_weight_sum(sample_weight, len(y))
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
            fold_weight_fraction = (
                _effective_weight_sum(weight_in, len(y_in))
                / full_weight_sum
            )
            self.cv_weight_fractions_.append(fold_weight_fraction)
            _evaluate_cached_cv_fold(
                estimator=self,
                wrapper_class=SHSTreeRegressor,
                base_estimator=base_est,
                X_in=X_in,
                y_prune=y_in,
                y_shrink=y_in,
                X_out=X_out,
                y_out=y_out,
                weight_in=weight_in,
                weight_out=weight_out,
                sp_alpha_list=sp_alpha_list,
                reg_param_list=reg_param_list,
                scorer=scorer,
                fold_weight_fraction=fold_weight_fraction,
                decimals=decimals,
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
        _finalize_cv_selection(self, param_list)
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
                "selection_rule": self.selection_rule,
                "reg_param_mode": self.reg_param_mode,
            }
        )
        return params


class SPTreeRegressor(SHSTreeRegressor):
    """Sparse-pruned regression tree, with hierarchical shrinkage off by default.

    ``sp_alpha`` (default 1) is the normalized hiCAP penalty; ``reg_param``
    defaults to zero. All solver choices and other parameters follow
    :class:`SHSTreeRegressor` and :class:`SHSTree`. ``solver="auto"`` uses
    exact structural events for eligible infinity-penalty regression trees.

    The fitted ``estimator_`` predicts using retained CART node means. Optional
    ``coef_`` and ``coefficient_path_`` describe the penalized original-stump
    problem, not the compact pruned tree's prediction values.

    Examples
    --------
    >>> model = SPTreeRegressor(solver="coefficient_path", random_state=0)
    >>> model.reg_param
    0
    """
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
        ord: int | str = np.inf,
        max_leaf_nodes: int | None = None,
        random_state: int | None = None,
        support_tol: float | None = None,
        prefit: bool = False,
        solver: str = "auto",
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
            solver=solver,
        )


class SPTreeClassifier(SHSTreeClassifier):
    """Sparse-pruned binary/multiclass classifier, without default HS.

    Parameters follow :class:`SHSTreeClassifier` and :class:`SHSTree`, except
    ``reg_param`` defaults to zero. ``solver="auto"`` selects the exact
    structural solver for eligible infinity-penalty CART trees. Optional
    ``proximal`` and ``coefficient_path`` expose logistic/softmax coefficients;
    APA remains binary-only. ``sp_alpha=0`` retains the original tree structure.
    """
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
        ord: int | str = np.inf,
        max_leaf_nodes: int | None = None,
        random_state: int | None = None,
        support_tol: float | None = None,
        prefit: bool = False,
        solver: str = "auto",
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
            solver=solver,
        )


class SPTreeRegressorCV(SHSTreeRegressorCV):
    """Select a sparse-pruned regression tree by structural-path CV.

    Parameters and fitted attributes follow :class:`SHSTreeRegressorCV`.
    The only changed default is ``reg_param_list=(0,)``, which disables HS.
    ``sp_alpha_list="auto"`` scores training-fold structural states; numeric
    lists request grid CV. ``solver`` selects the final-fit backend, so
    ``solver="coefficient_path"`` exposes final ``coef_`` and all coefficient
    knots while keeping eligible automatic CV structural-only.

    Examples
    --------
    >>> model = SPTreeRegressorCV(max_leaf_nodes=128, random_state=0)
    >>> model.sp_alpha_list, model.reg_param_list
    ('auto', (0,))
    """
    def __init__(
        self,
        estimator_: BaseEstimator | None = None,
        sp_alpha_list: Sequence[float] | str = "auto",
        reg_param_list: Sequence[float] | str = (0,),
        max_leaf_nodes: int = 20,
        cv: int = 3,
        scoring=None,
        selection_rule: str = "one_se",
        reg_param_mode: str = "normalized",
        prune_set: str = "auto",
        random_state: int | None = None,
        gamma1: float = 1.0,
        a: float = 1.0,
        max_iter: int = 2000,
        tol: float = 1e-6,
        ord: int | str = np.inf,
        support_tol: float | None = None,
        prefit: bool = False,
        solver: str = "auto",
    ) -> None:
        super().__init__(
            estimator_=estimator_,
            sp_alpha_list=sp_alpha_list,
            reg_param_list=reg_param_list,
            max_leaf_nodes=max_leaf_nodes,
            cv=cv,
            scoring=scoring,
            selection_rule=selection_rule,
            reg_param_mode=reg_param_mode,
            prune_set=prune_set,
            random_state=random_state,
            gamma1=gamma1,
            a=a,
            max_iter=max_iter,
            tol=tol,
            ord=ord,
            support_tol=support_tol,
            prefit=prefit,
            solver=solver,
        )


class SPTreeClassifierCV(SHSTreeClassifierCV):
    """Select a sparse-pruned classifier by structural CV, without default HS.

    Parameters and fitted attributes follow :class:`SHSTreeClassifierCV`,
    with ``reg_param_list=(0,)`` by default. ``sp_alpha_list="auto"`` uses
    fold-local structural knots for eligible binary/multiclass trees; numeric
    lists and explicit APA use grid CV. Optional coefficient paths describe
    nonlinear logistic/softmax samples, not exact linear interpolation.
    """
    def __init__(
        self,
        estimator_: BaseEstimator | None = None,
        sp_alpha_list: Sequence[float] | str = "auto",
        reg_param_list: Sequence[float] = (0,),
        max_leaf_nodes: int = 20,
        cv: int = 3,
        scoring=None,
        selection_rule: str = "one_se",
        reg_param_mode: str = "normalized",
        prune_set: str = "auto",
        random_state: int | None = None,
        gamma1: float = 1.0,
        a: float = 1.0,
        max_iter: int = 2000,
        tol: float = 1e-6,
        ord: int | str = np.inf,
        support_tol: float | None = None,
        prefit: bool = False,
        solver: str = "auto",
    ) -> None:
        super().__init__(
            estimator_=estimator_,
            sp_alpha_list=sp_alpha_list,
            reg_param_list=reg_param_list,
            max_leaf_nodes=max_leaf_nodes,
            cv=cv,
            scoring=scoring,
            selection_rule=selection_rule,
            reg_param_mode=reg_param_mode,
            prune_set=prune_set,
            random_state=random_state,
            gamma1=gamma1,
            a=a,
            max_iter=max_iter,
            tol=tol,
            ord=ord,
            support_tol=support_tol,
            prefit=prefit,
            solver=solver,
        )
