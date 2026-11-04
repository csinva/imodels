"""Iterative random forests, signed interactions, and lo-siRF local importance.

Based on Basu et al. (2018), Supplement Algorithm 2, with classification and
regression following the paper-era R ``iRF`` package:
https://arxiv.org/abs/1706.08457. No dependency on the historical ``irf`` package
or private scikit-learn tree builders.

Signed scores follow Kumbier et al., arXiv:1810.07287v2, with regression
targets binarized at a fixed cutoff as in R iRF v3;
local stability follows Wang et al. (2025), doi:10.1038/s44161-025-00656-8.
"""

from imodels.util.introspection import TextMixin
from collections import Counter
from numbers import Integral, Real
import warnings

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils import check_random_state, resample
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_array, check_consistent_length, check_is_fitted

try:
    from sklearn.utils.validation import validate_data
except ImportError:  # scikit-learn < 1.6
    def validate_data(estimator, *args, **kwargs):
        return estimator._validate_data(*args, **kwargs)

from ._forest import _WeightedForest, _forest_paths
from ._interactions import (
    _interaction_terms, _interaction_scores, _local_stability, _permutation_ranking,
    _benjamini_hochberg,
)
from ._rit import _random_intersection_trees


def _check_integer(name, value, minimum):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, Integral) or value < minimum):
        raise ValueError(f"{name} must be an integer >= {minimum}.")


class _IRFBase(TextMixin, BaseEstimator):
    """Shared iterative reweighting, outer bootstrap, and RIT stability."""

    _task = None

    def _check_parameters(self, n_features):
        if not isinstance(self.signed, (bool, np.bool_)):
            raise ValueError("signed must be a boolean.")
        if self.feature_groups is not None:
            groups = np.asarray(self.feature_groups)
            if (groups.shape != (n_features,) or groups.dtype.kind not in "iu"
                    or np.any(groups < 0)):
                raise ValueError(
                    "feature_groups must contain one nonnegative integer group ID per feature.")
        for name, minimum in (
            ("n_estimators", 1), ("n_iterations", 1), ("n_bootstraps", 0),
            ("min_samples_split", 2), ("min_samples_leaf", 1), ("n_rit", 1),
            ("rit_depth", 2), ("rit_branching", 1),
        ):
            _check_integer(name, getattr(self, name), minimum)
        if self.max_depth is not None:
            _check_integer("max_depth", self.max_depth, 1)
        if self.n_estimators_bootstrap is not None:
            _check_integer("n_estimators_bootstrap", self.n_estimators_bootstrap, 1)
        if not isinstance(self.bootstrap, (bool, np.bool_)):
            raise ValueError("bootstrap must be a boolean.")
        if self.n_jobs is not None:
            if (isinstance(self.n_jobs, (bool, np.bool_))
                    or not isinstance(self.n_jobs, Integral) or self.n_jobs == 0):
                raise ValueError("n_jobs must be a nonzero integer or None.")
        fraction = self.bootstrap_fraction
        if (isinstance(fraction, (bool, np.bool_)) or not isinstance(fraction, Real)
                or not np.isfinite(fraction) or not 0 < fraction <= 1):
            raise ValueError("bootstrap_fraction must be in (0, 1].")
        features = self.max_features
        if features is None:
            return
        if isinstance(features, str):
            if features in ("sqrt", "log2"):
                return
        elif not isinstance(features, (bool, np.bool_)):
            if isinstance(features, Integral) and 1 <= features <= n_features:
                return
            if isinstance(features, Real) and not isinstance(features, Integral):
                if np.isfinite(features) and 0 < features <= 1:
                    return
        raise ValueError(
            "max_features must be 'sqrt', 'log2', None, an integer in "
            "[1, n_features], or a float in (0, 1]."
        )

    def _new_forest(self, n_estimators, seed):
        return _WeightedForest(
            n_estimators=n_estimators, max_features=self.max_features,
            max_depth=self.max_depth, min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf, bootstrap=self.bootstrap,
            n_jobs=self.n_jobs, random_state=seed, task=self._task,
        )

    # Task-specific hooks.
    def _validate_targets(self, X, y):
        raise NotImplementedError

    def _prepare_targets(self, y):
        """Return targets as passed to the forest backend."""
        raise NotImplementedError

    def _resolve_leaf_selection(self, y):
        """Resolve fitted state used by ``_select_leaf``."""

    def _forest_n_classes(self):
        return None

    def _stratify_labels(self, y_fit):
        return None

    def _select_leaf(self, prediction):
        """Return whether a leaf with this prediction enters RIT."""
        raise NotImplementedError

    def _discover_interactions(self, leaves, seed):
        paths, masses = [], []
        for path, prediction, mass in leaves:
            if mass > 0 and self._select_leaf(prediction):
                paths.append(path)
                masses.append(mass)
        return _random_intersection_trees(
            paths, masses, n_trees=self.n_rit, max_depth=self.rit_depth,
            n_children=self.rit_branching, random_state=seed,
        )

    def fit(self, X, y, sample_weight=None):
        """Fit forests and interaction stability; return the fitted estimator.

        ``sample_weight`` supplies nonnegative, finite observation weights.
        Prediction uses the final full-data forest.
        """
        X, y = self._validate_targets(X, y)
        self._check_parameters(X.shape[1])
        y_fit = self._prepare_targets(y)

        if sample_weight is None:
            weights = np.ones(X.shape[0], dtype=float)
        elif np.isscalar(sample_weight):
            weights = np.full(X.shape[0], sample_weight, dtype=float)
        else:
            weights = check_array(sample_weight, ensure_2d=False, dtype=float)
        if weights.ndim != 1 or len(weights) != len(X):
            raise ValueError("sample_weight must have one value per input row.")
        if not np.all(np.isfinite(weights)) or np.any(weights < 0):
            raise ValueError("sample_weight must contain finite nonnegative values.")
        if not np.any(weights > 0):
            raise ValueError("sample_weight cannot be all zero; total weight must be positive.")
        # All uses of observation weights are invariant to a common scale.
        # Normalize before impurity calculations to avoid overflow/underflow
        # for otherwise valid weights such as [1e308, 1e308].
        positive = weights > 0
        weights = weights / weights.max()
        if np.any(positive & (weights == 0)):
            raise ValueError("sample_weight has an unsupported numerical dynamic range.")
        original_indices = np.flatnonzero(weights > 0)
        X, y_fit, weights = (
            X[original_indices], y_fit[original_indices], weights[original_indices]
        )
        self.signed_ = bool(self.signed)
        self.feature_groups_ = (np.arange(self.n_features_in_) if self.feature_groups is None
                                else np.asarray(self.feature_groups).copy())
        for name in ("interaction_scores_", "bootstrap_interaction_scores_"):
            self.__dict__.pop(name, None)
        self._resolve_leaf_selection(y_fit)
        if self.signed_ and not self.bootstrap:
            warnings.warn(
                "signed=True with bootstrap=False leaves no out-of-bag rows; "
                "precision and MIP cannot be computed (NaN). Their zero stability "
                "is not evidence against an interaction.", UserWarning, stacklevel=2)

        n_outer = int(np.ceil(self.bootstrap_fraction * len(X)))
        stratify = self._stratify_labels(y_fit)
        if self.n_bootstraps and stratify is not None:
            if n_outer < len(np.unique(stratify)):
                raise ValueError(
                    "bootstrap_fraction gives fewer outer rows than observed classes."
                )

        rng = (np.random.RandomState() if self.random_state is None
               else check_random_state(self.random_state))
        max_seed = np.iinfo(np.int32).max
        feature_weights = np.full(self.n_features_in_, 1.0 / self.n_features_in_)
        weights_history, importances_history = [], []
        for _ in range(self.n_iterations):
            weights_history.append(feature_weights.copy())
            forest = self._new_forest(self.n_estimators, int(rng.randint(max_seed)))
            forest.fit(X, y_fit, feature_weights=feature_weights,
                       sample_weight=weights, n_classes=self._forest_n_classes())
            importances = forest.feature_importances_.copy()
            importances_history.append(importances)
            if importances.sum() > 0:
                feature_weights = importances / importances.sum()

        self.forest_ = forest
        self.estimators_ = forest.estimators_
        self.feature_weights_history_ = np.asarray(weights_history)
        self.feature_importances_history_ = np.asarray(importances_history)
        # Freeze INPUT weights of forest K, not output importances (K+1).
        self.feature_weights_ = self.feature_weights_history_[-1].copy()
        self.feature_importances_ = importances_history[-1].copy()
        self.bootstrap_interactions_ = []
        self.bootstrap_samples_ = []
        signed = self.signed_
        if signed:
            leaves = (
                leaf for tree in self.estimators_
                for leaf in tree.terminal_paths(
                    X, sample_weight=weights, signed=True,
                    feature_groups=self.feature_groups_)
            )
            candidates = sorted(self._discover_interactions(
                leaves, int(rng.randint(max_seed))))
            terms = _interaction_terms(candidates)
            self.bootstrap_interaction_scores_ = []
            score_y = (y_fit if self._task == "classification"
                       else (y_fit >= self.leaf_threshold_).astype(np.intp))
            in_target = score_y == self._target_index
            if in_target.all() or not in_target.any():
                side = "inside" if in_target.all() else "outside"
                warnings.warn(
                    f"Every positive-weight training row is {side} the signed scoring "
                    "target (the interaction class, or y >= leaf_threshold_ for "
                    "regression); prevalence, precision, FSD and MIP are degenerate.",
                    UserWarning, stacklevel=2)
        n_trees = (self.n_estimators if self.n_estimators_bootstrap is None
                   else self.n_estimators_bootstrap)
        replicates = []
        for _ in range(self.n_bootstraps):
            sample_seed, forest_seed, rit_seed = rng.randint(max_seed, size=3)
            indices = resample(
                np.arange(len(X)), replace=True, n_samples=n_outer,
                stratify=stratify, random_state=int(sample_seed),
            )
            self.bootstrap_samples_.append(original_indices[indices])
            replicates.append((indices, int(forest_seed), int(rit_seed)))
        # Trees of all outer forests are fit in one pool of workers, which
        # also measure leaf masses on the outer sample; only paths return.
        outer_forests = [self._new_forest(n_trees, forest_seed)
                         for _, forest_seed, _ in replicates]

        def outer_samples():
            for indices, _, _ in replicates:
                data = (X[indices], y_fit[indices], self.feature_weights_,
                        weights[indices], self._forest_n_classes())
                yield (data + (original_indices[indices], score_y[indices])
                       if signed else data)

        signed_options = (dict(feature_groups=self.feature_groups_, candidates=candidates,
                               target_class=self._target_index) if signed else {})
        if not signed and self.feature_groups is not None:
            signed_options["feature_groups"] = self.feature_groups_
        leaves_per_forest = _forest_paths(
            outer_forests, outer_samples(), self.n_jobs, **signed_options)
        score_names = ("prevalence", "precision", "fsd", "mip")
        score_values = []
        for (_, _, rit_seed), result in zip(replicates, leaves_per_forest):
            if signed:
                leaves, statistics = result
                scores = _interaction_scores(candidates, terms, statistics)
                score_values.append(scores)
                self.bootstrap_interaction_scores_.append({
                    interaction: dict(zip(score_names, map(float, values)))
                    for interaction, values in zip(candidates, scores)
                })
            else:
                leaves = result
            interactions = self._discover_interactions(leaves, rit_seed)
            self.bootstrap_interactions_.append(interactions)

        counts = Counter(
            interaction for replicate in self.bootstrap_interactions_
            for interaction in replicate
        )
        self.interactions_ = sorted(candidates if signed else counts,
                                    key=lambda s: (-counts[s], -len(s), s))
        self.interaction_stability_ = {
            interaction: counts[interaction] / self.n_bootstraps
            for interaction in self.interactions_
        } if self.n_bootstraps else {}
        if signed:
            self.interaction_scores_ = {}
            if candidates and self.n_bootstraps:
                values = np.asarray(score_values)
                valid = np.sum(~np.isnan(values), axis=0)
                with np.errstate(invalid="ignore"):
                    means = np.divide(
                        np.nansum(values, axis=0), valid,
                        out=np.full(values.shape[1:], np.nan), where=valid > 0)
                for j, interaction in enumerate(candidates):
                    self.interaction_scores_[interaction] = dict(
                        zip(score_names, map(float, means[j])),
                        fsd_stability=float(np.mean(values[:, j, 2] > 0)),
                        mip_stability=float(np.mean(values[:, j, 3] > 0)),
                    )
        return self

    def _validate_interactions(self, interactions, first_only=False):
        check_is_fitted(self, "forest_")
        if not self.signed_:
            raise ValueError("Interaction analysis requires a model fitted with signed=True.")
        if not isinstance(first_only, (bool, np.bool_)):
            raise ValueError("first_only must be a boolean.")
        if interactions is None:
            return self.interactions_
        groups = set(self.feature_groups_.tolist())
        if isinstance(interactions, str):
            if interactions != "features":
                raise ValueError("The only supported interactions shortcut is 'features'.")
            return [((int(group), sign),) for group in sorted(groups) for sign in (-1, 1)]
        result = []
        try:
            for interaction in interactions:
                atoms = []
                for group, sign in interaction:
                    if (isinstance(group, (bool, np.bool_)) or not isinstance(group, Integral)
                            or group not in groups or isinstance(sign, (bool, np.bool_))
                            or not isinstance(sign, Integral) or sign not in (-1, 1)):
                        raise ValueError
                    atoms.append((int(group), int(sign)))
                if not atoms or len(set(atoms)) != len(atoms):
                    raise ValueError
                result.append(tuple(sorted(atoms)))
            if len(set(result)) != len(result):
                raise ValueError
        except (TypeError, ValueError):
            raise ValueError(
                "interactions must be distinct, nonempty sequences of (group_id, sign) "
                "pairs, with fitted group IDs, signs -1 or +1, and no repeated pairs."
            ) from None
        return result

    def local_stability(self, X, interactions=None, *, first_only=False):
        """Fraction of trees using each signed interaction to predict each row.

        ``interactions`` is a sequence of interactions, each a sequence of
        ``(feature_or_group_id, sign)`` pairs; signs are -1 (left, <=) and +1
        (right, >). Defaults to ``interactions_``. Singletons are allowed;
        ``interactions="features"`` selects both signs of every feature/group,
        ordered by group ID then sign (-1, +1).
        Returns an array of shape (n_samples, n_interactions), in input order.

        Uses all splits on the visited path, as in lo-siRF, including repeated
        features with opposite directions. Discovery uses only the first split
        per feature/group. Scores use all trees of the final full-data forest,
        independently of their predicted class. Set ``first_only=True`` to
        match discovery's first-split encoding. X may contain unseen rows.
        """
        interactions = self._validate_interactions(interactions, first_only)
        X = validate_data(self, X, reset=False, dtype=np.float32, accept_sparse=False)
        return _local_stability(self.estimators_, X, interactions, self.feature_groups_,
                                first_only=first_only)

    def rank_interactions(self, X, y, interactions=None, n_permutations=10000,
                          random_state=None, *, first_only=False):
        """Rank signed interactions by held-out differences in local stability.

        Supply evaluation rows and labels that were not used to fit the model.
        The statistic is mean local stability for ``interaction_class_`` minus
        the mean for the other classes. For regression, the target is
        ``y >= leaf_threshold_``, using the fitted cutoff. Both groups must
        occur in y. Candidate
        selection can use a separate dataset; no data splitting is implicit.
        ``interactions`` and ``first_only`` have the same meanings as in
        ``local_stability``; ``interactions="features"`` ranks single features.

        Returns a dict keyed by interaction, with ``difference``, unadjusted
        ``pvalue``, and Benjamini-Hochberg-adjusted ``pvalue_bh`` entries. The
        adjustment covers all interactions in this call. Results are ordered
        by increasing raw p-value, then decreasing absolute
        difference. The two-sided Monte Carlo p-value is (1 + number of label
        permutations with abs(statistic) >= abs(observed)) / (n_permutations + 1).
        Permutation comparisons use integer tree counts, preserving exact ties.
        These nominal values prioritize interactions in a fixed forest.
        """
        _check_integer("n_permutations", n_permutations, 1)
        interactions = self._validate_interactions(interactions, first_only)
        y = check_array(y, ensure_2d=False,
                        dtype=float if self._task == "regression" else None)
        if y.ndim != 1:
            raise ValueError("y must be one-dimensional.")
        check_consistent_length(X, y)
        if self._task == "regression":
            target = y >= self.leaf_threshold_
        else:
            check_classification_targets(y)
            if not np.all(np.isin(y, self.classes_)):
                raise ValueError("y must contain labels seen during fitting.")
            target = y == self.interaction_class_
        if not np.any(target) or np.all(target):
            raise ValueError("y must contain both target and non-target observations.")
        X = validate_data(self, X, reset=False, dtype=np.float32, accept_sparse=False)
        local = _local_stability(self.estimators_, X, interactions,
                                 self.feature_groups_, normalize=False,
                                 first_only=first_only)
        difference, pvalue = _permutation_ranking(
            local, target, n_permutations, random_state, n_trees=len(self.estimators_))
        adjusted = _benjamini_hochberg(pvalue)
        order = sorted(range(len(interactions)),
                       key=lambda j: (pvalue[j], -abs(difference[j]), interactions[j]))
        return {interactions[j]: {"difference": float(difference[j]),
                                  "pvalue": float(pvalue[j]),
                                  "pvalue_bh": float(adjusted[j])} for j in order}

    def select_interactions(self, threshold=0.5):
        """Return signed candidates passing all three bootstrap screens.

        Recovery, positive-FSD, and positive-MIP stability must each be at
        least ``threshold`` (between 0 and 1). Keeps ``interactions_`` order.
        Requires a signed fit with outer bootstraps. With no OOB precision,
        positive-MIP stability is zero. Each metric must be available in at
        least one replicate, including when ``threshold=0``.
        """
        interactions = self._validate_interactions(None)
        if (isinstance(threshold, (bool, np.bool_)) or not isinstance(threshold, Real)
                or not np.isfinite(threshold) or not 0 <= threshold <= 1):
            raise ValueError("threshold must be a finite number in [0, 1].")
        if not self.bootstrap_interaction_scores_:
            raise ValueError("select_interactions requires a fit with n_bootstraps > 0.")
        return [s for s in interactions
                if self.interaction_stability_[s] >= threshold
                and self.interaction_scores_[s]["fsd_stability"] >= threshold
                and self.interaction_scores_[s]["mip_stability"] >= threshold
                and all(any(not np.isnan(scores[s][metric])
                            for scores in self.bootstrap_interaction_scores_)
                        for metric in ("fsd", "mip"))]


class IRFClassifier(ClassifierMixin, _IRFBase):
    """Iteratively reweight forests and discover stable feature interactions.

    Set ``signed=True`` for siRF discovery and scores. The fitted estimator's
    ``local_stability`` and ``rank_interactions`` methods provide lo-siRF local
    analysis. The default ``signed=False`` gives unsigned iRF.

    At each node, candidate features are sampled without replacement with
    probabilities proportional to the preceding forest's Gini importances.
    All reweighting iterations use the full training data, with ordinary
    per-tree bootstrapping. Interaction discovery then fits new forests on
    outer bootstrap samples, holding the final iteration's *input* feature
    weights fixed. Random intersection trees (RIT) search each outer forest;
    an interaction receives one vote per outer replicate.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of trees in each reweighting forest.
    n_iterations : int, default=5
        Number of full-data fits, including the initial uniform forest. At one
        iteration, prediction and interaction discovery use uniform weights.
    n_bootstraps : int, default=10
        Number of outer bootstrap forests. Zero skips stability and scores;
        signed full-data candidates remain available. Ten gives stability
        increments of 0.1; consider 50 for final analyses.
    max_features : {"sqrt", "log2", None}, int or float, default="sqrt"
        Candidate features per node. A float in (0, 1] specifies a fraction of
        input features; None uses all. Zero-weight features are excluded.
    max_depth : int or None, default=None
        Maximum decision-tree depth; None allows unlimited depth.
    min_samples_split : int, default=2
        Minimum distinct in-bag observations needed to split a node.
    min_samples_leaf : int, default=1
        Minimum distinct in-bag observations at a leaf.
    bootstrap : bool, default=True
        Bootstrap rows for each tree (inner bootstrap), independently of the
        outer bootstrap controlled by ``n_bootstraps``.
    n_estimators_bootstrap : int or None, default=None
        Trees per outer forest; None uses ``n_estimators``.
    bootstrap_fraction : float, default=1.0
        Outer sample size as a fraction of training rows, rounded up. Sampling
        is with replacement. Original R uses 1.0; historical Python uses 0.2.
    stratify_bootstrap : bool, default=True
        Preserve class proportions in outer samples, as in original R.
        Full-size samples preserve class counts.
    n_rit : int, default=100
        Random intersection trees per outer forest.
    rit_depth : int, default=5
        Maximum number of sampled paths intersected along a RIT branch,
        following historical R. Must be at least 2. Pairs are retained
        immediately and stop their branch; larger sets continue to this depth.
    rit_branching : int, default=2
        Children per RIT node.
    interaction_class : object or None, default=None
        Discover paths whose leaf predicts this label. None selects the last
        label in ``classes_`` (1 for labels 0 and 1). Multiclass prediction is
        supported; interaction discovery targets one class. R's default,
        ``class.id=1``, selects the second sorted label instead; the two agree
        for binary labels.
    n_jobs : int or None, default=None
        Parallel jobs for fitting trees. None uses one; -1 uses all processors.
    random_state : int, RandomState or None, default=None
        Controls forests, outer samples, and RIT. An integer gives reproducible
        results across ``n_jobs`` values. Successive forests use new seeds.
    signed : bool, default=False
        Discover signed interactions and compute siRF scores. Signed atoms are
        ``(feature_or_group_id, direction)`` pairs: -1 means left (<= split
        threshold), +1 means right (> threshold). Discovery keeps the first
        direction per feature/group. Candidates come from the final full-data
        forest and are evaluated across the outer bootstrap forests.
    feature_groups : array-like of int or None, default=None
        One nonnegative group ID per input column, for discovery and local
        importance. None uses zero-based feature indices. Available for both
        unsigned and signed discovery. Grouping affects interpretation only;
        forests still split on individual features.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Class labels in probability-column order.
    forest_ : object
        Final full-data weighted forest, used for prediction.
    estimators_ : list
        Trees in ``forest_``.
    feature_weights_ : ndarray of shape (n_features,)
        Input sampling weights of the final forest and every outer forest.
    feature_weights_history_ : ndarray of shape (n_iterations, n_features)
        Input weights of full-data forests, starting with uniform weights.
    feature_importances_ : ndarray of shape (n_features,)
        Normalized Gini importances produced by the final full-data forest.
    feature_importances_history_ : ndarray of shape (n_iterations, n_features)
        Output importances from each full-data forest.
    interaction_stability_ : dict
        Sorted interaction tuples mapped to the fraction of outer replicates
        recovering that exact set. With ``signed=True``, keys contain signed
        atoms and include full-data candidates with zero bootstrap recovery.
        Otherwise keys contain feature indices, or group IDs when
        ``feature_groups`` is set. Sets have at least two atoms.
        Stability holds the learned feature weights fixed. Empty when
        ``n_bootstraps=0``.
    interactions_ : list of tuple
        Sets ordered by descending stability, then descending size, then
        feature or group IDs (signed atoms in signed mode). Thresholds are not retained.
    interaction_scores_ : dict
        Available with ``signed=True``. Each candidate maps to bootstrap-mean
        ``prevalence``, ``precision``, ``fsd`` (feature-selection dependence),
        and ``mip`` (minimum increase in precision), plus ``fsd_stability`` and
        ``mip_stability``: fractions of all bootstraps with positive scores.
        Undefined scores are NaN and do not count as positive. Empty when
        ``n_bootstraps=0``.
    bootstrap_interaction_scores_ : list of dict
        Per-bootstrap candidate scores before averaging, with ``signed=True``.
    signed_ : bool
        Whether the fitted model used signed discovery.
    feature_groups_ : ndarray of shape (n_features,)
        Fitted mapping from input columns to nonnegative group IDs.
    bootstrap_interactions_ : list of set
        Distinct recovered intersections from each outer replicate.
    bootstrap_samples_ : list of ndarray
        Original input row indices for each outer sample, retaining duplicates.
    interaction_class_ : object
        Resolved target class for interaction discovery.

    Notes
    -----
    Supports dense, finite numeric inputs and single-output classification.
    Nodes are split by a NumPy search that
    reproduces public scikit-learn stumps exactly, falling back to the stumps
    themselves; this can be slower than a compiled random forest. Default tree
    budgets are smaller than the paper's 500 forest trees and 500 RITs.

    Positive ``sample_weight`` values affect splits, probabilities, and leaf
    masses. Zero-weight rows are excluded before resampling. Row sampling is
    uniform (stratified for outer samples by default). All-zero Gini importances
    cause an iteration to retain its input feature weights.
    Observation weights are scaled by a common factor for numerical stability.

    RIT leaf masses are computed by routing every observation of the outer
    sample through every tree, including outer-bootstrap duplicates. They are
    not the individual trees' inner-bootstrap training counts.

    RIT follows the paper-era R implementation: two paths initialize each
    root, pairs stop early, and singletons are discarded. The supplement's
    terminal-only RIT pseudocode has different interaction sampling behavior.
    Predictions average leaf probabilities, whereas historical R averages
    hard tree votes. A leaf whose top classes tie is labeled by a uniform
    random draw among them, as in R, so tied leaves can enter RIT.

    Signed scores follow the revised siRF paper: prevalence conditions on
    observed class; precision uses tree-OOB rows, excluding all outer copies
    of an in-bag observation. Tree ratios with no support are NaN and omitted
    from the corresponding mean. With ``bootstrap=False``, OOB precision and
    MIP are unavailable (NaN). FSD is log prevalence minus the sum of singleton
    log prevalences; MIP is log precision minus the largest leave-one-out log
    precision. Zero/zero is NaN; one-sided zeros give signed infinity. Bootstrap
    means omit NaN, but positive-score stability always divides by all B
    replicates. These definitions differ from the historical R score formulas.

    References
    ----------
    Basu, S., Kumbier, K., Brown, J. B., and Yu, B. (2018). Iterative random
    forests to discover predictive and stable high-order interactions.
    PNAS, 115(8), 1943-1948. doi:10.1073/pnas.1711236115.

    Examples
    --------
    >>> from imodels import IRFClassifier
    >>> X = [[0, 0], [0, 1], [1, 0], [1, 1]] * 8
    >>> y = [0, 0, 0, 1] * 8
    >>> model = IRFClassifier(n_estimators=10, n_iterations=2,
    ...                       n_bootstraps=3, random_state=0).fit(X, y)
    >>> model.predict([[1, 1]]).tolist()
    [1]
    """

    _task = "classification"

    def __init__(
        self, n_estimators=100, n_iterations=5, n_bootstraps=10,
        max_features="sqrt", max_depth=None, min_samples_split=2,
        min_samples_leaf=1, bootstrap=True, n_estimators_bootstrap=None,
        bootstrap_fraction=1.0, stratify_bootstrap=True, n_rit=100,
        rit_depth=5, rit_branching=2, interaction_class=None,
        n_jobs=None, random_state=None, signed=False, feature_groups=None,
    ):
        self.n_estimators = n_estimators
        self.n_iterations = n_iterations
        self.n_bootstraps = n_bootstraps
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.bootstrap = bootstrap
        self.n_estimators_bootstrap = n_estimators_bootstrap
        self.bootstrap_fraction = bootstrap_fraction
        self.stratify_bootstrap = stratify_bootstrap
        self.n_rit = n_rit
        self.rit_depth = rit_depth
        self.rit_branching = rit_branching
        self.interaction_class = interaction_class
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.signed = signed
        self.feature_groups = feature_groups

    def _check_parameters(self, n_features):
        if not isinstance(self.stratify_bootstrap, (bool, np.bool_)):
            raise ValueError("stratify_bootstrap must be a boolean.")
        super()._check_parameters(n_features)

    def _validate_targets(self, X, y):
        X, y = validate_data(self, X, y, dtype=np.float32, accept_sparse=False)
        check_classification_targets(y)
        return X, y

    def _prepare_targets(self, y):
        self.classes_, y_encoded = np.unique(y, return_inverse=True)
        self.n_classes_ = len(self.classes_)
        return y_encoded

    def _resolve_leaf_selection(self, y):
        target = self.classes_[-1] if self.interaction_class is None else self.interaction_class
        matches = np.flatnonzero(self.classes_ == target)
        if len(matches) != 1:
            raise ValueError("interaction_class must be a label present in y.")
        self._target_index = int(matches[0])
        self.interaction_class_ = self.classes_[self._target_index]

    def _forest_n_classes(self):
        return self.n_classes_

    def _stratify_labels(self, y_fit):
        return y_fit if self.stratify_bootstrap else None

    def _select_leaf(self, prediction):
        return prediction == self._target_index

    def predict_proba(self, X):
        """Return class probabilities from the final full-data forest."""
        check_is_fitted(self, "forest_")
        X = validate_data(self, X, reset=False, dtype=np.float32, accept_sparse=False)
        return self.forest_.predict_proba(X)

    def predict(self, X):
        """Predict labels using the final full-data forest."""
        probabilities = self.predict_proba(X)
        return self.classes_[np.argmax(probabilities, axis=1)]


class IRFRegressor(RegressorMixin, _IRFBase):
    """Iterative random forest regression with stable feature interactions.

    Set ``signed=True`` for siRF discovery and scores. The fitted estimator's
    ``local_stability`` and ``rank_interactions`` methods provide lo-siRF local
    analysis. The default ``signed=False`` gives unsigned iRF.

    Follows the regression branch of the paper-era R ``iRF`` package. Forests
    are reweighted by residual-sum-of-squares importance (R's
    ``IncNodePurity``). Every outer-bootstrap leaf, or every leaf whose mean
    exceeds ``leaf_threshold``, supplies a decision path to random
    intersection trees (RIT). An interaction receives one vote per outer
    replicate.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of trees in each reweighting forest.
    n_iterations : int, default=5
        Number of full-data fits, including the initial uniform forest.
    n_bootstraps : int, default=10
        Number of outer bootstrap forests. Zero skips stability and scores;
        signed full-data candidates remain available. Ten gives stability
        increments of 0.1; consider 50 for final analyses.
    max_features : {"sqrt", "log2", None}, int or float, default=1/3
        Candidate features per node. The default gives R's regression
        ``mtry = max(floor(p / 3), 1)``. A float in (0, 1] is a fraction of
        input features, rounded down; None uses all.
    max_depth : int or None, default=None
        Maximum decision-tree depth; None allows unlimited depth.
    min_samples_split : int, default=6
        Minimum in-bag draws needed to split a node. Bootstrap duplicates
        count separately, as in R. The default matches R's ``nodesize=5``,
        which stops nodes holding five or fewer draws.
    min_samples_leaf : int, default=1
        Minimum in-bag draws at a leaf.
    bootstrap : bool, default=True
        Bootstrap rows for each tree (inner bootstrap).
    n_estimators_bootstrap : int or None, default=None
        Trees per outer forest; None uses ``n_estimators``.
    bootstrap_fraction : float, default=1.0
        Outer sample size as a fraction of training rows, rounded up. Sampling
        is with replacement and unstratified, as in R.
    n_rit : int, default=100
        Random intersection trees per outer forest.
    rit_depth : int, default=5
        Maximum number of sampled paths intersected along a RIT branch.
    rit_branching : int, default=2
        Children per RIT node.
    leaf_threshold : float or None, default=None
        Use only leaves whose in-bag mean is strictly greater than this value,
        like R's ``rit.param$class.cut``. In signed mode, None resolves to the
        median of positive-weight training responses, frozen before resampling.
        Signed scores and ranking use observed targets at or above this cutoff.
        In unsigned mode, None uses every leaf.
    n_jobs : int or None, default=None
        Parallel jobs for fitting trees. None uses one; -1 uses all processors.
    random_state : int, RandomState or None, default=None
        Controls forests, outer samples, and RIT.
    signed : bool, default=False
        Discover and score signed interactions as in ``IRFClassifier``.
        Forest fitting and prediction continue to use continuous responses.
    feature_groups : array-like of int or None, default=None
        One nonnegative group ID per column, for unsigned or signed discovery.
        None uses zero-based feature indices. Grouping affects interpretation;
        trees still split on individual columns.

    Attributes
    ----------
    forest_ : object
        Final full-data weighted forest, used for prediction.
    estimators_ : list
        Trees in ``forest_``.
    feature_weights_ : ndarray of shape (n_features,)
        Input sampling weights of the final forest and every outer forest.
    feature_weights_history_ : ndarray of shape (n_iterations, n_features)
        Input weights of full-data forests, starting with uniform weights.
    feature_importances_ : ndarray of shape (n_features,)
        Normalized residual-sum-of-squares decreases of the final forest.
    feature_importances_history_ : ndarray of shape (n_iterations, n_features)
        Output importances from each full-data forest.
    interaction_stability_ : dict
        Sorted feature-index tuples (group IDs when ``feature_groups`` is set)
        mapped to the fraction of outer replicates recovering that exact set.
        In signed mode, keys contain ``(feature_or_group_id, direction)``
        atoms. Empty with no bootstraps.
    interactions_ : list of tuple
        Sets ordered by descending stability, then descending size, then
        feature or group IDs (signed atoms in signed mode).
    bootstrap_interactions_ : list of set
        Distinct recovered intersections from each outer replicate.
    bootstrap_samples_ : list of ndarray
        Original input row indices for each outer sample, retaining duplicates.
    interaction_scores_ : dict
        Signed bootstrap-mean prevalence, precision, FSD, MIP, and separate
        FSD/MIP stability fractions, as documented for ``IRFClassifier``.
        Available with ``signed=True``; empty with no bootstraps.
    bootstrap_interaction_scores_ : list of dict
        Per-bootstrap signed scores before averaging.
    leaf_threshold_ : float or None
        Fitted cutoff for leaf selection and signed response binarization.
    signed_ : bool
        Whether the fit used signed discovery.
    feature_groups_ : ndarray of shape (n_features,)
        Fitted column-to-group mapping.

    Notes
    -----
    Regression nodes follow R's ``regTree``. Each bootstrap draw is a separate
    case, so duplicates count toward node size. A node splits only when the
    best candidate strictly reduces the residual sum of squares; classification
    instead keeps zero-gain splits. Leaves predict the in-bag mean, and the
    forest averages leaf means. Split search does not depend on the scale or
    offset of the target: targets are shifted and rescaled within each node
    before searching, while predictions and importances keep original units.

    Supports dense, finite numeric inputs and a single continuous target.
    R's options ``wt.pred.accuracy`` and ``cutoff.unimp.feature`` are not
    implemented. Signed scores use the revised siRF formulas and iRF v3's
    regression cutoff conventions; RIT retains count-based path sampling.

    References
    ----------
    Basu, S., Kumbier, K., Brown, J. B., and Yu, B. (2018). Iterative random
    forests to discover predictive and stable high-order interactions.
    PNAS, 115(8), 1943-1948. doi:10.1073/pnas.1711236115.

    Examples
    --------
    >>> import numpy as np
    >>> from imodels import IRFRegressor
    >>> rng = np.random.RandomState(0)
    >>> X = rng.uniform(size=(200, 4))
    >>> y = (X[:, 0] > 0.5) * (X[:, 1] > 0.5) + 0.1 * rng.normal(size=200)
    >>> model = IRFRegressor(n_estimators=20, n_iterations=3, n_bootstraps=3,
    ...                      leaf_threshold=0.5, random_state=0).fit(X, y)
    >>> model.interactions_[0]
    (0, 1)
    """

    _task = "regression"

    def __init__(
        self, n_estimators=100, n_iterations=5, n_bootstraps=10,
        max_features=1 / 3, max_depth=None, min_samples_split=6,
        min_samples_leaf=1, bootstrap=True, n_estimators_bootstrap=None,
        bootstrap_fraction=1.0, n_rit=100, rit_depth=5, rit_branching=2,
        leaf_threshold=None, n_jobs=None, random_state=None,
        signed=False, feature_groups=None,
    ):
        self.n_estimators = n_estimators
        self.n_iterations = n_iterations
        self.n_bootstraps = n_bootstraps
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.bootstrap = bootstrap
        self.n_estimators_bootstrap = n_estimators_bootstrap
        self.bootstrap_fraction = bootstrap_fraction
        self.n_rit = n_rit
        self.rit_depth = rit_depth
        self.rit_branching = rit_branching
        self.leaf_threshold = leaf_threshold
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.signed = signed
        self.feature_groups = feature_groups

    def _check_parameters(self, n_features):
        threshold = self.leaf_threshold
        if threshold is not None and (
                isinstance(threshold, (bool, np.bool_)) or not isinstance(threshold, Real)
                or not np.isfinite(threshold)):
            raise ValueError("leaf_threshold must be a finite number or None.")
        super()._check_parameters(n_features)

    def _validate_targets(self, X, y):
        X, y = validate_data(self, X, y, dtype=np.float32, accept_sparse=False,
                             y_numeric=True)
        y = np.asarray(y, dtype=np.float64)
        if not np.all(np.isfinite(y)):
            raise ValueError("y must contain finite values.")
        return X, y

    def _prepare_targets(self, y):
        return y

    def _resolve_leaf_selection(self, y):
        self.leaf_threshold_ = (float(np.median(y))
                                if self.signed_ and self.leaf_threshold is None
                                else self.leaf_threshold)
        self._target_index = 1

    def _select_leaf(self, prediction):
        return self.leaf_threshold_ is None or prediction > self.leaf_threshold_

    def predict(self, X):
        """Predict by averaging leaf means of the final full-data forest."""
        check_is_fitted(self, "forest_")
        X = validate_data(self, X, reset=False, dtype=np.float32, accept_sparse=False)
        return self.forest_.predict(X)
