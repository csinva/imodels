"""Iterative random forests and bootstrap-stable feature interactions.

Based on Basu et al. (2018), Supplement Algorithm 2, with classification and
regression following the paper-era R ``iRF`` package:
https://arxiv.org/abs/1706.08457. No dependency on the historical ``irf`` package
or private scikit-learn tree builders.
"""

from imodels.util.introspection import TextMixin
from collections import Counter
from numbers import Integral, Real

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils import check_random_state, resample
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_array, check_is_fitted

try:
    from sklearn.utils.validation import validate_data
except ImportError:  # scikit-learn < 1.6
    def validate_data(estimator, *args, **kwargs):
        return estimator._validate_data(*args, **kwargs)

from ._forest import _WeightedForest, _forest_paths
from ._rit import _random_intersection_trees


def _check_integer(name, value, minimum):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, Integral) or value < minimum):
        raise ValueError(f"{name} must be an integer >= {minimum}.")


class _IRFBase(TextMixin, BaseEstimator):
    """Shared iterative reweighting, outer bootstrap, and RIT stability."""

    _task = None

    def _check_parameters(self, n_features):
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

    def _resolve_leaf_selection(self):
        """Resolve fitted state used by ``_select_leaf``."""

    def _forest_n_classes(self):
        return None

    def _stratify_labels(self, y_fit):
        return None

    def _select_leaf(self, prediction):
        """Return whether a leaf with this prediction enters RIT."""
        raise NotImplementedError

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
        self._resolve_leaf_selection()

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
        outer_samples = (
            (X[indices], y_fit[indices], self.feature_weights_, weights[indices],
             self._forest_n_classes())
            for indices, _, _ in replicates
        )
        leaves_per_forest = _forest_paths(outer_forests, outer_samples, self.n_jobs)
        for (_, _, rit_seed), leaves in zip(replicates, leaves_per_forest):
            paths, masses = [], []
            for path, prediction, mass in leaves:
                if mass > 0 and self._select_leaf(prediction):
                    paths.append(path)
                    masses.append(mass)
            interactions = _random_intersection_trees(
                paths, masses, n_trees=self.n_rit, max_depth=self.rit_depth,
                n_children=self.rit_branching, random_state=rit_seed,
            )
            self.bootstrap_interactions_.append(interactions)

        counts = Counter(
            interaction for replicate in self.bootstrap_interactions_
            for interaction in replicate
        )
        self.interactions_ = sorted(counts, key=lambda s: (-counts[s], -len(s), s))
        self.interaction_stability_ = {
            interaction: counts[interaction] / self.n_bootstraps
            for interaction in self.interactions_
        }
        return self


class IRFClassifier(ClassifierMixin, _IRFBase):
    """Iteratively reweight forests and discover stable feature interactions.

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
        Number of outer bootstrap forests; zero enables prediction only.
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
        Sorted feature-index tuples mapped to the fraction of outer replicates
        recovering that exact set. Sets contain at least two features. Stability holds
        the learned feature weights fixed.
    interactions_ : list of tuple
        Sets ordered by descending stability, then descending size, then
        feature indices. No split directions or thresholds are retained.
    bootstrap_interactions_ : list of set
        Distinct recovered intersections from each outer replicate.
    bootstrap_samples_ : list of ndarray
        Original input row indices for each outer sample, retaining duplicates.
    interaction_class_ : object
        Resolved target class for interaction discovery.

    Notes
    -----
    Supports dense, finite numeric inputs and single-output classification.
    This is original unsigned iRF. Nodes are split by a NumPy search that
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
        n_jobs=None, random_state=None,
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

    def _resolve_leaf_selection(self):
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
        Number of outer bootstrap forests; zero enables prediction only.
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
        like R's ``rit.param$class.cut``. None uses every leaf, R's default.
    n_jobs : int or None, default=None
        Parallel jobs for fitting trees. None uses one; -1 uses all processors.
    random_state : int, RandomState or None, default=None
        Controls forests, outer samples, and RIT.

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
        Sorted feature-index tuples mapped to the fraction of outer replicates
        recovering that exact set.
    interactions_ : list of tuple
        Sets ordered by descending stability, then descending size, then
        feature indices.
    bootstrap_interactions_ : list of set
        Distinct recovered intersections from each outer replicate.
    bootstrap_samples_ : list of ndarray
        Original input row indices for each outer sample, retaining duplicates.

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
    R's options ``wt.pred.accuracy``, ``cutoff.unimp.feature`` and
    ``varnames.grp`` are not implemented.

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

    def _select_leaf(self, prediction):
        return self.leaf_threshold is None or prediction > self.leaf_threshold

    def predict(self, X):
        """Predict by averaging leaf means of the final full-data forest."""
        check_is_fitted(self, "forest_")
        X = validate_data(self, X, reset=False, dtype=np.float32, accept_sparse=False)
        return self.forest_.predict(X)
