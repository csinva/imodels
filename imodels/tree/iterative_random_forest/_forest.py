"""Weighted random forests used internally by iterative random forests.

Each node draws its own candidate features without replacement. Public sklearn
stumps perform the numerical CART split search; no private sklearn tree builder
or compiled extension is required.

Classification and regression follow the node rules of R's randomForest as
modified by the paper-era iRF package. Classification nodes hold distinct
in-bag rows weighted by bootstrap multiplicity and keep zero-gain splits.
Regression nodes hold every bootstrap draw, so duplicates count toward node
size, and a split must reduce the residual sum of squares.
"""

import numbers

import numpy as np
from joblib import Parallel, delayed
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils import check_random_state


_TASKS = ("classification", "regression")


def _rescale_for_split_search(node_y, mean):
    """Shift and rescale a node's targets before sklearn's split search.

    The best residual-sum-of-squares split is unchanged by shifting and
    rescaling y. Doing so within each node keeps sklearn's absolute impurity
    tolerance and its variance formula from hiding splits for very small, very
    large, or offset targets, including deep nodes whose spread is tiny
    relative to the root's. The shift is the observed value nearest the mean
    and the scale is a power of two, so integer-valued targets stay exact and
    equal-gain splits are broken as they would be on the raw targets.
    """
    distance = np.abs(node_y - mean)
    shift = node_y[distance == distance.min()].min()
    _, exponent = np.frexp(np.abs(node_y - shift).max())
    return np.ldexp(node_y - shift, -exponent)


def _goes_left(column, threshold):
    """Route rows as sklearn does: compare float32 inputs in float64.

    Casting the column is required on every NumPy version. NumPy 2 compares a
    float32 array with a Python float in float32, and NumPy 1 does the same
    even for a float64 scalar. A midpoint threshold between adjacent float32
    values can then round up, sending the upper value left.
    """
    return np.asarray(column, dtype=np.float64) <= np.float64(threshold)


class _WeightedTree:
    """A CART tree with weighted candidate-feature draws at each node."""

    def __init__(self, max_features, max_depth, min_samples_split,
                 min_samples_leaf, bootstrap, random_state,
                 task="classification"):
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.task = task

    def fit(self, X, y, feature_weights, sample_weight, n_classes=None):
        if self.task not in _TASKS:
            raise ValueError(f"task must be one of {_TASKS}.")
        regression = self.task == "regression"
        rng = (np.random.RandomState() if self.random_state is None
               else check_random_state(self.random_state))
        n_samples, self.n_features_in_ = X.shape
        self.n_classes_ = None if regression else n_classes
        if self.bootstrap:
            self.bootstrap_indices_ = rng.randint(n_samples, size=n_samples)
        else:
            self.bootstrap_indices_ = np.arange(n_samples)
        if regression:
            # R's regTree copies every bootstrap draw into the node, so node
            # sizes count duplicates. Weights then carry observation weights only.
            weights = sample_weight
            active = self.bootstrap_indices_[weights[self.bootstrap_indices_] > 0]
        else:
            # R's classification tree keeps distinct in-bag rows and stores
            # bootstrap multiplicity as a case weight.
            multiplicity = np.bincount(self.bootstrap_indices_, minlength=n_samples)
            weights = sample_weight * multiplicity
            active = np.flatnonzero(weights > 0)
        if not active.size:
            raise ValueError("The tree bootstrap sample has zero total sample weight.")

        support = np.flatnonzero(feature_weights > 0)
        probabilities = feature_weights[support] / feature_weights[support].sum()
        n_candidates = min(self.max_features, support.size)
        left, right, features, thresholds = [], [], [], []
        values, masses, impurities, counts, class_masses = [], [], [], [], []
        raw_importances = np.zeros(self.n_features_in_)
        # Each task records parent, side, observation indices, and node depth.
        tasks = [(-1, False, active, 0)]
        while tasks:
            parent, is_left, indices, depth = tasks.pop()
            node_id = len(features)
            if parent >= 0:
                (left if is_left else right)[parent] = node_id
            left.append(-1)
            right.append(-1)
            features.append(-2)
            thresholds.append(-2.0)
            node_weights = weights[indices]
            mass = node_weights.sum()
            if regression:
                node_y = y[indices]
                value = np.dot(node_weights, node_y) / mass
                impurity = max(0.0, np.dot(node_weights, (node_y - value) ** 2) / mass)
                pure = node_y.min() == node_y.max()
            else:
                class_mass = np.bincount(y[indices], weights=node_weights,
                                         minlength=n_classes)
                class_masses.append(class_mass)
                value = class_mass / mass
                impurity = max(0.0, 1.0 - np.dot(value, value))
                pure = np.count_nonzero(class_mass) <= 1
            values.append(value)
            masses.append(mass)
            impurities.append(impurity)
            counts.append(indices.size)
            if (pure
                    or indices.size < self.min_samples_split
                    or indices.size < 2 * self.min_samples_leaf
                    or (self.max_depth is not None and depth >= self.max_depth)):
                continue

            candidates = rng.choice(support, size=n_candidates, replace=False,
                                    p=probabilities)
            stump_class = DecisionTreeRegressor if regression else DecisionTreeClassifier
            stump = stump_class(
                criterion="squared_error" if regression else "gini", max_depth=1,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                random_state=rng.randint(np.iinfo(np.int32).max),
            )
            if regression:
                centered = node_y - value
                stump_y = _rescale_for_split_search(node_y, value)
            else:
                stump_y = y[indices]
            stump.fit(X[np.ix_(indices, candidates)], stump_y,
                      sample_weight=node_weights)
            split = stump.tree_
            if split.node_count == 1:
                continue
            feature = int(candidates[split.feature[0]])
            threshold = float(split.threshold[0])
            goes_left = _goes_left(X[indices, feature], threshold)
            if goes_left.all() or not goes_left.any():
                # Unreachable while routing matches sklearn; guards against an
                # endless chain of one-sided splits if it ever does not.
                continue
            if regression:
                # RSS decrease (R's IncNodePurity) in original units, computed
                # from child means: never negative, and zero only when the
                # child means are equal. R's regTree requires a strict decrease.
                w_left = node_weights[goes_left].sum()
                w_right = mass - w_left
                mean_left = np.dot(node_weights[goes_left], centered[goes_left]) / w_left
                mean_right = np.dot(node_weights[~goes_left], centered[~goes_left]) / w_right
                decrease = w_left * w_right / mass * (mean_left - mean_right) ** 2
                if not decrease > 0:
                    continue
            else:
                # Weighted Gini decrease (R's MeanDecreaseGini). Zero-gain
                # splits are retained: they are necessary to discover XOR.
                decrease = max(0.0, split.weighted_n_node_samples[0] * split.impurity[0]
                               - split.weighted_n_node_samples[1] * split.impurity[1]
                               - split.weighted_n_node_samples[2] * split.impurity[2])
            left_indices = indices[goes_left]
            right_indices = indices[~goes_left]
            features[node_id] = feature
            thresholds[node_id] = threshold
            raw_importances[feature] += decrease
            tasks.append((node_id, False, right_indices, depth + 1))
            tasks.append((node_id, True, left_indices, depth + 1))

        self.children_left_ = np.asarray(left, dtype=np.intp)
        self.children_right_ = np.asarray(right, dtype=np.intp)
        self.feature_ = np.asarray(features, dtype=np.intp)
        self.threshold_ = np.asarray(thresholds)
        self.value_ = np.asarray(values)
        self.weighted_n_node_samples_ = np.asarray(masses)
        self.n_node_samples_ = np.asarray(counts, dtype=np.intp)
        self.impurity_ = np.asarray(impurities)
        self.node_count_ = len(features)
        self.is_leaf_ = self.children_left_ < 0
        if regression:
            self.leaf_prediction_ = self.value_.copy()
        else:
            self.leaf_prediction_ = self._leaf_classes(np.asarray(class_masses), rng)
        # R's mean decrease Gini aggregates raw reductions across trees before
        # normalizing features, rather than normalizing each tree separately.
        self.raw_feature_importances_ = raw_importances
        total = raw_importances.sum()
        self.feature_importances_ = (raw_importances / total if total > 0
                                     else raw_importances)
        return self

    def _leaf_classes(self, class_masses, rng):
        """Label each leaf by its majority class, breaking ties at random.

        R's randomForest breaks exact ties in leaf class mass uniformly at
        random. Always taking the lowest class index would exclude every tied
        binary leaf from RIT. Drawing only on ties leaves the random stream
        otherwise unchanged.
        """
        labels = np.argmax(class_masses, axis=1)
        for node_id in np.flatnonzero(self.is_leaf_):
            class_mass = class_masses[node_id]
            tied = np.flatnonzero(class_mass == class_mass.max())
            if tied.size > 1:
                labels[node_id] = tied[rng.randint(tied.size)]
        return labels

    def apply(self, X):
        """Return terminal node IDs for each observation."""
        X = np.asarray(X, dtype=np.float32)
        leaves = np.zeros(X.shape[0], dtype=np.intp)
        tasks = [(0, np.arange(X.shape[0]))]
        while tasks:
            node_id, indices = tasks.pop()
            if not indices.size:
                continue
            if self.children_left_[node_id] < 0:
                leaves[indices] = node_id
                continue
            goes_left = _goes_left(X[indices, self.feature_[node_id]],
                                   self.threshold_[node_id])
            tasks.append((self.children_right_[node_id], indices[~goes_left]))
            tasks.append((self.children_left_[node_id], indices[goes_left]))
        return leaves

    def predict_proba(self, X):
        if self.task == "regression":
            raise AttributeError("predict_proba is not available for regression trees.")
        return self.value_[self.apply(X)]

    def predict(self, X):
        """Return leaf means (regression) or tie-broken leaf classes."""
        return self.leaf_prediction_[self.apply(X)]

    def terminal_paths(self, X=None, sample_weight=None):
        """Return (feature set, leaf prediction, weighted leaf mass).

        The leaf prediction is the tie-broken class index for classification
        and the in-bag mean for regression.

        If X is supplied, route every row through this tree to measure leaf
        population, as required by the iRF path-sampling procedure. Otherwise
        report the inner bootstrap's weighted training population.
        """
        if X is None:
            masses = self.weighted_n_node_samples_
        else:
            masses = np.bincount(self.apply(X), weights=sample_weight,
                                 minlength=self.node_count_)
        paths = []
        tasks = [(0, frozenset())]
        while tasks:
            node_id, path = tasks.pop()
            if self.children_left_[node_id] < 0:
                prediction = self.leaf_prediction_[node_id]
                prediction = (float(prediction) if self.task == "regression"
                              else int(prediction))
                paths.append((path, prediction, float(masses[node_id])))
            else:
                child_path = path | {int(self.feature_[node_id])}
                tasks.append((self.children_right_[node_id], child_path))
                tasks.append((self.children_left_[node_id], child_path))
        return paths


class _WeightedForest:
    """Internal forest backend.

    For classification, labels must already be encoded as integers.
    """

    def __init__(self, n_estimators=100, max_features="sqrt", max_depth=None,
                 min_samples_split=2, min_samples_leaf=1, bootstrap=True,
                 n_jobs=None, random_state=None, task="classification"):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.bootstrap = bootstrap
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.task = task

    def fit(self, X, y, feature_weights, sample_weight=None, n_classes=None):
        if self.task not in _TASKS:
            raise ValueError(f"task must be one of {_TASKS}.")
        X = np.asarray(X, dtype=np.float32, order="C")
        self.n_features_in_ = X.shape[1]
        if self.task == "regression":
            y = np.asarray(y, dtype=np.float64)
            self.n_classes_ = None
        else:
            y = np.asarray(y, dtype=np.intp)
            self.n_classes_ = int(y.max()) + 1 if n_classes is None else n_classes
            self.classes_ = np.arange(self.n_classes_)
        feature_weights = np.asarray(feature_weights, dtype=float)
        if (feature_weights.shape != (self.n_features_in_,)
                or not np.isfinite(feature_weights).all()
                or (feature_weights < 0).any() or feature_weights.sum() <= 0):
            raise ValueError("feature_weights must be finite, nonnegative, and have positive sum.")
        self.feature_weights_ = feature_weights / feature_weights.sum()
        if sample_weight is None:
            sample_weight = np.ones(X.shape[0], dtype=float)
        else:
            sample_weight = np.asarray(sample_weight, dtype=float)
        if self.max_features is None:
            max_features = self.n_features_in_
        elif self.max_features == "sqrt":
            max_features = max(1, int(np.sqrt(self.n_features_in_)))
        elif self.max_features == "log2":
            max_features = max(1, int(np.log2(self.n_features_in_)))
        elif isinstance(self.max_features, numbers.Integral):
            max_features = self.max_features
        else:
            # Floor with a relative guard so fractions such as 1/3 reproduce
            # R's floor(p / 3) despite binary rounding.
            max_features = max(1, int(np.floor(
                self.max_features * self.n_features_in_ * (1 + 1e-12))))
        self.max_features_ = max_features
        rng = (np.random.RandomState() if self.random_state is None
               else check_random_state(self.random_state))
        seeds = rng.randint(np.iinfo(np.int32).max, size=self.n_estimators)
        self.estimators_ = Parallel(n_jobs=self.n_jobs, prefer="threads")(
            delayed(_WeightedTree(
                max_features, self.max_depth, self.min_samples_split,
                self.min_samples_leaf, self.bootstrap, int(seed), self.task,
            ).fit)(X, y, self.feature_weights_, sample_weight, self.n_classes_)
            for seed in seeds
        )
        importances = np.mean([tree.raw_feature_importances_
                               for tree in self.estimators_], axis=0)
        total = importances.sum()
        self.raw_feature_importances_ = importances
        self.feature_importances_ = (importances / total if total > 0 else importances)
        return self

    def predict_proba(self, X):
        if self.task == "regression":
            raise AttributeError("predict_proba is not available for regression forests.")
        probabilities = np.zeros((len(X), self.n_classes_))
        for tree in self.estimators_:
            probabilities += tree.predict_proba(X)
        return probabilities / len(self.estimators_)

    def predict(self, X):
        if self.task == "regression":
            return np.mean([tree.predict(X) for tree in self.estimators_], axis=0)
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def apply(self, X):
        return np.column_stack([tree.apply(X) for tree in self.estimators_])
