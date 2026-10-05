"""Weighted random forests used internally by iterative random forests.

Each node draws its own candidate features without replacement. A NumPy search
reproduces the split a public sklearn stump would choose, including sklearn's
tie-breaking; nodes where rounding could make the two differ are passed to the
stump itself. No private sklearn tree builder or compiled extension is required.

Classification and regression follow the node rules of R's randomForest as
modified by the paper-era iRF package. Classification nodes hold distinct
in-bag rows weighted by bootstrap multiplicity and keep zero-gain splits.
Regression nodes hold every bootstrap draw, so duplicates count toward node
size, and a split must reduce the residual sum of squares.
"""

import numbers
import threading

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
    shifted = node_y - shift
    _, exponent = np.frexp(np.abs(shifted).max())
    return np.ldexp(shifted, -exponent)


def _goes_left(column, threshold):
    """Route rows as sklearn does: compare float32 inputs in float64.

    Casting the column is required on every NumPy version. NumPy 2 compares a
    float32 array with a Python float in float32, and NumPy 1 does the same
    even for a float64 scalar. A midpoint threshold between adjacent float32
    values can then round up, sending the upper value left.
    """
    return np.asarray(column, dtype=np.float64) <= np.float64(threshold)


_EPS = np.finfo(np.float64).eps
_INT32_MAX = int(np.iinfo(np.int32).max)
_RAND_R_MAX = 2147483647
# sklearn skips split points between values closer than this. Recent versions
# add it in float32 and older ones in float64; points where the two disagree
# are left to sklearn.
_FEATURE_THRESHOLD = 1e-7
# Larger nodes (rows x candidates x classes) go to sklearn, whose per-call
# overhead is then small relative to the search.
_MAX_FAST_CELLS = 200_000
# Regression ties are replayed in Python, so only in small nodes.
_MAX_REPLAY_ROWS = 64
_UNRESOLVED = object()
_thread_state = threading.local()


def _weighted_sample(rng, population, size, probabilities):
    """Return ``rng.choice(population, size, replace=False, p=probabilities)``.

    Makes the same draws as NumPy's legacy ``RandomState.choice``, whose
    stream NumPy keeps fixed, without its per-call input validation. Callers
    must ensure at least ``size`` probabilities are positive.
    """
    p = probabilities.copy()
    found = []
    while len(found) < size:
        draws = rng.random_sample(size - len(found))
        if found:
            p[found] = 0
        cdf = np.cumsum(p)
        cdf /= cdf[-1]
        # New indices in order of first occurrence, as np.unique(return_index).
        found.extend(dict.fromkeys(cdf.searchsorted(draws, side="right").tolist()))
    return population[found]


def _sklearn_feature_order(seed, constant):
    """Order in which sklearn's best splitter evaluates non-constant columns.

    Mirrors the Fisher-Yates draw in ``node_split_best`` at the root, using
    sklearn's ``our_rand_r`` generator seeded as ``Splitter.init`` does.
    """
    # Reseeding is much faster than constructing a RandomState per node.
    generator = getattr(_thread_state, "generator", None)
    if generator is None:
        generator = _thread_state.generator = np.random.RandomState()
    generator.seed(seed)
    state = int(generator.randint(0, _RAND_R_MAX))
    n_features = len(constant)
    features = list(range(n_features))
    remaining, n_constant, n_visited, order = n_features, 0, 0, []
    while remaining > n_constant and (n_visited < n_features or n_visited <= n_constant):
        n_visited += 1
        if state == 0:
            state = 1
        state ^= (state << 13) & 0xFFFFFFFF
        state ^= state >> 17
        state ^= (state << 5) & 0xFFFFFFFF
        j = state % (_RAND_R_MAX + 1) % (remaining - n_constant) + n_constant
        if constant[features[j]]:
            features[j], features[n_constant] = features[n_constant], features[j]
            n_constant += 1
            continue
        remaining -= 1
        features[remaining], features[j] = features[j], features[remaining]
        order.append(features[remaining])
    return order


def _fma(x, y, z):
    """Return ``x * y + z`` rounded once, as a fused multiply-add does.

    Float ratios have power-of-two denominators and Python's integer true
    division rounds correctly, so only the final rounding occurs.
    """
    xn, xd = float(x).as_integer_ratio()
    yn, yd = float(y).as_integer_ratio()
    zn, zd = float(z).as_integer_ratio()
    return (xn * yn * zd + zn * xd * yd) / (xd * yd * zd)


def _sklearn_gini_proxies(w_left, w_right, impurity_left, impurity_right):
    """sklearn's Gini proxy ``-w_right * i_right - w_left * i_left``, rounded
    without a fused multiply-add and with either product fused, since the
    compiler may contract one of them."""
    right, left = w_right * impurity_right, w_left * impurity_left
    return (-right - left, _fma(-w_left, impurity_left, -right),
            _fma(-w_right, impurity_right, -left))


def _sklearn_mse_proxies(x_column, y_column, positions, node_total):
    """sklearn's MSE proxy at each split position of one sorted column.

    Replays ``RegressionCriterion.update`` for unit weights: sums continue
    forward from the previous position, or run backward from the node total
    when that is shorter. Returns None unless equal feature values share y, so
    that sklearn's order within ties cannot change the sums.
    """
    if np.any((x_column[1:] == x_column[:-1]) & (y_column[1:] != y_column[:-1])):
        return None
    y_column = y_column.tolist()
    n = len(y_column)
    pos, sum_left, w_left, proxies = 0, 0.0, 0.0, {}
    for p in positions.tolist():
        if p - pos <= n - p:
            for q in range(pos, p):
                sum_left += y_column[q]
                w_left += 1.0
        else:
            sum_left, w_left = node_total, float(n)
            for q in range(n - 1, p - 1, -1):
                sum_left -= y_column[q]
                w_left -= 1.0
        pos = p
        sum_right = node_total - sum_left
        proxies[p] = (sum_left * sum_left / w_left
                      + sum_right * sum_right / (n - w_left))
    return proxies


def _fast_split(X, y, weights, regression, n_classes, min_samples_leaf, seed):
    """Return sklearn's depth-one best split without calling sklearn.

    Returns ``(column, threshold, weighted_n, impurity)`` for the root and its
    left and right children, None when sklearn would not split, or
    ``_UNRESOLVED`` when the result cannot be guaranteed to match sklearn.
    Classification requires integer weights, so class sums are exact and the
    returned impurities match sklearn bit for bit. Regression callers use only
    the column and threshold. Near ties are settled by reproducing sklearn's
    rounding and feature order.
    """
    n, k = X.shape
    if not regression:
        total_weight = weights.sum()
        if (n * k * n_classes > _MAX_FAST_CELLS or total_weight >= 2 ** 26
                or not np.array_equal(weights, np.rint(weights))):
            return _UNRESOLVED
    order = np.argsort(X, axis=0, kind="stable")
    x_sorted = X[order, np.arange(k)]
    lower, upper = x_sorted[:-1], x_sorted[1:]
    valid32 = upper > lower + np.float32(_FEATURE_THRESHOLD)
    valid64 = (upper.astype(np.float64)
               > lower.astype(np.float64) + _FEATURE_THRESHOLD)
    candidate = valid32 | valid64
    uncertain = valid32 != valid64
    if min_samples_leaf > 1:
        # Row r splits after r + 1 samples; both sides need min_samples_leaf.
        for mask in (candidate, uncertain):
            mask[:min_samples_leaf - 1] = False
            mask[n - min_samples_leaf:] = False
    if not candidate.any():
        return None

    w_left = np.cumsum(weights[order], axis=0)[:-1]
    if regression:
        weighted_y = weights * y
        total_weight, total_sum = weights.sum(), weighted_y.sum()
        sum_left = np.cumsum(weighted_y[order], axis=0)[:-1]
        sum_right = total_sum - sum_left
        w_right = total_weight - w_left
        proxy = sum_left * sum_left / w_left + sum_right * sum_right / w_right
        # Bounds the rounding error of these sums and of sklearn's.
        tolerance = 64 * n * _EPS * max(total_weight, np.abs(weighted_y).sum())
        impurity = np.dot(weighted_y, y) / total_weight - (total_sum / total_weight) ** 2
        if impurity < 1e-9:
            return _UNRESOLVED
    else:
        counts = np.zeros((n, n_classes))
        counts[np.arange(n), y] = weights
        left = np.cumsum(counts[order], axis=0)[:-1]
        total = counts.sum(axis=0)
        right = total - left
        w_right = total_weight - w_left
        impurity_left = 1.0 - (left * left).sum(axis=-1) / (w_left * w_left)
        impurity_right = 1.0 - (right * right).sum(axis=-1) / (w_right * w_right)
        proxy = -w_right * impurity_right - w_left * impurity_left
        tolerance = 64 * _EPS * total_weight
        impurity = 1.0 - (total * total).sum() / (total_weight * total_weight)
        if impurity <= _EPS:
            return None

    best = proxy[candidate].max()
    near = candidate & (proxy >= best - tolerance)
    if (near & uncertain).any():
        return _UNRESOLVED
    rows, columns = np.nonzero(near)
    if rows.size == 1:
        row, column = rows[0], columns[0]
    else:
        # Rounding decides between these candidates, so reproduce sklearn's
        # arithmetic. It keeps the first maximum in feature order, then row.
        tied = sorted(set(columns.tolist()))
        rank = {tied[0]: 0}
        if len(tied) > 1:
            constant32 = x_sorted[-1] <= x_sorted[0] + np.float32(_FEATURE_THRESHOLD)
            constant64 = (x_sorted[-1].astype(np.float64)
                          <= x_sorted[0].astype(np.float64) + _FEATURE_THRESHOLD)
            if (constant32 != constant64).any():
                return _UNRESOLVED
            rank = {c: i for i, c in enumerate(_sklearn_feature_order(seed, constant32))}
        if regression:
            if (n > _MAX_REPLAY_ROWS or uncertain[:, tied].any()
                    or not np.all(weights == 1.0)):
                return _UNRESOLVED
            node_total = np.cumsum(y)[-1]  # sklearn sums in node order
            replayed = {}
            for c in tied:
                replayed[c] = _sklearn_mse_proxies(
                    x_sorted[:, c], y[order[:, c]],
                    np.flatnonzero(candidate[:, c]) + 1, node_total)
                if replayed[c] is None:
                    return _UNRESOLVED
            variants = [[replayed[c][r + 1] for r, c in zip(rows, columns)]]
        else:
            variants = list(zip(*(
                _sklearn_gini_proxies(w_left[r, c], w_right[r, c],
                                      impurity_left[r, c], impurity_right[r, c])
                for r, c in zip(rows, columns))))
        winners = {max(range(rows.size),
                       key=lambda i: (values[i], -rank[columns[i]], -rows[i]))
                   for values in variants}
        if len(winners) > 1:
            return _UNRESOLVED
        i = winners.pop()
        row, column = rows[i], columns[i]

    # sklearn becomes a leaf if the improvement rounds below -EPSILON; leave
    # (near-)zero improvements to sklearn.
    if regression:
        if best - total_sum * total_sum / total_weight <= tolerance:
            return _UNRESOLVED
        weighted_n = impurities = None
    else:
        wl, wr = w_left[row, column], w_right[row, column]
        il, ir = impurity_left[row, column], impurity_right[row, column]
        if impurity - wr / total_weight * ir - wl / total_weight * il <= 64 * _EPS:
            return _UNRESOLVED
        weighted_n = (float(total_weight), float(wl), float(wr))
        impurities = (float(impurity), float(il), float(ir))
    threshold = (float(x_sorted[row, column]) / 2.0
                 + float(x_sorted[row + 1, column]) / 2.0)
    return int(column), threshold, weighted_n, impurities


def _best_split(X, y, weights, regression, n_classes, min_samples_split,
                min_samples_leaf, seed):
    """Return the split of a depth-one sklearn tree fit with ``seed``, or None."""
    split = _fast_split(X, y, weights, regression, n_classes, min_samples_leaf, seed)
    if split is _UNRESOLVED:
        split = _sklearn_split(X, y, weights, regression, min_samples_split,
                               min_samples_leaf, seed)
    return split


def _sklearn_split(X, y, weights, regression, min_samples_split, min_samples_leaf,
                   seed):
    """Fit a public sklearn stump; same return format as ``_fast_split``."""
    stump_class = DecisionTreeRegressor if regression else DecisionTreeClassifier
    stump = stump_class(
        criterion="squared_error" if regression else "gini", max_depth=1,
        min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
        random_state=seed,
    ).fit(X, y, sample_weight=weights)
    tree = stump.tree_
    if tree.node_count == 1:
        return None
    return (int(tree.feature[0]), float(tree.threshold[0]),
            tuple(tree.weighted_n_node_samples[:3].tolist()),
            tuple(tree.impurity[:3].tolist()))


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
        # With too few nonzero probabilities, NumPy's choice raises at the
        # first split; otherwise the unvalidated copy makes the same draws.
        validated_draws = np.count_nonzero(probabilities) < n_candidates
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

            if validated_draws:
                candidates = rng.choice(support, size=n_candidates, replace=False,
                                        p=probabilities)
            else:
                candidates = _weighted_sample(rng, support, n_candidates, probabilities)
            seed = rng.randint(_INT32_MAX)
            if regression:
                centered = node_y - value
                stump_y = _rescale_for_split_search(node_y, value)
            else:
                stump_y = y[indices]
            split = _best_split(
                X[indices[:, None], candidates], stump_y, node_weights, regression,
                n_classes, self.min_samples_split, self.min_samples_leaf, seed,
            )
            if split is None:
                continue
            column, threshold, weighted_n, split_impurity = split
            feature = int(candidates[column])
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
                decrease = max(0.0, weighted_n[0] * split_impurity[0]
                               - weighted_n[1] * split_impurity[1]
                               - weighted_n[2] * split_impurity[2])
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
        is_max = class_masses == class_masses.max(axis=1, keepdims=True)
        for node_id in np.flatnonzero(self.is_leaf_ & (is_max.sum(axis=1) > 1)):
            tied = np.flatnonzero(is_max[node_id])
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


def _fit_tree_paths(tree, X, y, feature_weights, sample_weight, n_classes):
    """Fit a tree and return its terminal paths measured on its training rows."""
    tree.fit(X, y, feature_weights, sample_weight, n_classes)
    return tree.terminal_paths(X, sample_weight=sample_weight)


def _forest_paths(forests, datasets, n_jobs=None):
    """Fit forests in one pool of workers; yield each forest's terminal paths.

    ``datasets`` gives ``(X, y, feature_weights, sample_weight, n_classes)``
    for each forest and is consumed lazily. Workers return the trees'
    ``terminal_paths`` on their training rows rather than the trees, and
    results stream back in order, so memory holds about one forest's paths
    at a time. Sharing one pool avoids idle workers between forests.
    """
    if not forests:
        return

    def tasks():
        for forest, data in zip(forests, datasets):
            trees, fit_args = forest._prepare(*data)
            for tree in trees:
                yield delayed(_fit_tree_paths)(tree, *fit_args)

    try:
        results = Parallel(n_jobs=n_jobs, return_as="generator")(tasks())
    except TypeError:  # joblib < 1.3 cannot stream results
        results = iter(Parallel(n_jobs=n_jobs)(tasks()))
    for forest in forests:
        yield [leaf for _ in range(forest.n_estimators) for leaf in next(results)]


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
        trees, fit_args = self._prepare(X, y, feature_weights, sample_weight, n_classes)
        # Tree building is mostly Python, so processes (joblib's default
        # backend) run in parallel where threads would contend for the GIL.
        self.estimators_ = Parallel(n_jobs=self.n_jobs)(
            delayed(tree.fit)(*fit_args) for tree in trees)
        importances = np.mean([tree.raw_feature_importances_
                               for tree in self.estimators_], axis=0)
        total = importances.sum()
        self.raw_feature_importances_ = importances
        self.feature_importances_ = (importances / total if total > 0 else importances)
        return self

    def _prepare(self, X, y, feature_weights, sample_weight=None, n_classes=None):
        """Validate inputs; return unfitted trees and their shared fit arguments."""
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
        seeds = rng.randint(_INT32_MAX, size=self.n_estimators)
        trees = [_WeightedTree(max_features, self.max_depth, self.min_samples_split,
                               self.min_samples_leaf, self.bootstrap, int(seed), self.task)
                 for seed in seeds]
        return trees, (X, y, self.feature_weights_, sample_weight, self.n_classes_)

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
