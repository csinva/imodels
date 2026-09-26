"""Certifiably optimal sparse decision trees, discovered by autoresearch.

`FastSmallTreeClassifier` returns the tree minimising misclassification rate plus
a penalty per leaf, and certifies that no other tree on the same binarized
features scores better. The search behind it lives in `solver.py`.
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree._tree import Tree

from imodels.tree.optimal_tree.solver import (HAVE_NUMBA, ST_LB, ST_UB, BinaryEncoder,
                                              BitDataset, CompiledOptimizer, TargetEncoder,
                                              TreeClassifier, cluster_rows)
from imodels.util.arguments import (check_fit_arguments, check_predict_X,
                                    decode_labels)
from imodels.util.introspection import RuleInspectionMixin

NUMBA_HINT = (
    "FastSmallTreeClassifier needs numba, which is not installed. Install it with "
    "`pip install numba` (or `pip install imodels[optional]`). The search itself is "
    "compiled, and interpreting it is orders of magnitude slower, so there is no "
    "pure-Python fallback."
)


def _class_weights(costs: np.ndarray):
    """Per-class weights w when the objective is a reweighting of the counts.

    That is the case when a mistake on a row of class j costs w[j] whatever was
    predicted (zero diagonal, each column constant off it): the default objective
    and ``balance``. Returns None for any other cost matrix.
    """
    C = np.asarray(costs, dtype=np.float64)
    K = C.shape[0]
    if K < 2 or np.any(np.diag(C) != 0):
        return None
    off = ~np.eye(K, dtype=bool)
    w = np.array([C[off[:, j], j][0] for j in range(K)])
    if not all(np.all(C[off[:, j], j] == w[j]) for j in range(K)):
        return None
    return w


def _leaf_value(dist: np.ndarray, costs: np.ndarray, weights) -> np.ndarray:
    """A node's sklearn ``value``: normalised, with ``argmax`` the solver's choice."""
    if weights is not None:
        v = weights * dist
    else:
        cost = np.asarray(costs, dtype=np.float64) @ dist
        v = cost.max() - cost
    if v.sum() <= 0:              # an empty node, or one where every label costs the same
        v = np.zeros_like(dist)
        v[int(np.argmin(np.asarray(costs, dtype=np.float64) @ dist))] = 1.0
    return v / v.sum()


class FastSmallTreeClassifier(RuleInspectionMixin, ClassifierMixin, BaseEstimator):
    """A decision tree that is certifiably optimal for its objective.

    Fits the tree minimising

        misclassification rate + ``regularization`` * (number of leaves)

    over *every* decision tree on the binarized features, and proves that no
    other such tree scores better. This is the objective of `GOSDT
    <https://arxiv.org/abs/2006.08690>`_ (Lin et al., ICML 2020); the
    branch-and-bound search is described in `solver.py` and was developed by
    autoresearch (see `Agentic-imodels <https://github.com/csinva/agentic-imodels>`_).

    Greedy trees (CART, C4.5) pick each split without regard to the splits below
    it, so they can miss the best tree of a given size. This model does not:
    when ``optimal_`` is True, the objective it reports is a proven minimum.

    Depth is not a parameter. ``regularization`` sets the price of a leaf, and
    the optimal tree at that price is as deep as it needs to be. Since a leaf
    that does not improve the misclassification rate by ``regularization`` is
    never worth adding, the trees stay small.

    Parameters
    ----------
    regularization : float, default=0.05
        Penalty added to the objective for each leaf, in units of
        misclassification rate. Larger values give smaller trees. Values below
        ``1 / n_samples`` are not useful, since a leaf fixing a single training
        point already pays for itself.
    time_limit : float, default=60.0
        Seconds after which the search returns the best tree found so far rather
        than continuing. ``optimal_`` is then False and a warning is raised.
        ``0`` means no limit, which can run for a very long time on wide or
        continuous data.
    balance : bool, default=False
        Weigh the classes equally, optimising balanced accuracy rather than
        accuracy.
    costs : array-like of shape (n_classes, n_classes), default=None
        ``costs[i, j]`` is the cost of predicting class ``i`` when the truth is
        class ``j``. Overrides ``balance``.
    memory_limit : int, default=0
        Resident memory in bytes above which the search stops, as the time limit
        does. ``0`` means no limit. Memory grows with the number of subproblems
        kept, which is what makes hard instances expensive.
    verbose : bool, default=False
        Print search statistics after fitting.

    Attributes
    ----------
    optimal_ : bool
        Whether optimality was certified. False means a limit was reached first,
        so the tree is the best one found rather than a proven optimum.
    objective_ : float
        Objective value of the returned tree: its misclassification rate plus
        ``regularization`` times its number of leaves.
    lowerbound_, upperbound_ : float
        Interval the search closed on the optimal objective; equal when
        ``optimal_`` is True.
    n_leaves_ : int
        Number of leaves in the returned tree.
    complexity_ : int
        Number of splits in the returned tree, the imodels measure of size.
    n_binary_features_ : int
        Number of binary features the encoder produced from X. A continuous
        column becomes one feature per distinct value, which is what makes
        continuous data expensive to solve exactly.
    estimator_ : sklearn.tree.DecisionTreeClassifier
        The fitted tree as an sklearn decision tree, which `predict` delegates
        to, so sklearn's tree tooling (``plot_tree``, ``export_text``,
        ``feature_importances_``, dtreeviz) works on it directly. Each split is
        placed midway between the training values on either side of the rule, as
        sklearn places its own, so it routes every training row exactly as the
        certified tree does. A new value strictly between two consecutive
        training values of a column is routed by that midpoint, sklearn's
        convention; the objective is defined on the training rows, so this
        changes nothing the certificate covers.
    tree_ : dict
        The same tree as nested dicts of named rules, which printing the model
        shows.
    time_ : float
        Seconds spent in the search.
    classes_ : ndarray
        Class labels seen during fit.

    Notes
    -----
    The compiled search takes about 20 seconds to build the first time it runs
    on a machine. The result is cached on disk, so later processes load it in
    about a second. Set ``OPTTREE_NUMBA_CACHE=0`` to skip the cache if its
    directory is unwritable.

    Examples
    --------
    >>> import numpy as np
    >>> from imodels import FastSmallTreeClassifier
    >>> X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]] * 8)
    >>> y = (X[:, 0] == X[:, 1]).astype(int)
    >>> model = FastSmallTreeClassifier(regularization=0.01).fit(X, y)
    >>> model.optimal_
    True
    """

    def __init__(self, regularization: float = 0.05, time_limit: float = 60.0,
                 balance: bool = False, costs=None, memory_limit: int = 0,
                 verbose: bool = False):
        self.regularization = regularization
        self.time_limit = time_limit
        self.balance = balance
        self.costs = costs
        self.memory_limit = memory_limit
        self.verbose = verbose

    def fit(self, X, y, feature_names=None):
        """Fit the optimal tree.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
        feature_names : list of str, optional
            Names used when printing the model; taken from X's columns when it
            is a DataFrame.

        Returns
        -------
        self
        """
        if not HAVE_NUMBA:
            raise ImportError(NUMBA_HINT)
        X, y, feature_names = check_fit_arguments(self, X, y, feature_names)
        # the certificate rests on bounds of the form "a tree with a leaves costs at least
        # a * regularization" and "a tree's loss cannot fall below zero", which need a
        # nonnegative penalty and nonnegative, finite costs
        lam = float(self.regularization)
        if not np.isfinite(lam) or lam < 0.0:
            raise ValueError(f"regularization must be a finite number >= 0, got {self.regularization!r}")
        if self.costs is not None:
            C = np.asarray(self.costs, dtype=np.float64)
            if not np.all(np.isfinite(C)) or np.any(C < 0.0):
                raise ValueError("costs must be finite and nonnegative")
        frame = pd.DataFrame(X, columns=list(feature_names))

        self.encoder_ = BinaryEncoder().fit(frame)
        Xb = self.encoder_.transform(frame)
        self.target_encoder_ = TargetEncoder().fit(y)
        y_idx = self.target_encoder_.transform(y)
        # the objective does not depend on the row order, but the kernels skip the
        # empty words of a capture mask, so rows are clustered to keep the capture
        # sets of deep nodes contiguous
        perm = cluster_rows(Xb, y_idx, len(self.classes_), self.encoder_.groups)
        data = BitDataset(Xb[perm], y_idx[perm], len(self.classes_),
                          costs=self.costs, balance=self.balance)
        self.n_binary_features_ = data.m

        # n_jobs=1: the search has a parallel phase, but a library model should not
        # take every core of the caller's machine without being asked
        opt = CompiledOptimizer(data, self.regularization, groups=self.encoder_.groups,
                                time_limit=self.time_limit, memory_limit=self.memory_limit,
                                verbose=self.verbose, n_jobs=1)
        root = opt.run()          # a row index into the compiled engine's node store
        self.optimal_ = opt.optimal
        self.stop_reason_ = opt.stop_reason
        self.time_ = opt.elapsed
        self.iterations_ = opt.iterations
        self.lowerbound_ = float(opt.st[ST_LB][root])
        self.upperbound_ = float(opt.st[ST_UB][root])
        if not self.optimal_:
            warnings.warn(
                f"{self.stop_reason_} limit reached before optimality was certified; "
                "returning the best tree found. Raise time_limit, or raise "
                "regularization to search over smaller trees.", RuntimeWarning)

        self.tree_ = self._decode(opt.extract(root), data)
        opt.release()             # the node store is large and is not needed past extraction
        certified = TreeClassifier(self.tree_)
        self.objective_ = certified.risk()
        self.n_leaves_ = certified.leaves()
        self.complexity_ = max(self.n_leaves_ - 1, 0)
        self.estimator_, self._node_proba_ = self._to_sklearn(self.tree_, frame, data.costs)
        self._check_sklearn_tree(certified, frame)
        if self.verbose:
            print(f"objective {self.objective_:.6g} in {self.time_:.3g}s "
                  f"({self.iterations_} subproblems, optimal={self.optimal_})")
        return self

    def _decode(self, node: dict, data: BitDataset) -> dict:
        """Turn the solver's tree of capture sets into one of named rules.

        Each leaf keeps the class counts of the training rows it captures, so
        that `predict_proba` has something to report.
        """
        if "prediction" in node:
            count, dist, max_loss, _, _, prediction = data.leaf_stats(node["key"])
            return {
                "prediction": self.target_encoder_.inverse(prediction),
                "name": "class",
                "loss": float(max_loss),
                "complexity": float(self.regularization),
                "count": int(count),
                "dist": np.asarray(dist, dtype=float).tolist(),
            }
        rule = self.encoder_.rules[node["feature"]]
        return {
            "feature": int(rule["feature"]),
            "name": rule["name"],
            "relation": rule["relation"],
            "reference": rule["reference"],
            "type": rule["type"],
            "true": self._decode(node["true"], data),
            "false": self._decode(node["false"], data),
        }

    def _to_sklearn(self, tree: dict, frame: pd.DataFrame, costs: np.ndarray):
        """The certified tree as a fitted `DecisionTreeClassifier`.

        sklearn sends a row left when ``x[feature] <= threshold``, so a rule's
        false side becomes the left child and its true side the right one. Every
        rule is a threshold on a numeric column (``check_fit_arguments`` admits
        nothing else), so every rule maps onto one sklearn split.

        sklearn predicts ``argmax`` of a leaf's ``value``, so ``value`` must pick
        the class the solver certified. The solver picks ``argmin(costs @ dist)``
        with the first class winning ties, which for the default objective and
        for ``balance`` is ``argmax`` of the class-weighted counts, what sklearn
        itself stores under ``class_weight``. A general ``costs`` matrix has no
        such weighting, so there ``value`` holds each class's saving over the
        costliest one, which ``argmax`` ranks the same way.

        Returns the estimator and each node's empirical class frequencies, which
        `predict_proba` reports.
        """
        K = len(self.classes_)
        weights = _class_weights(costs)
        values_of = {}                      # column -> its sorted training values

        def threshold(node):
            j = node["feature"]
            if node["type"] == "categorical":
                raise ValueError("categorical rules cannot be expressed as sklearn splits")
            if j not in values_of:
                values_of[j] = np.unique(frame.iloc[:, j].to_numpy(dtype=np.float64))
            vals, ref = values_of[j], node["reference"]
            # on the training values, ">= ref" and "== ref" (a two-valued column,
            # ref its larger value) both hold exactly for the values >= ref
            return 0.5 * (vals[vals < ref].max() + vals[vals >= ref].min())

        rows = []

        def add(node, depth):
            i = len(rows)
            rows.append(None)
            if "prediction" in node:
                dist = np.asarray(node["dist"], dtype=np.float64)
                rows[i] = (-1, -1, -2, -2.0, dist, int(node["prediction"]))
                return dist, depth
            left = len(rows)
            left_dist, left_depth = add(node["false"], depth + 1)
            right = len(rows)
            right_dist, right_depth = add(node["true"], depth + 1)
            dist = left_dist + right_dist
            rows[i] = (left, right, int(node["feature"]), threshold(node), dist, None)
            return dist, max(left_depth, right_depth)

        _, max_depth = add(tree, 0)

        est = Tree(self.n_features_in_, np.array([K], dtype=np.intp), 1)
        nodes = np.zeros(len(rows), dtype=est.__getstate__()["nodes"].dtype)
        values = np.zeros((len(rows), 1, K))
        node_proba = np.zeros((len(rows), K))
        for i, (left, right, feature, thr, dist, prediction) in enumerate(rows):
            n = dist.sum()
            freq = dist / n if n > 0 else np.full(K, 1.0 / K)
            nodes[i]["left_child"], nodes[i]["right_child"] = left, right
            nodes[i]["feature"], nodes[i]["threshold"] = feature, thr
            nodes[i]["impurity"] = 1.0 - float(np.sum(freq ** 2))
            nodes[i]["n_node_samples"] = nodes[i]["weighted_n_node_samples"] = n
            if "missing_go_to_left" in nodes.dtype.names:
                nodes[i]["missing_go_to_left"] = 1   # a missing value fails a rule: its false side
            values[i, 0] = _leaf_value(dist, costs, weights)
            node_proba[i] = freq
            if prediction is not None and int(np.argmax(values[i, 0])) != prediction:
                raise AssertionError("sklearn leaf value does not pick the certified class")
        est.__setstate__({"max_depth": max_depth, "node_count": len(rows),
                          "nodes": nodes, "values": values})

        clf = DecisionTreeClassifier()
        clf.tree_ = est
        clf.classes_ = self.classes_
        clf.n_classes_ = K
        clf.n_outputs_ = 1
        clf.n_features_in_ = self.n_features_in_
        clf.max_features_ = self.n_features_in_
        if hasattr(self, "feature_names_in_"):
            clf.feature_names_in_ = self.feature_names_in_
        return clf, node_proba

    def _check_sklearn_tree(self, certified, frame: pd.DataFrame):
        """Warn if the sklearn tree routes any training row differently.

        sklearn compares X as float32, so two training values of one column closer
        than float32 resolves can land on the same side of a split the certified
        tree drew between them. On ordinary data this never happens.
        """
        ours = decode_labels(self, certified.predict_fast(frame).astype(int))
        theirs = self.estimator_.predict(self._sklearn_X(frame.to_numpy()))
        n_diff = int(np.sum(np.asarray(ours) != np.asarray(theirs)))
        if n_diff:
            warnings.warn(
                f"the sklearn tree predicts {n_diff} training rows differently from the "
                "certified tree: some column has training values closer together than "
                "float32 can separate", RuntimeWarning)

    def _sklearn_X(self, X):
        """X as `estimator_` expects it: named when the model was fitted on names."""
        X = np.asarray(X, dtype=float)
        if hasattr(self, "feature_names_in_"):
            return pd.DataFrame(X, columns=self.feature_names_in_)
        return X

    def predict(self, X):
        """Predict the class of each row of X, with sklearn's tree."""
        X = check_predict_X(self, X)  # checks fitted-ness before any attribute of ours
        return self.estimator_.predict(self._sklearn_X(X))

    def predict_proba(self, X):
        """Class probabilities, read off the training rows in each leaf.

        A leaf of an optimal tree predicts one class, so these are the empirical
        class frequencies of the training rows that reached the leaf rather than
        a calibrated probability.
        """
        X = check_predict_X(self, X)  # checks fitted-ness before any attribute of ours
        return self._node_proba_[self.estimator_.apply(self._sklearn_X(X))]

    def _display_tree(self, node: dict) -> dict:
        """A copy of the tree whose leaves carry the caller's class labels.

        The tree itself predicts the integer codes `check_fit_arguments` made of
        y, which is what routes rows; only printing wants the labels back.
        """
        if "prediction" in node:
            label = self.classes_[int(node["prediction"])]
            # unwrap numpy scalars so the printed label reads as the caller wrote it
            return {**node, "prediction": getattr(label, "item", lambda: label)()}
        return {**node,
                "true": self._display_tree(node["true"]),
                "false": self._display_tree(node["false"])}

    def __str__(self):
        if not hasattr(self, "tree_"):
            return f"{type(self).__name__}(unfitted)"
        status = "certified optimal" if self.optimal_ else "NOT certified optimal"
        return (f"> ------------------------------\n"
                f"> FastSmallTree: {self.n_leaves_} leaves, "
                f"objective {self.objective_:.4g} ({status})\n"
                f"> ------------------------------\n"
                f"{TreeClassifier(self._display_tree(self.tree_))}")
