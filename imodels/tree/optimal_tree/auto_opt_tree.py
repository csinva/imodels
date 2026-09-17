"""Certifiably optimal sparse decision trees, discovered by autoresearch.

`AutoOptTreeClassifier` returns the tree minimising misclassification rate plus
a penalty per leaf, and certifies that no other tree on the same binarized
features scores better. The search behind it lives in `solver.py`.
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

from imodels.tree.optimal_tree.solver import (HAVE_NUMBA, ST_LB, ST_UB, BinaryEncoder,
                                              BitDataset, CompiledOptimizer, TargetEncoder,
                                              TreeClassifier, cluster_rows)
from imodels.util.arguments import (check_fit_arguments, check_predict_X,
                                    decode_labels)

NUMBA_HINT = (
    "AutoOptTreeClassifier needs numba, which is not installed. Install it with "
    "`pip install numba` (or `pip install imodels[optional]`). The search itself is "
    "compiled, and interpreting it is orders of magnitude slower, so there is no "
    "pure-Python fallback."
)


class AutoOptTreeClassifier(ClassifierMixin, BaseEstimator):
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
    tree_ : dict
        The fitted tree, as nested dicts of splits and leaves.
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
    >>> from imodels import AutoOptTreeClassifier
    >>> X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]] * 8)
    >>> y = (X[:, 0] == X[:, 1]).astype(int)
    >>> model = AutoOptTreeClassifier(regularization=0.01).fit(X, y)
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
        self.tree = TreeClassifier(self.tree_)
        self.objective_ = self.tree.risk()
        self.n_leaves_ = self.tree.leaves()
        self.complexity_ = max(self.n_leaves_ - 1, 0)
        self._leaf_proba_, self._leaf_tree_ = self._leaf_lookup(self.tree_)
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

    def _leaf_lookup(self, tree: dict):
        """Class frequencies per leaf, and a copy of the tree predicting leaf ids.

        Running the solver's own traversal over the copy is what routes rows to
        leaves for `predict_proba`, so the two agree with `predict` by
        construction.
        """
        probs = []

        def rec(node):
            if "prediction" in node:
                probs.append(node["dist"])
                return {**node, "prediction": len(probs) - 1}
            return {**node, "true": rec(node["true"]), "false": rec(node["false"])}

        indexed = rec(tree)
        counts = np.asarray(probs, dtype=float)
        totals = counts.sum(axis=1, keepdims=True)
        uniform = np.full_like(counts, 1.0 / max(counts.shape[1], 1))
        return np.divide(counts, totals, out=uniform, where=totals > 0), indexed

    def _frame(self, X):
        """X as a DataFrame named the way the tree's rules are."""
        X = check_predict_X(self, X)
        return pd.DataFrame(np.asarray(X, dtype=float),
                            columns=list(self.feature_names_))

    def predict(self, X):
        """Predict the class of each row of X."""
        frame = self._frame(X)  # checks fitted-ness before any attribute of ours
        preds = self.tree.predict_fast(frame)
        return decode_labels(self, preds.astype(int))

    def predict_proba(self, X):
        """Class probabilities, read off the training rows in each leaf.

        A leaf of an optimal tree predicts one class, so these are the empirical
        class frequencies of the training rows that reached the leaf rather than
        a calibrated probability.
        """
        frame = self._frame(X)  # checks fitted-ness before any attribute of ours
        leaf_ids = TreeClassifier(self._leaf_tree_).predict_fast(frame)
        return self._leaf_proba_[leaf_ids.astype(int)]

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
                f"> AutoOptTree: {self.n_leaves_} leaves, "
                f"objective {self.objective_:.4g} ({status})\n"
                f"> ------------------------------\n"
                f"{TreeClassifier(self._display_tree(self.tree_))}")
