"""Export tree-based imodels models as fitted scikit-learn estimators.

`to_sklearn` turns a fitted tree-based model into the scikit-learn object with the
same predictions: a ``DecisionTreeClassifier`` / ``DecisionTreeRegressor`` for a
single tree, a ``RandomForestClassifier`` / ``RandomForestRegressor`` for an
iterative random forest, and a list of ``DecisionTreeRegressor`` (one per tree,
added up) for FIGS. Anything that reads scikit-learn trees then works on imodels
trees too: `imodels.viz` draws tree-based models only through this export, and
dtreeviz can draw the result directly (see `imodels.shadow_tree`).

The exported trees keep each node's sample counts. Pass the training data as
``X`` to recount them, which matters for TaoTree (whose splits are rewritten after
the counts were taken) and C4.5 (which stores no counts).

scikit-learn compares inputs in float32, while some imodels trees compare in
float64. Thresholds are rounded so every row is routed the same way, except rows
within float32 rounding (about 1e-7 relative) of a threshold.
"""

import copy

import numpy as np

__all__ = ["to_sklearn", "build_sklearn_tree"]


def build_sklearn_tree(children_left, children_right, feature, threshold, value, *,
                       classes=None, n_node_samples=None, impurity=None, missing_go_to_left=None,
                       n_features_in=None, feature_names_in=None):
    """A fitted scikit-learn decision tree from node arrays.

    Parameters
    ----------
    children_left, children_right : array of int, -1 at leaves
    feature, threshold : arrays; a node sends x left when ``x[feature] <= threshold``
        (compared in float32, as scikit-learn does)
    value : array (n_nodes, n_classes) of class probabilities (classifier) or
        (n_nodes, n_outputs) of predictions (regressor)
    classes : class labels for a classifier; None builds a regressor
    n_node_samples : samples reaching each node; defaults to the number of leaves below it
    impurity, missing_go_to_left : optional per-node arrays (impurity defaults to NaN, unknown;
        missing values go right by default)
    n_features_in, feature_names_in : the input features
    """
    from sklearn import __version__
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    from sklearn.tree._tree import Tree

    left = np.asarray(children_left, dtype=np.intp)
    right = np.asarray(children_right, dtype=np.intp)
    n = len(left)
    value = np.asarray(value, dtype=float).reshape(n, -1)
    is_clf = classes is not None
    if n_node_samples is None:
        n_node_samples = np.zeros(n)
        for i in reversed(range(n)):  # children come after their parent
            n_node_samples[i] = 1 if left[i] < 0 else n_node_samples[left[i]] + n_node_samples[right[i]]
    depth = np.zeros(n, dtype=int)
    for i in range(n):
        if left[i] >= 0:
            depth[left[i]] = depth[right[i]] = depth[i] + 1
    if n_features_in is None:
        n_features_in = int(np.max(feature)) + 1 if np.any(left >= 0) else 1

    if is_clf:
        n_outputs, n_classes = 1, np.array([value.shape[1]], dtype=np.intp)
        values = value.reshape(n, 1, -1)
    else:
        n_outputs, n_classes = value.shape[1], np.ones(value.shape[1], dtype=np.intp)
        values = value.reshape(n, -1, 1)
    tree = Tree(int(n_features_in), n_classes, n_outputs)
    nodes = np.zeros(n, dtype=tree.__getstate__()["nodes"].dtype)
    nodes["left_child"], nodes["right_child"] = left, right
    nodes["feature"] = np.where(left >= 0, np.asarray(feature), -2)
    nodes["threshold"] = np.where(left >= 0, np.asarray(threshold, dtype=float), -2.0)
    # unknown impurities stay NaN rather than 0, so nothing reads them as pure nodes
    nodes["impurity"] = np.nan if impurity is None else np.asarray(impurity, dtype=float)
    nodes["n_node_samples"] = np.asarray(n_node_samples).astype(np.intp)
    nodes["weighted_n_node_samples"] = np.asarray(n_node_samples, dtype=float)
    if "missing_go_to_left" in nodes.dtype.names:
        nodes["missing_go_to_left"] = 0 if missing_go_to_left is None else np.asarray(missing_go_to_left)
    tree.__setstate__({"max_depth": int(depth.max()), "node_count": n, "nodes": nodes,
                       "values": np.ascontiguousarray(values)})

    est = DecisionTreeClassifier() if is_clf else DecisionTreeRegressor()
    est.tree_ = tree
    est.n_outputs_ = n_outputs
    est.n_features_in_ = int(n_features_in)
    est.max_features_ = int(n_features_in)
    est._sklearn_version = __version__
    if is_clf:
        est.classes_ = np.asarray(classes)
        est.n_classes_ = len(classes)
    if feature_names_in is not None:
        est.feature_names_in_ = np.asarray(feature_names_in, dtype=object)
    return est


def _f32_le(t):
    """float32 threshold for ``x <= t``: every x <= t goes left."""
    return float(np.float32(t))


def _f32_lt(t):
    """float32 threshold for ``x < t``: x equal to t goes right (float32(t) itself must go right)."""
    return float(np.nextafter(np.float32(t), np.float32(-np.inf)))


def _state(est):
    st = est.tree_.__getstate__()
    st["nodes"] = st["nodes"].copy()
    return st


def recount(est, X):
    """Set each node's sample counts of a scikit-learn tree to the rows of X reaching it (in place)."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # fitted with or without feature names
        counts = np.asarray(est.decision_path(X).sum(axis=0)).ravel()
    st = _state(est)
    st["nodes"]["n_node_samples"] = counts.astype(np.intp)
    st["nodes"]["weighted_n_node_samples"] = counts.astype(float)
    est.tree_.__setstate__(st)
    return est


# ---------------------------------------------------------------- per model
def _figs(m):
    """FIGS: one regression tree per FIGS tree, holding the values the model adds up."""
    names = getattr(m, "feature_names_in_", None)
    out = []
    for root in m.trees_:
        rows = []

        def add(nd):
            me = len(rows)
            rows.append(None)
            val = np.asarray(nd.value, dtype=float).ravel()
            if nd.left is None or nd.right is None:
                rows[me] = (-1, -1, -2, -2.0, val, nd)
            else:
                lft = add(nd.left)
                rgt = add(nd.right)
                rows[me] = (lft, rgt, int(nd.feature), _f32_le(nd.threshold), val, nd)
            return me

        add(root)
        l, r, f, t, v, nds = zip(*rows)
        out.append(build_sklearn_tree(
            l, r, f, t, np.array(v), n_node_samples=[int(getattr(nd, "n_samples_", 0) or 0) for nd in nds],
            impurity=[getattr(nd, "impurity", np.nan) for nd in nds], n_features_in=m.n_features_in_,
            feature_names_in=names))
    return out


def _c45(m, X):
    """C4.5: leaves keep P(class 1) (binary, including after shrinkage) or their label (multiclass)."""
    xml_names = list(m.feature_names)
    to_name = getattr(m, "xml_name_to_feature_name_", {})
    K = len(m.classes_)
    rows = []

    def add(el):
        me = len(rows)
        rows.append(None)
        kids = [c for c in el.childNodes if c.nodeType == c.ELEMENT_NODE]
        if not kids:
            v = float(el.firstChild.nodeValue)
            rows[me] = (-1, -1, -2, -2.0, np.array([1 - v, v]) if K == 2 else np.eye(K)[int(v)])
            return me
        flags = {c.getAttribute("flag"): c for c in kids}
        if set(flags) != {"l", "r"}:
            raise NotImplementedError("C4.5 trees with categorical (multiway) splits cannot be exported.")
        lc, rc = flags["l"], flags["r"]
        f = xml_names.index(lc.tagName) if lc.tagName in xml_names else \
            [to_name.get(nm, nm) for nm in xml_names].index(to_name.get(lc.tagName, lc.tagName))
        lft, rgt = add(lc), add(rc)  # 'l' holds x < threshold
        rows[me] = (lft, rgt, f, _f32_lt(float(lc.getAttribute("feature"))), None)
        return me

    add(m.root)
    n = len(rows)
    left = np.array([r[0] for r in rows])
    right = np.array([r[1] for r in rows])
    leaves = np.zeros(n)
    value = np.zeros((n, K))
    for i in reversed(range(n)):  # inner nodes: the average of their leaves
        if left[i] < 0:
            value[i], leaves[i] = rows[i][4], 1
        else:
            leaves[i] = leaves[left[i]] + leaves[right[i]]
            value[i] = (value[left[i]] * leaves[left[i]] + value[right[i]] * leaves[right[i]]) / leaves[i]
    return build_sklearn_tree(left, right, [r[2] for r in rows], [r[3] for r in rows], value,
                              classes=m.classes_, n_features_in=len(xml_names))


def _irf(m):
    """Iterative random forest: its weighted trees as a scikit-learn random forest."""
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    forest = m.forest_
    clf = getattr(forest, "task", "classification") != "regression"
    classes = np.arange(forest.n_classes_) if clf else None
    trees = []
    for t in forest.estimators_:
        if clf:
            val = np.asarray(t.value_, dtype=float).reshape(len(t.value_), -1)
        else:
            val = np.asarray(getattr(t, "leaf_prediction_", t.value_), dtype=float).reshape(len(t.value_), -1)
        trees.append(build_sklearn_tree(
            t.children_left_, t.children_right_, t.feature_, t.threshold_, val, classes=classes,
            n_node_samples=t.n_node_samples_, impurity=t.impurity_, n_features_in=m.n_features_in_))
    rf = RandomForestClassifier(n_estimators=len(trees)) if clf else RandomForestRegressor(n_estimators=len(trees))
    rf.estimators_ = trees
    rf.estimator_ = trees[0]
    rf.n_outputs_ = 1
    rf.n_features_in_ = m.n_features_in_
    if clf:
        rf.classes_ = np.asarray(m.classes_)
        rf.n_classes_ = len(m.classes_)
        for t in trees:
            t.classes_ = rf.classes_
    if hasattr(m, "feature_names_in_"):
        rf.feature_names_in_ = m.feature_names_in_
    return rf


def to_sklearn(model, X=None):
    """The scikit-learn equivalent of a fitted tree-based imodels model.

    Parameters
    ----------
    model
        A fitted tree-based model: GreedyTree, DecisionTreeCCP, HSTree (and its CV
        and CCP variants), TaoTree, FastSmallTree, C4.5 (and HSC45), FIGS (and
        FIGSCV), IRF, or a scikit-learn tree.
    X : array-like, optional
        Training data. When given, every node's sample counts are recounted on it.

    Returns
    -------
    DecisionTreeClassifier or DecisionTreeRegressor
        For single trees. ``predict`` / ``predict_proba`` match the model's.
    RandomForestClassifier or RandomForestRegressor
        For IRF.
    list of DecisionTreeRegressor
        For FIGS, one per tree. Their predictions add up to the model's raw score:
        the prediction for a regressor, and for a classifier the per-class scores
        whose softmax is ``predict_proba`` (``sum(t.predict(X) for t in trees)``).

    Examples
    --------
    >>> from imodels import FastSmallTreeClassifier, to_sklearn   # doctest: +SKIP
    >>> tree = to_sklearn(FastSmallTreeClassifier().fit(X, y))     # doctest: +SKIP
    >>> sklearn.tree.plot_tree(tree)                               # doctest: +SKIP
    """
    from imodels.util.model_trees import is_sklearn_tree

    name = type(model).__name__
    if name in ("FIGSClassifierCV", "FIGSRegressorCV"):
        return to_sklearn(model.figs, X)
    if hasattr(model, "trees_"):  # FIGS
        if getattr(model, "_encoder", None) is not None:
            raise NotImplementedError("FIGS with categorical_features cannot be exported: its splits are on encoded columns.")
        return _figs(model)
    if name in ("IRFClassifier", "IRFRegressor"):
        return _irf(model)
    if name in ("HSC45TreeClassifier", "HSC45TreeClassifierCV"):
        model = model.estimator_
        name = type(model).__name__
    if name == "C45TreeClassifier":
        est = _c45(model, X)
    elif name == "FastSmallTreeClassifier":
        est = copy.deepcopy(model.estimator_)
        # tree_.value is a cost-weighted device for argmax; predict_proba reads the leaf frequencies
        st = _state(est)
        st["values"] = np.ascontiguousarray(np.asarray(model._node_proba_, dtype=float)[:, None, :])
        est.tree_.__setstate__(st)
    elif is_sklearn_tree(model):  # sklearn trees and GreedyTree
        est = copy.deepcopy(model)
    else:
        inner = getattr(model, "model", None) if name.startswith("TaoTree") else getattr(model, "estimator_", None)
        if inner is None or not (is_sklearn_tree(inner) or hasattr(inner, "estimators_")):
            raise TypeError(f"{name} is not a fitted tree-based model that can be exported to scikit-learn.")
        est = copy.deepcopy(inner)  # TaoTree, DecisionTreeCCP, HSTree (shrunk values live in estimator_)
        if not is_sklearn_tree(est):  # hierarchical shrinkage of a forest
            if X is not None:
                for e in est.estimators_:
                    recount(getattr(e, "estimator_", e), X)
            return est
    if X is not None:
        recount(est, np.asarray(X, dtype=float) if not hasattr(X, "columns") else X)
    return est
