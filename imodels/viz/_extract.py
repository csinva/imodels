"""Pull a fitted sklearn tree (and optional training data) into plain Python objects."""

import warnings
from dataclasses import dataclass, field

import numpy as np

from ._text import fmt

# conditions are (feature, op, value); a split node sends a sample LEFT when all its conditions hold
OPS = {"<=": np.less_equal, "<": np.less, ">": np.greater, ">=": np.greater_equal,
       "==": np.equal, "!=": np.not_equal,
       "isnan": lambda a, v: np.isnan(a), "notnan": lambda a, v: ~np.isnan(a)}  # value missing / present
NEG = {"<=": ">", "<": ">=", ">": "<=", ">=": "<", "==": "!=", "!=": "==", "isnan": "notnan", "notnan": "isnan"}
SYM = {"<=": "\u2264", "<": "<", ">": ">", ">=": "\u2265", "==": "=", "!=": "\u2260", "isnan": "missing", "notnan": "present"}


def sk_threshold(t):
    """float64 cut c with (x <= c) == (float32(x) <= t): sklearn trees compare inputs as float32."""
    t = float(t)
    a = np.float32(t)
    if float(a) > t:
        a = np.nextafter(a, np.float32(-np.inf))
    b = np.nextafter(a, np.float32(np.inf))
    return float(np.nextafter((float(a) + float(b)) / 2, -np.inf))


def cond_mask(X, conds):
    """Rows of X satisfying every (feature, op, value) condition."""
    m = np.ones(len(X), dtype=bool)
    for f, op, v in conds:
        with np.errstate(invalid="ignore"):
            m &= OPS[op](X[:, f], v)
    return m


def cond_holds(x, conds):
    return all(bool(OPS[op](x[f], v)) for f, op, v in conds)


@dataclass
class Node:
    id: int
    depth: int
    parent: int
    left: int
    right: int
    feature: int
    threshold: float
    n: int  # unweighted sample count
    weight: float  # weighted sample count
    impurity: float
    counts: np.ndarray  # class counts (classification) or [mean] (regression)
    idx: np.ndarray = None  # rows of X reaching this node (when data is given)
    conds: list = None  # [(feature, op, value)]: go left when all hold; None means feature <= threshold
    label: str = None  # optional display title (e.g. "Rule 3")
    nan_left: bool = None  # where a missing value goes at a one-feature split (None: compare as usual)

    @property
    def is_leaf(self):
        return self.left < 0

    @property
    def split(self):
        """The split as a list of conditions (left branch = all hold)."""
        if self.is_leaf:
            return []
        return self.conds if self.conds else [(self.feature, "<=", self.threshold)]

    @property
    def simple(self):
        """True when the split is one condition on one feature (charts and intervals apply)."""
        return not self.is_leaf and len(self.split) == 1 and self.split[0][1] in ("<=", "<", ">", ">=")

    @property
    def features(self):
        return [f for f, _, _ in self.split]


@dataclass
class TreeInfo:
    nodes: list
    task: str  # "classification" | "regression"
    class_names: list
    feature_names: list
    criterion: str
    model_name: str
    max_depth: int
    n_leaves: int
    X: np.ndarray = None
    y: np.ndarray = None  # class index (classification) or float target
    feature_kind: dict = field(default_factory=dict)  # feature -> binary|integer|continuous
    value_range: tuple = (0.0, 1.0)
    target_name: str = "target"
    roots: list = field(default_factory=lambda: [0])  # several roots = an ensemble of trees
    combine: str = "single"  # "single" | "sum" (leaf values add up) | "mean" (tree predictions are averaged)
    shown_roots: list = None  # the trees to draw (default: all); predictions always use every tree
    layout: str = "tree"  # "tree" | "cascade" (rule lists)
    link: str = "identity"  # how summed leaf values map to the output ("identity" | "logistic")
    intercept: float = 0.0
    leaf_kind: str = None  # "counts" | "value" (what a leaf's numbers mean)

    @property
    def is_clf(self):
        return self.task == "classification"

    @property
    def root(self):
        return self.nodes[0]

    def summary(self):
        """One-line description used as the default subtitle."""
        n = f"{int(self.root.n):,} training samples" if self.root.n else ""
        if self.layout == "cascade":
            k = sum(1 for nd in self.nodes if not nd.is_leaf)
            parts = [self.model_name, f"{k} rule{'s' if k != 1 else ''} + else"]
        elif len(self.roots) > 1:
            how = "averaged" if self.combine == "mean" else "added together"
            shown = self.shown_roots or self.roots
            more = f" (first {len(shown)} drawn)" if len(shown) < len(self.roots) else ""
            parts = [self.model_name, f"{len(self.roots)} trees {how}{more}", f"{self.n_leaves} leaves"]
        else:
            parts = [self.model_name, f"depth {self.max_depth}", f"{self.n_leaves} leaves"]
        return "  \u00b7  ".join(parts + ([n] if n else []))

    def prediction(self, node):
        """Predicted class index (classification) or value (regression)."""
        if self.is_clf:
            pred = getattr(node, "pred", None)  # a model's own leaf label, when it stores one
            return int(pred) if pred is not None else int(np.argmax(node.counts))
        return float(node.counts[0])

    def path_to(self, nid):
        out = []
        while nid >= 0:
            out.append(nid)
            nid = self.nodes[nid].parent
        return out[::-1]

    def n_descendants(self, nid):
        node = self.nodes[nid]
        if node.is_leaf:
            return 0
        return 2 + self.n_descendants(node.left) + self.n_descendants(node.right)

    def cond_text(self, f, op, v, sig=3, with_name=True, negate=False):
        """Readable text for one condition, e.g. 'age >= 40' or 'smoker = 1'."""
        if negate:
            op = NEG[op]
        kind = self.feature_kind.get(f, "continuous")
        name = f"{self.feature_names[f]} " if with_name else ""
        if kind == "binary" and op in ("<=", "<", ">", ">=") and 0 < v < 1:
            return f"{name}= {0 if op in ('<=', '<') else 1}"
        if kind == "integer" and op in ("<=", ">"):
            k = int(np.floor(v))
            return f"{name}\u2264 {fmt(k, 12)}" if op == "<=" else f"{name}\u2265 {fmt(k + 1, 12)}"
        return f"{name}{SYM[op]} {fmt(v, sig if kind == 'continuous' else 12)}"

    def edge_label(self, nid, sig=3):
        """Condition text on the edge that leads into ``nid`` (e.g. '<= 2.45', or yes / no)."""
        node = self.nodes[nid]
        if node.parent < 0:
            return ""
        p = self.nodes[node.parent]
        left = p.left == nid
        if len(p.split) > 1:
            return "yes" if left else "no"
        f, op, v = p.split[0]
        return self.cond_text(f, op, v, sig, with_name=False, negate=not left)

    def split_text(self, nid, sig=3):
        """The node's split, e.g. 'age >= 40 and bmi > 30'."""
        return " and ".join(self.cond_text(f, op, v, sig) for f, op, v in self.nodes[nid].split)

    def rules(self, nid, sig=3):
        """Simplified conjunction of conditions from the root to ``nid``.

        Repeated splits on one feature collapse into a single interval.
        """
        lo, hi, order, extra = {}, {}, [], []
        path = self.path_to(nid)
        for a, b in zip(path[:-1], path[1:]):
            p = self.nodes[a]
            went_left = p.left == b
            if not p.simple:
                if went_left:
                    extra += [self.cond_text(f, op, v, sig) for f, op, v in p.split]
                elif len(p.split) == 1:
                    f, op, v = p.split[0]
                    extra.append(self.cond_text(f, op, v, sig, negate=True))
                else:
                    extra.append("not (" + self.split_text(a, sig) + ")")
                continue
            f, op, t = p.split[0]
            upper = (op in ("<=", "<")) == went_left  # this edge bounds f from above
            if f not in order:
                order.append(f)
            if upper:
                hi[f] = min(hi.get(f, np.inf), t)
            else:
                lo[f] = max(lo.get(f, -np.inf), t)
        out = []
        for f in order:
            name = self.feature_names[f]
            kind = self.feature_kind.get(f, "continuous")
            l, h = lo.get(f, -np.inf), hi.get(f, np.inf)
            if kind == "binary":
                out.append(f"{name} = {0 if np.isfinite(h) else 1}")
            elif kind == "integer":
                li = int(np.floor(l)) + 1 if np.isfinite(l) else None
                hi_ = int(np.floor(h)) if np.isfinite(h) else None
                if li is not None and hi_ is not None:
                    out.append(f"{name} = {li}" if li == hi_ else f"{li} ≤ {name} ≤ {hi_}")
                elif li is not None:
                    out.append(f"{name} ≥ {li}")
                else:
                    out.append(f"{name} ≤ {hi_}")
            else:
                if np.isfinite(l) and np.isfinite(h):
                    out.append(f"{fmt(l, sig)} < {name} ≤ {fmt(h, sig)}")
                elif np.isfinite(l):
                    out.append(f"{name} > {fmt(l, sig)}")
                else:
                    out.append(f"{name} ≤ {fmt(h, sig)}")
        return out + extra

    def importances(self):
        """Per-feature importance and how it was computed.

        Impurity decrease weighted by samples (sklearn's definition) when node impurities are
        known; otherwise the number of samples reaching each split. Conjunction splits share
        their credit equally among their features.
        """
        imp = np.zeros(len(self.feature_names))
        ok = all(np.isfinite(nd.impurity) for nd in self.nodes)
        for nd in self.nodes:
            if nd.is_leaf:
                continue
            fs = sorted(set(nd.features))
            if ok:
                L, R = self.nodes[nd.left], self.nodes[nd.right]
                g = nd.weight * nd.impurity - L.weight * L.impurity - R.weight * R.impurity
            else:
                g = nd.weight
            for f in fs:
                imp[f] += max(g, 0.0) / len(fs)
        tot = imp.sum()
        return (imp / tot if tot > 0 else imp), ("impurity" if ok else "samples")

    def route(self, X):
        """Assign rows of X to nodes by following each split (used for non-sklearn trees)."""
        for r in self.roots:
            stack = [(r, np.arange(len(X)))]
            while stack:
                nid, idx = stack.pop()
                nd = self.nodes[nid]
                nd.idx = idx
                if nd.is_leaf:
                    continue
                m = cond_mask(X[idx], nd.split)
                if nd.nan_left is not None:  # sklearn sends missing values to a fixed side
                    nan = np.isnan(X[idx, nd.split[0][0]])
                    m[nan] = nd.nan_left
                stack += [(nd.left, idx[m]), (nd.right, idx[~m])]

    def output(self, X):
        """Model output from this representation: class probabilities (n, K), P(positive) for
        sums of trees, or values. Used to check that a view reproduces its model."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if self.combine == "mean":  # forests: average the trees' predictions
            outs = []
            for r in self.roots:
                leaves = [self.nodes[self.walk(x, r)[-1]] for x in X]
                outs.append([nd.counts / (nd.counts.sum() or 1) for nd in leaves] if self.is_clf
                            else [nd.counts[0] for nd in leaves])
            return np.mean(np.asarray(outs, dtype=float), axis=0)
        if self.combine == "sum":
            s = np.full(len(X), self.intercept)
            for i, x in enumerate(X):
                for r in self.roots:
                    nd = self.nodes[self.walk(x, r)[-1]]
                    s[i] += nd.value if getattr(nd, "value", None) is not None else nd.counts[0]
            return 1 / (1 + np.exp(-s)) if self.link == "logistic" else s
        leaves = [self.nodes[self.walk(x)[-1]] for x in X]
        if self.is_clf:
            K = len(self.class_names)
            return np.array([np.eye(K)[nd.pred] if getattr(nd, "pred", None) is not None
                             else nd.counts / (nd.counts.sum() or 1) for nd in leaves])
        return np.array([nd.counts[0] for nd in leaves])

    def walk(self, x, root=None):
        """Node ids visited by one sample from ``root`` (default: first root) to a leaf."""
        nid = self.roots[0] if root is None else root
        path = [nid]
        while not self.nodes[nid].is_leaf:
            nd = self.nodes[nid]
            f = nd.split[0][0]
            if nd.nan_left is not None and np.isnan(x[f]):
                nid = nd.left if nd.nan_left else nd.right
            else:
                nid = nd.left if cond_holds(x, nd.split) else nd.right
            path.append(nid)
        return path


def _unwrap(model):
    try:
        from sklearn.pipeline import Pipeline

        if isinstance(model, Pipeline):
            model = model[-1]
    except ImportError:  # pragma: no cover
        pass
    if hasattr(model, "tree_"):
        return model
    if hasattr(model, "estimators_"):
        raise TypeError(
            "Got an ensemble. Pass one of its trees instead, e.g. "
            "model.estimators_[0] (random forest) or model.estimators_[0, 0] (gradient boosting)."
        )
    raise TypeError(f"Expected a fitted sklearn decision tree, got {type(model).__name__}.")


def _feature_kind(col):
    col = col[np.isfinite(col)]
    if col.size == 0:
        return "continuous"
    u = np.unique(col)
    if np.all(np.isin(u, [0.0, 1.0])):
        return "binary"
    if np.all(u == np.round(u)):
        return "integer"
    return "continuous"


def extract(model, X=None, y=None, feature_names=None, class_names=None, target_name=None, output=0):
    est = _unwrap(model)
    t = est.tree_
    from sklearn.base import is_classifier

    clf = is_classifier(est)
    if t.n_outputs > 1:
        warnings.warn(f"Multi-output tree: showing output {output} only.")

    # names
    if feature_names is None and X is not None and hasattr(X, "columns"):
        feature_names = [str(c) for c in X.columns]
    if feature_names is None and hasattr(est, "feature_names_in_"):
        feature_names = [str(c) for c in est.feature_names_in_]
    if feature_names is None:
        feature_names = [f"x{i}" for i in range(t.n_features)]
    feature_names = list(feature_names)

    classes = None
    if clf:
        classes = est.classes_[output] if t.n_outputs > 1 else est.classes_
        if class_names is None:
            class_names = [str(c) for c in classes]
        elif isinstance(class_names, dict):
            class_names = [str(class_names.get(c, c)) for c in classes]
        else:
            class_names = [str(c) for c in class_names]
    if target_name is None:
        target_name = getattr(y, "name", None) or ("class" if clf else "target")

    # node values: sklearn >= 1.4 stores class fractions, older versions counts
    vals = t.value[:, output, :]
    w = t.weighted_n_node_samples
    if clf:
        sums = vals.sum(axis=1, keepdims=True)
        sums[sums == 0] = 1
        counts = vals / sums * w[:, None]
    else:
        counts = vals[:, :1]

    depth = np.zeros(t.node_count, dtype=int)
    parent = -np.ones(t.node_count, dtype=int)
    for i in range(t.node_count):
        for c in (t.children_left[i], t.children_right[i]):
            if c >= 0:
                parent[c] = i
                depth[c] = depth[i] + 1

    nodes = [
        Node(
            id=i,
            depth=int(depth[i]),
            parent=int(parent[i]),
            left=int(t.children_left[i]),
            right=int(t.children_right[i]),
            feature=int(t.feature[i]),
            threshold=sk_threshold(t.threshold[i]) if t.children_left[i] >= 0 else float(t.threshold[i]),
            n=int(t.n_node_samples[i]),
            weight=float(w[i]),
            impurity=float(t.impurity[i]),
            counts=np.asarray(counts[i], dtype=float),
        )
        for i in range(t.node_count)
    ]

    mg = getattr(t, "missing_go_to_left", None)
    if mg is not None:  # sklearn >= 1.3 routes missing values to a fixed side at each split
        for nd, go in zip(nodes, np.asarray(mg).astype(bool)):
            if not nd.is_leaf:
                nd.nan_left = bool(go)

    info = TreeInfo(
        nodes=nodes,
        task="classification" if clf else "regression",
        class_names=class_names,
        feature_names=feature_names,
        criterion=str(getattr(est, "criterion", "impurity")),
        model_name=type(est).__name__,
        max_depth=int(t.max_depth),
        n_leaves=int(t.n_leaves),
        target_name=str(target_name),
    )

    if X is not None:
        Xa = np.asarray(X, dtype=float)
        if Xa.ndim != 2 or Xa.shape[1] != t.n_features:
            raise ValueError(f"X must have shape (n, {t.n_features}), got {Xa.shape}.")
        info.X = Xa
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # fitted with feature names
            ind = est.decision_path(Xa).tocsc()
        for j, node in enumerate(nodes):
            node.idx = ind.indices[ind.indptr[j] : ind.indptr[j + 1]]
        used = sorted({f for nd in nodes for f in nd.features})
        info.feature_kind = {f: _feature_kind(Xa[:, f]) for f in used}
        if y is not None:
            ya = np.asarray(y)
            if ya.ndim > 1:
                ya = ya[:, output]
            if clf:
                lookup = {c: i for i, c in enumerate(classes)}
                info.y = np.array([lookup[v] for v in ya])
            else:
                info.y = ya.astype(float)

    if not clf:
        leaf_vals = [nd.counts[0] for nd in nodes]
        lo, hi = float(np.min(leaf_vals)), float(np.max(leaf_vals))
        if info.y is not None:
            lo, hi = min(lo, float(info.y.min())), max(hi, float(info.y.max()))
        info.value_range = (lo, hi if hi > lo else lo + 1.0)
    return info
