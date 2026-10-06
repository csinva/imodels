"""Build the shared tree representation from model-agnostic node specs (used by the imodels adapters)."""

import numpy as np

from ._extract import Node, TreeInfo, _feature_kind


def make_info(specs, *, task, feature_names, model_name, class_names=None, X=None, y=None, roots=None,
              combine="single", layout="tree", criterion="", target_name=None, link="identity",
              intercept=0.0, leaf_kind=None):
    """Turn node specs into a TreeInfo.

    Each spec is a dict with ``left``/``right`` (child ids, -1 for leaves) and either ``conds``
    ([(feature, op, value)], left when all hold) or ``feature``/``threshold`` (left when <=).
    Optional: ``proba`` (class probabilities, classification), ``value`` (regression value or
    additive leaf contribution), ``n`` (training samples), ``impurity``, ``label``.
    With X (and y), samples are routed through the nodes to fill counts and chart data.
    """
    clf = task == "classification"
    n_nodes = len(specs)
    roots = roots or [0]
    parent = -np.ones(n_nodes, dtype=int)
    for i, sp in enumerate(specs):
        for c in (sp.get("left", -1), sp.get("right", -1)):
            if c >= 0:
                parent[c] = i
    depth = np.zeros(n_nodes, dtype=int)
    for r in roots:
        stack = [r]
        while stack:
            i = stack.pop()
            for c in (specs[i].get("left", -1), specs[i].get("right", -1)):
                if c >= 0:
                    depth[c] = depth[i] + 1
                    stack.append(c)

    nodes = []
    for i, sp in enumerate(specs):
        conds = sp.get("conds")
        feat, thr = sp.get("feature", -1), sp.get("threshold", np.nan)
        if conds and len(conds) == 1:
            feat, thr = int(conds[0][0]), float(conds[0][2])
        if conds and len(conds) == 1 and conds[0][1] == "<=":
            conds = None  # plain threshold split
        nodes.append(Node(id=i, depth=int(depth[i]), parent=int(parent[i]), left=int(sp.get("left", -1)),
                          right=int(sp.get("right", -1)), feature=int(feat), threshold=float(thr),
                          n=int(sp.get("n", 0) or 0), weight=float(sp.get("n", 0) or 0),
                          impurity=float(sp.get("impurity", np.nan)), counts=np.zeros(1),
                          conds=[(int(f), op, float(v)) for f, op, v in conds] if conds else None,
                          label=sp.get("label")))

    info = TreeInfo(nodes=nodes, task=task, class_names=class_names, feature_names=list(feature_names),
                    criterion=criterion, model_name=model_name,
                    max_depth=int(depth.max()) if n_nodes else 0,
                    n_leaves=sum(1 for nd in nodes if nd.is_leaf), roots=list(roots), combine=combine,
                    layout=layout, link=link, intercept=float(intercept),
                    leaf_kind=leaf_kind or ("counts" if clf else "value"),
                    target_name=target_name or ("class" if clf else "target"))

    k = len(class_names) if clf else 1
    if X is not None:
        Xa = np.asarray(X, dtype=float)
        info.X = Xa
        info.route(Xa)
        used = sorted({f for nd in nodes for f in nd.features})
        info.feature_kind = {f: _feature_kind(Xa[:, f]) for f in used}
        if y is not None:
            info.y = np.asarray(y).astype(int if clf else float)
        for nd in nodes:
            if nd.n == 0 and nd.idx is not None:
                nd.n = nd.weight = len(nd.idx)

    def fill(i):  # models that count samples only at leaves: internal nodes hold the sum
        nd = nodes[i]
        if not nd.is_leaf:
            fill(nd.left)
            fill(nd.right)
            if nd.n == 0:
                nd.n = nodes[nd.left].n + nodes[nd.right].n
                nd.weight = nodes[nd.left].weight + nodes[nd.right].weight
    for r in roots:
        fill(r)
    for nd, sp in zip(nodes, specs):
        if clf:
            if sp.get("proba") is not None:  # the model's own estimate, scaled to the node size
                nd.counts = np.asarray(sp["proba"], dtype=float) * max(nd.n, 1)
            elif info.y is not None and nd.idx is not None:
                nd.counts = np.bincount(info.y[nd.idx], minlength=k).astype(float)
            else:
                nd.counts = np.ones(k) / k * max(nd.n, 1)
        else:
            v = sp.get("value")
            if v is None and info.y is not None and nd.idx is not None and len(nd.idx):
                v = float(info.y[nd.idx].mean())
            nd.counts = np.array([float(v) if v is not None else np.nan])
        if np.isnan(nd.impurity) and info.y is not None and nd.idx is not None and len(nd.idx):
            if clf:
                p = np.bincount(info.y[nd.idx], minlength=k) / len(nd.idx)
                nd.impurity = float(1 - (p ** 2).sum())
                info.criterion = info.criterion or "gini"
            else:
                nd.impurity = float(info.y[nd.idx].var())
                info.criterion = info.criterion or "squared_error"

    if all(nd.weight == 0 for nd in nodes):  # no sample counts at all: size flows by leaves below each node
        def leaves(i):
            nd = nodes[i]
            nd.weight = 1.0 if nd.is_leaf else leaves(nd.left) + leaves(nd.right)
            return nd.weight
        for r in info.roots:
            leaves(r)
    def agg(i):  # internal nodes without their own numbers summarize their children
        nd, sp = nodes[i], specs[i]
        if nd.is_leaf:
            return
        L, R = nodes[nd.left], nodes[nd.right]
        agg(nd.left)
        agg(nd.right)
        if clf and sp.get("proba") is None and nd.idx is None:
            nd.counts = L.counts + R.counts
        if not clf and not np.isfinite(nd.counts[0]):
            w = (L.weight + R.weight) or 1.0
            nd.counts = np.array([(L.weight * L.counts[0] + R.weight * R.counts[0]) / w])
    for r in roots:
        agg(r)
    if not info.criterion and any(np.isfinite(nd.impurity) for nd in nodes):
        info.criterion = "impurity"
    if not clf:
        vals = [nd.counts[0] for nd in nodes if nd.is_leaf and np.isfinite(nd.counts[0])]
        lo, hi = (min(vals), max(vals)) if vals else (0.0, 1.0)
        if info.y is not None and combine == "single":
            lo, hi = min(lo, float(info.y.min())), max(hi, float(info.y.max()))
        info.value_range = (lo, hi if hi > lo else lo + 1.0)
    return info


def rule_list_specs(rules, default, **kw):
    """Specs for an ordered rule list: [(conds, outcome)] then the final 'else' outcome.

    ``outcome`` is a dict with ``proba`` or ``value`` (and optionally ``n``).
    """
    specs = []
    for i, (conds, out) in enumerate(rules):
        me = len(specs)
        specs.append(dict(conds=conds, left=me + 1, right=me + 2, label=f"Rule {i + 1}"))
        specs.append(dict(**out))
    specs.append(dict(**default))
    return specs
