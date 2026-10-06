"""Plain-text rendering of a fitted model: what ``print(model)`` shows for imodels estimators.

It reads the same views as the figures (TreeInfo for trees, rule lists and sums of trees; AdditiveView
for rule sets, scorecards, GAMs and linear models), which tests check against each model's own
predictions, so the text describes exactly what the model computes.
"""

import numpy as np

from ._text import fmt, fmt_count

SPARK = "▁▂▃▄▅▆▇█"

# one line on how each kind of model predicts, shown under the title
DESCRIBE = {
    "GreedyTreeClassifier": "CART decision tree", "GreedyTreeRegressor": "CART decision tree",
    "DecisionTreeCCPClassifier": "decision tree pruned to a target size by cost-complexity pruning",
    "DecisionTreeCCPRegressor": "decision tree pruned to a target size by cost-complexity pruning",
    "HSTreeClassifier": "decision tree with hierarchical shrinkage (each node's value is pulled toward its parent's)",
    "HSTreeRegressor": "decision tree with hierarchical shrinkage (each node's value is pulled toward its parent's)",
    "HSTreeClassifierCV": "decision tree with hierarchical shrinkage, its strength picked by cross-validation",
    "HSTreeRegressorCV": "decision tree with hierarchical shrinkage, its strength picked by cross-validation",
    "TaoTreeClassifier": "decision tree refined by tree alternating optimization (TAO)",
    "FastSmallTreeClassifier": "optimal small decision tree",
    "C45TreeClassifier": "C4.5 decision tree",
    "FIGSClassifier": "FIGS: a sum of small trees", "FIGSRegressor": "FIGS: a sum of small trees",
    "FIGSClassifierCV": "FIGS: a sum of small trees", "FIGSRegressorCV": "FIGS: a sum of small trees",
    "IRFClassifier": "iterative random forest", "IRFRegressor": "iterative random forest",
    "GreedyRuleListClassifier": "greedy rule list: rules are checked in order and the first that holds decides",
    "OneRClassifier": "one-rule model: a rule list on a single feature",
    "FastFrugalTreeClassifier": "Fast-and-frugal tree: each cue either decides, or passes the case to the next cue",
    "BayesianRuleListClassifier": "Bayesian rule list: rules are checked in order and the first that holds decides",
    "RuleFitClassifier": "RuleFit: a sparse linear model over rules (and linear terms)",
    "RuleFitRegressor": "RuleFit: a sparse linear model over rules (and linear terms)",
    "FPLassoClassifier": "a lasso over rules mined with FP-growth", "FPLassoRegressor": "a lasso over rules mined with FP-growth",
    "SkopeRulesClassifier": "Skope rules: precise rules, weighted by their out-of-bag precision",
    "FPSkopeClassifier": "Skope rules over rules mined with FP-growth",
    "BoostedRulesClassifier": "boosted rules: one-split rules combined by AdaBoost",
    "SlipperClassifier": "SLIPPER: boosted rules",
    "BayesianRuleSetClassifier": "Bayesian rule set: predicts positive when any rule holds",
    "FastRiskScoreClassifier": "risk score: add the points of every condition that holds",
    "SLIMClassifier": "SLIM: a linear model with small integer coefficients",
    "SLIMRegressor": "SLIM: a linear model with small integer coefficients",
    "MarginalShrinkageLinearRegressor": "linear model shrunk toward each feature's marginal effect",
    "TreeGAMClassifier": "TreeGAM: an additive model with one shape function per feature",
    "TreeGAMRegressor": "TreeGAM: an additive model with one shape function per feature",
    "GPGamRegressor": "GPGam: an additive model with one Gaussian-process shape function per feature",
}


def num(v, sig=4):
    """A number for text: ASCII minus sign, no exponent below 1e15."""
    if v is None:
        return ""
    v = float(v)
    if np.isfinite(v) and 1e6 <= abs(v) < 1e15:
        digits = max(sig - len(str(int(abs(v)))), 0)
        s = f"{v:,.{digits}f}"
        s = s.rstrip("0").rstrip(".") if "." in s else s
    else:
        s = fmt(v, sig)
    return s.replace("−", "-")


def signed(v, sig=4):
    s = num(v, sig)
    return s if s.startswith("-") or s == "0" else "+" + s


def _clean(s):
    return s.replace("−", "-")


def _feature_sig(pairs, base):
    """Significant digits per feature, raised until no two of its thresholds print alike."""
    by_f = {}
    for f, v in pairs:
        by_f.setdefault(f, set()).add(float(v))
    out = {}
    for f, vals in by_f.items():
        sig = base
        while sig < 15 and len({fmt(v, sig) for v in vals}) < len(vals):
            sig += 1
        out[f] = sig
    return out


def _table(rows, aligns, gap=3, indent=2):
    """Rows of cells as aligned text columns ('<' left, '>' right)."""
    widths = [max(len(r[i]) for r in rows) for i in range(len(aligns))]
    lines = []
    for r in rows:
        cells = [c.rjust(w) if a == ">" else c.ljust(w) for c, w, a in zip(r, widths, aligns)]
        lines.append((" " * indent + (" " * gap).join(cells)).rstrip())
    return lines


def _header(name, describe, facts, how):
    lines = [name]
    if describe:
        lines.append("  " + describe[0].upper() + describe[1:] + ("" if describe.endswith(".") else "."))
    if facts:
        lines.append("  " + "  ·  ".join(facts))
    if how:
        lines.append("  " + how)
    return lines


# ---------------------------------------------------------------- trees
def _leaf_text(info, nd, sig):
    contrib = getattr(nd, "value", None) if info.combine == "sum" else None
    if contrib is not None:
        return signed(contrib, sig)
    if info.is_clf:
        tot = nd.counts.sum() or 1
        k = info.prediction(nd)
        return f"{info.class_names[k]}  ({nd.counts[k] / tot:.0%})"
    return num(nd.counts[0], max(sig, 4))


def _tree_lines(info, root, sig, fsig):
    """One tree as lines of (text, samples)."""
    rows = []

    def cond(nd):
        return _clean(" and ".join(info.cond_text(f, op, v, fsig.get(f, sig)) for f, op, v in nd.split))

    def walk(nid, prefix, label, last, top):
        nd = info.nodes[nid]
        branch = "" if top else ("└─ " if last else "├─ ")
        lead = prefix + branch + (f"{label:<4}" if label else "")
        body = ("→ " + _leaf_text(info, nd, sig)) if nd.is_leaf else (cond(nd) + " ?")
        rows.append((lead + body, fmt_count(nd.n) if nd.n else ""))
        if not nd.is_leaf:
            child_prefix = prefix + ("" if top else ("   " if last else "│  "))
            walk(nd.left, child_prefix, "yes", False, False)
            walk(nd.right, child_prefix, "no", True, False)

    walk(root, "", "", True, True)
    return rows


def _rows_with_samples(rows, indent=2):
    w = max(len(t) for t, _ in rows)
    sw = max([len(s) for _, s in rows] + [1])
    out = []
    for i, (t, s) in enumerate(rows):
        out.append((" " * indent + t.ljust(w) + ("   " + s.rjust(sw) + (" samples" if i == 0 else "")) if s else " " * indent + t).rstrip())
    return out


def _cascade_lines(info, sig, fsig):
    """Rule lists: IF / ELSE IF rows with their outcome and coverage."""
    rows, nid, first = [], info.roots[0], True
    while True:
        nd = info.nodes[nid]
        if nd.is_leaf:
            rows.append(["ELSE", "", "→", _leaf_text(info, nd, sig), fmt_count(nd.n) if nd.n else ""])
            break
        side_left = info.nodes[nd.left].is_leaf
        side, main = (nd.left, nd.right) if side_left else (nd.right, nd.left)
        conds = nd.split
        if side_left:
            text = " and ".join(info.cond_text(f, op, v, fsig.get(f, sig)) for f, op, v in conds)
        elif len(conds) == 1:
            f, op, v = conds[0]
            text = info.cond_text(f, op, v, fsig.get(f, sig), negate=True)
        else:
            text = "not (" + " and ".join(info.cond_text(f, op, v, fsig.get(f, sig)) for f, op, v in conds) + ")"
        sd = info.nodes[side]
        rows.append(["IF" if first else "ELSE IF", _clean(text), "→", _leaf_text(info, sd, sig),
                     fmt_count(sd.n) if sd.n else ""])
        first = False
        nid = main
    has_n = any(r[4] for r in rows)
    if has_n:
        rows.insert(0, ["", "", "", "", "samples"])
    else:
        rows = [r[:4] for r in rows]
    return _table(rows, ["<", "<", "<", "<", ">"][:len(rows[0])], gap=2)


def tree_text(info, sig=3, max_trees=3):
    if not getattr(info, "counts_known", True):
        for nd in info.nodes:
            nd.n = 0
    name = info.model_name
    describe = DESCRIBE.get(name.split(" (")[0])
    facts = []
    if info.layout == "cascade":
        k = sum(1 for nd in info.nodes if not nd.is_leaf)
        facts.append(f"{k} rule{'s' if k != 1 else ''} + else")
    elif len(info.roots) > 1:
        facts.append(f"{len(info.roots)} trees, " + ("averaged" if info.combine == "mean" else "added together"))
        facts.append(f"{info.n_leaves} leaves")
    else:
        facts += [f"depth {info.max_depth}", f"{info.n_leaves} leaves"]
    if info.root.n:
        facts.append(f"{fmt_count(info.root.n)} training samples")
    how = ""
    if info.is_clf and info.class_names:
        facts.append("classes: " + ", ".join(info.class_names))
    if info.combine == "sum":
        out = f"P({info.class_names[-1]}) = sigmoid(sum)" if info.is_clf else "the prediction"
        base = f"{num(info.intercept, sig)} + " if info.intercept else ""
        how = f"Each tree adds the value of the leaf a sample reaches; {base}sum of leaf values gives {out}."
        if info.is_clf:
            how = (f"Each tree adds the value of the leaf a sample reaches; "
                   f"P({info.class_names[-1]}) = sigmoid({base}sum).")
    elif info.combine == "mean":
        how = "The trees' predictions are averaged."
    lines = _header(name, describe, facts, how)
    splits = [(f, v) for nd in info.nodes if not nd.is_leaf for f, op, v in nd.split]
    fsig = _feature_sig(splits, sig)
    lines.append("")
    if info.layout == "cascade":
        lines += _cascade_lines(info, sig, fsig)
        return "\n".join(lines)
    roots = info.roots
    for k, r in enumerate(roots[:max_trees]):
        if len(roots) > 1:
            lines.append(f"  Tree {k + 1} of {len(roots)}" + (" (+)" if info.combine == "sum" and k else ""))
        lines += _rows_with_samples(_tree_lines(info, r, sig, fsig), indent=4 if len(roots) > 1 else 2)
        if len(roots) > 1:
            lines.append("")
    if len(roots) > max_trees:
        lines.append(f"  ... and {len(roots) - max_trees} more trees (print with imodels.viz.text(model, max_trees=...) to see them)")
    return "\n".join(lines).rstrip()


# ---------------------------------------------------------------- additive models
def _how_additive(v, sig):
    if v.note:
        return _clean(v.note)
    if v.link == "logistic" and v.is_clf:
        return f"P({v.class_names[-1]}) = sigmoid(score)"
    return f"prediction = {v.score_name}"


def _rule_or_term(v, t, sig, fsig):
    if t.kind == "rule":
        if not t.conds:
            return "always"
        return _clean(" and ".join(v.cond_text(f, op, val, fsig.get(f, sig)) for f, op, val in t.conds))
    name = v.feature_names[t.feature]
    if t.kind == "linear":
        clip = f"  (x clipped to [{num(t.clip[0], sig)}, {num(t.clip[1], sig)}])" if t.clip else ""
        return f"× {name}{clip}"
    return name


def _ruleset_lines(v, sig, fsig, scorecard=False):
    terms = sorted(v.terms, key=lambda t: -v.importance(t))
    has_cov = any(t.kind == "rule" and np.isfinite(t.support) for t in terms)
    rule_ws = [t.weight for t in terms if t.kind == "rule"]
    integer = scorecard and all(float(w).is_integer() for w in rule_ws)
    kinds = {t.kind for t in terms}
    head = ["points" if scorecard else ("coef" if kinds == {"linear"} else "effect"),
            "rule" if kinds == {"rule"} else ("feature" if kinds == {"linear"} else "term")]
    rows = [head + (["coverage"] if has_cov else [])]
    for t in terms:
        if t.kind == "rule":
            w = f"{int(t.weight):+d}" if integer else signed(t.weight, sig)
            cells = [w, ("IF " if not scorecard else "") + _rule_or_term(v, t, sig, fsig)]
        else:
            cells = [signed(t.weight, sig), _rule_or_term(v, t, sig, fsig)]
        if has_cov:
            cells.append(f"{t.support:.0%}" if t.kind == "rule" and np.isfinite(t.support) else "")
        rows.append(cells)
    if v.intercept and not scorecard:
        rows.append([signed(v.intercept, sig), "baseline (always added)"] + ([""] if has_cov else []))
    return _table(rows, [">", "<", ">"][:len(rows[0])])


def _risk_lines(v, per_row=11):
    scores = [s for s, _ in v.risk]
    probs = [p for _, p in v.risk]
    lines = [f"  P({v.class_names[-1]}) for each total score:"]
    for i in range(0, len(scores), per_row):
        s_row = ["score"] + [(f"{int(s):+d}" if s else "0") if float(s).is_integer() else num(s, 3)
                             for s in scores[i:i + per_row]]
        p_row = ["risk"] + [f"{p:.1%}" if p < 0.995 and p > 0.005 else f"{p:.0%}" for p in probs[i:i + per_row]]
        lines += _table([s_row, p_row], ["<"] + [">"] * (len(s_row) - 1), gap=2, indent=4)
    return lines


def _domain(ts, v):
    """(lo, hi, label): the feature's range in the data, else the span of the model's split points."""
    if v.X is not None:
        col = v.X[:, ts[0].feature]
        col = col[np.isfinite(col)]
        if len(col):
            return float(np.quantile(col, 0.005)), float(np.quantile(col, 0.995)), "data"
    pts = []
    for t in ts:
        pts += list(np.ravel(t.grid)) if t.kind == "curve" else list(np.ravel(t.edges))
    pts = np.asarray(pts, float)
    pts = pts[np.isfinite(pts)]
    if len(pts) == 0:
        return 0.0, 1.0, "splits"
    return float(pts.min()), float(pts.max()), "splits"


def _gam_lines(v, sig):
    shapes = [t for t in v.terms if t.kind in ("shape", "curve")]
    by_f = {}
    for t in shapes:
        by_f.setdefault(t.feature, []).append(t)
    curves = {}
    for f, ts in by_f.items():
        lo, hi, source = _domain(ts, v)
        pad = (hi - lo) * 0.05 if source == "splits" else 0.0  # show both sides of the outer splits
        grid = np.linspace(lo - pad, hi + pad, 24)
        Xg = np.zeros((len(grid), len(v.feature_names)))
        Xg[:, f] = grid
        curves[f] = (lo, hi, sum(t.contribution(Xg) for t in ts))
    if not curves:
        return []
    allv = np.concatenate([c for _, _, c in curves.values()])
    gmin, gmax = float(allv.min()), float(allv.max())
    span = (gmax - gmin) or 1.0
    order = sorted(curves, key=lambda f: -float(np.ptp(curves[f][2])))
    rows = [["feature", "effect as the feature rises", "effect range",
             "feature range" if v.X is not None else "split points"]]
    for f in order:
        lo, hi, c = curves[f]
        spark = "".join(SPARK[min(7, int((x - gmin) / span * 7.999))] for x in c)
        rows.append([v.feature_names[f], spark, f"{signed(c.min(), sig)} to {signed(c.max(), sig)}",
                     f"{num(lo, sig)} to {num(hi, sig)}"])
    lines = _table(rows, ["<", "<", "<", "<"])
    lines.append(f"  (each bar of the sparkline is one step along the feature's range; heights share one scale, "
                 f"{signed(gmin, sig)} to {signed(gmax, sig)})")
    return lines


def additive_text(v, sig=3):
    name = v.model_name
    describe = DESCRIBE.get(name.split(" (")[0])
    n_terms = len(v.terms)
    facts = []
    if v.family == "gam":
        facts.append(f"{len({t.feature for t in v.terms})} shape functions")
    elif v.family == "scorecard":
        facts.append(f"{n_terms} scoring items")
    else:
        kinds = {t.kind for t in v.terms}
        facts.append(f"{n_terms} {'rules' if kinds == {'rule'} else 'terms'}")
    if v.y is not None:
        facts.append(f"{len(v.y):,} training samples")
    if v.is_clf and v.class_names:
        facts.append("classes: " + ", ".join(v.class_names))
    lines = _header(name, describe, facts, _how_additive(v, sig))
    pairs = [(f, val) for t in v.terms if t.kind == "rule" for f, op, val in t.conds]
    fsig = _feature_sig(pairs, sig)
    lines.append("")
    if v.family == "gam":
        lines += _gam_lines(v, sig)
        others = [t for t in v.terms if t.kind not in ("shape", "curve")]
        if others:
            lines.append("")
            lines += _ruleset_lines(type(v)(**{**v.__dict__, "terms": others}), sig, fsig)
        lines.append(f"  baseline (always added): {num(v.intercept, sig)}")
    elif v.family == "scorecard":
        lines += _ruleset_lines(v, sig, fsig, scorecard=True)
        if v.risk:
            lines.append("")
            lines += _risk_lines(v)
    else:
        lines += _ruleset_lines(v, sig, fsig)
    return "\n".join(lines).rstrip()


def model_text(view, sig=3, max_trees=3, extra=None):
    from ._views import AdditiveView

    out = additive_text(view, sig) if isinstance(view, AdditiveView) else tree_text(view, sig, max_trees)
    if extra:
        head, rest = out.split("\n", 1) if "\n" in out else (out, "")
        out = head + "\n  " + extra + ("\n" + rest if rest else "")
    return out


def text(model, X=None, y=None, *, feature_names=None, class_names=None, target_name=None, precision=4, max_trees=3):
    """Write a fitted model out as readable text: what ``print(model)`` shows for imodels estimators.

    Trees print their conditions and leaf outcomes, rule lists their IF / ELSE IF rows, rule sets
    their rules ranked by effect, scorecards their points and the risk of each total, and additive
    models a sparkline of each shape function.

    Parameters
    ----------
    model : estimator
        A fitted model: an imodels estimator (tree, sum of trees, rule list, rule set, scoring
        system or additive model) or a scikit-learn tree, forest, gradient-boosting, linear or
        isotonic model, or a Pipeline whose earlier steps only scale features.
    X : array-like of shape (n_samples, n_features), optional
        Training (or held-out) data. With it, split nodes show their feature's distribution,
        rules show their coverage, and thresholds on binary or integer features read naturally.
    y : array-like of shape (n_samples,), optional
        Targets for ``X``, used for class mixes and target ranges.
    feature_names : list of str, optional
        Feature names. Default: the columns of ``X``, else the names the model was fitted with.
    class_names : list or dict, optional
        Class names, as a list or as a dict from class label to name. Default: the model's classes.
    target_name : str, optional
        Name of the target (regression). Default: the name of ``y``, else "target".
    precision : int, default=4
        Significant digits for values. Thresholds get more digits where two would print alike.
    max_trees : int, default=3
        For ensembles, how many trees to write out.

    Returns
    -------
    str
        The model as text.
    """
    from ._adapt import to_view

    view = to_view(model, X, y, feature_names, class_names, target_name)
    extra = None
    if hasattr(model, "optimal_") and hasattr(model, "objective_"):  # FastSmallTree
        status = "certified optimal" if model.optimal_ else "not certified optimal (the time limit was reached)"
        extra = f"{status}, objective {model.objective_:.4g}"
    return model_text(view, precision, max_trees, extra)
