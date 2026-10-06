"""Adapters from imodels estimators to imodels.viz views.

Tree-based models (trees, FIGS, IRF) are exported to scikit-learn with ``imodels.to_sklearn`` and drawn
by the scikit-learn readers, so no model-specific tree code lives here. The other adapters read the
fitted structure (following imodels' own predict logic) into a TreeInfo (rule lists) or an AdditiveView
(rule sets, scorecards, GAMs, linear models). tests/viz_imodels_test.py checks that every view
reproduces the model's predictions.
"""

import re
import warnings

import numpy as np

from ._extract import extract, sk_threshold
from ._ir import make_info, rule_list_specs
from ._views import AdditiveView, Term


# ---------------------------------------------------------------- helpers
def _names(model, X, feature_names, n=None):
    if feature_names is not None:
        return [str(f) for f in feature_names]
    if X is not None and hasattr(X, "columns"):
        return [str(c) for c in X.columns]
    for attr in ("feature_names_in_", "feature_names_", "feature_names"):
        v = getattr(model, attr, None)
        if v is not None and len(v) and not str(v[0]).startswith("X_"):
            return [str(c) for c in v]
    n = n or getattr(model, "n_features_in_", None) or (np.asarray(X).shape[1] if X is not None else 0)
    return [f"x{i}" for i in range(n)]


def _classes(model, y):
    c = getattr(model, "classes_", None)
    if c is None:
        c = np.unique(y) if y is not None else np.array([0, 1])
    return np.asarray(c)


def _class_names(classes, class_names):
    if class_names is None:
        return [str(c) for c in classes]
    if isinstance(class_names, dict):
        return [str(class_names.get(c, c)) for c in classes]
    return [str(c) for c in class_names]


def _y_index(y, classes):
    if y is None:
        return None
    ya = np.asarray(y)
    lookup = {c: i for i, c in enumerate(classes)}
    try:
        return np.array([lookup[v] for v in ya])
    except KeyError:  # e.g. classes_ stored as floats while y is int
        lookup = {float(c): i for i, c in enumerate(classes)}
        return np.array([lookup[float(v)] for v in ya])


def _X(X):
    return None if X is None else np.asarray(X, dtype=float)


def _parse_agg(agg_dict, names):
    """imodels Rule.agg_dict {(feature, op): value_str} -> [(index, op, value)]."""
    conds = []
    for (f, op), v in agg_dict.items():
        m = re.fullmatch(r"X_(\d+)", str(f))
        j = int(m.group(1)) if m else names.index(f)
        conds.append((j, op, float(v)))
    return conds


# ---------------------------------------------------------------- tree-based models
def _exported(m, X, y, fn, cn, tn):
    """Trees, FIGS and IRF, drawn through their scikit-learn export."""
    from imodels.util.sklearn_export import to_sklearn

    from . import _sklearn

    est = to_sklearn(m, X)  # with X, node sample counts are recounted on the data
    name = type(m).__name__
    if isinstance(est, list):  # FIGS: regression trees that add up
        clf = hasattr(m, "classes_")
        info = _sklearn.sum_of_trees(est, X, y, _names(m, X, fn, est[0].n_features_in_), cn, tn, name,
                                     classes=_classes(m, y) if clf else None)
    else:
        names = _names(m, X, fn, est.n_features_in_)
        info = _sklearn.adapt(est, X, y, names, cn, tn)  # forests (IRF, shrunk forests)
        if info is None:
            info = extract(est, X, y, names, cn, tn)
    info.model_name = name
    if hasattr(m, "reg_param") and getattr(m, "reg_param", None) is not None:
        info.model_name = f"{name} (shrinkage {m.reg_param:g})"
    return info


# ---------------------------------------------------------------- rule lists
def _restore_model_proba(info, specs):
    """Rule lists predict the stored probability; keep it even when data was routed."""
    for nd, sp in zip(info.nodes, specs):
        if sp.get("proba") is not None:
            nd.counts = np.asarray(sp["proba"], dtype=float) * max(nd.n, 1)
    return info


def _rule_list_dicts(m, X, y, fn, cn, tn, strict):
    """GreedyRuleList / OneR (strict: x < c if flip else x >= c) and FastFrugalTree (x <= c / x > c)."""
    names = _names(m, X, fn)
    classes = _classes(m, y)
    rules = m.rules_
    out = []
    for r in rules[:-1]:
        op = ("<" if r["flip"] else ">=") if strict else ("<=" if r["flip"] else ">")
        p = float(r["val_right"])
        out.append(([(int(r["index_col"]), op, float(r["cutoff"]))],
                    dict(proba=[1 - p, p], n=int(r.get("num_pts_right", 0) or 0))))
    p = float(rules[-1]["val"])
    specs = rule_list_specs(out, dict(proba=[1 - p, p], n=int(rules[-1].get("num_pts", 0) or 0)))
    info = make_info(specs, task="classification", feature_names=names, model_name=type(m).__name__,
                     class_names=_class_names(classes, cn), X=_X(X), y=_y_index(y, classes), layout="cascade",
                     target_name=tn)
    return _restore_model_proba(info, specs)


def _brl(m, X, y, fn, cn, tn):
    names = _names(m, X, fn)
    classes = _classes(m, y)
    theta = np.asarray(m.theta, dtype=float)
    out = [(_parse_agg(r.agg_dict, names), dict(proba=[1 - theta[k], theta[k]]))
           for k, r in enumerate(m.rules_without_feature_names_)]
    specs = rule_list_specs(out, dict(proba=[1 - theta[-1], theta[-1]]))
    info = make_info(specs, task="classification", feature_names=names, model_name=type(m).__name__,
                     class_names=_class_names(classes, cn), X=_X(X), y=_y_index(y, classes), layout="cascade",
                     target_name=tn)
    return _restore_model_proba(info, specs)


# ---------------------------------------------------------------- rule sets
def _rulefit(m, X, y, fn, cn, tn):
    clf = hasattr(m, "classes_")
    names = _names(m, X, fn)
    rules = m.rules_without_feature_names_
    coef = np.ravel(np.asarray(m.coef, dtype=float))
    nlin = len(coef) - len(rules)
    terms = []
    if nlin and getattr(m, "include_linear", True):
        lims = np.asarray(m.winsorizer.winsor_lims) if m.lin_standardise else None
        mult = np.asarray(m.friedscale.scale_multipliers) if m.lin_standardise else np.ones(nlin)
        for j in range(nlin):
            w = float(coef[j]) * float(mult[j])
            if w:
                clip = (float(lims[0, j]), float(lims[1, j])) if lims is not None else None
                terms.append(Term("linear", w, feature=j, clip=clip))
    for r, c in zip(rules, coef[nlin:]):
        if c:
            terms.append(Term("rule", float(c), conds=_parse_agg(r.agg_dict, names)))
    if clf:  # RuleFit's classifier: P = softmax([1 - f, f]) = sigmoid(2f - 1)
        classes = _classes(m, y)
        cls = _class_names(classes, cn)
        v = AdditiveView("ruleset", "classification", names, type(m).__name__, terms, intercept=float(np.ravel(m.intercept)[0]),
                         link="logistic", link_scale=2.0, link_offset=-1.0, class_names=cls, score_name="score f",
                         note=f"f = baseline + rules that hold + linear terms;  P({cls[-1]}) = sigmoid(2f − 1).")
        return v.attach(_X(X), _y_index(y, classes))
    v = AdditiveView("ruleset", "regression", names, type(m).__name__, terms, intercept=float(np.ravel(m.intercept)[0]),
                     target_name=tn or "target", score_name=tn or "prediction",
                     note="Prediction = baseline + rules that hold + linear terms (linear terms use winsorized x).")
    return v.attach(_X(X), y)


def _skope(m, X, y, fn, cn, tn):
    names = _names(m, X, fn)
    classes = _classes(m, y)
    rules = m.rules_without_feature_names_
    tot = sum(float(r.args[0]) for r in rules) or 1.0
    terms = [Term("rule", float(r.args[0]) / tot, conds=_parse_agg(r.agg_dict, names)) for r in rules]
    cls = _class_names(classes, cn)
    v = AdditiveView("ruleset", "classification", names, type(m).__name__, terms, link="clip", class_names=cls,
                     score_name=f"P({cls[-1]})",
                     note=f"P({cls[-1]}) = precision-weighted share of the rules that hold "
                          "(each rule's weight is its out-of-bag precision, normalized to sum to 1).")
    return v.attach(_X(X), _y_index(y, classes))


def _point(conds, p, holds=True):
    """A row where all of ``conds`` hold (holds=True) or where the first one fails."""
    x = np.zeros(p)
    if not holds:
        f, op, v = conds[0]
        x[f] = v + 1 + abs(v) if op in ("<=", "<") else v - 1 - abs(v)
        return x
    for f in {c[0] for c in conds}:
        lo = max([v for g, op, v in conds if g == f and op in (">", ">=")], default=-np.inf)
        hi = min([v for g, op, v in conds if g == f and op in ("<=", "<")], default=np.inf)
        if np.isfinite(lo) and np.isfinite(hi):
            x[f] = (lo + hi) / 2
        elif np.isfinite(hi):
            x[f] = hi if all(op == "<=" for g, op, _ in conds if g == f) else hi - 1 - abs(hi)
        else:
            x[f] = lo if all(op == ">=" for g, op, _ in conds if g == f) else lo + 1 + abs(lo)
    return x


def _adaboost_rules(m, conds_of):
    """Boosted rules as (intercept, [(conds, weight)]): each estimator's share of the decision function
    where its rule holds and where it does not, read from the model itself (with only that estimator
    kept), so it is exact whichever AdaBoost variant (SAMME, SAMME.R) the installed sklearn uses."""
    est, w = list(m.estimators_), np.asarray(m.estimator_weights_, dtype=float)
    W = float(np.sum(w[:len(est)])) or 1.0
    p = m.n_features_in_
    intercept, rules = 0.0, []
    try:
        for k, e in enumerate(est):
            conds = conds_of(e)
            m.estimators_, m.estimator_weights_ = [e], w[k:k + 1]
            pts = np.array([_point(conds, p, True), _point(conds, p, False)]) if conds else np.zeros((2, p))
            d = np.ravel(m.decision_function(pts)) * w[k] / W
            intercept += float(d[1])
            if conds and d[0] != d[1]:
                rules.append((conds, float(d[0] - d[1])))
    finally:
        m.estimators_, m.estimator_weights_ = est, w
    return intercept, rules


def _boosted(m, X, y, fn, cn, tn):
    names = _names(m, X, fn)
    classes = _classes(m, y)
    if len(classes) != 2:
        raise NotImplementedError("Multiclass BoostedRules is not supported yet.")

    def conds_of(e):  # a stump sends x left when x <= t
        t = e.tree_
        return [] if t.node_count == 1 else [(int(t.feature[0]), "<=", sk_threshold(t.threshold[0]))]

    intercept, rules = _adaboost_rules(m, conds_of)
    terms = [Term("rule", wt, conds=c) for c, wt in rules]
    cls = _class_names(classes, cn)
    v = AdditiveView("ruleset", "classification", names, type(m).__name__, terms, intercept=intercept,
                     link="logistic", class_names=cls, score_name="decision",
                     note=f"Each boosted stump moves the decision toward the class of its leaf; "
                          f"P({cls[-1]}) = sigmoid(sum). Stumps are shown as one rule plus a share of the baseline.")
    return v.attach(_X(X), _y_index(y, classes))


def _slipper(m, X, y, fn, cn, tn):
    names = _names(m, X, fn)
    classes = _classes(m, y)
    intercept, rules = _adaboost_rules(
        m, lambda e: [(int(c["feature"]), c["operator"], float(c["pivot"])) for c in e.rule])
    terms = [Term("rule", wt, conds=c) for c, wt in rules]
    cls = _class_names(classes, cn)
    v = AdditiveView("ruleset", "classification", names, type(m).__name__, terms, intercept=intercept,
                     link="logistic", class_names=cls, score_name="decision",
                     note=f"Each rule adds its weight to the decision when it holds; P({cls[-1]}) = sigmoid(sum).")
    return v.attach(_X(X), _y_index(y, classes))


def _brs(m, X, y, fn, cn, tn):
    attr = [str(a) for a in m.attr_names_orig] if hasattr(m, "attr_names_orig") else None
    names = _names(m, X, fn, len(attr) if attr else None)
    classes = np.array([0, 1])
    terms = []
    for rule in m.rules_:
        conds = []
        for item in rule:
            neg = item.endswith("_neg")
            col = item[:-4] if neg else item
            conds.append(((attr or names).index(col), "<=" if neg else ">", 0.5))
        terms.append(Term("rule", 1.0, conds=conds))
    cls = _class_names(classes, cn)
    v = AdditiveView("ruleset", "classification", names, type(m).__name__, terms, link="threshold",
                     threshold=0.5, class_names=cls, score_name="rules that hold",
                     note=f"Predicts {cls[-1]} when any rule holds (the model gives no probabilities).")
    return v.attach(_X(X), _y_index(y, classes) if y is not None else None)


# ---------------------------------------------------------------- scoring systems
def _frs_encode(X, categorical, categories):
    """X as floats the way FastRiskScore reads it: numeric columns via ``pd.to_numeric``, categorical
    columns as codes into ``categories[j]`` (levels the model does not use are appended), missing as NaN."""
    import pandas as pd

    frame = X if isinstance(X, pd.DataFrame) else pd.DataFrame(np.asarray(X, dtype=object))
    out = np.full(frame.shape, np.nan)
    for j in range(frame.shape[1]):
        col = frame.iloc[:, j]
        if j < len(categorical) and categorical[j]:
            try:
                from imodels.algebraic.risk_score.fast_risk_score import _levels
                levels = _levels(col)
            except ImportError:  # older imodels
                levels = col.astype(object).astype(str).to_numpy(object)
            miss = col.isna().to_numpy()
            known = categories.setdefault(j, [])
            for lv in pd.unique(levels[~miss]):
                if lv not in known:
                    known.append(lv)
            lookup = {lv: k for k, lv in enumerate(known)}
            out[:, j] = [np.nan if mi else lookup[lv] for lv, mi in zip(levels, miss)]
        else:
            out[:, j] = pd.to_numeric(col, errors="coerce").to_numpy(float)
    return out


def _fastriskscore(m, X, y, fn, cn, tn):
    names = list(getattr(m, "feature_names_", None) or _names(m, X, fn))
    classes = _classes(m, y)
    terms, categories = [], {}
    binz = getattr(m, "binarizer_", None) if getattr(m, "binarize", True) else None
    categorical = list(getattr(binz, "categorical_", [])) if binz is not None else []
    if binz is not None:
        # rules_: (column, kind, value, name); column is a position (imodels >= 3.0.3) or a name (older)
        col_of = lambda c: int(c) if isinstance(c, (int, np.integer)) else names.index(c)
        rules = {name: (col_of(c), kind, val) for c, kind, val, name in binz.rules_}
        for c, kind, val, _ in binz.rules_:  # the model's levels come first, in its order
            if kind == "eq_str":
                categories.setdefault(col_of(c), []).append(val)
        for name, pts in m.points_.items():
            if not pts:
                continue
            j, kind, val = rules[name]
            if kind == "le":
                conds = [(j, "<=", float(val))]
            elif kind == "eq":
                conds = [(j, "==", float(val))]
            elif kind == "missing":
                conds = [(j, "isnan", 0.0)]
            elif kind == "eq_str":
                conds = [(j, "==", float(categories[j].index(val)))]
            else:
                raise NotImplementedError(f"FastRiskScore indicator kind {kind!r} is not supported.")
            terms.append(Term("rule", float(pts), conds=conds))
    else:
        for j, pts in enumerate(np.asarray(m.coef_).ravel()):
            if pts:
                terms.append(Term("linear", float(pts), feature=j))
    Xa = _frs_encode(X, categorical, categories) if X is not None else None
    cls = _class_names(classes, cn)
    sign = "+" if m.intercept_ >= 0 else "\u2212"
    v = AdditiveView("scorecard", "classification", names, type(m).__name__, terms, link="logistic",
                     link_scale=float(m.scale_), link_offset=float(m.intercept_), class_names=cls, score_name="score",
                     categories={j: [str(lv) for lv in lvs] for j, lvs in categories.items()},
                     note=f"P({cls[-1]}) = sigmoid({m.scale_:.3g} \u00d7 score {sign} {abs(m.intercept_):.3g})")
    v.attach(Xa, _y_index(y, classes))
    if all(t.kind == "rule" for t in terms):
        lo = sum(min(0, t.weight) for t in terms)
        hi = sum(max(0, t.weight) for t in terms)
        v.risk = [(float(s), float(v.output(s))) for s in np.arange(lo, hi + 1)]
    return v


def _slim(m, X, y, fn, cn, tn):
    est = m.model_
    clf = hasattr(est, "classes_")
    names = _names(m, X, fn)
    coef = np.asarray(est.coef_, dtype=float)
    if clf and coef.ndim > 1 and coef.shape[0] > 1:
        raise NotImplementedError("Multiclass SLIM is not supported.")
    Xa = _X(X)
    terms = []
    for j, w in enumerate(coef.ravel()):
        if not w:
            continue
        if Xa is not None and np.isin(np.unique(Xa[:, j]), [0.0, 1.0]).all():
            # SLIM's intended use: 0/1 features, so each one is a scorecard item worth w points
            terms.append(Term("rule", float(w), conds=[(j, ">", 0.5)]))
        else:
            terms.append(Term("linear", float(w), feature=j))
    b = float(np.ravel(est.intercept_)[0])
    exact = not getattr(m, "_fit_backup_used", False)
    if clf:
        classes = _classes(est, y)
        cls = _class_names(classes, cn)
        sign = "+" if b >= 0 else "\u2212"
        # the score is the sum of integer points; the (possibly fractional) intercept moves into the link
        v = AdditiveView("scorecard", "classification", names, type(m).__name__, terms, link="logistic",
                         link_offset=b, class_names=cls, score_name="score",
                         note=f"P({cls[-1]}) = sigmoid(score {sign} {abs(b):.3g})")
        v.attach(Xa, _y_index(y, classes))
        if all(t.kind == "rule" for t in terms) and terms:
            lo = int(sum(min(0, t.weight) for t in terms))
            hi = int(sum(max(0, t.weight) for t in terms))
            v.risk = [(float(k), float(v.output(k))) for k in range(lo, hi + 1)]
        elif v.X is not None:
            sc = v.score(v.X)
            grid = np.unique(np.round(np.linspace(np.quantile(sc, 0.01), np.quantile(sc, 0.99), 9), 1))
            v.risk = [(float(k), float(v.output(k))) for k in grid]
        return v
    v = AdditiveView("ruleset", "regression", names, type(m).__name__, terms, intercept=b, target_name=tn or "target",
                     score_name=tn or "prediction", note="Prediction = intercept + integer coefficient \u00d7 feature.")
    return v.attach(Xa, y)


def _marginal_shrinkage(m, X, y, fn, cn, tn):
    names = _names(m, X, fn)
    sx, sy = m.scalar_X_, m.scalar_y_
    coef = np.asarray(m.est_main_.coef_, dtype=float).ravel()
    w = coef / sx.scale_ * sy.scale_[0]
    b = float(sy.mean_[0] - np.sum(sx.mean_ / sx.scale_ * coef) * sy.scale_[0])
    terms = [Term("linear", float(wj), feature=j) for j, wj in enumerate(w) if wj]
    v = AdditiveView("ruleset", "regression", names, type(m).__name__, terms, intercept=b, target_name=tn or "target",
                     score_name=tn or "prediction", note="Prediction = intercept + coefficient × feature (raw units).")
    return v.attach(_X(X), y)


# ---------------------------------------------------------------- additive models
def _tree_of(e):
    return getattr(e, "estimator_", e).tree_


def _shape_from_trees(trees_by_f, n_features):
    """Merge single-feature trees [(coef, estimator)] into one step function per feature."""
    terms = []
    for f, items in sorted(trees_by_f.items()):
        cuts = sorted({sk_threshold(t) for _, e in items for t in _tree_of(e).threshold[_tree_of(e).feature == f]})
        if not cuts:
            continue
        edges = np.array([-np.inf] + cuts + [np.inf])
        # evaluate each bin (edge_i, edge_i+1] strictly inside it: sklearn compares in float32, so a
        # value sitting exactly on a cut can round to the other side
        c = np.array(cuts)
        span = max(1.0, float(np.ptp(c)))
        reps = np.concatenate([[c[0] - span], (c[:-1] + c[1:]) / 2, [c[-1] + span]])
        Xg = np.zeros((len(reps), n_features))
        Xg[:, f] = reps
        vals = sum(c * e.predict(Xg) for c, e in items)
        terms.append(Term("shape", feature=f, edges=edges, values=np.asarray(vals, dtype=float)))
    return terms


def _treegam(m, X, y, fn, cn, tn):
    clf = hasattr(m, "classes_")
    names = _names(m, X, fn)
    p = m.n_features_in_
    if getattr(m, "estimators_marginal", None):
        raise NotImplementedError("TreeGAM with marginal estimators is not supported yet.")
    coefs = getattr(m, "cyclic_coef_", None)
    if coefs is None:
        coefs = np.ones(len(m.estimators_)) * m.learning_rate
    by_f, const = {}, float(m.bias_)
    for c, e in zip(coefs, m.estimators_):
        f = int(_tree_of(e).feature[0])
        if f < 0:  # a constant tree just shifts the baseline
            const += float(c * e.predict(np.zeros((1, p)))[0])
        else:
            by_f.setdefault(f, []).append((float(c), e))
    terms = _shape_from_trees(by_f, p)
    if clf:
        classes = _classes(m, y)
        cls = _class_names(classes, cn)
        v = AdditiveView("gam", "classification", names, type(m).__name__, terms, intercept=const, link="clip",
                         class_names=cls, score_name=f"P({cls[-1]})",
                         note=f"P({cls[-1]}) = baseline {const:.3g} + sum of shape functions, clipped to [0, 1].")
        return v.attach(_X(X), _y_index(y, classes))
    v = AdditiveView("gam", "regression", names, type(m).__name__, terms, intercept=const, target_name=tn or "target",
                     score_name=tn or "prediction", note=f"Prediction = baseline {const:.4g} + sum of shape functions.")
    return v.attach(_X(X), y)


def _gpgam(m, X, y, fn, cn, tn):
    if getattr(m, "pairs_", None):
        raise NotImplementedError("GPGam with pairwise interactions is not supported; fit it with n_pairs=0.")
    if getattr(m, "log_target_", False):
        raise NotImplementedError("GPGam with a log target is not supported.")
    names = _names(m, X, fn)
    terms = []
    for u, j in enumerate(m.units_):
        vals = np.asarray(m.main_values_[m.main_offsets_[u]:m.main_offsets_[u + 1]], dtype=float) * m.y_std_
        step = bool(m.cats_[j]) if hasattr(m, "cats_") else False
        if step or len(vals) < 3:
            # lookup via searchsorted(edges, x, 'right'), i.e. bins [e_i, e_i+1): nudge edges for our (e_i, e_i+1] bins
            e = np.concatenate([[-np.inf], np.nextafter(np.asarray(m.edges_[j], dtype=float), -np.inf), [np.inf]])
            terms.append(Term("shape", feature=int(j), edges=e, values=vals))
        else:
            tied = getattr(m, "tied_bins_", {}).get(j)
            terms.append(Term("curve", feature=int(j), grid=np.asarray(m.grids_[j], dtype=float), values=vals,
                              edges=np.asarray(m.edges_[j], dtype=float) if tied is not None else None,
                              tied=np.asarray(tied, dtype=bool) if tied is not None else None))
    const = float(m.y_mean_ + m.bias_)
    v = AdditiveView("gam", "regression", names, type(m).__name__, terms, intercept=const, target_name=tn or "target",
                     score_name=tn or "prediction", note=f"Prediction = {const:.4g} + sum of shape functions.")
    return v.attach(_X(X), y)


# ---------------------------------------------------------------- dispatch
TREES = ("GreedyTreeClassifier", "GreedyTreeRegressor", "DecisionTreeCCPClassifier", "DecisionTreeCCPRegressor",
         "HSTreeClassifier", "HSTreeRegressor", "HSTreeClassifierCV", "HSTreeRegressorCV",
         "HSDecisionTreeCCPClassifierCV", "HSDecisionTreeCCPRegressorCV", "TaoTreeClassifier",
         "FastSmallTreeClassifier", "C45TreeClassifier", "HSC45TreeClassifier", "HSC45TreeClassifierCV",
         "FIGSClassifier", "FIGSRegressor", "FIGSClassifierCV", "FIGSRegressorCV", "IRFClassifier", "IRFRegressor")
ADAPTERS = {
    **{name: _exported for name in TREES},
    "GreedyRuleListClassifier": lambda m, *a: _rule_list_dicts(m, *a, strict=True),
    "OneRClassifier": lambda m, *a: _rule_list_dicts(m, *a, strict=True),
    "FastFrugalTreeClassifier": lambda m, *a: _rule_list_dicts(m, *a, strict=False),
    "BayesianRuleListClassifier": _brl,
    "RuleFitClassifier": _rulefit, "RuleFitRegressor": _rulefit,
    "FPLassoClassifier": _rulefit, "FPLassoRegressor": _rulefit,
    "SkopeRulesClassifier": _skope, "FPSkopeClassifier": _skope,
    "BoostedRulesClassifier": _boosted,
    "SlipperClassifier": _slipper,
    "BayesianRuleSetClassifier": _brs,
    "FastRiskScoreClassifier": _fastriskscore,
    "SLIMClassifier": _slim, "SLIMRegressor": _slim,
    "MarginalShrinkageLinearModelRegressor": _marginal_shrinkage,  # name before imodels 3.0.3
    "MarginalShrinkageLinearRegressor": _marginal_shrinkage,
    "TreeGAMClassifier": _treegam, "TreeGAMRegressor": _treegam,
    "GPGamRegressor": _gpgam,
}
UNSUPPORTED = {
    "BoostedRulesRegressor": "its prediction is a weighted median of stumps, which is not additive",
    "TaoTreeRegressor": "imodels does not fit it yet",
}


def adapt(model, X, y, feature_names, class_names, target_name):
    """Return a view for an imodels estimator, or None if the model is not one we adapt."""
    name = type(model).__name__
    if name in ("AutoInterpretableClassifier", "AutoInterpretableRegressor"):
        inner = model.est_.best_estimator_.named_steps["est"]
        return adapt(inner, X, y, feature_names, class_names, target_name)
    if name in UNSUPPORTED:
        raise NotImplementedError(f"{name} is not supported: {UNSUPPORTED[name]}.")
    fn = ADAPTERS.get(name)
    if fn is None:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return fn(model, X, y, feature_names, class_names, target_name)
