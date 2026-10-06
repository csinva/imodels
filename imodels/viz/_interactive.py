"""Interactive mode: a self-contained HTML page (no network access needed)."""

import html
import os

import numpy as np

from ._adapt import to_view
from ._extract import TreeInfo, _unwrap
from ._render import BAND, Artist, defs, resolve_orientation
from ._static import _resolve_style
from ._text import esc, fmt, fmt_count, new_uid, text_width
from ._theme import Paint


def _tooltip(info, artist, nid, sig):
    nd = info.nodes[nid]
    P = artist.P
    rows = []
    if nd.is_leaf and info.combine == "sum":
        v = getattr(nd, "value", None)
        v = float(nd.counts[0]) if v is None else v
        to = f" toward {esc(info.class_names[1 if v >= 0 else 0])}" if info.is_clf else ""
        head = f'<span class="tt-k">Leaf</span> adds <b>{"+" if v > 0 else ""}{esc(fmt(v, max(sig, 4)))}</b>{to}'
    elif nd.is_leaf:
        k = info.prediction(nd)
        pred = info.class_names[k] if info.is_clf else fmt(nd.counts[0], max(sig, 4))
        head = f'<span class="tt-k">Leaf</span> predicts <b>{esc(pred)}</b>'
    else:
        head = (f'<span class="tt-k">Split</span> <b>{esc(info.feature_names[nd.feature])}</b> '
                f'{esc(info.edge_label(nd.left, sig))}' if nd.simple else
                f'<span class="tt-k">Split</span> <b>{esc(info.split_text(nd.id, sig))}</b>')
    rules = info.rules(nid, sig)
    rule_html = "".join(f"<li>{esc(r)}</li>" for r in rules) or "<li class='tt-m'>all samples (root)</li>"
    share = nd.weight / (info.root.weight or 1)
    stats = [("samples", f"{fmt_count(nd.n)} <span class='tt-m'>({share:.1%})</span>")]
    if np.isfinite(nd.impurity):
        stats.append((info.criterion.replace("_", " "), fmt(nd.impurity, sig)))
    if info.is_clf:
        tot = nd.counts.sum() or 1
        for k, c in enumerate(nd.counts):
            if c <= 0:
                continue
            rows.append(
                f"<tr><td><i style='background:{P.cls(k)}'></i>{esc(info.class_names[k])}</td>"
                f"<td class='num'>{fmt_count(c)}</td><td class='num'>{c / tot:.0%}</td>"
                f"<td class='bar'><span style='width:{c / tot * 100:.1f}%;background:{P.cls(k)}'></span></td></tr>")
    else:
        stats.append(("mean", fmt(nd.counts[0], max(sig, 4))))
        if info.y is not None and nd.idx is not None and len(nd.idx):
            ys = info.y[nd.idx]
            stats += [("std", fmt(ys.std(), sig)), ("range", f"{fmt(ys.min(), sig)} to {fmt(ys.max(), sig)}")]
    stat_html = "".join(f"<tr><td>{esc(k)}</td><td class='num' colspan=3>{v}</td></tr>" for k, v in stats)
    return (f"<div class='tt-h'>{head}</div><ul class='tt-r'>{rule_html}</ul>"
            f"<table>{stat_html}{rows and '<tr class=sep><td colspan=4></td></tr>' or ''}{''.join(rows)}</table>")


from ._page import build_page, class_legend, finite as _finite  # noqa: E402


def _marginal(info, artist, f, thr):
    """Histogram of feature ``f`` over the data (class-stacked for classifiers) for the HUD."""
    if info.X is None:
        lo, hi = min(thr), max(thr)
        span = (hi - lo) or abs(hi) or 1.0
        return dict(hlo=lo - 0.25 * span, hhi=hi + 0.25 * span, hist=None, ends=None)
    col = info.X[:, f]
    ok = np.isfinite(col)
    col = col[ok]
    kind = info.feature_kind.get(f, "continuous")
    lo, hi, ends = artist._range(col, robust=kind == "continuous")
    lo, hi = min(lo, min(thr)), max(hi, max(thr))
    if kind in ("binary", "integer", "categorical") and hi - lo <= 30:
        ends = (fmt(col.min(), 12), fmt(col.max(), 12))
        lo, hi = float(np.floor(lo) - 0.5), float(np.ceil(hi) + 0.5)
        edges = np.arange(lo, hi + 1e-9, 1.0)
    else:
        edges = np.linspace(lo, hi, 29)
    col = np.clip(col, lo, hi)
    if info.is_clf and info.y is not None:
        ys = info.y[ok]
        hist = [np.histogram(col[ys == k], bins=edges)[0].tolist() for k in range(len(info.class_names))]
        hist = [list(b) for b in zip(*hist)]  # per bin: counts by class
    else:
        hist = [[int(c)] for c in np.histogram(col, bins=edges)[0]]
    return dict(hlo=float(lo), hhi=float(hi), hist=hist, ends=list(ends))


class InteractiveTree:
    """An interactive page for a model, returned by `interactive`: one self-contained HTML file.
    Displays inline in Jupyter.

    Attributes
    ----------
    html : str
        The page as HTML text.
    height : int
        Height in pixels when shown inline in Jupyter.
    """

    def __init__(self, html_text, height=720):
        self.html = html_text
        self.height = height

    def save(self, path):
        """Write the page to an .html file.

        Parameters
        ----------
        path : str or path-like
            Output file, ending in .html or .htm.

        Returns
        -------
        path : str or path-like
            The path written.
        """
        if os.path.splitext(str(path))[1].lower() not in (".html", ".htm"):
            raise ValueError("Interactive trees save to .html")
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.html)
        return path

    def _repr_html_(self):
        return (f'<iframe srcdoc="{html.escape(self.html, quote=True)}" '
                f'style="width:100%;height:{self.height}px;border:0;border-radius:12px" loading="lazy"></iframe>')


def interactive(model, X=None, y=None, *, feature_names=None, class_names=None, target_name=None,
                title=None, subtitle=None, theme="light", orientation="auto", style="auto",
                initial_depth=None, precision=3, max_samples=400, simple="auto", output=0, height=720,
                max_trees=None):
    """Build an interactive page for a fitted model: one self-contained HTML file.

    Click a split to fold it, drag to pan, scroll to zoom, and hover for the full rule. The Predict
    panel routes a typed or sampled input through the model, shows how its prediction is built,
    and lists the smallest changes that would flip it. The Features panel shows each feature's
    importance and highlights where it is used. Shows inline in Jupyter; save with ``.save(path)``.

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
    title, subtitle : str, optional
        Page title, and a subtitle (default: a one-line summary of the model).
    theme : {"light", "dark", "auto"}, default="light"
        Color theme; "auto" follows the viewer's OS. The page has a toggle either way.
    orientation : {"auto", "LR", "TB"}, default="auto"
        Left to right, or top to bottom. "auto" lays a single tree out left to right and
        several trees (forests, boosting, FIGS) top to bottom.
    style : {"auto", "detailed", "compact"}, default="auto"
        Card style for trees.
    initial_depth : int, optional
        Levels expanded on load (default: all if the tree has at most 63 nodes, else 3).
    precision : int, default=3
        Significant digits for thresholds and values.
    max_samples : int, default=400
        Rows of ``X`` embedded for the "random sample" button (0 embeds none).
    simple : bool or "auto", default="auto"
        Start in simple mode (plain boxes filled by class mix); "auto" does so for trees with
        more than 32 leaves. The page has a toggle either way.
    output : int, default=0
        For multi-output trees, which output to show.
    height : int, default=720
        Height in pixels when shown inline in Jupyter.
    max_trees : int, optional
        For ensembles (forests, boosting, FIGS), how many trees to draw (default 6).
        Predictions always use every tree.

    Returns
    -------
    InteractiveTree
        The page, with ``.html`` (the HTML text) and ``.save(path)``.
    """
    info = to_view(model, X, y, feature_names, class_names, target_name, output)
    from ._views import AdditiveView

    if isinstance(info, AdditiveView):
        from ._interactive_view import interactive_view

        return interactive_view(info, title=title, subtitle=subtitle, theme=theme, precision=precision,
                                max_samples=max_samples, height=height)
    style = _resolve_style(style, info, None) if style != "auto" or info.X is None else (
        "detailed" if info.n_leaves <= 300 else "compact")
    P = Paint("light", use_vars=True)
    uid = new_uid()
    # big trees embed every card, so keep the per-node scatter lighter
    A = Artist(info, P, style=style, sig=precision, uid=uid, max_points=260 if len(info.nodes) <= 63 else 110)
    A.set_path([])

    if max_trees is not None:
        info.shown_roots = info.roots[:max(1, int(max_trees))]
    shown = set(info.shown_roots or info.roots)
    root_of = {}
    for r in info.roots:
        stack = [r]
        while stack:
            i = stack.pop()
            root_of[i] = r
            if not info.nodes[i].is_leaf:
                stack += [info.nodes[i].left, info.nodes[i].right]

    nodes = []
    for nd in info.nodes:
        lab = info.edge_label(nd.id, precision)
        if root_of.get(nd.id) not in shown:  # a tree that is not drawn: only what prediction needs
            nodes.append(dict(
                id=nd.id, p=nd.parent, l=nd.left, r=nd.right, d=nd.depth, hidden=True, leaf=nd.is_leaf,
                c=[[int(f), op, float(v)] for f, op, v in nd.split], fs=nd.features, f=nd.feature if nd.simple else -1,
                nl=nd.nan_left, n=nd.n, wt=nd.weight, lab=lab,
                v=getattr(nd, "value", None) if getattr(nd, "value", None) is not None
                else (float(nd.counts[0]) if not info.is_clf else None),
                pr=[round(float(q), 6) for q in nd.counts / (nd.counts.sum() or 1)] if info.is_clf else None,
                pc=info.prediction(nd) if info.is_clf else None))
            continue
        c = A.card(nd.id)
        sc = A.card(nd.id, simple=True)
        nodes.append(dict(nl=nd.nan_left,
            id=nd.id, p=nd.parent, l=nd.left, r=nd.right, d=nd.depth, w=round(c["w"], 1), h=round(c["h"], 1),
            svg=c["svg"], sw=round(sc["w"], 1), sh=round(sc["h"], 1), ssvg=sc["svg"],
            mix=[[col, round(f, 5)] for col, f in A.mix(nd)], n=nd.n, wt=nd.weight, col=A.node_color(nd), lab=lab,
            labw=round(text_width(lab, 10.5, True) + 14, 1), f=nd.feature if nd.simple else -1, t=nd.threshold,
            c=[[int(f), op, float(v)] for f, op, v in nd.split], fs=nd.features,
            st=info.split_text(nd.id, precision) if not nd.is_leaf else "",
            desc=info.n_descendants(nd.id), leaf=nd.is_leaf, chart=c["chart"],
            v=getattr(nd, "value", None) if getattr(nd, "value", None) is not None
            else (float(nd.counts[0]) if not info.is_clf else None),
            tip=_tooltip(info, A, nd.id, precision),
            pr=[round(float(q), 6) for q in nd.counts / (nd.counts.sum() or 1)] if info.is_clf else None,
            pc=info.prediction(nd) if info.is_clf else None,
        ))
    if info.layout == "cascade":  # name each rule's outcome after its rule
        for nd in info.nodes:
            if not nd.is_leaf and nd.label:
                side = nd.left if info.nodes[nd.left].is_leaf else nd.right
                nodes[side]["ruleLabel"] = f"{nd.label} outcome"

    imp, imp_kind = info.importances()
    if not isinstance(model, TreeInfo) and info.combine == "single" and info.layout == "tree":
        try:  # sklearn trees: use the estimator's own numbers
            imp, imp_kind = _unwrap(model).feature_importances_, "impurity"
        except (TypeError, AttributeError):
            pass
    used = sorted({f for nd in info.nodes for f in nd.features}, key=lambda f: -imp[f])
    feats = []
    for f in used:
        thr = [v for nd in info.nodes for g, _, v in nd.split if g == f]
        if info.X is not None:
            col = info.X[:, f]
            col = col[np.isfinite(col)]
            lo, hi, med = float(col.min()), float(col.max()), float(np.median(col))
        else:
            lo, hi = min(thr), max(thr)
            span = (hi - lo) or abs(hi) or 1.0
            lo, hi, med = lo - 0.25 * span, hi + 0.25 * span, float(np.median(thr))
        kind = info.feature_kind.get(f, "continuous")
        step = 1 if kind in ("binary", "integer") else float(f"{(hi - lo) / 200:.2g}") or 0.01
        feats.append(dict(i=f, name=info.feature_names[f], imp=float(imp[f]), lo=lo, hi=hi,
                          v=round(med) if kind != "continuous" else med, step=step, kind=kind,
                          nsplit=len(thr), thr=sorted(thr), **_marginal(info, A, f, thr)))

    samples = []
    if info.X is not None and max_samples:
        rng = np.random.default_rng(0)
        rows = rng.choice(len(info.X), min(max_samples, len(info.X)), replace=False)
        for r in rows:
            truth = None
            if info.y is not None:
                truth = info.class_names[int(info.y[r])] if info.is_clf else fmt(info.y[r], max(precision, 4))
            samples.append(dict(x={str(f): float(info.X[r, f]) for f in used}, y=truth, row=int(r)))

    sub = subtitle
    if sub is None:
        sub = info.summary()
    if info.is_clf:
        note = {"sum": "leaf values add up to the log-odds", "mean": "the trees' predictions are averaged"}.get(info.combine)
        legend = class_legend(info.class_names, P, note)
    else:
        label = "leaf value, added across trees" if info.combine == "sum" else f"predicted {info.target_name}"
        legend = [dict(k="grad", label=label, lo=info.value_range[0], hi=info.value_range[1],
                       stops=[f"var(--dti-s{j})" for j in range(9)])]
    crit = info.criterion.replace("_", " ") or "impurity"
    if imp_kind == "impurity":
        imp_note = (f"<b>%</b> is the feature's share of the model's total {esc(crit)} reduction, summed over its "
                    "splits and weighted by samples (sklearn's <code>feature_importances_</code> definition). ")
    else:
        imp_note = "<b>%</b> is the feature's share of all samples reaching a split on it (impurities were not available). "
    imp_note += "Bars are relative to the top feature; ticks under each distribution mark split thresholds."
    data = dict(
        nodes=nodes, feats=feats, samples=samples, task=info.task,
        classes=[dict(name=n, col=P.cls(k)) for k, n in enumerate(info.class_names or [])],
        legend=legend, impNote=imp_note, fileName=info.model_name,
        target=info.target_name, orientation=resolve_orientation(orientation, info),
        initialDepth=initial_depth if initial_depth is not None else (99 if len(nodes) <= 63 else 3),
        maxDepth=info.max_depth, simple=bool(info.n_leaves > 32 if simple == "auto" else simple),
        defs=defs(uid, P), sig=precision, bandMax=BAND,
        roots=info.roots, shown=list(info.shown_roots or info.roots), combine=info.combine, layout=info.layout, link=info.link, intercept=info.intercept,
    )
    page_title = title or f"{info.model_name} ({info.n_leaves} leaves)"
    out = build_page("tree", title=page_title, subtitle=sub, theme=theme, data=data)
    return InteractiveTree(out, height)
