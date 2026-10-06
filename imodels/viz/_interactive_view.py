"""Interactive page for additive views (rule sets, scorecards, GAMs)."""

from types import SimpleNamespace

import numpy as np

from ._interactive import InteractiveTree, _marginal
from ._page import build_page, class_legend
from ._render import Artist, defs
from ._render_views import ViewPainter, gam_body, ruleset_body, scorecard_body, signed, view_summary
from ._text import FONT_STACK, esc, fmt, new_uid
from ._theme import Paint


def _term_tip(v, t, sig):
    head = esc(v.term_text(t, sig))
    rows = []
    if t.kind == "rule":
        rows.append(("adds", f"{signed(t.weight, sig)} to the {esc(v.score_name)} when it holds"))
        if np.isfinite(t.support):
            rows.append(("coverage", f"{t.support:.1%} of samples"))
        if v.is_clf and v.y is not None and t.idx is not None and len(t.idx):
            cnt = np.bincount(v.y[t.idx], minlength=len(v.class_names))
            rows += [(esc(v.class_names[k]), f"{int(c):,} ({c / cnt.sum():.0%})") for k, c in enumerate(cnt) if c]
    elif t.kind == "linear":
        rows.append(("adds", f"{signed(t.weight, sig)} per unit of {esc(v.feature_names[t.feature])}"))
        if t.center:
            rows.append(("centered at", fmt(t.center, sig)))
    else:
        rows.append(("range", f"{signed(float(np.min(t.values)), sig)} to {signed(float(np.max(t.values)), sig)}"))
    body = "".join(f"<tr><td>{k}</td><td class='num' colspan=3>{val}</td></tr>" for k, val in rows)
    return f"<div class='tt-h'><span class='tt-k'>{t.kind}</span> <b>{head}</b></div><table>{body}</table>"


def interactive_view(v, *, title=None, subtitle=None, theme="auto", precision=3, max_samples=400, height=720):
    P = Paint("light", use_vars=True)
    uid = new_uid()
    vp = ViewPainter(v, P, precision, uid)
    chart, rows_top = None, 0
    if v.family == "scorecard":
        body, W, H, chart = scorecard_body(vp)
    elif v.family == "gam":
        body, W, H, chart = gam_body(vp)
    else:
        body, W, H, rows_top = ruleset_body(vp)
    body_w = W - 56  # bodies are laid out inside the 28px figure margins

    terms = []
    for t in v.terms:
        terms.append(dict(
            k=t.kind, w=float(t.weight), c=[[int(f), op, float(val)] for f, op, val in (t.conds or [])],
            f=int(t.feature), ctr=float(t.center),
            e=[float(e) for e in t.edges] if t.edges is not None else None,
            vals=[float(a) for a in t.values] if t.values is not None else None,
            g=[float(a) for a in t.grid] if t.grid is not None else None,
            clip=[float(a) for a in t.clip] if t.clip is not None else None,
            tied=[bool(b) for b in t.tied] if t.tied is not None else None,
            fs=t.features, txt=v.term_text(t, precision), tip=_term_tip(v, t, precision),
            sup=float(t.support) if np.isfinite(t.support) else None))

    # features: importance = share of the summed spread of the terms that use them
    imp = np.zeros(len(v.feature_names))
    for t in v.terms:
        fs = sorted(set(t.features))
        for f in fs:
            imp[f] += v.importance(t) / len(fs)
    imp = imp / (imp.sum() or 1)
    used = sorted(v.used_features(), key=lambda f: -imp[f])
    shim = SimpleNamespace(sig=precision, _range=lambda vals, t=None, robust=False: Artist._range(shim, vals, t, robust))
    feats = []
    for f in used:
        thr = [val for t in v.terms if t.kind == "rule" for g, _, val in t.conds if g == f]
        thr += [float(e) for t in v.terms if t.kind == "shape" and t.feature == f for e in t.edges[1:-1]
                if np.isfinite(e)]
        if v.X is not None:
            col = v.X[:, f][np.isfinite(v.X[:, f])]
            lo, hi, med = float(col.min()), float(col.max()), float(np.median(col))
        else:
            ref = thr or [0.0, 1.0]
            span = (max(ref) - min(ref)) or 1.0
            lo, hi, med = min(ref) - 0.25 * span, max(ref) + 0.25 * span, float(np.median(ref))
        kind = v.feature_kind.get(f, "continuous")
        step = 1 if kind in ("binary", "integer", "categorical") else float(f"{(hi - lo) / 200:.2g}") or 0.01
        levels = v.categories.get(f)
        if levels:  # categorical: codes 0..k-1, default to the most common level
            lo, hi, thr = 0.0, float(len(levels) - 1), []
            codes = v.X[:, f][np.isfinite(v.X[:, f])].astype(int) if v.X is not None else np.zeros(1, int)
            med = float(np.bincount(codes, minlength=len(levels)).argmax())
        missing = any(op in ("isnan", "notnan") for t in v.terms if t.kind == "rule" for g, op, _ in t.conds if g == f)
        mg = _marginal(v, shim, f, thr or [med]) if v.X is not None else dict(hlo=lo, hhi=hi, hist=None, ends=None)
        if levels:
            mg["ends"] = [levels[0], levels[-1]]
        feats.append(dict(i=int(f), name=v.feature_names[f], imp=float(imp[f]), lo=lo, hi=hi,
                          v=round(med) if kind != "continuous" else med, step=step, kind=kind,
                          levels=levels, missing=missing,
                          nsplit=sum(1 for t in v.terms if f in t.features), thr=sorted(thr), **mg))

    samples = []
    if v.X is not None and max_samples:
        rng = np.random.default_rng(0)
        for r in rng.choice(len(v.X), min(max_samples, len(v.X)), replace=False):
            truth = None
            if v.y is not None:
                truth = v.class_names[int(v.y[r])] if v.is_clf else fmt(v.y[r], max(precision, 4))
            samples.append(dict(x={str(f): float(v.X[r, f]) for f in used}, y=truth))

    up, down = vp.direction_words()
    if v.is_clf:
        legend = class_legend(v.class_names, P, "bars are colored by the class a term favors")
    else:
        legend = [dict(k="dot", col=vp.pos, label=up), dict(k="dot", col=vp.neg, label=down)]
    noun = {"gam": "shape function", "scorecard": "item"}.get(v.family, "term")
    imp_note = (f"<b>%</b> is the feature's share of how much the model's {noun}s move the {esc(v.score_name)} across "
                f"the training data (standard deviation of each {noun}'s contribution, split evenly among the features it "
                "uses). Bars are relative to the top feature; ticks mark thresholds the model uses.")
    data = dict(
        family=v.family, task=v.task, terms=terms, intercept=float(v.intercept), link=v.link,
        threshold=float(v.threshold), linkScale=float(v.link_scale), linkOffset=float(v.link_offset),
        feats=feats, samples=samples, scoreName=v.score_name, legend=legend, impNote=imp_note,
        classes=[dict(name=n, col=P.cls(k)) for k, n in enumerate(v.class_names or [])],
        pos=vp.pos, neg=vp.neg, target=v.target_name, chart=chart, sig=precision, rowsTop=rows_top,
        note=v.note, fileName=v.model_name,
    )
    sub = view_summary(v) if subtitle is None else subtitle
    svg = (f'<svg id="canvas" xmlns="http://www.w3.org/2000/svg" viewBox="-4 -4 {body_w + 8:.0f} {H + 8:.0f}" '
           f'width="{body_w + 8:.0f}" font-family="{esc(FONT_STACK)}">{defs(uid, P)}{body}</svg>')
    out = build_page("view", title=title or v.model_name, subtitle=sub, theme=theme, data=data, stage=svg)
    return InteractiveTree(out, height)
