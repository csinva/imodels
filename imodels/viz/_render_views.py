"""Static SVG for additive views: rule sets, scorecards and shape-function (GAM) grids."""

import numpy as np

from ._render import _n, _st, _text, _top_round_bar, defs
from ._text import FONT_STACK, esc, fmt, fmt_count, text_width, truncate

M = 28  # outer margin


def signed(v, sig=3):
    s = fmt(abs(v), sig)
    return ("+" if v > 0 else "−" if v < 0 else "") + s


class ViewPainter:
    """Shared pieces: polarity colors, header, chips, wrapping, small bars."""

    def __init__(self, view, P, sig=3, uid="dti", x=None):
        self.v, self.P, self.sig, self.uid = view, P, sig, uid
        self.x = None if x is None else np.asarray(x, dtype=float).ravel()
        if view.is_clf:
            self.pos, self.neg = P.cls(1 if len(view.class_names) > 1 else 0), P.cls(0)
        else:
            self.pos, self.neg = P.cls(0), P.cls(7)  # blue raises, red lowers

    def pol(self, w):
        return self.pos if w >= 0 else self.neg

    def direction_words(self):
        v = self.v
        if v.is_clf:
            return f"toward {v.class_names[-1]}", f"toward {v.class_names[0]}"
        return f"raises {v.target_name}", f"lowers {v.target_name}"

    def header(self, width, title, subtitle):
        P, parts, y = self.P, [], 0
        if title:
            parts.append(_text(0, 18, title, 18, P("ink"), 650))
            y = 28
        if subtitle:
            parts.append(_text(0, y + 13, subtitle, 12, P("ink2")))
            y += 22
        y += 6
        x = 0.0
        if self.v.is_clf:
            items = [(self.P.cls(k), n) for k, n in enumerate(self.v.class_names)]
        else:
            up, down = self.direction_words()
            items = [(self.pos, up), (self.neg, down)]
        for col, lab in items:
            parts.append(f'<circle cx="{_n(x + 5)}" cy="{_n(y + 6)}" r="5" style="{_st(col)}"/>')
            parts.append(_text(x + 15, y + 10, lab, 11.5, P("ink2")))
            x += text_width(lab, 11.5) + 26
        if self.v.is_clf:
            parts.append(_text(x + 6, y + 10, "bars are colored by the class a term favors", 11.5, P("muted")))
        y += 18
        if self.x is not None:
            y += 6
            parts.append(_text(0, y + 11, self.prediction_text(), 12, P("ink"), 600))
            y += 18
        return "".join(parts), y + 18

    def prediction_text(self):
        v = self.v
        s = float(v.score(self.x[None])[0])
        out = float(v.output(s))
        if v.is_clf:
            k = 1 if out >= 0.5 else 0
            if v.link == "threshold":
                return f"Prediction for this sample: {v.class_names[k]} ({v.score_name} {fmt(s, self.sig)})"
            return f"Prediction for this sample: {v.class_names[k]} (P({v.class_names[-1]}) = {out:.0%}, {v.score_name} {fmt(s, self.sig)})"
        return f"Prediction for this sample: {fmt(out, max(self.sig, 4))}"

    def chip(self, x, y, text, on=False):
        P = self.P
        w = text_width(text, 11.5) + 18
        fill = _st(P("hl") if on else P("grid"), fo=1 if on else 0.7)
        col = P("card") if on else P("ink")
        return (f'<rect x="{_n(x)}" y="{_n(y)}" width="{_n(w)}" height="22" rx="11" style="{fill}"/>'
                + _text(x + w / 2, y + 15, text, 11.5, col, 500, "middle", "font-variant-numeric:tabular-nums")), w

    def chips(self, x, y, texts, max_w, joiner="and", on=None):
        """Wrap condition chips joined by a small word; returns (svg, height)."""
        out, cx, cy = [], x, y
        jw = text_width(joiner, 10.5) + 12
        for i, t in enumerate(texts):
            t = truncate(t, 11.5, max_w - 20)
            w = text_width(t, 11.5) + 18
            if i:
                if cx + jw + w > x + max_w:
                    cx, cy = x, cy + 28
                else:
                    out.append(_text(cx + jw / 2, cy + 15, joiner, 10.5, self.P("muted"), anchor="middle"))
                    cx += jw
            svg, w = self.chip(cx, cy, t, on=bool(on and on[i]))
            out.append(f'<g class="chip" data-c="{i}">{svg}</g>')
            cx += w
        return "".join(out), (cy - y) + 22

    def class_bar(self, x, y, w, h, counts, cid):
        P = self.P
        tot = counts.sum() or 1.0
        out = [f'<clipPath id="{cid}"><rect x="{_n(x)}" y="{_n(y)}" width="{_n(w)}" height="{_n(h)}" rx="{h / 2:g}"/></clipPath>',
               f'<g clip-path="url(#{cid})"><rect x="{_n(x)}" y="{_n(y)}" width="{_n(w)}" height="{_n(h)}" style="{_st(P("grid"))}"/>']
        nz = [k for k in range(len(counts)) if counts[k] > 0]
        gap = 2.0 if len(nz) > 1 else 0.0
        avail, cx = w - gap * (len(nz) - 1), x
        for k in nz:
            seg = avail * counts[k] / tot
            out.append(f'<rect x="{_n(cx)}" y="{_n(y)}" width="{_n(max(seg, 0.5))}" height="{_n(h)}" style="{_st(P.cls(k))}"/>')
            cx += seg + gap
        return "".join(out) + "</g>"

    def frame(self, body, W, H, title, subtitle, aria):
        P = self.P
        head, hh = self.header(W - 2 * M, title, subtitle)
        Ht = H + hh + 2 * M
        return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{_n(W)}" height="{_n(Ht)}" viewBox="0 0 {_n(W)} {_n(Ht)}" '
                f'role="img" aria-label="{esc(aria)}" font-family="{esc(FONT_STACK)}">'
                + defs(self.uid, P) + f'<rect width="100%" height="100%" style="{_st(P("surface"))}"/>'
                + f'<g transform="translate({M},{M})">{head}</g>'
                + f'<g transform="translate({M},{_n(M + hh)})">{body}</g></svg>')


# ---------------------------------------------------------------- rule sets / linear terms
def _card(w, h, P, uid, x=0, y=0):
    return (f'<rect x="{_n(x)}" y="{_n(y)}" width="{_n(w)}" height="{_n(h)}" rx="12" filter="url(#{uid}-sh)" '
            f'style="{_st(P("card"), P("ring"), 1)}"/>')


def _label(x, y, text, P, anchor="start"):
    return _text(x, y, text.upper(), 10, P("muted"), 600, anchor, "letter-spacing:.08em")


def ruleset_body(vp, max_rows=30):
    """Ranked table of rules (and linear terms) with effect, coverage and class mix, inside one card.

    Rows are groups positioned with a transform so the interactive page can re-sort them.
    """
    v, P, sig = vp.v, vp.P, vp.sig
    terms = [t for t in v.terms if abs(t.weight) > 1e-12]
    order = sorted(range(len(terms)), key=lambda i: -v.importance(terms[i]))
    shown, hidden = order[:max_rows], len(order) - max_rows
    has_mix = v.is_clf and v.y is not None and any(t.kind == "rule" for t in terms)
    has_cov = any(np.isfinite(t.support) for t in terms if t.kind == "rule")
    c_tag, c_rank, c_rule, rule_w = 16, 38, 48, 410
    c_eff, eff_w = c_rule + rule_w + 26, 200
    c_cov = c_eff + eff_w + 30
    c_mix = c_cov + 110
    BW = (c_mix + 120 if has_mix else c_cov + 100 if has_cov else c_eff + eff_w) + 20
    W = BW + 2 * M
    eff_val = {i: (terms[i].weight if terms[i].kind == "rule" else
                   np.sign(terms[i].weight) * v.importance(terms[i])) for i in shown}
    maxe = max([abs(e) for e in eff_val.values()] + [1e-12])
    mid = c_eff + eff_w / 2
    out = []
    y = 18
    linear_only = all(t.kind == "linear" for t in terms)
    out.append(_label(c_rule, y + 10, "Feature" if linear_only else "Rule", P))
    out.append(_label(mid, y + 10, f"Effect on {v.score_name}", P, "middle"))
    if has_cov:
        out.append(_label(c_cov, y + 10, "Coverage", P))
    if has_mix:
        out.append(_label(c_mix, y + 10, "Classes covered", P))
    y += 24
    out.append(f'<line x1="0" x2="{BW}" y1="{y}" y2="{y}" style="{_st(stroke=P("grid"), sw=1)}"/>')
    y += 10
    svg, h = vp.chips(c_rule, y, ["baseline (intercept)"], rule_w)
    out.append(svg)
    out.append(_text(mid, y + 15, signed(v.intercept, sig) if v.intercept else "0", 11.5, P("ink2"), 600, "middle",
                     "font-variant-numeric:tabular-nums"))
    y += h + 10
    rows_top = y
    for rank, i in enumerate(shown, 1):
        t = terms[i]
        # row content is drawn at local coordinates; the group's transform places it
        _, h = vp.chips(c_rule, 10, [v.cond_text(f, op, val, sig) for f, op, val in t.conds] if t.kind == "rule"
                        else ["x"], rule_w)
        rh = max(h, 22) + 20
        kind = "term rule" if t.kind == "rule" else "term"
        row = [f'<g class="{kind}" data-t="{v.terms.index(t)}" data-rank="{rank}" data-y="{_n(y)}" data-h="{_n(rh)}" '
               f'transform="translate(0,{_n(y)})" style="transform:translate(0px,{_n(y)}px)">',
               f'<rect class="rowbg" x="0" y="0" width="{BW}" height="{_n(rh)}" style="{_st(P("page"), fo=0)}"/>',
               f'<line x1="0" x2="{BW}" y1="0" y2="0" style="{_st(stroke=P("grid"), sw=1)}"/>',
               f'<circle class="tag on" cx="{c_tag}" cy="21" r="4" style="{_st(P("hl"), extra="display:none")}"/>',
               f'<circle class="tag near" cx="{c_tag}" cy="21" r="3.5" style="{_st("none", P("hl"), 1.5, extra="display:none")}"/>',
               _text(c_rank, 25, str(rank), 11, P("muted"), 500, "end", "font-variant-numeric:tabular-nums")]
        if t.kind == "rule":
            texts = [v.cond_text(f, op, val, sig) for f, op, val in t.conds] or ["always"]
            on = [_holds(vp.x, f, op, val) for f, op, val in t.conds] if vp.x is not None else None
            svg, h = vp.chips(c_rule, 10, texts, rule_w, on=on)
        else:
            label = v.feature_names[t.feature] if linear_only else f"{v.feature_names[t.feature]} (linear)"
            svg, h = vp.chips(c_rule, 10, [label], rule_w)
        row.append(svg)
        e = eff_val[i]  # effect bar, centered at zero
        bw = abs(e) / maxe * (eff_w / 2 - 34)
        bx = mid if e >= 0 else mid - bw
        cy = 21
        row.append(f'<line x1="{_n(mid)}" x2="{_n(mid)}" y1="4" y2="{_n(rh - 4)}" style="{_st(stroke=P("axis"), sw=1)}"/>')
        row.append(f'<rect x="{_n(bx)}" y="{_n(cy - 6)}" width="{_n(max(bw, 1))}" height="12" rx="3" style="{_st(vp.pol(e))}"/>')
        lab = signed(t.weight, sig) if t.kind == "rule" else f"{signed(t.weight, sig)} / unit"
        lx = mid + bw + 6 if e >= 0 else mid - bw - 6
        row.append(_text(lx, cy + 4, lab, 11, P("ink"), 600, "start" if e >= 0 else "end", "font-variant-numeric:tabular-nums"))
        if has_cov and t.kind == "rule" and np.isfinite(t.support):
            row.append(f'<rect x="{c_cov}" y="{_n(cy - 3)}" width="60" height="6" rx="3" style="{_st(P("grid"))}"/>')
            row.append(f'<rect x="{c_cov}" y="{_n(cy - 3)}" width="{_n(max(60 * t.support, 1))}" height="6" rx="3" style="{_st(P("ink2"))}"/>')
            row.append(_text(c_cov + 68, cy + 4, f"{t.support:.0%}", 11, P("ink2"), extra="font-variant-numeric:tabular-nums"))
        if has_mix and t.kind == "rule" and t.idx is not None and len(t.idx):
            counts = np.bincount(v.y[t.idx], minlength=len(v.class_names)).astype(float)
            row.append(vp.class_bar(c_mix, cy - 3, 90, 6, counts, f"{vp.uid}-m{i}"))
        row.append("</g>")
        out += row
        y += rh
    out.append(f'<line x1="0" x2="{BW}" y1="{_n(y)}" y2="{_n(y)}" style="{_st(stroke=P("grid"), sw=1)}"/>')
    y += 10
    if hidden > 0:
        out.append(_text(c_rule, y + 12, f"+{hidden} smaller terms not shown", 11, P("muted")))
        y += 20
    if v.note:
        out.append(_text(c_rule, y + 12, v.note, 11.5, P("ink2")))
        y += 22
    y += 8
    return _card(BW, y, P, vp.uid) + "".join(out), W, y, rows_top


def _holds(x, f, op, val):
    from ._extract import OPS

    return bool(OPS[op](x[f], val))


# ---------------------------------------------------------------- scorecards
def scorecard_body(vp):
    v, P, sig = vp.v, vp.P, vp.sig
    terms = [t for t in v.terms if abs(t.weight) > 1e-12]
    tbl_w, chart_x, chart_w, chart_h = 380, 470, 400, 210
    W = chart_x + chart_w + 2 * M + 10
    out, y = [], 0
    out.append(_text(0, 10, "IF", 10, P("muted"), 600, extra="letter-spacing:.08em"))
    out.append(_text(tbl_w, 10, "POINTS", 10, P("muted"), 600, "end", "letter-spacing:.08em"))
    y = 22
    for t in terms:
        out.append(f'<line x1="0" x2="{tbl_w}" y1="{_n(y)}" y2="{_n(y)}" style="{_st(stroke=P("grid"), sw=1)}"/>')
        y += 9
        if t.kind == "rule":
            texts = [v.cond_text(f, op, val, sig) for f, op, val in t.conds]
        else:
            texts = [f"{v.feature_names[t.feature]}, per unit"]
        _, h = vp.chips(0, y, texts, tbl_w - 70)
        g = [f'<g class="term{" rule" if t.kind == "rule" else ""}" data-t="{v.terms.index(t)}">',
             f'<rect class="rowbg" x="-8" y="{_n(y - 5)}" width="{tbl_w + 16}" height="{_n(h + 10)}" rx="6" style="{_st(P("page"), fo=0)}"/>']
        svg, h = vp.chips(0, y, texts, tbl_w - 70)
        g.append(svg)
        pts = signed(t.weight, sig)
        pw = text_width(pts, 12, True) + 20
        g.append(f'<rect class="pts" x="{_n(tbl_w - pw)}" y="{_n(y)}" width="{_n(pw)}" height="22" rx="11" style="{_st(vp.pol(t.weight), fo=0.16)}"/>')
        g.append(_text(tbl_w - pw / 2, y + 15.5, pts, 12, P("ink"), 700, "middle", "font-variant-numeric:tabular-nums"))
        g.append("</g>")
        out += g
        y += h + 9
    out.append(f'<line x1="0" x2="{tbl_w}" y1="{_n(y)}" y2="{_n(y)}" style="{_st(stroke=P("ink"), sw=1.5)}"/>')
    y += 8
    if v.intercept:
        out.append(_text(0, y + 14, "start from", 12, P("ink2")))
        out.append(_text(tbl_w - 10, y + 14, signed(v.intercept, sig), 12, P("ink2"), 600, "end"))
        y += 22
    out.append(_text(0, y + 15, "Total score", 13, P("ink"), 650))
    out.append(_text(tbl_w - 10, y + 15, "= sum of points", 12, P("muted"), anchor="end"))
    y += 30
    if v.note:
        out.append(_text(0, y + 10, v.note, 11, P("muted")))
        y += 18
    table_h = y
    chart_start = len(out)

    # risk curve over every reachable score
    risk = v.risk
    if risk is None:
        lo = v.intercept + sum(min(0, t.weight) for t in terms if t.kind == "rule")
        hi = v.intercept + sum(max(0, t.weight) for t in terms if t.kind == "rule")
        grid = np.arange(np.floor(lo), np.ceil(hi) + 1)
        risk = [(float(s), float(v.output(s))) for s in grid]
    ss = np.array([s for s, _ in risk])
    ps = np.array([p for _, p in risk])
    cy0, cx0 = 22, chart_x
    smin, smax = ss.min(), ss.max() if ss.max() > ss.min() else ss.min() + 1
    sx = lambda s: cx0 + 14 + (s - smin) / (smax - smin) * (chart_w - 28)
    sy = lambda p: cy0 + chart_h - p * chart_h
    out.append(_text(cx0, 6, f"RISK BY SCORE: P({v.class_names[-1].upper()})" if v.is_clf else "OUTPUT BY SCORE", 10, P("muted"), 600, extra="letter-spacing:.08em"))
    for frac in (0, 0.25, 0.5, 0.75, 1):
        yy = sy(frac)
        out.append(f'<line x1="{cx0}" x2="{cx0 + chart_w}" y1="{_n(yy)}" y2="{_n(yy)}" style="{_st(stroke=P("grid"), sw=1)}"/>')
        out.append(_text(cx0 - 6, yy + 3.5, f"{frac:.0%}", 10, P("muted"), anchor="end"))
    # training-score distribution, stacked by class, under the curve
    if v.X is not None:
        sc = v.score(v.X)
        hist_h = 46
        base = cy0 + chart_h + 34 + hist_h
        k = len(v.class_names) if v.is_clf and v.y is not None else 1
        H = np.zeros((k, len(ss)))
        idx = np.clip(np.round((sc - smin)).astype(int), 0, len(ss) - 1) if np.allclose(ss, np.round(ss)) else \
            np.clip(np.searchsorted(ss, sc), 0, len(ss) - 1)
        labs = v.y if k > 1 else np.zeros(len(sc), dtype=int)
        np.add.at(H, (labs, idx), 1)
        top = H.sum(0).max() or 1
        bw = min(22, (chart_w - 28) / len(ss) - 3)
        for j in range(len(ss)):
            yb = base
            nz = [c for c in range(k) if H[c, j] > 0]
            for q, c in enumerate(nz):
                hh = H[c, j] / top * hist_h
                yb -= hh
                out.append(f'<path d="{_top_round_bar(sx(ss[j]) - bw / 2, yb, bw, hh, 3 if q == len(nz) - 1 else 0)}" style="{_st(P.cls(c) if k > 1 else P("ink2"))}"/>')
        out.append(f'<line x1="{cx0}" x2="{cx0 + chart_w}" y1="{_n(base + 0.5)}" y2="{_n(base + 0.5)}" style="{_st(stroke=P("axis"), sw=1)}"/>')
        out.append(_text(cx0, base + 16, "training samples by score", 10, P("muted")))
    pts = " ".join(f"{_n(sx(s))},{_n(sy(p))}" for s, p in risk)
    out.append(f'<polyline points="{pts}" style="{_st("none", P("ink"), 2, extra="stroke-linejoin:round")}"/>')
    label_every = max(1, int(np.ceil(len(risk) / 12)))
    for j, (s, p) in enumerate(risk):
        out.append(f'<circle class="rk" data-s="{s:g}" cx="{_n(sx(s))}" cy="{_n(sy(p))}" r="4.5" style="{_st(vp.pol(p - 0.5) if v.is_clf else P("ink"), P("surface"), 2)}"/>')
        out.append(_text(sx(s), cy0 + chart_h + 16, fmt(s, 4), 10.5, P("ink2"), anchor="middle", extra="font-variant-numeric:tabular-nums"))
        if j % label_every == 0:  # label above the dot, or below it near the top of the chart
            ly = sy(p) - 9 if sy(p) - 9 > cy0 + 4 else sy(p) + 17
            out.append(_text(sx(s), ly, f"{p:.0%}", 10, P("ink"), 600, "middle"))
    if vp.x is not None:
        s = float(v.score(vp.x[None])[0])
        out.append(f'<line x1="{_n(sx(s))}" x2="{_n(sx(s))}" y1="{cy0}" y2="{cy0 + chart_h}" style="{_st(stroke=P("hl"), sw=2)}"/>')
    chart_h_all = cy0 + chart_h + 34 + (64 if v.X is not None else 0)
    # two cards: the points table and the risk curve; content sits 16px inside each
    ox_t, ox_c, oy = 16, 20, 16
    tbl_card = _card(tbl_w + 2 * ox_t, table_h + 2 * oy, P, vp.uid)
    chart_card = _card(chart_w + 2 * ox_c + 20, chart_h_all + 2 * oy, P, vp.uid, x=chart_x - ox_c - 20)
    body = (tbl_card + chart_card
            + f'<g transform="translate({ox_t},{oy})">' + "".join(out[:chart_start]) + "</g>"
            + f'<g transform="translate({ox_c - 20 + 20},{oy})">' + "".join(out[chart_start:]) + "</g>")
    total_h = max(table_h, chart_h_all) + 2 * oy
    chart = dict(x0=cx0 + 14 + ox_c, x1=cx0 + chart_w - 14 + ox_c, smin=float(smin), smax=float(smax),
                 y0=cy0 + oy, y1=cy0 + chart_h + oy)
    return body, chart_x + chart_w + 20 + 2 * M + 4, total_h, chart


# ---------------------------------------------------------------- shape functions (GAM)
def gam_body(vp, cols=3, max_panels=24):
    v, P, sig = vp.v, vp.P, vp.sig
    by_f = {}
    for t in v.terms:  # several terms on one feature add up into one shape
        by_f.setdefault(t.feature, []).append(t)
    feats = sorted(by_f, key=lambda f: -float(np.std(sum(t.contribution(v.X) for t in by_f[f])))
                   if v.X is not None else -sum(v.importance(t) for t in by_f[f]))[:max_panels]
    pw, ph, gx, gy = 280, 150, 26, 34
    cols = min(cols, max(1, len(feats)))
    W = cols * pw + (cols - 1) * gx + 2 * M
    grids, curves = {}, {}
    for f in feats:
        if v.X is not None:
            col = v.X[:, f]
            lo, hi = np.quantile(col, [0.005, 0.995]) if len(np.unique(col)) > 20 else (col.min(), col.max())
        else:
            edges = np.concatenate([t.edges[np.isfinite(t.edges)] for t in by_f[f] if t.kind == "shape"] or [np.array([0, 1])])
            lo, hi = edges.min(), edges.max()
        if hi <= lo:
            lo, hi = lo - 0.5, hi + 0.5
        xs = np.linspace(lo, hi, 241)
        Xg = np.zeros((len(xs), len(v.feature_names)))
        Xg[:, f] = xs
        curves[f] = sum(t.contribution(Xg) for t in by_f[f])
        grids[f] = (lo, hi, xs)
    ymax = max([np.max(np.abs(c)) for c in curves.values()] + [1e-12])
    out, meta = [], []
    for k, f in enumerate(feats):
        r, c = divmod(k, cols)
        ox, oy = c * (pw + gx), r * (ph + gy + 40)
        lo, hi, xs = grids[f]
        ys = curves[f]
        imp = float(np.std(sum(t.contribution(v.X) for t in by_f[f]))) if v.X is not None else float(np.ptp(ys))
        g = [f'<g class="panel" data-f="{f}" transform="translate({ox},{oy})">']
        g.append(f'<rect width="{pw}" height="{ph + 40}" rx="10" filter="url(#{vp.uid}-sh)" style="{_st(P("card"), P("ring"), 1)}"/>')
        g.append(_text(12, 20, truncate(v.feature_names[f], 12.5, pw - 110, True), 12.5, P("ink"), 600))
        g.append(_text(pw - 12, 20, f"±{fmt(imp, 2)} typical", 10.5, P("muted"), anchor="end", extra="font-variant-numeric:tabular-nums"))
        x0, x1, y0, y1 = 12, pw - 12, 32, 32 + ph - 40
        sx = lambda q: x0 + (q - lo) / (hi - lo) * (x1 - x0)
        sy = lambda q: y0 + (y1 - y0) / 2 - q / ymax * (y1 - y0) / 2
        zero = sy(0)
        # tinted areas above / below zero
        for sign, col in ((1, vp.pos), (-1, vp.neg)):
            clipped = np.where(sign * ys > 0, ys, 0)
            d = f"M{_n(sx(xs[0]))},{_n(zero)}" + "".join(f"L{_n(sx(a))},{_n(sy(b))}" for a, b in zip(xs, clipped)) + f"L{_n(sx(xs[-1]))},{_n(zero)}Z"
            g.append(f'<path d="{d}" style="{_st(col, fo=0.16)}"/>')
        g.append(f'<line x1="{x0}" x2="{x1}" y1="{_n(zero)}" y2="{_n(zero)}" style="{_st(stroke=P("axis"), sw=1)}"/>')
        line = "M" + " L".join(f"{_n(sx(a))},{_n(sy(b))}" for a, b in zip(xs, ys))
        g.append(f'<path d="{line}" style="{_st("none", P("ink"), 2, extra="stroke-linejoin:round")}"/>')
        g.append(_text(x0, y0 + 2, signed(ymax, 2), 9.5, P("muted")))
        g.append(_text(x0, y1 + 1, signed(-ymax, 2), 9.5, P("muted")))
        # data distribution strip
        if v.X is not None:
            col = np.clip(v.X[:, f], lo, hi)
            H = np.histogram(col, bins=40, range=(lo, hi))[0]
            top = H.max() or 1
            bw = (x1 - x0) / 40
            for j, cnt in enumerate(H):
                if cnt:
                    hh = cnt / top * 14
                    g.append(f'<rect x="{_n(x0 + j * bw + 0.5)}" y="{_n(y1 + 22 - hh)}" width="{_n(bw - 1)}" height="{_n(hh)}" style="{_st(P("axis"))}"/>')
        g.append(_text(x0, ph + 30, fmt(lo, sig), 10, P("muted")))
        g.append(_text(x1, ph + 30, fmt(hi, sig), 10, P("muted"), anchor="end"))
        if vp.x is not None:
            xv = float(np.clip(vp.x[f], lo, hi))
            yv = float(np.interp(xv, xs, ys))
            g.append(f'<line x1="{_n(sx(xv))}" x2="{_n(sx(xv))}" y1="{y0}" y2="{y1}" style="{_st(stroke=P("hl"), sw=1.5)}"/>')
            g.append(f'<circle cx="{_n(sx(xv))}" cy="{_n(sy(yv))}" r="4" style="{_st(P("hl"), P("card"), 1.5)}"/>')
        g.append("</g>")
        out += g
        meta.append(dict(f=int(f), ox=ox, oy=oy, x0=x0, x1=x1, y0=y0, y1=y1, lo=float(lo), hi=float(hi), ymax=float(ymax)))
    rows = (len(feats) + cols - 1) // cols
    H = rows * (ph + gy + 40) - gy
    y = H + 12
    if v.note:
        out.append(_text(0, y + 12, v.note, 11.5, P("ink2")))
        y += 20
    return "".join(out), W, y, meta


def render_view(view, P, sig=3, uid="dti", x=None, title=None, subtitle=None):
    """Static SVG for an AdditiveView."""
    vp = ViewPainter(view, P, sig, uid, x)
    if view.family == "scorecard":
        body, W, H, _ = scorecard_body(vp)
    elif view.family == "gam":
        body, W, H, _ = gam_body(vp)
    else:
        body, W, H, _ = ruleset_body(vp)
    if subtitle is None:
        subtitle = view_summary(view)
    return vp.frame(body, W, H, title, subtitle, title or view.model_name)


def view_summary(v):
    n_rules = sum(1 for t in v.terms if t.kind == "rule" and abs(t.weight) > 1e-12)
    n_lin = sum(1 for t in v.terms if t.kind == "linear" and abs(t.weight) > 1e-12)
    parts = [v.model_name]
    if v.family == "gam":
        parts.append(f"{len({t.feature for t in v.terms})} shape functions added together")
    elif v.family == "scorecard":
        parts.append(f"{n_rules + n_lin} scoring items")
    else:
        if n_rules:
            parts.append(f"{n_rules} rule{'s' if n_rules != 1 else ''}")
        if n_lin:
            parts.append(f"{n_lin} linear term{'s' if n_lin != 1 else ''}")
    if v.X is not None:
        parts.append(f"{fmt_count(len(v.X))} training samples")
    return "  ·  ".join(parts)
