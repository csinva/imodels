"""SVG drawing of node cards, ribbons and the figure header."""

import numpy as np

from ._layout import tidy
from ._text import FONT_STACK, esc, fmt, fmt_count, text_width, truncate

PAD = 10
RADIUS = 10
CHART_W = 160
CHART_H = 38
BAND = 120.0  # ribbon width (px) carrying all root samples
MAX_CARD_W = 240


def _st(fill=None, stroke=None, sw=None, fo=None, so=None, op=None, extra=""):
    parts = []
    if fill is not None:
        parts.append(f"fill:{fill}")
    if fo is not None:
        parts.append(f"fill-opacity:{fo:g}")
    if stroke is not None:
        parts.append(f"stroke:{stroke}")
    if sw is not None:
        parts.append(f"stroke-width:{sw:g}")
    if so is not None:
        parts.append(f"stroke-opacity:{so:g}")
    if op is not None:
        parts.append(f"opacity:{op:g}")
    if extra:
        parts.append(extra)
    return ";".join(parts)


def _n(v):
    return f"{v:.1f}".rstrip("0").rstrip(".")


def _text(x, y, s, size, color, weight=400, anchor="start", extra=""):
    a = f' text-anchor="{anchor}"' if anchor != "start" else ""
    return (
        f'<text x="{_n(x)}" y="{_n(y)}"{a} style="font-size:{size:g}px;font-weight:{weight};'
        f'fill:{color}{";" + extra if extra else ""}">{esc(s)}</text>'
    )


def _top_round_bar(x, y, w, h, r):
    """Bar path with rounded top corners and a square base."""
    r = max(0.0, min(r, w / 2, h))
    if r <= 0.01:
        return f"M{_n(x)},{_n(y + h)}V{_n(y)}H{_n(x + w)}V{_n(y + h)}Z"
    return (
        f"M{_n(x)},{_n(y + h)}V{_n(y + r)}Q{_n(x)},{_n(y)} {_n(x + r)},{_n(y)}"
        f"H{_n(x + w - r)}Q{_n(x + w)},{_n(y)} {_n(x + w)},{_n(y + r)}V{_n(y + h)}Z"
    )


class Artist:
    """Draws one node card at the origin. Shared by the static and interactive modes."""

    def __init__(self, info, paint, style="detailed", sig=3, uid="dti", x=None, max_points=260, simple=False):
        self.simple = simple
        self.info, self.P, self.style, self.sig, self.uid = info, paint, style, sig, uid
        self.x = None if x is None else np.asarray(x, dtype=float).ravel()
        self.max_points = max_points
        self.has_data = info.X is not None and info.y is not None and style == "detailed"
        self.N = info.root.weight
        lo, hi = info.value_range
        self._vlo, self._vspan = lo, (hi - lo) or 1.0

    # ---- color helpers
    def vfrac(self, v):
        return (v - self._vlo) / self._vspan

    def node_color(self, node):
        contrib = getattr(node, "value", None) if self.info.combine == "sum" else None
        if self.info.is_clf and contrib is not None:
            return self.P.cls(1 if contrib >= 0 else 0)
        if self.info.is_clf:
            return self.P.cls(self.info.prediction(node))
        return self.P.seq(self.vfrac(node.counts[0]))

    # ---- pieces
    def _class_bar(self, x, y, w, h, counts):
        P = self.P
        tot = counts.sum() or 1.0
        cid = f"{self.uid}-cb{self._cur}"
        out = [f'<clipPath id="{cid}"><rect x="{_n(x)}" y="{_n(y)}" width="{_n(w)}" height="{_n(h)}" rx="{h / 2:g}"/></clipPath>',
               f'<g clip-path="url(#{cid})">',
               f'<rect x="{_n(x)}" y="{_n(y)}" width="{_n(w)}" height="{_n(h)}" style="{_st(P("grid"))}"/>']
        cx, nz = x, [k for k in range(len(counts)) if counts[k] > 0]
        gap = 2.0 if len(nz) > 1 else 0.0
        avail = w - gap * (len(nz) - 1)
        for i, k in enumerate(nz):
            seg = avail * counts[k] / tot
            out.append(f'<rect x="{_n(cx)}" y="{_n(y)}" width="{_n(max(seg, 0.5))}" height="{_n(h)}" style="{_st(P.cls(k))}"/>')
            cx += seg + gap
        out.append("</g>")
        return "".join(out)

    def _value_track(self, x, y, w, v):
        P = self.P
        f = float(np.clip(self.vfrac(v), 0, 1))
        cx = x + f * w
        return (
            f'<line x1="{_n(x)}" y1="{_n(y)}" x2="{_n(x + w)}" y2="{_n(y)}" style="{_st(stroke=P("grid"), sw=4, extra="stroke-linecap:round")}"/>'
            f'<line x1="{_n(x)}" y1="{_n(y)}" x2="{_n(cx)}" y2="{_n(y)}" style="{_st(stroke=P.seq(f), sw=4, extra="stroke-linecap:round")}"/>'
            f'<circle cx="{_n(cx)}" cy="{_n(y)}" r="4.5" style="{_st(P.seq(f), P("card"), 2)}"/>'
        )

    def _axis(self, x0, w, ybase, lo, hi, t, tlabel, ends=None):
        """Baseline, threshold caret and tick labels under a node chart."""
        P = self.P
        sx = lambda v: x0 + (v - lo) / (hi - lo) * w
        xt = float(np.clip(sx(t), x0, x0 + w))
        out = [f'<line x1="{_n(x0)}" y1="{_n(ybase + 0.5)}" x2="{_n(x0 + w)}" y2="{_n(ybase + 0.5)}" style="{_st(stroke=P("axis"), sw=1)}"/>',
               f'<path d="M{_n(xt)},{_n(ybase + 2)}l4.5,6h-9z" style="{_st(P("ink"))}"/>']
        tw = text_width(tlabel, 10.5, True)
        tx = float(np.clip(xt, x0 + tw / 2, x0 + w - tw / 2))
        out.append(_text(tx, ybase + 19, tlabel, 10.5, P("ink"), 600, "middle"))
        lo_s, hi_s = ends or (fmt(lo, self.sig), fmt(hi, self.sig))
        if lo_s and x0 + text_width(lo_s, 10) + 8 < tx - tw / 2:
            out.append(_text(x0, ybase + 19, lo_s, 10, P("muted")))
        if hi_s and x0 + w - text_width(hi_s, 10) - 8 > tx + tw / 2:
            out.append(_text(x0 + w, ybase + 19, hi_s, 10, P("muted"), anchor="end"))
        return "".join(out)

    def _range(self, vals, t=None, robust=False):
        """Axis range covering ``vals`` and threshold ``t``.

        With ``robust``, a long tail beyond the 0.5/99.5% quantiles is clipped so the bulk of
        the data stays readable; the end label then gets a '+' (or a leading '<').
        Returns (lo, hi, (lo_label, hi_label)).
        """
        lo, hi = float(np.min(vals)), float(np.max(vals))
        cut_lo = cut_hi = False
        if robust and len(vals) >= 20:
            qlo, qhi = np.quantile(vals, [0.005, 0.995])
            spread = qhi - qlo
            if spread > 0 and qlo - lo > 0.5 * spread:
                lo, cut_lo = float(qlo), True
            if spread > 0 and hi - qhi > 0.5 * spread:
                hi, cut_hi = float(qhi), True
        if t is not None:
            if t < lo:
                lo, cut_lo = t, False
            if t > hi:
                hi, cut_hi = t, False
        if hi - lo < 1e-12:
            lo, hi = lo - 0.5, hi + 0.5
        labels = (("<" if cut_lo else "") + fmt(lo, self.sig), fmt(hi, self.sig) + ("+" if cut_hi else ""))
        return lo, hi, labels

    def _hist_clf(self, node, x0, y0, w, h):
        P, info = self.P, self.info
        idx = node.idx
        vals = info.X[idx, node.feature]
        # classifiers stack by class; other models (e.g. sums of trees) get a plain histogram
        labs = info.y[idx] if info.is_clf else np.zeros(len(idx), dtype=int)
        ok = np.isfinite(vals)
        vals, labs = vals[ok], labs[ok]
        kind = info.feature_kind.get(node.feature, "continuous")
        lo, hi, labels = self._range(vals, node.threshold, robust=kind == "continuous")
        if kind in ("binary", "integer") and hi - lo <= 30:
            labels = (fmt(np.min(vals), 12), fmt(np.max(vals), 12))
            if kind == "binary":
                labels = ("", "")
            lo, hi = np.floor(lo) - 0.5, np.ceil(hi) + 0.5
            edges = np.arange(lo, hi + 1e-9, 1.0)
        else:
            edges = np.linspace(lo, hi, 25)
        vals = np.clip(vals, lo, hi)
        nb = len(edges) - 1
        k = len(info.class_names) if info.is_clf else 1
        H = np.zeros((k, nb))
        for c in range(k):
            H[c] = np.histogram(vals[labs == c], bins=edges)[0]
        tot = H.sum(0)
        top = tot.max() or 1
        gap = 1.5 if w / nb > 4 else 0.75
        bw = min(w / nb - gap, 22.0)  # thin marks: cap bar width
        out = []
        for b in range(nb):
            if tot[b] == 0:
                continue
            bx = x0 + (b + 0.5) * (w / nb) - bw / 2
            ycur = y0 + h
            nzc = [c for c in range(k) if H[c, b] > 0]
            for j, c in enumerate(nzc):
                sh = H[c, b] / top * h
                ycur -= sh
                r = min(2.0, bw / 2) if j == len(nzc) - 1 else 0
                out.append(f'<path d="{_top_round_bar(bx, ycur, bw, sh, r)}" style="{_st(P.cls(c) if info.is_clf else P.seq(0.55))}"/>')
        return "".join(out), lo, hi, labels

    def _scatter_reg(self, node, x0, y0, w, h):
        P, info = self.P, self.info
        idx = node.idx
        vals, ys = info.X[idx, node.feature], info.y[idx]
        ok = np.isfinite(vals)
        vals, ys = vals[ok], ys[ok]
        kind = info.feature_kind.get(node.feature, "continuous")
        lo, hi, labels = self._range(vals, node.threshold, robust=kind == "continuous")
        vals = np.clip(vals, lo, hi)
        ylo, yhi, _ = self._range(ys, robust=True)
        ys_plot = np.clip(ys, ylo, yhi)
        if len(vals) > self.max_points:
            sel = np.random.default_rng(node.id).choice(len(vals), self.max_points, replace=False)
            pv, py = vals[sel], ys_plot[sel]
        else:
            pv, py = vals, ys_plot
        sx = lambda v: x0 + (v - lo) / (hi - lo) * w
        sy = lambda v: y0 + h - 3 - (min(max(v, ylo), yhi) - ylo) / (yhi - ylo) * (h - 6)
        out = []
        for a, b in zip(pv, py):
            out.append(f'<circle cx="{_n(sx(a))}" cy="{_n(sy(b))}" r="1.9" style="{_st(P.seq(self.vfrac(b)), fo=0.75)}"/>')
        t = node.threshold
        L, R = info.nodes[node.left], info.nodes[node.right]
        xt = sx(t)
        for (a, b), ch in (((lo, t), L), ((t, hi), R)):
            yy = sy(ch.counts[0])
            out.append(f'<line x1="{_n(sx(a))}" y1="{_n(yy)}" x2="{_n(sx(b))}" y2="{_n(yy)}" style="{_st(stroke=P("ink"), sw=2, extra="stroke-linecap:round")}"/>')
        out.append(f'<line x1="{_n(xt)}" y1="{_n(y0)}" x2="{_n(xt)}" y2="{_n(y0 + h)}" style="{_st(stroke=P("ink"), sw=1, so=0.35)}"/>')
        return "".join(out), lo, hi, labels

    def _hist_leaf_reg(self, node, x0, y0, w, h):
        P, info = self.P, self.info
        ys = info.y[node.idx]
        lo, hi = info.value_range
        edges = np.linspace(lo, hi, 31)
        H = np.histogram(ys, bins=edges)[0]
        top = H.max() or 1
        bw = w / 30
        out = [f'<line x1="{_n(x0)}" y1="{_n(y0 + h + 0.5)}" x2="{_n(x0 + w)}" y2="{_n(y0 + h + 0.5)}" style="{_st(stroke=P("axis"), sw=1)}"/>']
        for b in range(30):
            if H[b] == 0:
                continue
            sh = H[b] / top * h
            c = P.seq(self.vfrac((edges[b] + edges[b + 1]) / 2))
            out.append(f'<path d="{_top_round_bar(x0 + b * bw + 0.5, y0 + h - sh, bw - 1, sh, 1.5)}" style="{_st(c)}"/>')
        m = x0 + float(np.clip(self.vfrac(node.counts[0]), 0, 1)) * w
        out.append(f'<path d="M{_n(m)},{_n(y0 + h + 2)}l4,5.5h-8z" style="{_st(P("ink"))}"/>')
        return "".join(out)

    def _marker(self, x0, y0, w, h, lo, hi, v):
        P = self.P
        xv = x0 + float(np.clip((v - lo) / (hi - lo), 0, 1)) * w
        return (
            f'<line x1="{_n(xv)}" y1="{_n(y0 - 3)}" x2="{_n(xv)}" y2="{_n(y0 + h)}" style="{_st(stroke=P("hl"), sw=2)}"/>'
            f'<circle cx="{_n(xv)}" cy="{_n(y0 - 3)}" r="3.5" style="{_st(P("hl"), P("card"), 1.5)}"/>'
        )

    def _frame(self, w, h, node, leaf, truncated=False):
        P = self.P
        cid = f"{self.uid}-cc{node.id}"
        out = []
        if truncated:
            for d in (8, 4):
                out.append(f'<rect x="{d}" y="{d}" width="{_n(w)}" height="{_n(h)}" rx="{RADIUS}" style="{_st(P("card"), P("ring"), 1)}"/>')
        out.append(f'<rect class="dti-frame" width="{_n(w)}" height="{_n(h)}" rx="{RADIUS}" filter="url(#{self.uid}-sh)" style="{_st(P("card"), P("ring"), 1)}"/>')
        if leaf:
            col = self.node_color(node)
            out.append(f'<clipPath id="{cid}"><rect width="{_n(w)}" height="{_n(h)}" rx="{RADIUS}"/></clipPath>')
            out.append(f'<g clip-path="url(#{cid})"><rect width="{_n(w)}" height="{_n(h)}" style="{_st(col, fo=0.07)}"/>'
                       f'<rect width="{_n(w)}" height="4" style="{_st(col)}"/></g>')
        return "".join(out)

    def _footer_text(self, node):
        info = self.info
        s = f"{fmt_count(node.n)} samples" if node.n else ""
        if node.n and abs(node.weight - node.n) > 1e-6:
            s = f"{fmt_count(node.n)} samples (weighted {fmt(node.weight)})"
        if not np.isfinite(node.impurity):
            return s
        crit = {"squared_error": "mse", "friedman_mse": "mse", "absolute_error": "mae", "poisson": "poisson dev"}.get(info.criterion, info.criterion)
        imp = f"{crit} {fmt(node.impurity, 3)}"
        return f"{s}  \u00b7  {imp}" if s else imp

    # ---- cards
    def mix(self, node):
        """[(color, fraction)] of the samples at ``node``: class shares, or one value color."""
        if not self.info.is_clf:
            return [(self.node_color(node), 1.0)]
        tot = node.counts.sum() or 1.0
        return [(self.P.cls(k), c / tot) for k, c in enumerate(node.counts) if c > 0]

    def card(self, nid, truncated=False, simple=None):
        """Return dict(w, h, svg, chart) for node ``nid`` drawn at the origin."""
        self._cur = nid
        node = self.info.nodes[nid]
        if self.simple if simple is None else simple:
            return self._simple(node, truncated)
        if node.is_leaf:
            return self._leaf(node)
        return self._split(node, truncated)

    def _simple(self, node, truncated=False):
        """A plain box filled by class proportions (or the predicted value) with a small label."""
        P, info = self.P, self.info
        contrib = getattr(node, "value", None) if info.combine == "sum" else None
        if node.is_leaf and contrib is not None:  # trees that add up: a leaf is what it adds, not a class
            label = ("+" if contrib > 0 else "") + fmt(contrib, self.sig)
        elif node.is_leaf:
            k = info.prediction(node)
            label = info.class_names[k] if info.is_clf else fmt(node.counts[0], max(self.sig, 4))
        else:
            label = node.label or (info.feature_names[node.feature] if node.simple else info.split_text(node.id, self.sig))
        label = truncate(label, 11.5, 200, True)
        h = 30.0
        w = max(64.0, text_width(label, 11.5, True) + 34)
        cid = f"{self.uid}-cs{node.id}"
        out = []
        if truncated:
            for d in (6, 3):
                out.append(f'<rect x="{d}" y="{d}" width="{_n(w)}" height="{_n(h)}" rx="9" style="{_st(P("card"), P("ring"), 1)}"/>')
        out.append(f'<clipPath id="{cid}"><rect width="{_n(w)}" height="{_n(h)}" rx="9"/></clipPath>')
        out.append(f'<rect class="dti-frame" width="{_n(w)}" height="{_n(h)}" rx="9" filter="url(#{self.uid}-sh)" style="{_st(P("card"), P("ring"), 1)}"/>')
        out.append(f'<g clip-path="url(#{cid})">')
        x, parts = 0.0, self.mix(node)
        gap = 2.0 if len(parts) > 1 else 0.0
        avail = w - gap * (len(parts) - 1)
        for col, f in parts:
            out.append(f'<rect x="{_n(x)}" width="{_n(max(avail * f, 0.5))}" height="{_n(h)}" style="{_st(col)}"/>')
            x += avail * f + gap
        out.append("</g>")
        pw = text_width(label, 11.5, True) + 16
        out.append(f'<rect x="{_n((w - pw) / 2)}" y="5" width="{_n(pw)}" height="20" rx="10" style="{_st(P("card"), fo=0.94)}"/>')
        out.append(_text(w / 2, 19, label, 11.5, P("ink"), 600, "middle"))
        if self.x is not None and self._on_path(node.id):
            out.append(f'<rect x="-3" y="-3" width="{_n(w + 6)}" height="{_n(h + 6)}" rx="12" style="{_st("none", P("hl"), 2)}"/>')
        tip = f"{label}\n{self._footer_text(node)}"
        if info.is_clf:
            tip += "\n" + "\n".join(f"{info.class_names[j]}: {fmt_count(c)}" for j, c in enumerate(node.counts) if c > 0)
        return dict(w=w, h=h, svg=f"<title>{esc(tip)}</title>" + "".join(out), chart=None)

    def _split(self, node, truncated):
        P, info = self.P, self.info
        compact = self.style == "compact"
        tsize = 12 if compact else 13
        # one-feature threshold splits get a chart; anything else lists its conditions
        lines = [] if node.simple else [info.cond_text(f, op, v, self.sig) for f, op, v in node.split]
        if node.simple:  # a one-feature split is titled by its feature; a rule number becomes a small label
            name, eyebrow = info.feature_names[node.feature], (node.label or "").upper()
        else:
            name, eyebrow = node.label or ("If" if len(lines) == 1 else "If all of"), ""
        footer = (f"{fmt_count(node.n)} samples" if node.n else "") if compact else self._footer_text(node)
        if truncated:
            footer = f"{fmt_count(node.n)} samples  ·  +{info.n_descendants(node.id)} nodes"
        fsize = 10 if compact else 10.5
        on_path = self.x is not None and self._on_path(node.id)
        has_chart = node.simple and self.has_data and node.idx is not None and len(node.idx)
        # the sample count sits in the title row (one line shorter) unless that row holds something else
        count = f"n={fmt_count(node.n)}" if (node.n and not eyebrow and not truncated
                                              and not (on_path and not has_chart and node.simple)) else ""
        if count:
            footer = ""
        cw = text_width(count, fsize) + 8 if count else 0
        base_w = 92 if compact else CHART_W + 2 * PAD
        w = max(base_w, text_width(name, tsize, True) + 2 * PAD + 2 + cw + (text_width(eyebrow, 9.5, True) + 12 if eyebrow else 0),
                text_width(footer, fsize) + 2 * PAD if footer else 0,
                *[text_width(l, 11.5) + 2 * PAD for l in lines])
        w = min(w, MAX_CARD_W if not compact else 180)
        inner = w - 2 * PAD
        body, chart = [], None
        y = PAD
        title = truncate(name, tsize, inner - cw, True)
        if count:
            body.append(_text(w - PAD, y + tsize - 2.5, count, fsize, P("muted"), anchor="end",
                              extra="font-variant-numeric:tabular-nums"))
        if eyebrow:
            ew = text_width(eyebrow, 9.5, True) + 8
            title = truncate(name, tsize, inner - ew - 4, True)
            body.append(_text(w - PAD, y + tsize - 3, eyebrow, 9.5, P("muted"), 600, "end", "letter-spacing:.06em"))
        body.append(_text(PAD, y + tsize - 2.5, title, tsize, P("ink"), 600))
        y += tsize + 4
        for i, line in enumerate(lines):
            y += 3
            body.append(_text(PAD, y + 11, truncate(line, 11.5, inner), 11.5, P("ink2"), extra="font-variant-numeric:tabular-nums"))
            y += 14
        if lines:
            y += 2
        if node.simple and self.has_data and node.idx is not None and len(node.idx):
            y += 6
            if info.is_clf:
                g, lo, hi, ends = self._hist_clf(node, PAD, y, inner, CHART_H)
            else:
                chart_fn = self._hist_clf if info.combine == "sum" else self._scatter_reg
                g, lo, hi, ends = chart_fn(node, PAD, y, inner, CHART_H)
            body.append(g)
            kind = info.feature_kind.get(node.feature, "continuous")
            tl = fmt(node.threshold, self.sig)
            if kind == "binary":
                tl = "0 | 1"
            elif kind == "integer":
                k = int(np.floor(node.threshold))
                tl = f"{k} | {k + 1}"
            body.append(self._axis(PAD, inner, y + CHART_H, lo, hi, node.threshold, tl, ends))
            chart = dict(x0=PAD, x1=PAD + inner, y0=y, y1=y + CHART_H, lo=lo, hi=hi)
            if on_path:
                body.append(self._marker(PAD, y, inner, CHART_H, lo, hi, self.x[node.feature]))
            y += CHART_H + 23
        else:
            y += 4
            if info.is_clf:
                body.append(self._class_bar(PAD, y, inner, 6 if compact else 8, node.counts))
                y += (6 if compact else 8) + 9
            else:
                body.append(self._value_track(PAD + 4.5, y + 4.5, inner - 9, node.counts[0]))
                y += 9 + 9
        if footer:
            body.append(_text(PAD, y + fsize - 1, truncate(footer, fsize, inner), fsize, P("muted"), extra="font-variant-numeric:tabular-nums"))
            y += fsize + 2
        else:
            y -= 4
        if on_path and chart is None and node.simple:
            v = f"x = {fmt(self.x[node.feature], self.sig)}"
            body.append(_text(w - PAD, PAD + tsize - 2.5, v, 10.5, P("hl"), 700, "end"))
        h = y + PAD - 1
        tip = f"{info.split_text(node.id, self.sig)}\n{self._footer_text(node)}"
        svg = f"<title>{esc(tip)}</title>" + self._frame(w, h, node, False, truncated) + "".join(body)
        return dict(w=w, h=h, svg=svg, chart=chart)

    def _leaf(self, node):
        P, info = self.P, self.info
        compact = self.style == "compact"
        tsize = 12 if compact else 13.5
        k = info.prediction(node)
        contrib = getattr(node, "value", None) if info.combine == "sum" else None
        if info.is_clf and contrib is not None:  # sum of trees: the leaf adds to the log-odds
            label = ("+" if contrib > 0 else "") + fmt(contrib, self.sig)
            sub = f"toward {info.class_names[1 if contrib >= 0 else 0]}  \u00b7  {fmt_count(node.n)}"
        elif info.is_clf:
            tot = node.counts.sum() or 1
            label = info.class_names[k]
            sub = f"{node.counts[k] / tot:.0%} of {fmt_count(node.n)}" if node.n else f"probability {node.counts[k] / tot:.0%}"
        else:
            label = fmt(node.counts[0], max(self.sig, 4))
            if info.combine == "sum" and node.counts[0] > 0:
                label = "+" + label  # additive contribution
            sub = f"{fmt_count(node.n)} samples" if node.n else ""
        dot = 9
        w = max(90 if compact else 116, text_width(label, tsize, True) + 2 * PAD + dot + 6, text_width(sub, 11) + 2 * PAD)
        w = min(w, MAX_CARD_W)
        inner = w - 2 * PAD
        y = PAD + 2
        body = [f'<circle cx="{_n(PAD + 4.5)}" cy="{_n(y + tsize / 2 - 0.5)}" r="4.5" style="{_st(self.node_color(node))}"/>',
                _text(PAD + dot + 6, y + tsize - 2.5, truncate(label, tsize, inner - dot - 6, True), tsize, P("ink"), 600)]
        y += tsize + 5
        body.append(_text(PAD, y + 10, sub, 11, P("ink2"), extra="font-variant-numeric:tabular-nums"))
        y += 15
        if not compact:
            y += 4
            if info.is_clf:
                body.append(self._class_bar(PAD, y, inner, 6, node.counts))
                y += 6
            elif self.has_data and node.idx is not None and len(node.idx) and info.combine != "sum":
                body.append(self._hist_leaf_reg(node, PAD, y, inner, 22))
                y += 28
            else:
                body.append(self._value_track(PAD + 4.5, y + 4.5, inner - 9, node.counts[0]))
                y += 9
        h = y + PAD - (2 if compact else 0)
        if self.x is not None and self._on_path(node.id):
            body.append(f'<rect x="-3" y="-3" width="{_n(w + 6)}" height="{_n(h + 6)}" rx="{RADIUS + 3}" style="{_st("none", P("hl"), 2)}"/>')
        tip = f"{label}\n{self._footer_text(node)}"
        if info.is_clf:
            tip += "\n" + "\n".join(f"{info.class_names[j]}: {fmt_count(c)}" for j, c in enumerate(node.counts) if c > 0)
        svg = f"<title>{esc(tip)}</title>" + self._frame(w, h, node, True) + "".join(body)
        return dict(w=w, h=h, svg=svg, chart=None)

    def _on_path(self, nid):
        return nid in self.path

    def set_path(self, path):
        self.path = set(path)


def defs(uid, P):
    return (
        f'<defs><filter id="{uid}-sh" x="-20%" y="-20%" width="140%" height="160%">'
        f'<feDropShadow dx="0" dy="1" stdDeviation="1" style="flood-color:{P("shadow")};flood-opacity:0.06"/>'
        f'<feDropShadow dx="0" dy="6" stdDeviation="8" style="flood-color:{P("shadow")};flood-opacity:0.07"/>'
        "</filter></defs>"
    )


def ribbon_geometry(pbox, cbox, a0, a1, c0, c1, horizontal):
    """Path for a flow ribbon from span [a0, a1] on the parent to [c0, c1] on the child.

    Spans are offsets along the breadth axis relative to each box center.
    Boxes are (x, y, w, h).
    """
    px, py, pw, ph = pbox
    cx, cy, cw, ch = cbox
    if not horizontal:
        pc, pe = px + pw / 2, py + ph
        cc, ce = cx + cw / 2, cy
        m = (pe + ce) / 2
        A0, A1, C0, C1 = pc + a0, pc + a1, cc + c0, cc + c1
        d = (f"M{_n(A0)},{_n(pe)}C{_n(A0)},{_n(m)} {_n(C0)},{_n(m)} {_n(C0)},{_n(ce)}"
             f"L{_n(C1)},{_n(ce)}C{_n(C1)},{_n(m)} {_n(A1)},{_n(m)} {_n(A1)},{_n(pe)}Z")
        mid = ((A0 + A1 + C0 + C1) / 4, m)
    else:
        pc, pe = py + ph / 2, px + pw
        cc, ce = cy + ch / 2, cx
        m = (pe + ce) / 2
        A0, A1, C0, C1 = pc + a0, pc + a1, cc + c0, cc + c1
        d = (f"M{_n(pe)},{_n(A0)}C{_n(m)},{_n(A0)} {_n(m)},{_n(C0)} {_n(ce)},{_n(C0)}"
             f"L{_n(ce)},{_n(C1)}C{_n(m)},{_n(C1)} {_n(m)},{_n(A1)} {_n(pe)},{_n(A1)}Z")
        mid = (m, (A0 + A1 + C0 + C1) / 4)
    return d, mid


def band_width(weight, total, band=BAND):
    return max(1.5, band * weight / total)


def root_band(card, horizontal=False):
    """Ribbon width for all samples: never wider than the root card's edge it leaves from."""
    return min(BAND, 0.8 * (card["h"] if horizontal else card["w"]))


def pill(cx, cy, label, P):
    w = text_width(label, 10.5, True) + 14
    return (
        f'<g class="dti-pill"><rect x="{_n(cx - w / 2)}" y="{_n(cy - 9)}" width="{_n(w)}" height="18" rx="9" '
        f'style="{_st(P("card"), P("ring"), 1)}"/>'
        + _text(cx, cy + 3.7, label, 10.5, P("ink"), 600, "middle", "font-variant-numeric:tabular-nums")
        + "</g>"
    )


def cn_share(info, p, c):
    pn, cn = info.nodes[p], info.nodes[c]
    return cn.weight / pn.weight if pn.weight else 0.0


class Figure:
    """Lays out visible nodes and composes the full static SVG."""

    def __init__(self, info, artist, P, orientation="TB", max_depth=None, title=None,
                 subtitle=None, legend=True, background=True, prediction=None):
        self.info, self.A, self.P = info, artist, P
        self.horizontal = orientation.upper() == "LR"
        self.max_depth = max_depth
        self.title, self.subtitle, self.legend = title, subtitle, legend
        self.background, self.prediction = background, prediction

    def visible(self):
        kids, vis, trunc = {}, [], set()
        stack = list(reversed(self.info.shown_roots or self.info.roots))
        while stack:
            nid = stack.pop()
            nd = self.info.nodes[nid]
            vis.append(nid)
            if nd.is_leaf:
                continue
            if self.max_depth is not None and nd.depth >= self.max_depth:
                trunc.add(nid)
                continue
            kids[nid] = [nd.left, nd.right]
            stack += [nd.right, nd.left]
        return vis, kids, trunc

    def layout(self, vis, kids, cards):
        """Boxes (x, y, w, h) per node, per-edge orientation, glyphs (x, y, text, size) and total size.

        Ensembles are laid out in a grid of up to three trees per row (one column when horizontal);
        trees that add up are joined by '+', and every tree of an ensemble is labelled.
        """
        info, H = self.info, self.horizontal
        if info.layout == "cascade":
            return self._cascade(kids, cards)
        breadth = {n: (c["h"] if H else c["w"]) for n, c in cards.items()}
        dsize = {n: (c["w"] if H else c["h"]) for n, c in cards.items()}
        depth = {n: info.nodes[n].depth for n in vis}
        gap, lvl = (16, 96) if H else (14, 44)
        roots = info.shown_roots or info.roots
        multi = len(info.roots) > 1
        trees = [tidy(r, kids, breadth, depth, dsize, gap=gap, level_gap=lvl) for r in roots]
        per_row = 1 if H else min(3, len(roots))
        label_h = 20 if multi else 0
        boxes, glyphs = {}, []
        row_top, tot_w, tot_h = 0.0, 0.0, 0.0
        for start in range(0, len(roots), per_row):
            row = list(range(start, min(start + per_row, len(roots))))
            off = 0.0
            for k in row:
                r, (pos, top, tb, td) = roots[k], trees[k]
                if k != row[0] and info.combine == "sum":
                    glyphs.append((off - 22, row_top + label_h + cards[r]["h"] / 2 + 9, "+", 26))
                for n in pos:
                    c = cards[n]
                    if H:
                        boxes[n] = (top[n], row_top + label_h + off + pos[n] - c["h"] / 2, c["w"], c["h"])
                    else:
                        boxes[n] = (off + pos[n] - c["w"] / 2, row_top + label_h + top[n], c["w"], c["h"])
                if multi:
                    lx = off + tb / 2 if not H else 0
                    glyphs.append((lx, row_top + 14, f"tree {info.roots.index(r) + 1} of {len(info.roots)}", 11))
                off += (td if H else tb) + 44
            row_d = max((trees[k][2] if H else trees[k][3]) for k in row)
            tot_w = max(tot_w, off - 44)
            row_top += label_h + row_d + 32
            tot_h = row_top - 32
        horiz = {n: H for n in boxes}
        w, h = (max(t[3] for t in trees), tot_h) if H else (tot_w, tot_h)
        return boxes, horiz, glyphs, w, h

    def _cascade(self, kids, cards):
        """Rule-list layout: rules in a column, each rule's outcome to its right."""
        info = self.info
        boxes, horiz = {}, {}
        chain, nid = [], info.roots[0]
        while True:
            chain.append(nid)
            if nid not in kids:
                break
            nd = info.nodes[nid]
            side, main = (nd.left, nd.right) if info.nodes[nd.left].is_leaf else (nd.right, nd.left)
            chain[-1] = (nid, side)
            nid = main
        col_w = max(cards[c if isinstance(c, int) else c[0]]["w"] for c in chain)
        side_x = col_w + 84  # rules are centered in a column; outcomes sit to the right
        y, side_w = 0.0, 0.0
        for item in chain:
            nid, side = item if isinstance(item, tuple) else (item, None)
            c = cards[nid]
            row_h = c["h"]
            boxes[nid] = ((col_w - c["w"]) / 2, y, c["w"], c["h"])
            horiz[nid] = False
            if side is not None:
                s = cards[side]
                boxes[side] = (side_x, y + (c["h"] - s["h"]) / 2 if s["h"] < c["h"] else y, s["w"], s["h"])
                horiz[side] = True
                row_h = max(row_h, s["h"])
                side_w = max(side_w, s["w"])
            y += row_h + 40
        return boxes, horiz, [], side_x + side_w if side_w else col_w, y - 40

    def header(self, width):
        """Title, subtitle and legend. The legend sits on the subtitle's line when there is room."""
        P, info = self.P, self.info
        parts, y = [], 0
        sub = self.subtitle if self.subtitle is not None else info.summary()
        sub_top, sub_x = y, 0.0
        if self.title:
            parts.append(_text(0, 18, self.title, 18, P("ink"), 650))
            tw = text_width(self.title, 18, True) + 18
            if sub and tw + text_width(sub, 12) <= width:  # subtitle on the title's line
                sub_top, sub_x = 5, tw
            else:
                y = 26
                sub_top = y
        if sub:
            parts.append(_text(sub_x, sub_top + 13, sub, 12, P("ink2")))
            y = max(y, sub_top + 19) if sub_x == 0 else 26
        if self.legend:
            note = {"sum": "leaf values add up to the log-odds", "mean": "the trees' predictions are averaged"}.get(info.combine)
            if info.is_clf:
                widths = [text_width(n, 11.5) + 26 for n in info.class_names]
                leg_w = sum(widths) + (text_width(note, 11.5) + 6 if note else 0)
            else:
                lo, hi = info.value_range
                label = "leaf value, added across trees" if info.combine == "sum" else f"predicted {info.target_name}"
                leg_w = text_width(label, 11.5) + 10 + text_width(fmt(lo), 10.5) + 152 + text_width(fmt(hi), 10.5)
            inline = bool(sub) and sub_x + text_width(sub, 12) + 28 + leg_w <= width
            x0, ly = (sub_x + text_width(sub, 12) + 28, sub_top + 3) if inline else (0.0, y + 3)
            if info.is_clf:
                x = x0
                for k, name in enumerate(info.class_names):
                    if x + widths[k] > width and x > x0:
                        x, ly = 0.0, ly + 20
                    parts.append(f'<circle cx="{_n(x + 5)}" cy="{_n(ly + 6)}" r="5" style="{_st(P.cls(k))}"/>')
                    parts.append(_text(x + 15, ly + 10, name, 11.5, P("ink2")))
                    x += widths[k]
                if note:
                    parts.append(_text(x + 6, ly + 10, note, 11.5, P("muted")))
            else:
                gid = f"{self.A.uid}-grad"
                stops = "".join(f'<stop offset="{i / (len(P.seq_stops()) - 1):.3f}" stop-color="{c}"/>'
                                for i, c in enumerate(P.seq_stops()))
                lw = text_width(label, 11.5) + 10
                parts.append(f'<defs><linearGradient id="{gid}">{stops}</linearGradient></defs>')
                parts.append(_text(x0, ly + 10, label, 11.5, P("ink2")))
                parts.append(_text(x0 + lw, ly + 10, fmt(lo), 10.5, P("muted")))
                gx = x0 + lw + text_width(fmt(lo), 10.5) + 6
                parts.append(f'<rect x="{_n(gx)}" y="{_n(ly + 2)}" width="140" height="8" rx="4" style="fill:url(#{gid})"/>')
                parts.append(_text(gx + 146, ly + 10, fmt(hi), 10.5, P("muted")))
            y = max(y, ly + 17)
        if self.prediction:
            y += 4
            parts.append(_text(0, y + 11, self.prediction, 12, P("ink"), 600))
            y += 18
        return "".join(parts), (y + 12 if parts else 0)

    def svg(self):
        P, info, A = self.P, self.info, self.A
        vis, kids, trunc = self.visible()
        cards = {nid: A.card(nid, nid in trunc) for nid in vis}
        root0 = info.roots[0]
        band = root_band(cards[root0], self.horizontal and info.layout != "cascade")
        if info.layout == "cascade":  # flows turn sideways into outcome cards: keep them thinner than a card
            band = min(band, 0.8 * min(c["h"] for c in cards.values()))
        boxes, horiz, plus, tree_w, tree_h = self.layout(vis, kids, cards)
        M = 18
        head, head_h = self.header(max(tree_w, 480))
        W = max(tree_w, 520 if head_h else 0) + 2 * M
        Ht = tree_h + head_h + 2 * M + 10
        ox = M + (W - 2 * M - tree_w) / 2
        oy = M + head_h

        path = A.path if A.x is not None else set()
        ribbons, labels, nodes_svg = [], [], []
        for p, cs in kids.items():
            pn = info.nodes[p]
            pw = band_width(pn.weight, A.N, band)
            start = -pw / 2
            mixed = len({horiz[c] for c in cs}) > 1  # cascade: children leave from different sides
            for c in cs:
                H = horiz[c]
                if mixed:
                    start = -pw * cn_share(info, p, c) / 2
                cn = info.nodes[c]
                cw = band_width(cn.weight, A.N, band)
                share = pw * cn.weight / pn.weight if pn.weight else 0
                on = c in path
                fo = 0.6 if on else (0.08 if path else 0.32)
                tip = f"{esc(info.split_text(p, A.sig))}: {esc(info.edge_label(c, A.sig))}  \u2192  {fmt_count(cn.n)} samples"
                sub = []
                a0, c0 = start, -cw / 2
                for col, f in A.mix(cn):
                    d, _ = ribbon_geometry(boxes[p], boxes[c], a0, a0 + share * f, c0, c0 + cw * f, H)
                    sub.append(f'<path d="{d}" style="{_st(col)}"/>')
                    a0, c0 = a0 + share * f, c0 + cw * f
                _, mid = ribbon_geometry(boxes[p], boxes[c], start, start + share, -cw / 2, cw / 2, H)
                start += share
                ribbons.append(f'<g style="fill-opacity:{fo:g}"><title>{tip}</title>{"".join(sub)}</g>')
                labels.append(pill(mid[0], mid[1], info.edge_label(c, A.sig), P))
        for x, y, text, size in plus:  # "+" between trees that add up; "tree k of n" labels
            if text == "+":
                labels.append(_text(x, y, text, size, P("muted"), 300, "middle"))
            else:
                labels.append(_text(x, y, text, size, P("muted"), 600, "start" if self.horizontal else "middle",
                                    "letter-spacing:.06em"))
        for n in vis:
            x, y, w, h = boxes[n]
            dim = ' style="opacity:0.45"' if path and n not in path else ""
            nodes_svg.append(f'<g transform="translate({_n(x)},{_n(y)})"{dim}>{cards[n]["svg"]}</g>')

        bg = f'<rect width="100%" height="100%" style="{_st(P("surface"))}"/>' if self.background else ""
        aria = esc(self.title or f"{info.model_name} with {info.n_leaves} leaves")
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{_n(W)}" height="{_n(Ht)}" viewBox="0 0 {_n(W)} {_n(Ht)}" '
            f'role="img" aria-label="{aria}" font-family="{esc(FONT_STACK)}">'
            + defs(A.uid, P) + bg
            + f'<g transform="translate({M},{M})">{head}</g>'
            + f'<g transform="translate({_n(ox)},{_n(oy)})">'
            + "".join(ribbons) + "".join(nodes_svg) + "".join(labels)
            + "</g></svg>"
        )
