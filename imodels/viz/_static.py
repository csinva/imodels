"""Static mode: one self-contained SVG (exportable to PNG / PDF / HTML)."""

import os

import numpy as np

from ._adapt import to_view
from ._views import AdditiveView
from ._render import Artist, Figure
from ._text import new_uid, fmt
from ._theme import Paint


def _resolve_style(style, info, max_depth):
    if style != "auto":
        return style
    n_leaves = info.n_leaves if max_depth is None else min(info.n_leaves, 2 ** max_depth)
    return "detailed" if n_leaves <= 24 else "compact"


def instance_path(info, x):
    x = np.asarray(x, dtype=float).ravel()
    if any(nd.conds for nd in info.nodes) or len(info.roots) > 1:
        return [n for r in info.roots for n in info.walk(x, r)]
    nid, path = 0, [0]
    while not info.nodes[nid].is_leaf:
        nd = info.nodes[nid]
        v = x[nd.feature]
        if np.isnan(v):
            nid = nd.left if nd.nan_left is not False else nd.right
        else:
            nid = nd.left if v <= nd.threshold else nd.right
        path.append(nid)
    return path


def prediction_text(info, leaf, sig=3):
    nd = info.nodes[leaf]
    if info.is_clf:
        k = info.prediction(nd)
        p = nd.counts[k] / (nd.counts.sum() or 1)
        return f"Prediction for this sample: {info.class_names[k]} ({p:.0%} of {nd.n} training samples in its leaf)"
    return f"Prediction for this sample: {fmt(nd.counts[0], max(sig, 4))} (mean of {nd.n} training samples in its leaf)"


class TreeFigure:
    """A rendered tree. Displays inline in Jupyter; ``save`` writes .svg/.png/.pdf/.html."""

    def __init__(self, svg):
        self.svg = svg

    def _repr_svg_(self):
        return self.svg

    def __str__(self):
        return self.svg

    def save(self, path, scale=2.0):
        ext = os.path.splitext(str(path))[1].lower()
        if ext == ".svg":
            with open(path, "w", encoding="utf-8") as f:
                f.write(self.svg)
        elif ext in (".png", ".pdf"):
            try:
                import cairosvg
            except ImportError as e:  # pragma: no cover
                raise ImportError("PNG/PDF export needs cairosvg: pip install cairosvg") from e
            fn = cairosvg.svg2png if ext == ".png" else cairosvg.svg2pdf
            from ._text import EXPORT_FONT, FONT_STACK, esc

            svg = self.svg.replace(f'font-family="{esc(FONT_STACK)}"', f'font-family="{EXPORT_FONT}"', 1)
            fn(bytestring=svg.encode("utf-8"), write_to=str(path), scale=scale if ext == ".png" else 1.0)
        elif ext in (".html", ".htm"):
            with open(path, "w", encoding="utf-8") as f:
                f.write("<!doctype html><meta charset='utf-8'><title>Decision tree</title>"
                        "<body style='margin:0;display:flex;justify-content:center'>" + self.svg + "</body>")
        else:
            raise ValueError(f"Unsupported extension {ext!r}; use .svg, .png, .pdf or .html")
        return path


def draw(model, X=None, y=None, *, feature_names=None, class_names=None, target_name=None,
         x=None, max_depth=None, orientation="TB", theme="light", style="auto", title=None,
         subtitle=None, legend=True, background=True, precision=3, simple=False, output=0, max_trees=None):
    """Render a fitted sklearn decision tree as a static, publication-quality SVG.

    Parameters
    ----------
    model : fitted DecisionTreeClassifier / DecisionTreeRegressor / ExtraTree*, a Pipeline
        ending in one, or a single tree from an ensemble (``forest.estimators_[0]``).
    X, y : optional training (or held-out) data. When given, each split node shows the
        distribution of its split feature (stacked class histogram or feature/target
        scatter) and split thresholds on binary / integer features read naturally.
    x : optional single sample; its decision path is highlighted.
    max_depth : draw only the top ``max_depth`` levels; deeper subtrees are shown as stacked cards.
    orientation : "TB" (top to bottom) or "LR" (left to right).
    theme : "light" or "dark".
    style : "detailed", "compact" or "auto" (compact for trees with more than 24 leaves).
    precision : significant digits for thresholds and values.
    simple : draw each node as a plain box filled by its class proportions (or predicted
        value) instead of the detailed card with charts.
    max_trees : for ensembles (forests, boosting), how many trees to draw (default: 6).
    """
    info = to_view(model, X, y, feature_names, class_names, target_name, output)
    if isinstance(info, AdditiveView):
        from ._render_views import render_view

        return TreeFigure(render_view(info, Paint(theme), precision, new_uid(),
                                      x=x, title=title, subtitle=subtitle))
    if max_trees is not None:
        info.shown_roots = info.roots[:max(1, int(max_trees))]
    style = _resolve_style(style, info, max_depth)
    P = Paint(theme)
    artist = Artist(info, P, style=style, sig=precision, uid=new_uid(), x=x,
                    simple=simple)
    pred = None
    path = []
    if x is not None:
        path = instance_path(info, x)
        pred = prediction_text(info, path[-1], precision)
    artist.set_path(path)
    fig = Figure(info, artist, P, orientation=orientation, max_depth=max_depth, title=title,
                 subtitle=subtitle, legend=legend, background=background, prediction=pred)
    return TreeFigure(fig.svg())
