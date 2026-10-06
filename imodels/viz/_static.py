"""Static mode: one self-contained SVG (exportable to PNG / PDF / HTML)."""

import os

import numpy as np

from ._adapt import to_view
from ._views import AdditiveView
from ._render import Artist, Figure, resolve_orientation
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
    """A static figure of a model, returned by `draw`. Displays inline in Jupyter.

    Attributes
    ----------
    svg : str
        The figure as SVG text.
    """

    def __init__(self, svg):
        self.svg = svg

    def _repr_svg_(self):
        return self.svg

    def __str__(self):
        return self.svg

    def save(self, path, scale=2.0):
        """Write the figure to a file.

        Parameters
        ----------
        path : str or path-like
            Output file; its extension picks the format: .svg, .png, .pdf or .html.
            PNG and PDF need cairosvg.
        scale : float, default=2.0
            Resolution multiplier for PNG (PDF is vector, so it is unaffected).

        Returns
        -------
        path : str or path-like
            The path written.
        """
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
                f.write("<!doctype html><meta charset='utf-8'><title>Model</title>"
                        "<body style='margin:0;display:flex;justify-content:center'>" + self.svg + "</body>")
        else:
            raise ValueError(f"Unsupported extension {ext!r}; use .svg, .png, .pdf or .html")
        return path


def draw(model, X=None, y=None, *, feature_names=None, class_names=None, target_name=None,
         x=None, max_depth=None, orientation="auto", theme="light", style="auto", title=None,
         subtitle=None, legend=True, background=True, precision=3, simple=False, output=0, max_trees=None):
    """Draw a fitted model as a static figure.

    Returns a `TreeFigure`, which shows inline in Jupyter and saves to .svg, .png, .pdf or .html
    (`TreeFigure.save`). PNG and PDF need cairosvg.

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
    x : array-like of shape (n_features,), optional
        A single sample whose path through the model is highlighted.
    max_depth : int, optional
        Draw only the top ``max_depth`` levels; deeper subtrees are drawn as stacked cards.
    orientation : {"auto", "LR", "TB"}, default="auto"
        Left to right, or top to bottom. "auto" lays a single tree out left to right and
        several trees (forests, boosting, FIGS) top to bottom.
    theme : {"light", "dark"}, default="light"
        Color theme.
    style : {"auto", "detailed", "compact"}, default="auto"
        Card style; "auto" is compact for trees with more than 24 leaves.
    title, subtitle : str, optional
        Figure title, and a subtitle (default: a one-line summary of the model).
    legend : bool, default=True
        Whether to draw the legend.
    background : bool, default=True
        Whether to fill the background (False gives a transparent figure).
    precision : int, default=3
        Significant digits for thresholds and values.
    simple : bool, default=False
        Draw each node as a plain box filled by its class mix (or predicted value) instead of
        the detailed card with a chart.
    output : int, default=0
        For multi-output trees, which output to draw.
    max_trees : int, optional
        For ensembles (forests, boosting, FIGS), how many trees to draw (default 6).
        Predictions always use every tree.

    Returns
    -------
    TreeFigure
        The figure, with ``.svg`` (the SVG text) and ``.save(path)``.
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
    fig = Figure(info, artist, P, orientation=resolve_orientation(orientation, info), max_depth=max_depth, title=title,
                 subtitle=subtitle, legend=legend, background=background, prediction=pred)
    return TreeFigure(fig.svg())
