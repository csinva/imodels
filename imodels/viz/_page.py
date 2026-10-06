"""Assemble interactive pages from the shared shell (assets/page.html + shared.css/js) and a model script."""

import json
from importlib import resources

import numpy as np

from ._text import FONT_STACK, esc
from ._theme import css_vars


def _asset(name):
    return resources.files("imodels.viz").joinpath("assets", name).read_text(encoding="utf-8")


def finite(o):
    """JSON-safe copy: NaN and infinities become null."""
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, dict):
        return {k: finite(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [finite(v) for v in o]
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def class_legend(class_names, paint, note=None):
    items = [dict(k="dot", col=paint.cls(k), label=n) for k, n in enumerate(class_names)]
    return items + ([dict(k="note", label=note)] if note else [])


TOOLS = {
    "tree": """
      <label class="ctl" title="Expand the tree to this depth">Depth <input id="depth" type="range" min="1" step="1" aria-label="Depth"><b id="depthv"></b></label>
      <button id="b-fit" title="Fit to screen"><svg viewBox="0 0 16 16"><path d="M2 6V2h4M10 2h4v4M14 10v4h-4M6 14H2v-4"/></svg>Fit</button>
      <button id="b-simple" title="Simple boxes instead of charts"><svg viewBox="0 0 16 16"><rect x="2" y="4.5" width="12" height="7" rx="2"/><path d="M7 4.5v7"/></svg>Simple</button>
      <button id="b-orient" title="Switch orientation"><svg viewBox="0 0 16 16"><path d="M8 2v9M4.5 8 8 11.5 11.5 8M3 14h10"/></svg><span>Vertical</span></button>""",
    "view": """
      <label class="ctl" id="sortctl" title="Order the rows">Sort <select id="sort" aria-label="Sort rows">
        <option value="effect">by effect</option><option value="coverage">by coverage</option><option value="model">model order</option></select></label>""",
}
STAGE = {
    "tree": """<svg id="canvas" xmlns="http://www.w3.org/2000/svg"><g id="vp"><g id="g-rib"></g><g id="g-node"></g><g id="g-lab"></g></g></svg>
    <div class="hint-bar"><kbd>click</kbd>collapse / expand or load a leaf's row <kbd>drag</kbd>pan <kbd>scroll</kbd>zoom</div>""",
}


def build_page(kind, *, title, subtitle, theme, data, stage=None):
    """kind: "tree" or "view"."""
    light, dark = css_vars()
    payload = json.dumps(finite(data), separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    reps = {
        "__CSS_LIGHT__": light, "__CSS_DARK__": dark, "__FONT__": FONT_STACK,
        "__SHARED_CSS__": _asset("shared.css"), "__MODEL_CSS__": _asset(f"{kind}.css"),
        "__TOOLS__": TOOLS[kind], "__STAGE__": stage if stage is not None else STAGE[kind],
        "__SHARED_JS__": _asset("shared.js"), "__MODEL_JS__": _asset(f"{kind}.js"),
        "__TITLE__": esc(title), "__SUBTITLE__": esc(subtitle), "__THEME__": theme,
    }
    out = _asset("page.html")
    # data last: it may contain any text
    for k, v in reps.items():
        out = out.replace(k, v)
    return out.replace("__DATA__", payload)
