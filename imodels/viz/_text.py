"""Text measurement and formatting helpers.

Glyph widths are the Helvetica AFM metrics (units of 1/1000 em) for ASCII 32-126,
which match Arial / Helvetica closely enough to size node cards without a browser.
"""

import itertools
import math
from html import escape as _escape

_REG = [278, 278, 355, 556, 556, 889, 667, 222, 333, 333, 389, 584, 278, 333, 278, 278, 556, 556, 556, 556, 556, 556, 556, 556, 556, 556, 278, 278, 584, 584, 584, 556, 1015, 667, 667, 722, 722, 667, 611, 778, 722, 278, 500, 667, 556, 833, 722, 778, 667, 778, 722, 667, 611, 722, 667, 944, 667, 667, 611, 278, 278, 278, 469, 556, 222, 556, 556, 500, 556, 556, 278, 556, 556, 222, 222, 500, 222, 833, 556, 556, 556, 556, 333, 500, 278, 556, 500, 722, 500, 500, 500, 334, 260, 334, 584]
_BOLD = [278, 333, 474, 556, 556, 889, 722, 278, 333, 333, 389, 584, 278, 333, 278, 278, 556, 556, 556, 556, 556, 556, 556, 556, 556, 556, 333, 333, 584, 584, 584, 611, 975, 722, 722, 722, 722, 667, 611, 778, 722, 278, 556, 722, 611, 833, 722, 778, 667, 778, 722, 667, 611, 722, 667, 944, 667, 667, 611, 333, 278, 333, 584, 556, 278, 556, 611, 556, 611, 556, 333, 611, 611, 278, 278, 556, 278, 889, 611, 611, 611, 611, 389, 556, 333, 611, 556, 778, 556, 556, 500, 389, 280, 389, 584]

FONT_STACK = "Helvetica, 'Helvetica Neue', Arial, 'Nimbus Sans', 'Liberation Sans', sans-serif"
# cairo takes only the first family, so raster export swaps in a metric-compatible single face
EXPORT_FONT = "Helvetica"


def text_width(s, size, bold=False):
    """Approximate rendered width of ``s`` in px at font ``size``."""
    table = _BOLD if bold else _REG
    total = 0
    for ch in str(s):
        o = ord(ch)
        total += table[o - 32] if 32 <= o <= 126 else 600
    return total * size / 1000.0


def truncate(s, size, max_w, bold=False):
    """Shorten ``s`` with an ellipsis so it fits in ``max_w`` px."""
    s = str(s)
    if text_width(s, size, bold) <= max_w:
        return s
    while s and text_width(s + "\u2026", size, bold) > max_w:
        s = s[:-1]
    return s.rstrip() + "\u2026"


def esc(s):
    return _escape(str(s), quote=True)


def fmt(v, sig=3):
    """Format a number with ``sig`` significant digits, no trailing zeros."""
    if v is None:
        return ""
    v = float(v)
    if not math.isfinite(v):
        return str(v)
    if v == 0 or abs(v) < 1e-12:
        return "0"
    a = abs(v)
    if a >= 1e6 or a < 1e-3:
        return f"{v:.{max(sig - 1, 1)}e}".replace("e+0", "e").replace("e-0", "e-").replace("e+", "e")
    digits = max(sig - int(math.floor(math.log10(a))) - 1, 0)
    out = f"{round(v, digits):,.{digits}f}"
    if "." in out:
        out = out.rstrip("0").rstrip(".")
    return "\u2212" + out[1:] if out.startswith("-") else out


def fmt_count(n):
    n = float(n)
    return f"{int(round(n)):,}" if abs(n - round(n)) < 1e-9 else fmt(n)


_ids = itertools.count(1)


def new_uid():
    """Prefix for SVG element ids; a counter keeps output reproducible within a process."""
    return f"dti{next(_ids)}"
