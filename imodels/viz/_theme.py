"""Color tokens. Each token has a light and a dark value.

Static SVGs inline the hex for one theme; the interactive page references
``var(--dti-<token>)`` so the viewer can switch themes live.
The categorical slots and sequential ramp follow a palette validated for
color-vision deficiency (adjacent-pair CVD delta E >= 8 in both modes).
"""

import numpy as np

CHROME = {
    # token: (light, dark); "page" tints hovered and pinned rows, "surface" is the background
    "page": ("#f6f6f3", "#0d0d0d"),
    "surface": ("#ffffff", "#1a1a19"),
    "card": ("#ffffff", "#222220"),
    "ink": ("#0b0b0b", "#ffffff"),
    "ink2": ("#52514e", "#c3c2b7"),
    "muted": ("#898781", "#898781"),
    "grid": ("#e1e0d9", "#2c2c2a"),
    "axis": ("#c3c2b7", "#4a4a46"),
    "ring": ("#e4e3dd", "#34342f"),
    "ribbon": ("#b9b8b0", "#55554f"),
    "hl": ("#0b0b0b", "#ffffff"),
    "shadow": ("#1a1a1a", "#000000"),
}

CATEGORICAL = [
    ("#2a78d6", "#3987e5"),  # blue
    ("#eb6834", "#d95926"),  # orange
    ("#1baf7a", "#199e70"),  # aqua
    ("#eda100", "#c98500"),  # yellow
    ("#e87ba4", "#d55181"),  # magenta
    ("#008300", "#008300"),  # green
    ("#4a3aa7", "#9085e9"),  # violet
    ("#e34948", "#e66767"),  # red
    # beyond 8 classes identity leans on the legend and labels, not hue alone
    ("#7a6a58", "#a8957f"),  # umber
    ("#5f7f8f", "#86a6b6"),  # slate
    ("#9a9a2a", "#b8b84a"),  # olive
    ("#8a8a8a", "#a0a0a0"),  # gray
]

# single-hue blue ramp, steps 150..700; light runs pale->deep, dark runs deep->bright
_BLUE = ["#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
         "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
# keep the low end visible against each surface (>= ~2:1)
SEQ_LIGHT = _BLUE[3:]
SEQ_DARK = list(reversed(_BLUE[:9]))
N_SEQ = len(SEQ_LIGHT)


def css_vars(prefix="--dti-"):
    """CSS custom-property blocks for both themes (used by the interactive page)."""
    def block(i):
        lines = [f"{prefix}{k}:{v[i]};" for k, v in CHROME.items()]
        lines += [f"{prefix}c{j}:{c[i]};" for j, c in enumerate(CATEGORICAL)]
        seq = SEQ_LIGHT if i == 0 else SEQ_DARK
        lines += [f"{prefix}s{j}:{c};" for j, c in enumerate(seq)]
        return "".join(lines)
    return block(0), block(1)


class Paint:
    """Resolves token names to concrete colors (static) or CSS vars (interactive)."""

    def __init__(self, theme="light", use_vars=False):
        if theme not in ("light", "dark"):
            raise ValueError("theme must be 'light' or 'dark'")
        self.theme = theme
        self.i = 0 if theme == "light" else 1
        self.use_vars = use_vars

    def __call__(self, token):
        if self.use_vars:
            return f"var(--dti-{token})"
        return CHROME[token][self.i]

    def cls(self, k):
        k = k % len(CATEGORICAL)
        return f"var(--dti-c{k})" if self.use_vars else CATEGORICAL[k][self.i]

    def seq(self, frac):
        """Sequential color for a value scaled to [0, 1]."""
        j = int(np.clip(round(float(frac) * (N_SEQ - 1)), 0, N_SEQ - 1))
        if self.use_vars:
            return f"var(--dti-s{j})"
        return (SEQ_LIGHT if self.i == 0 else SEQ_DARK)[j]

    def seq_stops(self):
        return SEQ_LIGHT if self.i == 0 else SEQ_DARK
