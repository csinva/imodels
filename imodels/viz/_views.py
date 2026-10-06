"""Shared representation for additive models: rule sets, scorecards, GAMs and linear models.

A model is ``score(x) = intercept + sum(term.contribution(x))`` followed by a link:
"identity" (regression, or a probability for classifiers), "logistic" (log-odds), or
"threshold" (classify positive when the score crosses ``threshold``).
"""

from dataclasses import dataclass, field

import numpy as np

from ._extract import NEG, SYM, _feature_kind, cond_mask
from ._text import fmt


@dataclass
class Term:
    kind: str  # "rule" | "linear" | "shape" | "curve"
    weight: float = 0.0  # rule: added when it holds; linear: per unit of (x - center)
    conds: list = None  # rule: [(feature, op, value)]
    feature: int = -1  # linear / shape
    center: float = 0.0  # linear: contribution = weight * (x - center)
    edges: np.ndarray = None  # shape: bin edges (len k+1); values[i] applies on (edges[i], edges[i+1]]
    values: np.ndarray = None  # shape: contribution per bin; curve: contribution at each grid point
    grid: np.ndarray = None  # curve: x positions, linearly interpolated, flat beyond the ends
    clip: tuple = None  # linear: x is clipped to (lo, hi) first (winsorized linear terms)
    tied: np.ndarray = None  # curve: per bin of ``edges`` (x <= edge goes left), interpolate over the full
    # grid when the bin is tied, else only between untied grid points (GPGam's handling of repeated values)
    support: float = np.nan  # fraction of samples the rule covers (filled from data when given)
    label: str = None
    idx: np.ndarray = None  # rows covered (rules, with data)

    @property
    def features(self):
        if self.kind == "rule":
            return [f for f, _, _ in self.conds]
        return [self.feature]

    def contribution(self, X):
        X = np.atleast_2d(X)
        if self.kind == "rule":
            return np.where(cond_mask(X, self.conds), self.weight, 0.0)
        x = X[:, self.feature]
        if self.kind == "linear":
            if self.clip is not None:
                x = np.clip(x, *self.clip)
            return self.weight * (x - self.center)
        if self.kind == "curve":
            full = np.interp(x, self.grid, self.values)
            if self.tied is None or self.tied.all() or not self.tied.any():
                return full
            bins = np.searchsorted(self.edges, x, side="right")
            keep = ~self.tied
            return np.where(self.tied[bins], full, np.interp(x, self.grid[keep], np.asarray(self.values)[keep]))
        # right-inclusive bins match tree splits (x <= t goes left)
        i = np.clip(np.searchsorted(self.edges, x, side="left") - 1, 0, len(self.values) - 1)
        return np.asarray(self.values)[i]


@dataclass
class AdditiveView:
    family: str  # "ruleset" | "scorecard" | "gam" | "linear"
    task: str
    feature_names: list
    model_name: str
    terms: list
    intercept: float = 0.0
    link: str = "identity"  # "identity" | "logistic" | "threshold" | "clip" (probability in [0, 1]) | "exp"
    threshold: float = 0.0
    link_scale: float = 1.0  # logistic: P = sigmoid(link_scale * score + link_offset)
    link_offset: float = 0.0
    class_names: list = None
    target_name: str = "target"
    score_name: str = "score"  # what the summed score means, e.g. "log-odds of malignant"
    X: np.ndarray = None
    y: np.ndarray = None
    feature_kind: dict = field(default_factory=dict)
    risk: list = None  # scorecards: [(score, probability)] for every reachable score
    note: str = ""  # how the model combines its terms, in words
    categories: dict = field(default_factory=dict)  # feature -> level names; X holds level codes

    @property
    def is_clf(self):
        return self.task == "classification"

    def score(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        return self.intercept + sum((t.contribution(X) for t in self.terms), np.zeros(len(X)))

    def output(self, s):
        """Score -> prediction: P(positive class) for classifiers, the value for regressors."""
        s = np.asarray(s, dtype=float)
        if self.link == "logistic":
            return 1 / (1 + np.exp(-(self.link_scale * s + self.link_offset)))
        if self.link == "clip":
            return np.clip(s, 0.0, 1.0)
        if self.link == "exp":  # log-link GLMs
            return np.exp(s)
        if self.link == "threshold":
            return (s > self.threshold).astype(float)
        return s

    def predict(self, X):
        out = self.output(self.score(X))
        return (out >= 0.5).astype(int) if self.is_clf else out

    def used_features(self):
        return sorted({f for t in self.terms for f in t.features})

    def cond_text(self, f, op, v, sig=3, negate=False):
        if negate:
            op = NEG[op]
        kind = self.feature_kind.get(f, "continuous")
        name = self.feature_names[f]
        if op in ("isnan", "notnan"):
            return f"{name} {SYM[op]}"
        if f in self.categories and op in ("==", "!="):
            return f"{name} {SYM[op]} {self.categories[f][int(v)]}"
        if kind == "binary" and op in ("<=", "<", ">", ">=") and 0 < v < 1:
            return f"{name} = {0 if op in ('<=', '<') else 1}"
        if kind == "binary" and op in ("==", "!=") and v in (0, 1):
            return f"{name} = {int(v) if op == '==' else 1 - int(v)}"
        if kind == "integer" and op in ("<=", ">"):
            k = int(np.floor(v))
            return f"{name} ≤ {fmt(k, 12)}" if op == "<=" else f"{name} ≥ {fmt(k + 1, 12)}"
        return f"{name} {SYM[op]} {fmt(v, sig if kind == 'continuous' else 12)}"

    def term_text(self, t, sig=3):
        if t.label:
            return t.label
        if t.kind == "rule":
            if not t.conds:
                return "always (default rule)"
            return " and ".join(self.cond_text(f, op, v, sig) for f, op, v in t.conds)
        return self.feature_names[t.feature]

    def importance(self, t):
        """How much a term moves predictions across the data (std of its contribution)."""
        if self.X is not None:
            return float(np.std(t.contribution(self.X)))
        if t.kind == "rule":
            s = t.support if np.isfinite(t.support) else 0.5
            return abs(t.weight) * np.sqrt(s * (1 - s))
        if t.kind in ("shape", "curve"):
            return float(np.ptp(t.values))
        return abs(t.weight)

    def attach(self, X, y=None):
        """Attach data: coverage of rules, feature kinds and labels."""
        if X is None:
            return self
        Xa = np.asarray(X, dtype=float)
        self.X = Xa
        self.feature_kind = {f: "categorical" if f in self.categories else _feature_kind(Xa[:, f])
                             for f in self.used_features()}
        for t in self.terms:
            if t.kind == "rule":
                m = cond_mask(Xa, t.conds)
                t.idx = np.flatnonzero(m)
                t.support = float(m.mean())
        if y is not None:
            ya = np.asarray(y)
            self.y = ya.astype(int if self.is_clf else float)
        return self
