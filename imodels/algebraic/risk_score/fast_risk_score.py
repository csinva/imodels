"""FastRiskScore: sparse integer risk scores, found by autoresearch.

A risk score adds up a few integer points, one per feature that applies to a row,
and turns the total into a probability. FastRiskScoreClassifier picks at most ``k``
binary features (thresholds of numeric columns and levels of categorical ones)
and gives each an integer number of points in [-max_points, max_points], choosing
them to minimise the training log loss of the best map from total score to risk,

    min over points w, and real a, b, of  mean_i log(1 + exp(-y_i (a * x_i.w + b))).

This is the problem RiskSLIM and FasterRisk solve. The solver (``solver.py``) came out of
an autoresearch loop (agentic-imodels, evolve_slim) that started from FasterRisk; see
https://csinva.io/imodels/fastriskscore.html for the method and the benchmarks.
"""

from __future__ import annotations

import math
import numbers
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from scipy import sparse
from sklearn.utils.validation import check_array, check_is_fitted

from imodels.algebraic.risk_score import solver
from imodels.util.numba_compile import notify_first_compile
from imodels.util.arguments import (_finite_check_kwarg, check_binary_target, check_predict_X,
                                    set_feature_names_in)

NUMBA_HINT = (
    "FastRiskScoreClassifier needs numba, which is not installed. Install it with "
    "`pip install numba` (or `pip install imodels[optional]`). The search is compiled "
    "with numba once per machine (about two minutes) and cached after that."
)

DECILE_PROFILE_MAX = 19  # n_thresholds above this use the "fine" profile (measured: "decile" best at 9, "fine" at 99)


def calibrate(scores, y):
    """(a, b, loss): the minimum over real a, b of the mean log loss of sigmoid(a * score + b).

    Convex in (a, b); solved by damped Newton on the distinct score values."""
    y = np.asarray(y, float)
    u, inv = np.unique(np.round(np.asarray(scores, float), 9), return_inverse=True)
    pos = np.bincount(inv.ravel(), weights=y, minlength=len(u))
    cnt = np.bincount(inv.ravel(), minlength=len(u)).astype(float)
    n = cnt.sum()
    p = min(max(pos.sum() / n, 1e-12), 1 - 1e-12)
    a, b = 0.0, math.log(p / (1 - p))

    def f(a_, b_):
        z = a_ * u + b_
        return float((pos * np.logaddexp(0, -z) + (cnt - pos) * np.logaddexp(0, z)).sum() / n)

    cur = f(a, b)
    if len(u) > 1:
        for _ in range(200):
            q = 1 / (1 + np.exp(-(a * u + b)))
            r = cnt * q - pos
            w = cnt * q * (1 - q)
            g = np.array([r @ u, r.sum()]) / n
            H = np.array([[w @ (u * u), w @ u], [w @ u, w.sum()]]) / n + 1e-12 * np.eye(2)
            step = np.linalg.solve(H, g)
            t = 1.0
            while t > 1e-10:
                na, nb_ = a - t * step[0], b - t * step[1]
                new = f(na, nb_)
                if new <= cur - 1e-4 * t * float(g @ step):
                    break
                t *= 0.5
            if t <= 1e-10 or cur - new < 1e-13:
                if t > 1e-10:
                    a, b, cur = na, nb_, new
                break
            a, b, cur = na, nb_, new
    return a, b, cur


_MISSING = object()  # stands for a missing value among the levels of a categorical column; equal to no string


def _threshold(t, nxt):
    """(value, text) for the rule ``x <= t`` when the next value of x in the training data is ``nxt``: a short
    decimal d with t <= d < nxt, so that ``x <= d`` marks the same training rows as ``x <= t``, and the rule
    is applied with d, exactly as printed. It is the 6-significant-digit form of t when that lies in the
    gap, else the form with the fewest more digits that does, else ``repr(t)``. Thresholds of one column thus
    get distinct names however large or close their values are."""
    t, nxt = float(t), float(nxt)
    for text in [f"{t:g}"] + [f"{t:.{p}g}" for p in range(7, 18)]:
        d = float(text)
        if t <= d < nxt:
            return d, text
    return t, repr(t)


def _value_text(v):
    """The 6-significant-digit form of v when it reads back as v, else ``repr``."""
    v = float(v)
    text = f"{v:g}"
    return text if float(text) == v else repr(v)


def _unique_names(names):
    """Make names unique by appending " (2)", " (3)", ... to repeats (columns whose names collide with a rule
    of another column, e.g. a column called "a <= 1")."""
    seen, out = set(), []
    for name in names:
        new, i = name, 1
        while new in seen:
            i += 1
            new = f"{name} ({i})"
        seen.add(new)
        out.append(new)
    return out


class _Binarizer:
    """Indicator features: ``x <= t`` at up to ``n_thresholds`` quantiles of each numeric column,
    ``x = v`` for a two-valued column, one level per indicator for a categorical column (levels
    covering at least 1% of rows, at most 20), and ``x missing`` where the training data had
    missing values (for a categorical column, a missing value is never the same as a level called
    "missing"). Columns are used by position. A missing value at predict time in a numeric column
    without missing values in the training data is scored as above every threshold."""

    def __init__(self, n_thresholds=9):
        self.n_thresholds = n_thresholds

    def fit(self, frame, names=None):
        """``names``: the column names used in the rule names (default: those of ``frame``)."""
        names = [str(c) for c in frame.columns] if names is None else list(names)
        rules = []  # (column position, kind, value, name)
        self.categorical_ = []  # per column: its rules compare string levels
        qs = np.arange(1, self.n_thresholds + 1) / (self.n_thresholds + 1)
        for i, c in enumerate(names):
            col = frame.iloc[:, i]
            self.categorical_.append(not pd.api.types.is_numeric_dtype(col) or pd.api.types.is_bool_dtype(col))
            if self.categorical_[-1]:
                levels = _levels(col)
                levels[col.isna().to_numpy()] = _MISSING
                freq = pd.Series(levels, dtype=object).value_counts()
                for level in list(freq[freq >= 0.01 * len(col)].index[:20]):
                    if 0 < freq[level] < len(col):
                        if level is _MISSING:
                            rules.append((i, "missing", None, f"{c} missing"))
                        else:
                            rules.append((i, "eq_str", level, f"{c} = {level}"))
                continue
            x = pd.to_numeric(col, errors="coerce").to_numpy(float)
            miss = np.isnan(x)
            if miss.any() and not miss.all():
                rules.append((i, "missing", None, f"{c} missing"))
            vals = np.unique(x[~miss])
            if len(vals) <= 1:
                continue
            if len(vals) == 2:
                rules.append((i, "eq", vals[1], f"{c} = {_value_text(vals[1])}"))
                continue
            ts = np.unique(np.quantile(x[~miss], qs, method="lower"))
            ts = ts[ts < vals[-1]]
            nxt = vals[np.searchsorted(vals, ts, side="right")]
            for t, n in zip(ts, nxt):
                d, text = _threshold(t, n)
                rules.append((i, "le", d, f"{c} <= {text}"))
        names = _unique_names([r[3] for r in rules])
        self.rules_ = [r[:3] + (name,) for r, name in zip(rules, names)]
        self.names_ = names
        assert len(set(self.names_)) == len(self.names_)
        return self

    def transform(self, frame, rules=None):
        """Boolean matrix of the rules (all, or the indices ``rules``) for the rows of ``frame``. Each column
        is converted once and all its thresholds are compared at once."""
        rules = np.arange(len(self.rules_)) if rules is None else np.asarray(rules, dtype=int)
        out = np.zeros((len(frame), len(rules)), dtype=bool)
        by_col = {}
        for pos, j in enumerate(rules):
            c, kind, v, _ = self.rules_[j]
            by_col.setdefault(c, {}).setdefault(kind, []).append((pos, v))
        for c, kinds in by_col.items():
            col = frame.iloc[:, c]
            if self.categorical_[c]:
                miss = col.isna().to_numpy()
                if "missing" in kinds:
                    out[:, kinds["missing"][0][0]] = miss
                if "eq_str" in kinds:
                    pos, levels = map(list, zip(*kinds["eq_str"]))
                    codes = pd.Categorical(_levels(col), categories=levels).codes.copy()
                    codes[miss] = -1  # a missing value is no level, even one whose string form is a level
                    out[:, _block(pos)] = codes[:, None] == np.arange(len(levels))[None, :]
                continue
            x = pd.to_numeric(col, errors="coerce").to_numpy(float)
            for kind, items in kinds.items():
                pos, vals = map(list, zip(*items))
                if kind == "missing":
                    out[:, pos[0]] = np.isnan(x)
                elif kind == "eq":
                    out[:, pos[0]] = x == vals[0]
                else:
                    out[:, _block(pos)] = x[:, None] <= np.asarray(vals, float)[None, :]  # NaN compares False
        return out


def _block(pos):
    """Output columns ``pos`` as a slice when they are consecutive (writing a slice is much faster than a list)."""
    if pos[-1] - pos[0] == len(pos) - 1 and all(b - a == 1 for a, b in zip(pos, pos[1:])):
        return slice(pos[0], pos[-1] + 1)
    return pos


def _levels(col):
    """The values of a categorical column as strings (missing rows are masked by the callers)."""
    return col.astype(object).astype(str).to_numpy(object)


class FastRiskScoreClassifier(ClassifierMixin, BaseEstimator):
    """Sparse integer risk score (binary classification).

    The columns of X are turned into binary features (thresholds of numeric columns, levels of
    categorical ones), and the solver picks at most ``k`` of them with integer points. It runs a
    continuous beam search over supports (as FasterRisk does), rounds the best supports at many
    scales keeping the rounding with the smallest calibrated loss, and improves the best of them
    by an integer local search (value changes, additions, threshold slides and swaps, ranked by
    an estimate of the calibrated loss and checked exactly), refit swaps and, on near-separable
    data, an exact polish. The solver comes from an autoresearch loop (agentic-imodels, evolve_slim
    runs) that started from FasterRisk; the final version is n17_lean of run oct05-decile2. It has
    two settings profiles: "decile" (tuned with 9 thresholds per numeric column) and "fine" (99),
    chosen from ``n_thresholds``.

    Parameters
    ----------
    k: int
        The most features the score may use.
    max_points: int
        Every feature's points are an integer in [-max_points, max_points].
    n_thresholds: int
        Number of quantile thresholds per numeric column: 9 splits at the deciles, 99 at the
        percentiles (``x <= t`` indicators). More than 19 selects the solver's "fine" profile.
    binarize: bool
        If False, the columns of X are used as they are (they should then be binary or small
        integers), and the points apply to them directly; the solver then uses the "decile" profile.
    time_limit: float
        Seconds for the search. It stops early and returns its best points if the limit is
        reached (``stopped_early_`` is then True); on most data it finishes far sooner. Turning
        X into binary features before the search (and the one-time numba compilation) is not
        counted; ``fit_seconds_`` is the whole fit.

    Attributes
    ----------
    points_: dict
        Feature name -> points, for the features the score uses. Every binary feature has its
        own name: a threshold is printed with as many digits as it takes to tell it from the
        next value in the training data, and is applied as printed.
    coef_: ndarray
        Points for every binary feature (``features_``), zero for those not used.
    scale_, intercept_: float
        The risk of a row with total score s is ``1 / (1 + exp(-(scale_ * s + intercept_)))``.
    train_loss_: float
        Mean training log loss of that risk.
    profile_: str
        The solver's settings profile ("decile" or "fine").
    """

    def __init__(self, k: int = 5, max_points: int = 5, n_thresholds: int = 9, binarize: bool = True,
                 time_limit: float = 60.0):
        if not solver.HAVE_NUMBA:
            # say so when the model is built, as for the other models with optional dependencies;
            # fit raises the same message, since there is no pure-Python fallback
            warnings.warn(NUMBA_HINT + " Fitting will raise an ImportError until it is installed.")
        self.k = k
        self.max_points = max_points
        self.n_thresholds = n_thresholds
        self.binarize = binarize
        self.time_limit = time_limit

    # ------------------------------------------------------------------ fitting
    def _check_params(self):
        for name in ("k", "max_points", "n_thresholds"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, numbers.Integral) or v < 1:
                raise ValueError(f"{name} must be an integer of at least 1, got {v!r}")
        v = self.time_limit
        if isinstance(v, bool) or not isinstance(v, numbers.Real) or not (math.isfinite(v) and v > 0):
            raise ValueError(f"time_limit must be a positive finite number of seconds, got {v!r}")

    def _array(self, X):
        """X (not a DataFrame) as a 2-D array, by sklearn's check_array; sparse X is refused."""
        if sparse.issparse(X):
            raise TypeError(f"{type(self).__name__} does not support sparse X; pass a dense array "
                            "(X.toarray()) or a DataFrame")
        return check_array(X, dtype=None, accept_sparse=False, **_finite_check_kwarg(True))

    def fit(self, X, y, feature_names=None):
        if not solver.HAVE_NUMBA:
            raise ImportError(NUMBA_HINT)
        self._check_params()
        t0 = time.perf_counter()
        set_feature_names_in(self, X)  # also deletes a feature_names_in_ of an earlier fit on a DataFrame
        if isinstance(X, pd.DataFrame):
            frame = X
            self.feature_names_ = [str(c) for c in X.columns]
        else:
            X = self._array(X)
            frame = pd.DataFrame(X)
            if feature_names is not None and len(feature_names) != X.shape[1]:
                raise ValueError(f"feature_names has {len(feature_names)} names but X has {X.shape[1]} columns")
            self.feature_names_ = [str(c) for c in feature_names] if feature_names is not None else \
                [f"X{i}" for i in range(X.shape[1])]
        y = np.asarray(y)
        if y.ndim != 1 or len(y) != len(frame):
            raise ValueError("y must be 1-D with one entry per row of X")
        check_binary_target(self, y)
        self.classes_, y01 = np.unique(y, return_inverse=True)
        if len(self.classes_) < 2:
            raise ValueError("FastRiskScoreClassifier needs two classes in y")
        self.n_features_in_ = frame.shape[1]
        if self.binarize:
            self.binarizer_ = _Binarizer(self.n_thresholds).fit(frame, self.feature_names_)
            B = self.binarizer_.transform(frame)
            self.features_ = list(self.binarizer_.names_)
        else:
            B = frame.to_numpy(float)
            if np.isnan(B).any():
                raise ValueError("binarize=False needs X without missing values")
            self.features_ = list(self.feature_names_)
        fine = self.binarize and self.n_thresholds > DECILE_PROFILE_MAX
        self.profile_ = "fine" if fine else "decile"
        if B.shape[1] == 0:
            points, stopped = np.zeros(0), False
        else:
            if not solver._WARM:  # the first fit in this process compiles the search (or loads it)
                notify_first_compile(solver.__file__, "FastRiskScoreClassifier", "about 2 minutes",
                                     solver.NUMBA_CACHE, "RISKSCORE_NUMBA_CACHE")
            points, _, _, stopped = solver.solve(B, y01, int(self.k), bound=int(self.max_points),
                                                 time_limit=float(self.time_limit), profile=self.profile_)
        points = points.astype(np.int64)
        a, b, loss = calibrate(B @ points if len(points) else np.zeros(len(y01)), y01)
        if a < 0:  # orient the score so that more points always means more risk
            points, a = -points, -a
        self.coef_ = points
        self.scale_, self.intercept_, self.train_loss_ = float(a), float(b), float(loss)
        self.points_ = {self.features_[j]: int(points[j]) for j in np.flatnonzero(points)}
        assert len(self.points_) == np.count_nonzero(points)
        self.stopped_early_ = bool(stopped)
        self.fit_seconds_ = time.perf_counter() - t0
        return self

    # --------------------------------------------------------------- prediction
    def _frame(self, X):
        """X checked against the fit and as a DataFrame whose columns are used by position (as sklearn does)."""
        check_predict_X(self, X)
        if isinstance(X, pd.DataFrame):
            if getattr(self, "feature_names_in_", None) is None and X.shape[1] and \
                    all(isinstance(c, str) for c in X.columns):
                warnings.warn(f"X has feature names, but {type(self).__name__} was fitted without feature "
                              "names", UserWarning)
            return X
        X = self._array(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, but {type(self).__name__} is expecting "
                             f"{self.n_features_in_} features as input.")
        return pd.DataFrame(X)

    def total_score(self, X):
        """The total points of each row."""
        check_is_fitted(self)
        frame = self._frame(X)
        used = np.flatnonzero(self.coef_)  # only the features the score uses are computed
        if self.binarize:
            B = self.binarizer_.transform(frame, used)
        else:
            B = frame.iloc[:, used].to_numpy(float)
        return (B @ self.coef_[used]).astype(float) if len(used) else np.zeros(len(frame))

    def decision_function(self, X):
        s = self.total_score(X)
        return self.scale_ * s + self.intercept_

    def predict_proba(self, X):
        p = 1.0 / (1.0 + np.exp(-self.decision_function(X)))
        return np.column_stack([1 - p, p])

    def predict(self, X):
        z = self.decision_function(X)
        return self.classes_[(z > 0).astype(int)]

    def risk(self, score):
        """Probability of the second class for a total score."""
        return 1.0 / (1.0 + np.exp(-(self.scale_ * np.asarray(score, float) + self.intercept_)))

    # ----------------------------------------------------------------- display
    def __str__(self):
        if not hasattr(self, "coef_"):
            return "FastRiskScoreClassifier (not fitted)"
        if not self.points_:
            return (f"FastRiskScoreClassifier: no feature helps; the risk of "
                    f"class {self.classes_[1]} is {self.risk(0):.1%} for every row")
        width = max(len(f) for f in self.points_)
        lines = [f"FastRiskScoreClassifier: add the points of every line that applies"]
        for name, v in sorted(self.points_.items(), key=lambda t: -t[1]):
            lines.append(f"  {name:<{width}}  {v:+d}")
        pos = sum(v for v in self.points_.values() if v > 0)
        neg = sum(v for v in self.points_.values() if v < 0)
        scores = np.arange(neg, pos + 1)
        lines.append(f"risk of class {self.classes_[1]} for each total score:")
        cells = [(f"{s:+d}" if s else "0", f"{self.risk(s):.1%}") for s in scores]
        cw = max(max(len(c[0]), len(c[1])) for c in cells) + 1
        lines.append("  score " + "".join(c[0].rjust(cw) for c in cells))
        lines.append("  risk  " + "".join(c[1].rjust(cw) for c in cells))
        return "\n".join(lines)
