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
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
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


class _Binarizer:
    """Indicator features: ``x <= t`` at up to ``n_thresholds`` quantiles of each numeric column,
    ``x = v`` for a two-valued column, one level per indicator for a categorical column (levels
    covering at least 1% of rows, at most 20), and ``x missing`` where the training data had
    missing values."""

    def __init__(self, n_thresholds=9):
        self.n_thresholds = n_thresholds

    def fit(self, frame):
        self.rules_ = []  # (column, kind, value, name)
        qs = np.arange(1, self.n_thresholds + 1) / (self.n_thresholds + 1)
        for c in frame.columns:
            col = frame[c]
            if not pd.api.types.is_numeric_dtype(col) or pd.api.types.is_bool_dtype(col):
                col = col.astype(object).where(col.notna(), "missing").astype(str)
                freq = col.value_counts()
                for level in list(freq[freq >= 0.01 * len(col)].index[:20]):
                    if 0 < freq[level] < len(col):
                        self.rules_.append((c, "eq_str", level, f"{c} = {level}"))
                continue
            x = pd.to_numeric(col, errors="coerce").to_numpy(float)
            miss = np.isnan(x)
            if miss.any() and not miss.all():
                self.rules_.append((c, "missing", None, f"{c} missing"))
            vals = np.unique(x[~miss])
            if len(vals) <= 1:
                continue
            if len(vals) == 2:
                self.rules_.append((c, "eq", vals[1], f"{c} = {vals[1]:g}"))
                continue
            for t in np.unique(np.quantile(x[~miss], qs, method="lower")):
                if t < vals[-1]:
                    self.rules_.append((c, "le", t, f"{c} <= {t:g}"))
        self.names_ = [r[3] for r in self.rules_]
        return self

    def transform(self, frame):
        out = np.zeros((len(frame), len(self.rules_)))
        for j, (c, kind, v, _) in enumerate(self.rules_):
            col = frame[c]
            if kind == "eq_str":
                out[:, j] = (col.astype(object).where(col.notna(), "missing").astype(str) == v).to_numpy()
                continue
            x = pd.to_numeric(col, errors="coerce").to_numpy(float)
            if kind == "missing":
                out[:, j] = np.isnan(x)
            elif kind == "eq":
                out[:, j] = x == v
            else:
                out[:, j] = x <= v  # NaN compares False
        return out


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
        reached (``stopped_early_`` is then True); on most data it finishes far sooner.

    Attributes
    ----------
    points_: dict
        Feature name -> points, for the features the score uses.
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
    def _frame(self, X):
        if isinstance(X, pd.DataFrame):
            frame = X.copy()
            frame.columns = [str(c) for c in frame.columns]
            return frame
        X = check_array(X, dtype=None, **_finite_check_kwarg(True))
        return pd.DataFrame(X, columns=list(self.feature_names_))

    def fit(self, X, y, feature_names=None):
        if not solver.HAVE_NUMBA:
            raise ImportError(NUMBA_HINT)
        if int(self.k) < 1 or int(self.max_points) < 1:
            raise ValueError("k and max_points must be at least 1")
        t0 = time.perf_counter()
        set_feature_names_in(self, X)
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = [str(c) for c in X.columns]
        else:
            n_cols = np.shape(X)[1] if len(np.shape(X)) == 2 else None
            if n_cols is None:
                X = check_array(X)  # raises the usual sklearn error for 1-D input
            self.feature_names_ = list(feature_names) if feature_names is not None else \
                [f"X{i}" for i in range(n_cols)]
        y = np.asarray(y)
        if y.ndim != 1 or len(y) != len(X):
            raise ValueError("y must be 1-D with one entry per row of X")
        check_binary_target(self, y)
        self.classes_, y01 = np.unique(y, return_inverse=True)
        if len(self.classes_) < 2:
            raise ValueError("FastRiskScoreClassifier needs two classes in y")
        frame = self._frame(X)
        self.n_features_in_ = frame.shape[1]
        if self.binarize:
            self.binarizer_ = _Binarizer(self.n_thresholds).fit(frame)
            B = self.binarizer_.transform(frame)
            self.features_ = list(self.binarizer_.names_)
        else:
            B = frame.to_numpy(float)
            if np.isnan(B).any():
                raise ValueError("binarize=False needs X without missing values")
            self.features_ = list(self.feature_names_)
        fine = self.binarize and int(self.n_thresholds) > DECILE_PROFILE_MAX
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
        self.stopped_early_ = bool(stopped)
        self.fit_seconds_ = time.perf_counter() - t0
        return self

    # --------------------------------------------------------------- prediction
    def _binary(self, X):
        check_predict_X(self, X)
        if not isinstance(X, pd.DataFrame):
            X = check_array(X, dtype=None, **_finite_check_kwarg(True))
            X = pd.DataFrame(X, columns=list(self.feature_names_))
        else:
            X = X.copy()
            X.columns = [str(c) for c in X.columns]
        if self.binarize:
            return self.binarizer_.transform(X)
        return X.to_numpy(float)

    def total_score(self, X):
        """The total points of each row."""
        check_is_fitted(self)
        B = self._binary(X)
        return B @ self.coef_ if B.shape[1] else np.zeros(B.shape[0])

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
