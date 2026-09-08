"""Shared result container for sparse-pruning regularization paths."""
from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np


def _real_array_snapshot(values: Any, name: str) -> np.ndarray:
    """Own validated result storage so later caller edits cannot alter a path."""
    try:
        if np.iscomplexobj(values):
            raise ValueError("complex values are not supported")
        array = np.array(values, dtype=float, copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain real numeric values") from exc
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class RegularizationPath:
    """A coefficient path indexed by nonincreasing regularization strengths.

    Numeric arrays are copied into read-only snapshots at construction so
    changing the inputs cannot invalidate the stored path or its certificates.

    Parameters
    ----------
    lambdas:
        Regularization strengths in nonincreasing order.
    coefficients:
        Coefficient matrix with one row per value in ``lambdas``.
    intercepts:
        Optional intercept at each path point.  Solvers that receive an
        explicit intercept column may leave this as ``None``.
    penalties:
        Optional values of the unscaled structured penalty.
    exact:
        Whether linear interpolation describes the complete path between
        adjacent points.  This is true for a certified hiCAP homotopy path and
        false for a sampled APA-APG path.
    """

    lambdas: np.ndarray
    coefficients: np.ndarray
    intercepts: np.ndarray | None = None
    penalties: np.ndarray | None = None
    events: Sequence[Any] = field(default_factory=tuple)
    diagnostics: Sequence[Mapping[str, Any] | None] = field(default_factory=tuple)
    method: str = "unknown"
    exact: bool = False
    status: str = "complete"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        lambdas = _real_array_snapshot(self.lambdas, "lambdas")
        coefficients = _real_array_snapshot(self.coefficients, "coefficients")
        if lambdas.ndim != 1 or lambdas.size == 0:
            raise ValueError("lambdas must be a non-empty one-dimensional array")
        if coefficients.ndim != 2 or coefficients.shape[0] != lambdas.size:
            raise ValueError(
                "coefficients must be two-dimensional with one row per lambda"
            )
        if not np.all(np.isfinite(lambdas)) or np.any(lambdas < 0):
            raise ValueError("lambdas must be finite and nonnegative")
        if not np.all(np.isfinite(coefficients)):
            raise ValueError("coefficients must contain only finite values")
        if np.any(np.diff(lambdas) > 0):
            raise ValueError("lambdas must be in nonincreasing order")

        intercepts = self.intercepts
        if intercepts is not None:
            intercepts = _real_array_snapshot(intercepts, "intercepts")
            if intercepts.shape != lambdas.shape or not np.all(
                np.isfinite(intercepts)
            ):
                raise ValueError("intercepts must be finite with one value per lambda")

        penalties = self.penalties
        if penalties is not None:
            penalties = _real_array_snapshot(penalties, "penalties")
            if penalties.shape != lambdas.shape or not np.all(
                np.isfinite(penalties)
            ):
                raise ValueError("penalties must be finite with one value per lambda")

        events = tuple(self.events)
        diagnostics = tuple(self.diagnostics)
        if events and len(events) != lambdas.size:
            raise ValueError("events must be empty or have one entry per lambda")
        if diagnostics and len(diagnostics) != lambdas.size:
            raise ValueError("diagnostics must be empty or have one entry per lambda")

        object.__setattr__(self, "lambdas", lambdas)
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "intercepts", intercepts)
        object.__setattr__(self, "penalties", penalties)
        object.__setattr__(self, "events", events)
        object.__setattr__(self, "diagnostics", diagnostics)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def n_points(self) -> int:
        """Number of stored path points."""

        return self.lambdas.size

    def at(self, lam: float) -> tuple[np.ndarray, float | None]:
        """Linearly interpolate coefficients and the optional intercept.

        For sampled paths this is a numerical interpolation only.  Callers
        requiring an exact in-between solution should first check ``exact``.
        Lookup costs O(log K + p) for K stored knots and p coefficients;
        only the adjacent coefficient rows are accessed.
        """

        if isinstance(lam, (bool, np.bool_)) or not isinstance(
            lam, (int, float, np.integer, np.floating)
        ):
            raise ValueError("lam must be a nonnegative finite scalar")
        try:
            lam = float(lam)
        except (ValueError, OverflowError) as exc:
            raise ValueError("lam must be a nonnegative finite scalar") from exc
        if not np.isfinite(lam) or lam < 0:
            raise ValueError("lam must be a nonnegative finite scalar")
        upper = float(self.lambdas[0])
        lower = float(self.lambdas[-1])
        # Allow endpoint roundoff on each endpoint's own scale. An absolute
        # epsilon floor would accept lambda zero for tiny incomplete prefixes.
        relative_tolerance = 16 * np.finfo(float).eps
        upper_tolerance = relative_tolerance * upper
        lower_tolerance = relative_tolerance * lower
        if (lam > upper and lam - upper > upper_tolerance) or (
            lam < lower and lower - lam > lower_tolerance
        ):
            raise ValueError(
                f"lam={lam} lies outside the stored path [{lower}, {upper}]"
            )
        lam = min(max(lam, lower), upper)

        # Reversal is a view. Binary search preserves the previous np.unique
        # convention: use the first occurrence in increasing lambda order,
        # which is the last row of a duplicate batch in stored order.
        xp = self.lambdas[::-1]
        upper_position = bisect_left(xp, lam)
        upper_row = self.n_points - 1 - upper_position
        if xp[upper_position] == lam:
            intercept = (
                None if self.intercepts is None
                else float(self.intercepts[upper_row])
            )
            return self.coefficients[upper_row].copy(), intercept

        # The entry immediately below the insertion point may be the last
        # occurrence of a duplicate batch. Find its canonical first occurrence
        # without scanning the batch or copying the full coefficient matrix.
        lower_position = bisect_left(xp, xp[upper_position - 1])
        lower_row = self.n_points - 1 - lower_position
        fraction = float(
            (lam - xp[lower_position])
            / (xp[upper_position] - xp[lower_position])
        )
        beta = (
            (1.0 - fraction) * self.coefficients[lower_row]
            + fraction * self.coefficients[upper_row]
        )
        intercept = None
        if self.intercepts is not None:
            intercept = float(
                (1.0 - fraction) * self.intercepts[lower_row]
                + fraction * self.intercepts[upper_row]
            )
        return beta, intercept
