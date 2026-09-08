"""Cached sufficient statistics for normalized weighted squared loss."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class _QuadraticRegressionLoss:
    """Cached sufficient statistics for normalized weighted squared loss."""

    gram: np.ndarray | None
    linear: np.ndarray
    response_squared: float
    diagonal: np.ndarray | None
    lipschitz_constant: float
    max_off_diagonal: float | None
    max_off_diagonal_correlation: float | None
    weight_sum: float
    assumed_diagonal_gram: bool = False

    @property
    def backend(self) -> str:
        return "diagonal" if self.diagonal is not None else "gram"

    def matvec(self, beta: np.ndarray) -> np.ndarray:
        if self.diagonal is not None:
            return self.diagonal * beta
        if self.gram is None:  # pragma: no cover - invalid internal state
            raise RuntimeError("quadratic loss has neither Gram nor diagonal")
        return self.gram @ beta

    def loss(self, beta: np.ndarray) -> float:
        gram_beta = self.matvec(beta)
        value = 0.5 * float(
            beta @ gram_beta
            - 2.0 * self.linear @ beta
            + self.response_squared
        )
        scale = 0.5 * (
            abs(float(beta @ gram_beta))
            + 2.0 * abs(float(self.linear @ beta))
            + abs(self.response_squared)
        )
        if value < -128.0 * np.finfo(float).eps * max(1.0, scale):
            raise FloatingPointError(
                "cached quadratic loss became materially negative"
            )
        # The expression is a squared norm, but cancellation can leave a tiny
        # negative value at an exactly interpolating solution.
        return max(0.0, value)


def _make_quadratic_regression_loss(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    *,
    assume_diagonal_gram: bool = False,
) -> _QuadraticRegressionLoss:
    """Cache sufficient statistics for a normalized weighted quadratic loss.

    By default, the full Gram matrix is formed and numerical diagonality is
    verified before the diagonal backend is selected. Setting
    ``assume_diagonal_gram=True`` is an explicit trust-the-caller fast path:
    only ``diag(X'WX)``, ``X'Wy``, and ``y'Wy`` are computed. This uses
    ``O(p)`` storage instead of ``O(p**2)``, but is correct only when the
    weighted Gram matrix is known to be diagonal under these exact rows and
    weights. Unrepresentable sufficient statistics raise ``ValueError``;
    callers must rescale the problem rather than use a nonfinite cache.
    """
    if not isinstance(assume_diagonal_gram, (bool, np.bool_)):
        raise ValueError("assume_diagonal_gram must be a boolean")

    with np.errstate(over="ignore", invalid="ignore"):
        weight_sum = float(weights.sum())
    if not np.isfinite(weight_sum) or weight_sum <= 0:
        raise ValueError("quadratic loss requires a positive representable weight sum")
    normalized_weights = weights / weight_sum
    with np.errstate(over="ignore", invalid="ignore"):
        weighted_y = normalized_weights * y
        linear = X.T @ weighted_y
        response_squared = float(y @ weighted_y)
    if not np.all(np.isfinite(linear)) or not np.isfinite(response_squared):
        raise ValueError("quadratic response statistics are not representable; rescale X/y")

    if assume_diagonal_gram:
        with np.errstate(over="ignore", invalid="ignore"):
            diagonal = np.einsum(
                "ij,i,ij->j", X, normalized_weights, X, optimize=True
            )
        if not np.all(np.isfinite(diagonal)):
            raise ValueError("quadratic Gram diagonal is not representable; rescale X")
        return _QuadraticRegressionLoss(
            gram=None,
            linear=np.asarray(linear, dtype=float),
            response_squared=response_squared,
            diagonal=np.asarray(diagonal, dtype=float),
            lipschitz_constant=float(np.max(diagonal, initial=0.0)),
            max_off_diagonal=None,
            max_off_diagonal_correlation=None,
            weight_sum=weight_sum,
            assumed_diagonal_gram=True,
        )

    with np.errstate(over="ignore", invalid="ignore"):
        gram = X.T @ (normalized_weights[:, np.newaxis] * X)
        # Averaging by the difference preserves equal subnormal entries and
        # avoids overflowing a representable diagonal by doubling it first.
        gram = gram + 0.5 * (gram.T - gram)
    if not np.all(np.isfinite(gram)):
        raise ValueError("quadratic Gram matrix is not representable; rescale X")
    diagonal = np.diag(gram).copy()
    off_diagonal = gram.copy()
    np.fill_diagonal(off_diagonal, 0.0)
    max_off_diagonal = float(np.max(np.abs(off_diagonal), initial=0.0))
    column_norms = np.sqrt(np.maximum(diagonal, 0.0))
    correlation = np.abs(off_diagonal)
    # Divide separately: forming diag_i * diag_j can overflow (or underflow)
    # even when both Gram entries and the true correlation are representable.
    np.divide(
        correlation, column_norms[:, None], out=correlation,
        where=column_norms[:, None] > 0,
    )
    np.divide(
        correlation, column_norms[None, :], out=correlation,
        where=column_norms[None, :] > 0,
    )
    positive_scale = (column_norms[:, None] > 0) & (column_norms[None, :] > 0)
    correlation[~positive_scale & (off_diagonal != 0)] = np.inf
    max_off_diagonal_correlation = float(
        np.max(correlation, initial=0.0)
    )
    diagonal_tolerance = 32.0 * np.finfo(float).eps * max(1, gram.shape[0])
    if max_off_diagonal_correlation <= diagonal_tolerance:
        gram_diagonal: np.ndarray | None = diagonal
        lipschitz_constant = float(np.max(diagonal, initial=0.0))
    else:
        gram_diagonal = None
        with np.errstate(over="ignore", invalid="ignore"):
            largest_eigenvalue = float(np.linalg.eigvalsh(gram)[-1])
        if not np.isfinite(largest_eigenvalue):
            raise ValueError("quadratic spectral norm is not representable; rescale X")
        lipschitz_constant = max(0.0, largest_eigenvalue)
    return _QuadraticRegressionLoss(
        gram=np.asarray(gram, dtype=float),
        linear=np.asarray(linear, dtype=float),
        response_squared=response_squared,
        diagonal=gram_diagonal,
        lipschitz_constant=lipschitz_constant,
        max_off_diagonal=max_off_diagonal,
        max_off_diagonal_correlation=max_off_diagonal_correlation,
        weight_sum=weight_sum,
        assumed_diagonal_gram=False,
    )
