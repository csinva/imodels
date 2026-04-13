"""Optimization utilities for SparseForestry models."""
from __future__ import annotations

import numpy as np
from numpy.linalg import norm
from scipy.optimize import minimize
from sklearn.ensemble._forest import _generate_sample_indices, _generate_unsampled_indices


def proj_l1_ball(u: np.ndarray, tau: float) -> np.ndarray:
    if tau <= 0:
        return np.zeros_like(u)
    if np.sum(np.abs(u)) <= tau:
        return u.copy()

    abs_u = np.abs(u)
    s = -np.sort(-abs_u)
    css = np.cumsum(s)
    js = np.arange(1, len(s) + 1)
    cond = s - (css - tau) / js
    rho = np.nonzero(cond > 0)[0][-1]
    theta = (css[rho] - tau) / (rho + 1)
    return np.sign(u) * np.maximum(abs_u - theta, 0.0)


def hiCAP_regression(
    X: np.ndarray,
    y: np.ndarray,
    groups: list[np.ndarray],
    lam: float,
    beta_init: np.ndarray | None = None,
    gamma1: float = 1.0,
    a: float = 1.0,
    max_iter: int = 500,
    tol: float = 1e-6,
    ord: int | str = 2,
    verbose: bool = False,
) -> np.ndarray:
    n, d = X.shape
    K = len(groups)
    if K == 0:
        # No penalized groups (e.g., tree with no internal nodes).
        return np.zeros(d) if beta_init is None else beta_init.copy()
    alphas = [1 / K] * K + [0.0]
    groups = groups + [np.array(0)]

    sigma_max = norm(X, 2)
    Lf = (sigma_max**2) / (lam * K)

    counts = np.zeros(d)
    for G in groups:
        counts[G] += 1

    beta = np.zeros(d) if beta_init is None else beta_init.copy()
    beta_tilde = beta.copy()

    for k in range(max_iter):
        tau = 1.0 / (k + a)
        gamma = min(gamma1 * a / (k + a), 1.0 / Lf)
        beta_hat = (1 - tau) * beta + tau * beta_tilde
        grad = (X.T @ (X.dot(beta_hat) - y)) / (lam * K)
        v = beta_hat - gamma * grad
        beta_next = np.zeros(d)

        for alpha, G in zip(alphas, groups):
            vG = v[G]
            nG = norm(vG)
            if ord == 2:
                proxG = max(1 - (alpha * gamma) / nG, 0.0) * vG if nG > 0 else 0.0 * vG
            else:
                proxG = vG - proj_l1_ball(vG, alpha * gamma)
            beta_next[G] += proxG

        beta_next /= counts
        beta_tilde += (1.0 / tau) * (2 - gamma * Lf) * (beta_next - beta_hat)

        if norm(beta_next - beta) < tol:
            if verbose:
                print(f"Converged after {k + 1} iterations")
            beta = beta_next
            break

        beta = beta_next

    return beta


def hiCAP_classification(
    X: np.ndarray,
    y: np.ndarray,
    groups: list[np.ndarray],
    lam: float,
    beta_init: np.ndarray | None = None,
    gamma1: float = 1.0,
    a: float = 1.0,
    max_iter: int = 500,
    tol: float = 1e-6,
    ord: int | str = 2,
    verbose: bool = False,
) -> np.ndarray:
    n, d = X.shape
    K = len(groups)
    if K == 0:
        # No penalized groups (e.g., tree with no internal nodes).
        return np.zeros(d) if beta_init is None else beta_init.copy()
    alphas = [1 / K] * K + [0.0]
    groups = groups + [np.array(0)]

    sigma_max = norm(X, 2)
    Lf = (sigma_max**2) / (4 * lam * K)

    counts = np.zeros(d)
    for G in groups:
        counts[G] += 1

    beta = np.zeros(d) if beta_init is None else beta_init.copy()
    beta_tilde = beta.copy()

    for k in range(max_iter):
        tau = 1.0 / (k + a)
        gamma = min(gamma1 * a / (k + a), 1.0 / Lf)
        beta_hat = (1 - tau) * beta + tau * beta_tilde

        sigmoid = 1.0 / (1.0 + np.exp(-1 * (X.dot(beta_hat))))
        grad = (X.T @ (sigmoid - y)) / (lam * K)
        v = beta_hat - gamma * grad
        beta_next = np.zeros(d)

        for alpha, G in zip(alphas, groups):
            vG = v[G]
            nG = norm(vG)
            if ord == 2:
                proxG = max(1 - (alpha * gamma) / nG, 0.0) * vG if nG > 0 else 0.0 * vG
            else:
                proxG = vG - proj_l1_ball(vG, alpha * gamma)
            beta_next[G] += proxG

        beta_next /= counts
        beta_tilde += (1.0 / tau) * (2 - gamma * Lf) * (beta_next - beta_hat)

        if norm(beta_next - beta) < tol:
            if verbose:
                print(f"Converged after {k + 1} iterations")
            beta = beta_next
            break

        beta = beta_next

    return beta


def get_gcv_reg_param(tree, X: np.ndarray, y: np.ndarray) -> float:
    def vectorized_reorganize_row(row: np.ndarray) -> np.ndarray:
        non_zero_indices = row.nonzero()[0]
        non_zero_values = row[non_zero_indices]
        new_row = np.zeros_like(row)
        new_row[: len(non_zero_values)] = non_zero_values
        return new_row

    d_path = tree.decision_path(X.astype(np.float32)).toarray()
    node_obs = tree.n_node_samples.reshape(-1,)
    values = np.einsum("nl, l -> nl", d_path, tree.value.reshape(-1,))
    num_obs = np.einsum("nl, l -> nl", d_path, node_obs)
    seq_len = np.sum(d_path, axis=1)

    values = np.apply_along_axis(vectorized_reorganize_row, axis=1, arr=values)
    num_obs = np.apply_along_axis(vectorized_reorganize_row, axis=1, arr=num_obs)

    def gcv_opt(lam):
        seq_n = np.arange(len(values))
        root_contribution = values[:, 0]
        node_contribution = np.sum(
            (np.hstack((np.diff(values), np.zeros(len(seq_n)).reshape(-1, 1))) * num_obs)
            / (num_obs + lam),
            axis=1,
        )
        leaf_contribution = values[seq_n, seq_len - 1] * num_obs[seq_n, seq_len - 1] / (
            num_obs[seq_n, seq_len - 1] + lam
        )
        y_hat = root_contribution + node_contribution + leaf_contribution
        error = np.mean((y - y_hat) ** 2)
        denom = (1 - np.mean(node_obs / (lam + node_obs))) ** 2
        return error / denom

    return minimize(gcv_opt, 1)["x"][0]


def get_reg_set(prune_set: str, X: np.ndarray, y: np.ndarray, random_state, n_samples: int, n_samples_bootstrap: int):
    if prune_set == "ib":
        sampled_indices = _generate_sample_indices(random_state, n_samples, n_samples_bootstrap)
        X_prune = X[sampled_indices, :]
        y_prune = y[sampled_indices]
    elif prune_set == "full":
        X_prune = X
        y_prune = y
    else:
        unsampled_indices = _generate_unsampled_indices(random_state, n_samples, n_samples_bootstrap)
        X_prune = X[unsampled_indices, :]
        y_prune = y[unsampled_indices]
    return X_prune, y_prune
