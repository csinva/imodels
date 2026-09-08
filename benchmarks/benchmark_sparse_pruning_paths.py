#!/usr/bin/env python3
"""Benchmark sparse-pruning regularization-path solvers.

The benchmark compares three ways to solve the same normalized least-squares
problem with a nested-tree infinity-CAP penalty::

    0.5 * ||y - X beta||_2**2 / n
        + lambda * sum(group in groups) ||beta[group]||_inf

The compared methods are:

* independent (cold-started) APA-APG2 point solves;
* the warm-started APA-APG2 path API;
* the exact hiCAP homotopy path API.

Run this file from the repository root, for example::

    python benchmarks/benchmark_sparse_pruning_paths.py \
        --sizes 128x15 512x31 --path-points 40 --repeats 5

The default output is a compact human-readable table.  Pass ``--json`` for a
machine-readable record.  Solver imports are intentionally independent: a
missing path implementation is reported as unavailable without preventing the
remaining methods from running.

Accuracy and topology metrics use the requested common lambda range.  The
homotopy timing covers its complete path through lambda zero, whereas APA-APG2
timing covers the finite grid ending at ``--lambda-min-ratio``.  This is
intentional and should be kept in mind when interpreting the timing ratio.
"""
from __future__ import annotations

import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "imodels-mpl-cache"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(tempfile.gettempdir(), "imodels-xdg-cache"))

import argparse
import inspect
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
from scipy.optimize import linprog


# Allow the script to work from a source checkout without installing imodels.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


def _optional_import(
    module: str, name: str
) -> tuple[Callable[..., Any] | None, str | None]:
    # An unfinished optional implementation may fail during module import.
    try:
        imported_module = __import__(module, fromlist=[name])
        return getattr(imported_module, name), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


POINT_SOLVER, POINT_IMPORT_ERROR = _optional_import(
    "imodels.tree.sparse_pruning.optimization.apa_point", "hiCAP_regression"
)
APA_PATH_SOLVER, APA_IMPORT_ERROR = _optional_import(
    "imodels.tree.sparse_pruning.optimization.apa", "apa_apg_regression_path"
)
EXACT_PATH_SOLVER, EXACT_IMPORT_ERROR = _optional_import(
    "imodels.tree.sparse_pruning.optimization.hicap", "hicap_regression_path"
)


@dataclass
class Problem:
    """One deterministic synthetic nested-tree regression problem."""

    X: np.ndarray
    y: np.ndarray
    groups: list[np.ndarray]
    beta_true: np.ndarray
    lambdas: np.ndarray


@dataclass
class PathView:
    """Small adapter over both path APIs and the cold-start baseline."""

    lambdas: np.ndarray
    coefficients: np.ndarray
    diagnostics: tuple[Any, ...] = ()
    status: str = "complete"
    exact: bool = False


@dataclass
class MethodResult:
    """Serializable measurements for one solver on one problem size."""

    method: str
    available: bool
    status: str
    error: str | None = None
    wall_seconds_median: float | None = None
    wall_seconds_iqr: float | None = None
    wall_seconds: list[float] = field(default_factory=list)
    speedup_vs_cold: float | None = None
    seconds_per_topology: float | None = None
    native_points: int | None = None
    total_iterations: int | None = None
    unique_supports: int | None = None
    unique_topologies: int | None = None
    hierarchy_violation_fraction: float | None = None
    objective_median: float | None = None
    kkt_relative_median: float | None = None
    kkt_relative_max: float | None = None
    coefficient_relative_error_median: float | None = None
    coefficient_relative_error_max: float | None = None
    objective_relative_excess_median: float | None = None
    objective_relative_excess_max: float | None = None
    support_agreement_fraction: float | None = None
    topology_agreement_fraction: float | None = None
    exact_topology_recall: float | None = None


@dataclass
class CaseResult:
    """Serializable measurements for all methods on one problem size."""

    samples: int
    features: int
    groups: int
    path_points: int
    lambda_max: float
    lambda_min: float
    seed: int
    methods: list[MethodResult]


def nested_subtree_groups(n_features: int) -> list[np.ndarray]:
    """Return descendant groups for a complete, heap-indexed binary tree.

    Coordinate 0 is the root and children of coordinate ``j`` are ``2*j + 1``
    and ``2*j + 2`` when those coordinates exist.  Every node contributes its
    full descendant group, including singleton leaf groups.  The resulting
    collection is laminar: any two groups are nested or disjoint.
    """

    if n_features < 1:
        raise ValueError("n_features must be positive")

    descendants: list[np.ndarray] = []
    for root in range(n_features):
        nodes: list[int] = []
        stack = [root]
        while stack:
            node = stack.pop()
            if node >= n_features:
                continue
            nodes.append(node)
            stack.append(2 * node + 2)
            stack.append(2 * node + 1)
        descendants.append(np.asarray(sorted(nodes), dtype=np.intp))
    return descendants


def make_problem(
    n_samples: int,
    n_features: int,
    n_path_points: int,
    lambda_min_ratio: float,
    seed: int,
) -> Problem:
    """Construct a centered, correlated design with a rooted sparse signal."""

    if n_samples < 2 or n_features < 1:
        raise ValueError("each size must have at least 2 samples and 1 feature")
    if n_path_points < 2:
        raise ValueError("path_points must be at least 2")
    if not 0 < lambda_min_ratio < 1:
        raise ValueError("lambda_min_ratio must lie strictly between 0 and 1")

    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, n_features))

    # Correlation between a node and its parent makes this closer to the stump
    # designs encountered by sparse tree pruning than an orthogonal toy case.
    inherited_weight = 0.55
    innovation_weight = np.sqrt(1.0 - inherited_weight**2)
    for node in range(1, n_features):
        parent = (node - 1) // 2
        X[:, node] = (
            inherited_weight * X[:, parent] + innovation_weight * X[:, node]
        )
    X -= X.mean(axis=0)
    scales = X.std(axis=0)
    scales[scales == 0] = 1.0
    X /= scales

    # Activate a proper rooted subtree.  Coefficients decay with depth and use
    # mixed signs so that entry/drop behavior is not artificially trivial.
    active_count = min(n_features, max(2, n_features // 3))
    beta_true = np.zeros(n_features)
    for node in range(active_count):
        depth = int(np.floor(np.log2(node + 1)))
        beta_true[node] = rng.choice((-1.0, 1.0)) * 1.5 / (depth + 1)

    signal = X @ beta_true
    signal_scale = float(np.std(signal))
    noise_scale = signal_scale / 3.0 if signal_scale > 0 else 0.25
    y = signal + rng.normal(scale=noise_scale, size=n_samples)
    y -= y.mean()

    groups = nested_subtree_groups(n_features)
    gradient_at_zero = -(X.T @ y) / n_samples
    # Singleton groups make this a conservative zero-solution lambda.  Starting
    # slightly above it ensures every method is tested at beta == 0 even when
    # numerical conventions differ at the exact first knot.
    lambda_max = 1.05 * float(np.max(np.abs(gradient_at_zero)))
    lambda_max = max(lambda_max, np.finfo(float).eps)
    lambdas = np.geomspace(
        lambda_max, lambda_max * lambda_min_ratio, num=n_path_points
    )
    return Problem(X=X, y=y, groups=groups, beta_true=beta_true, lambdas=lambdas)


def _supported_kwargs(
    function: Callable[..., Any], kwargs: dict[str, Any]
) -> dict[str, Any]:
    """Filter optional benchmark controls against a solver's signature."""

    signature = inspect.signature(function)
    accepts_any = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if accepts_any:
        return kwargs
    return {key: value for key, value in kwargs.items() if key in signature.parameters}


def _path_lambda_keyword(function: Callable[..., Any]) -> str:
    parameters = inspect.signature(function).parameters
    for candidate in ("lambdas", "lambda_grid", "alphas"):
        if candidate in parameters:
            return candidate
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        return "lambdas"
    raise TypeError(
        "apa_apg_regression_path must accept one of: lambdas, lambda_grid, alphas"
    )


def _as_path_view(result: Any, *, exact_default: bool = False) -> PathView:
    """Normalize RegularizationPath-like, dict, and two-array results."""

    if isinstance(result, dict):
        lambda_values = next(
            (result[key] for key in ("lambdas", "alphas") if key in result), None
        )
        coefficients = next(
            (
                result[key]
                for key in ("coefficients", "coefs", "coef_path")
                if key in result
            ),
            None,
        )
        diagnostics = tuple(result.get("diagnostics", ()))
        status = str(result.get("status", "complete"))
        exact = bool(result.get("exact", exact_default))
    elif hasattr(result, "lambdas") and hasattr(result, "coefficients"):
        lambda_values = result.lambdas
        coefficients = result.coefficients
        diagnostics = tuple(getattr(result, "diagnostics", ()))
        status = str(getattr(result, "status", "complete"))
        exact = bool(getattr(result, "exact", exact_default))
    elif isinstance(result, tuple) and len(result) >= 2:
        lambda_values, coefficients = result[:2]
        diagnostics = ()
        status = "complete"
        exact = exact_default
    else:
        raise TypeError(
            "path solver must return RegularizationPath-like output, a mapping, "
            "or a (lambdas, coefficients) tuple"
        )

    if lambda_values is None or coefficients is None:
        raise ValueError("path result did not contain lambdas and coefficients")
    lambda_array = np.asarray(lambda_values, dtype=float)
    coefficient_array = np.asarray(coefficients, dtype=float)
    if lambda_array.ndim != 1 or coefficient_array.ndim != 2:
        raise ValueError("path lambdas must be 1D and coefficients must be 2D")
    if coefficient_array.shape[0] != lambda_array.size:
        if coefficient_array.shape[1] == lambda_array.size:
            coefficient_array = coefficient_array.T
        else:
            raise ValueError("coefficient path has no axis matching the lambda count")
    if not np.all(np.isfinite(lambda_array)) or not np.all(
        np.isfinite(coefficient_array)
    ):
        raise ValueError("path output contains non-finite values")

    order = np.argsort(-lambda_array, kind="stable")
    return PathView(
        lambdas=lambda_array[order],
        coefficients=coefficient_array[order],
        diagnostics=diagnostics,
        status=status,
        exact=exact,
    )


def _cold_path_runner(
    problem: Problem, max_iter: int, tol: float
) -> PathView:
    if POINT_SOLVER is None:
        raise ImportError(POINT_IMPORT_ERROR)
    coefficients: list[np.ndarray] = []
    diagnostics: list[dict[str, Any]] = []
    for lam in problem.lambdas:
        result = POINT_SOLVER(
            problem.X,
            problem.y,
            problem.groups,
            lam=float(lam),
            beta_init=None,
            max_iter=max_iter,
            tol=tol,
            ord="inf",
            return_info=True,
        )
        if isinstance(result, tuple) and len(result) == 2:
            beta, info = result
        else:
            beta, info = result, {}
        coefficients.append(np.asarray(beta, dtype=float))
        diagnostics.append(info)
    return PathView(
        lambdas=problem.lambdas.copy(),
        coefficients=np.asarray(coefficients),
        diagnostics=tuple(diagnostics),
        exact=False,
    )


def _apa_path_runner(
    problem: Problem, max_iter: int, tol: float
) -> PathView:
    if APA_PATH_SOLVER is None:
        raise ImportError(APA_IMPORT_ERROR)
    kwargs: dict[str, Any] = {
        _path_lambda_keyword(APA_PATH_SOLVER): problem.lambdas,
        "max_iter": max_iter,
        "tol": tol,
        "ord": "inf",
        "sample_weight": None,
    }
    result = APA_PATH_SOLVER(
        problem.X,
        problem.y,
        problem.groups,
        **_supported_kwargs(APA_PATH_SOLVER, kwargs),
    )
    return _as_path_view(result, exact_default=False)


def _exact_path_runner(
    problem: Problem, max_iter: int, tol: float
) -> PathView:
    if EXACT_PATH_SOLVER is None:
        raise ImportError(EXACT_IMPORT_ERROR)
    kwargs: dict[str, Any] = {
        "fit_intercept": False,
        "sample_weight": None,
        "tolerance": tol,
        "oracle_max_iter": max_iter,
        "lambda_min": float(problem.lambdas[-1]),
    }
    result = EXACT_PATH_SOLVER(
        problem.X,
        problem.y,
        problem.groups,
        **_supported_kwargs(EXACT_PATH_SOLVER, kwargs),
    )
    return _as_path_view(result, exact_default=True)


def evaluate_path(path: PathView, lambdas: np.ndarray) -> np.ndarray:
    """Interpolate a coefficient path at decreasing positive lambdas."""

    lambdas = np.asarray(lambdas, dtype=float)
    source_lambdas = path.lambdas[::-1]
    source_coefficients = path.coefficients[::-1]

    # Repeated lambda values represent simultaneous homotopy events.  Match
    # RegularizationPath.at's right-continuous convention.
    unique_lambdas, unique_indices = np.unique(source_lambdas, return_index=True)
    source_coefficients = source_coefficients[unique_indices]
    columns = [
        np.interp(lambdas, unique_lambdas, source_coefficients[:, column])
        for column in range(source_coefficients.shape[1])
    ]
    return np.column_stack(columns)


def group_penalty(beta: np.ndarray, groups: Sequence[np.ndarray]) -> float:
    return float(sum(np.max(np.abs(beta[group])) for group in groups))


def objective_values(problem: Problem, coefficients: np.ndarray) -> np.ndarray:
    values = []
    for lam, beta in zip(problem.lambdas, coefficients):
        residual = problem.y - problem.X @ beta
        loss = 0.5 * float(residual @ residual) / problem.X.shape[0]
        values.append(loss + float(lam) * group_penalty(beta, problem.groups))
    return np.asarray(values)


def kkt_relative_residual(
    X: np.ndarray,
    y: np.ndarray,
    groups: Sequence[np.ndarray],
    lam: float,
    beta: np.ndarray,
    face_tolerance: float,
) -> float:
    """Compute distance to the exact infinity-CAP KKT subdifferential.

    For each nonzero group, its infinity-norm subgradient is a convex
    combination of signed coordinates on the maximum-magnitude face.  At a
    zero group it is the unit L1 ball.  A small linear program chooses all
    overlapping group subgradients jointly and minimizes the infinity norm of
    ``gradient + lambda * sum(subgradients)``.
    """

    n_samples, n_features = X.shape
    beta = np.asarray(beta, dtype=float)
    gradient = X.T @ (X @ beta - y) / n_samples

    contribution_columns: list[np.ndarray] = []
    equality_blocks: list[list[int]] = []
    zero_group_blocks: list[list[int]] = []
    for group in groups:
        group = np.asarray(group, dtype=np.intp)
        group_values = beta[group]
        group_norm = float(np.max(np.abs(group_values)))
        scale = max(1.0, group_norm)
        if group_norm <= face_tolerance:
            block: list[int] = []
            for coordinate in group:
                positive = np.zeros(n_features)
                positive[coordinate] = 1.0
                contribution_columns.append(positive)
                block.append(len(contribution_columns) - 1)
                contribution_columns.append(-positive)
                block.append(len(contribution_columns) - 1)
            zero_group_blocks.append(block)
        else:
            face = np.flatnonzero(
                np.abs(group_values) >= group_norm - face_tolerance * scale
            )
            block = []
            for local_coordinate in face:
                contribution = np.zeros(n_features)
                coordinate = int(group[local_coordinate])
                contribution[coordinate] = np.sign(group_values[local_coordinate])
                contribution_columns.append(contribution)
                block.append(len(contribution_columns) - 1)
            equality_blocks.append(block)

    contributions = np.column_stack(contribution_columns)
    n_weights = contributions.shape[1]
    # Last variable is the nonnegative infinity-norm residual t.
    objective = np.zeros(n_weights + 1)
    objective[-1] = 1.0

    stationarity_positive = np.column_stack(
        (lam * contributions, -np.ones(n_features))
    )
    stationarity_negative = np.column_stack(
        (-lam * contributions, -np.ones(n_features))
    )
    upper_rows = [stationarity_positive, stationarity_negative]
    upper_bounds = [-gradient, gradient]
    for block in zero_group_blocks:
        row = np.zeros(n_weights + 1)
        row[block] = 1.0
        upper_rows.append(row[np.newaxis, :])
        upper_bounds.append(np.ones(1))

    if equality_blocks:
        equality_matrix = np.zeros((len(equality_blocks), n_weights + 1))
        for row_index, block in enumerate(equality_blocks):
            equality_matrix[row_index, block] = 1.0
        equality_rhs = np.ones(len(equality_blocks))
    else:
        equality_matrix = None
        equality_rhs = None

    result = linprog(
        objective,
        A_ub=np.vstack(upper_rows),
        b_ub=np.concatenate(upper_bounds),
        A_eq=equality_matrix,
        b_eq=equality_rhs,
        bounds=[(0.0, None)] * (n_weights + 1),
        method="highs",
    )
    if not result.success:
        return float("nan")
    raw_residual = max(0.0, float(result.fun))
    scale = max(float(np.max(np.abs(gradient))), float(lam), np.finfo(float).eps)
    return raw_residual / scale


def support_signature(beta: np.ndarray, tolerance: float) -> tuple[int, ...]:
    return tuple(np.flatnonzero(np.abs(beta) > tolerance).tolist())


def topology_signature(beta: np.ndarray, tolerance: float) -> tuple[int, ...]:
    """Return the ancestor closure of active heap-indexed tree coordinates."""

    topology: set[int] = set()
    for active in np.flatnonzero(np.abs(beta) > tolerance):
        node = int(active)
        while True:
            topology.add(node)
            if node == 0:
                break
            node = (node - 1) // 2
    return tuple(sorted(topology))


def hierarchy_is_violated(beta: np.ndarray, tolerance: float) -> bool:
    active = np.abs(beta) > tolerance
    for node in np.flatnonzero(active):
        if node and not active[(int(node) - 1) // 2]:
            return True
    return False


def _exact_state_coefficients(
    exact_path: PathView,
    lambda_min: float,
    lambda_max: float,
    support_tolerance: float,
) -> np.ndarray:
    """Evaluate knots, threshold crossings, and all resulting open segments."""

    knots = exact_path.lambdas[
        (exact_path.lambdas >= lambda_min) & (exact_path.lambdas <= lambda_max)
    ]
    initial_boundaries = np.unique(
        np.concatenate(([lambda_min, lambda_max], knots))
    )[::-1]
    crossing_lambdas: list[float] = []
    for upper, lower in zip(initial_boundaries[:-1], initial_boundaries[1:]):
        endpoint_coefficients = evaluate_path(
            exact_path, np.asarray([upper, lower])
        )
        upper_beta, lower_beta = endpoint_coefficients
        coefficient_change = upper_beta - lower_beta
        for target in (-support_tolerance, support_tolerance):
            changing = np.flatnonzero(np.abs(coefficient_change) > np.finfo(float).eps)
            fractions = (target - lower_beta[changing]) / coefficient_change[changing]
            for fraction in fractions[(fractions > 0.0) & (fractions < 1.0)]:
                crossing_lambdas.append(float(lower + fraction * (upper - lower)))

    boundaries = np.unique(
        np.concatenate((initial_boundaries, np.asarray(crossing_lambdas)))
    )[::-1]
    if boundaries.size > 1:
        midpoints = 0.5 * (boundaries[:-1] + boundaries[1:])
        evaluations = np.sort(np.concatenate((boundaries, midpoints)))[::-1]
    else:
        evaluations = boundaries
    return evaluate_path(exact_path, evaluations)


def _total_iterations(path: PathView) -> int | None:
    iterations: list[int] = []
    for diagnostic in path.diagnostics:
        if isinstance(diagnostic, dict) and "n_iter" in diagnostic:
            iterations.append(int(diagnostic["n_iter"]))
    return sum(iterations) if iterations else None


def _timed_runs(
    runner: Callable[[Problem, int, float], PathView],
    problem: Problem,
    repeats: int,
    max_iter: int,
    tol: float,
) -> tuple[PathView, list[float]]:
    path: PathView | None = None
    elapsed: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        path = runner(problem, max_iter, tol)
        elapsed.append(time.perf_counter() - start)
    assert path is not None
    return path, elapsed


def _unavailable_method(name: str, error: str | None) -> MethodResult:
    return MethodResult(
        method=name,
        available=False,
        status="unavailable",
        error=error or "solver is unavailable",
    )


def benchmark_case(
    problem: Problem,
    *,
    repeats: int,
    max_iter: int,
    tol: float,
    support_tolerance: float,
    kkt_face_tolerance: float,
    seed: int,
) -> CaseResult:
    """Benchmark all available solvers and calculate common-grid metrics."""

    specifications = [
        ("apa_apg2_cold", _cold_path_runner, POINT_SOLVER, POINT_IMPORT_ERROR),
        ("apa_apg2_warm", _apa_path_runner, APA_PATH_SOLVER, APA_IMPORT_ERROR),
        ("hicap_exact", _exact_path_runner, EXACT_PATH_SOLVER, EXACT_IMPORT_ERROR),
    ]
    paths: dict[str, PathView] = {}
    results: dict[str, MethodResult] = {}
    for name, runner, implementation, import_error in specifications:
        if implementation is None:
            results[name] = _unavailable_method(name, import_error)
            continue
        try:
            path, elapsed = _timed_runs(runner, problem, repeats, max_iter, tol)
            paths[name] = path
            q1, q3 = np.percentile(elapsed, [25.0, 75.0])
            results[name] = MethodResult(
                method=name,
                available=True,
                status=path.status,
                wall_seconds_median=float(np.median(elapsed)),
                wall_seconds_iqr=float(q3 - q1),
                wall_seconds=[float(value) for value in elapsed],
                native_points=int(path.lambdas.size),
                total_iterations=_total_iterations(path),
            )
        except Exception as exc:
            results[name] = MethodResult(
                method=name,
                available=True,
                status="failed",
                error=f"{type(exc).__name__}: {exc}",
            )

    evaluated: dict[str, np.ndarray] = {}
    objectives: dict[str, np.ndarray] = {}
    for name, path in paths.items():
        coefficients = evaluate_path(path, problem.lambdas)
        evaluated[name] = coefficients
        objective = objective_values(problem, coefficients)
        objectives[name] = objective
        kkt = np.asarray(
            [
                kkt_relative_residual(
                    problem.X,
                    problem.y,
                    problem.groups,
                    float(lam),
                    beta,
                    kkt_face_tolerance,
                )
                for lam, beta in zip(problem.lambdas, coefficients)
            ]
        )
        supports = {
            support_signature(beta, support_tolerance)
            for beta in path.coefficients
        }
        topologies = {
            topology_signature(beta, support_tolerance)
            for beta in path.coefficients
        }
        method_result = results[name]
        method_result.unique_supports = len(supports)
        method_result.unique_topologies = len(topologies)
        method_result.hierarchy_violation_fraction = float(
            np.mean(
                [
                    hierarchy_is_violated(beta, support_tolerance)
                    for beta in coefficients
                ]
            )
        )
        method_result.objective_median = float(np.median(objective))
        finite_kkt = kkt[np.isfinite(kkt)]
        if finite_kkt.size:
            method_result.kkt_relative_median = float(np.median(finite_kkt))
            method_result.kkt_relative_max = float(np.max(finite_kkt))

    has_certified_exact_path = (
        "hicap_exact" in paths
        and paths["hicap_exact"].exact
        and paths["hicap_exact"].status == "complete"
    )
    coefficient_reference_name = (
        "hicap_exact"
        if has_certified_exact_path
        else ("apa_apg2_cold" if "apa_apg2_cold" in evaluated else None)
    )
    if has_certified_exact_path:
        reference_objective = objectives["hicap_exact"]
    elif objectives:
        # Without exact hiCAP, the lower objective at each lambda is a useful
        # non-certified reference for comparing cold and warm continuation.
        reference_objective = np.min(np.vstack(list(objectives.values())), axis=0)
    else:
        reference_objective = None
    reference_coefficients = (
        evaluated[coefficient_reference_name]
        if coefficient_reference_name is not None
        else None
    )

    exact_topologies: set[tuple[int, ...]] | None = None
    if has_certified_exact_path:
        exact_states = _exact_state_coefficients(
            paths["hicap_exact"],
            float(problem.lambdas[-1]),
            float(problem.lambdas[0]),
            support_tolerance,
        )
        exact_topologies = {
            topology_signature(beta, support_tolerance) for beta in exact_states
        }
        results["hicap_exact"].unique_supports = len(
            {support_signature(beta, support_tolerance) for beta in exact_states}
        )
        results["hicap_exact"].unique_topologies = len(exact_topologies)

    for name, coefficients in evaluated.items():
        result = results[name]
        if reference_objective is not None:
            denominator = np.maximum(np.abs(reference_objective), np.finfo(float).eps)
            relative_excess = (objectives[name] - reference_objective) / denominator
            result.objective_relative_excess_median = float(np.median(relative_excess))
            result.objective_relative_excess_max = float(np.max(relative_excess))

        if reference_coefficients is not None:
            differences = np.linalg.norm(coefficients - reference_coefficients, axis=1)
            denominators = np.maximum(
                np.linalg.norm(reference_coefficients, axis=1),
                1.0,
            )
            relative_errors = differences / denominators
            result.coefficient_relative_error_median = float(np.median(relative_errors))
            result.coefficient_relative_error_max = float(np.max(relative_errors))

            supports = [
                support_signature(beta, support_tolerance) for beta in coefficients
            ]
            reference_supports = [
                support_signature(beta, support_tolerance)
                for beta in reference_coefficients
            ]
            topologies = [
                topology_signature(beta, support_tolerance) for beta in coefficients
            ]
            reference_topologies = [
                topology_signature(beta, support_tolerance)
                for beta in reference_coefficients
            ]
            result.support_agreement_fraction = float(
                np.mean(
                    [left == right for left, right in zip(supports, reference_supports)]
                )
            )
            result.topology_agreement_fraction = float(
                np.mean(
                    [
                        left == right
                        for left, right in zip(topologies, reference_topologies)
                    ]
                )
            )

        if exact_topologies:
            method_topologies = (
                exact_topologies
                if name == "hicap_exact"
                else {
                    topology_signature(beta, support_tolerance)
                    for beta in paths[name].coefficients
                }
            )
            result.exact_topology_recall = len(
                method_topologies & exact_topologies
            ) / len(exact_topologies)

    cold_time = results["apa_apg2_cold"].wall_seconds_median
    for result in results.values():
        if cold_time is not None and cold_time > 0:
            if (
                result.wall_seconds_median is not None
                and result.wall_seconds_median > 0
            ):
                result.speedup_vs_cold = cold_time / result.wall_seconds_median
        if result.unique_topologies and result.wall_seconds_median is not None:
            result.seconds_per_topology = (
                result.wall_seconds_median / result.unique_topologies
            )

    return CaseResult(
        samples=problem.X.shape[0],
        features=problem.X.shape[1],
        groups=len(problem.groups),
        path_points=problem.lambdas.size,
        lambda_max=float(problem.lambdas[0]),
        lambda_min=float(problem.lambdas[-1]),
        seed=seed,
        methods=[results[name] for name, *_ in specifications],
    )


def parse_size(value: str) -> tuple[int, int]:
    """Parse an ``NXP`` problem size."""

    pieces = value.lower().split("x")
    if len(pieces) != 2:
        raise argparse.ArgumentTypeError(
            "sizes must have the form NXP, for example 128x15"
        )
    try:
        n_samples, n_features = (int(piece) for piece in pieces)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("both N and P must be integers") from exc
    if n_samples < 2 or n_features < 1:
        raise argparse.ArgumentTypeError("N must be >= 2 and P must be >= 1")
    return n_samples, n_features


def _format_metric(value: float | int | None, precision: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{precision}g}"


def print_case(case: CaseResult) -> None:
    print(
        f"\nn={case.samples}, p={case.features}, groups={case.groups}, "
        f"grid={case.path_points}, "
        f"lambda=[{case.lambda_min:.3g}, {case.lambda_max:.3g}]"
    )
    headings = (
        "method",
        "status",
        "time med",
        "IQR",
        "speedup",
        "points",
        "supports",
        "topologies",
        "KKT med",
        "coef err",
        "topo recall",
    )
    rows: list[tuple[str, ...]] = []
    for method in case.methods:
        rows.append(
            (
                method.method,
                method.status,
                _format_metric(method.wall_seconds_median),
                _format_metric(method.wall_seconds_iqr),
                _format_metric(method.speedup_vs_cold),
                _format_metric(method.native_points),
                _format_metric(method.unique_supports),
                _format_metric(method.unique_topologies),
                _format_metric(method.kkt_relative_median),
                _format_metric(method.coefficient_relative_error_median),
                _format_metric(method.exact_topology_recall),
            )
        )
    widths = [
        max(len(headings[column]), *(len(row[column]) for row in rows))
        for column in range(len(headings))
    ]
    print("  ".join(value.ljust(width) for value, width in zip(headings, widths)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(value.ljust(width) for value, width in zip(row, widths)))
    for method in case.methods:
        if method.error:
            print(f"  {method.method}: {method.error}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=parse_size,
        default=[(128, 15), (512, 31)],
        metavar="NXP",
        help="sample-by-feature sizes (default: 128x15 512x31)",
    )
    parser.add_argument("--path-points", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--lambda-min-ratio", type=float, default=1e-3)
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--tol", type=float, default=1e-7)
    parser.add_argument("--support-tol", type=float, default=1e-6)
    parser.add_argument("--kkt-face-tol", type=float, default=1e-6)
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit JSON instead of the human-readable table",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.path_points < 2:
        raise SystemExit("--path-points must be at least 2")
    if args.repeats < 1:
        raise SystemExit("--repeats must be positive")
    if args.max_iter < 1:
        raise SystemExit("--max-iter must be positive")

    cases: list[CaseResult] = []
    for case_index, (n_samples, n_features) in enumerate(args.sizes):
        case_seed = args.seed + case_index
        problem = make_problem(
            n_samples=n_samples,
            n_features=n_features,
            n_path_points=args.path_points,
            lambda_min_ratio=args.lambda_min_ratio,
            seed=case_seed,
        )
        case = benchmark_case(
            problem,
            repeats=args.repeats,
            max_iter=args.max_iter,
            tol=args.tol,
            support_tolerance=args.support_tol,
            kkt_face_tolerance=args.kkt_face_tol,
            seed=case_seed,
        )
        cases.append(case)
        if not args.json:
            print_case(case)

    if args.json:
        from benchmarks._sparse_pruning_tree_scaling import environment_metadata

        environment = environment_metadata()
        print(json.dumps([
            {**asdict(case), "configuration": vars(args), "environment": environment}
            for case in cases
        ], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
