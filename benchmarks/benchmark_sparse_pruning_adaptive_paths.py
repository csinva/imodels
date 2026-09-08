#!/usr/bin/env python3
"""Compare exact hiCAP with fixed-grid and adaptive APA-APG2 paths.

The benchmark fits a real ``DecisionTreeRegressor`` to an sklearn regression
dataset and converts that tree to the local-stump basis used by sparse
pruning.  All three methods solve the same centered infinity-CAP problem.

In addition to native solver time and path size, each sampled path is compared
with exact hiCAP on a dense common lambda grid.  The JSON output retains the
native paths and pointwise errors needed to reproduce the companion figure;
the CSV output is a flat summary suitable for a spreadsheet.

Example (the bundled dataset requires no network access)::

    python \
      benchmarks/benchmark_sparse_pruning_adaptive_paths.py \
      --dataset diabetes --nodes 7 --samples 442 --repeats 3 \
      --output-dir benchmarks/adaptive_path_comparison/diabetes
"""
from __future__ import annotations

# Set reproducible/cache-safe defaults before importing NumPy, sklearn, or
# matplotlib (the plotter is imported lazily after all timings are complete).
import os
import tempfile

for _thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_thread_variable, "1")
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "imodels-mpl-cache"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(tempfile.gettempdir(), "imodels-xdg-cache"))

import argparse
import csv
import io
import json
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
from threadpoolctl import threadpool_limits


BENCHMARK_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY_ROOT = BENCHMARK_DIRECTORY.parent
for _source_directory in (REPOSITORY_ROOT, BENCHMARK_DIRECTORY):
    if str(_source_directory) not in sys.path:
        sys.path.insert(0, str(_source_directory))

import _sparse_pruning_tree_scaling as scaling
from benchmarks.experimental.adaptive import (
    apa_apg_adaptive_regression_path,
)
from imodels.tree.sparse_pruning.optimization.apa import apa_apg_regression_path
from imodels.tree.sparse_pruning.optimization.hicap import hicap_regression_path


SCHEMA_NAME = "imodels.sparse_pruning.adaptive_path_benchmark"
SCHEMA_VERSION = 1
METHOD_EXACT = "hicap_exact"
METHOD_FIXED = "apa_apg2_fixed"
METHOD_ADAPTIVE = "apa_apg2_adaptive"
METHODS = (METHOD_EXACT, METHOD_FIXED, METHOD_ADAPTIVE)


@dataclass
class _TimedPath:
    method: str
    wall_seconds: float
    path: Any | None
    error_type: str | None = None
    error_message: str | None = None


def _timed(method: str, solve: Callable[[], Any]) -> _TimedPath:
    started = time.perf_counter()
    try:
        path = solve()
    except Exception as exc:  # keep a durable row for numerical failures
        return _TimedPath(
            method=method,
            wall_seconds=float(time.perf_counter() - started),
            path=None,
            error_type=type(exc).__name__,
            error_message=str(exc),
        )
    return _TimedPath(
        method=method,
        wall_seconds=float(time.perf_counter() - started),
        path=path,
    )


def _evaluate_path(path: Any, lambdas: np.ndarray) -> np.ndarray:
    """Evaluate a native path with the benchmark's interpolation convention."""

    order = np.argsort(np.asarray(path.lambdas), kind="stable")
    coordinates = np.asarray(path.lambdas)[order]
    coefficients = np.asarray(path.coefficients)[order]
    unique_coordinates, unique_indices = np.unique(
        coordinates, return_index=True
    )
    coefficients = coefficients[unique_indices]
    return np.column_stack(
        [
            np.interp(lambdas, unique_coordinates, coefficients[:, column])
            for column in range(coefficients.shape[1])
        ]
    )


def _dense_lambda_grid(
    lambda_max: float, minimum_ratio: float, n_points: int
) -> np.ndarray:
    if n_points < 3:
        raise ValueError("dense_points must be at least three")
    positive = np.geomspace(
        lambda_max, lambda_max * minimum_ratio, num=n_points - 1
    )
    return np.r_[positive, 0.0]


def _gram_diagnostics(problem: scaling.TreePathProblem) -> dict[str, Any]:
    gram = problem.X.T @ problem.X / problem.n_samples
    diagonal = np.maximum(np.diag(gram), 0.0)
    off_diagonal = gram.copy()
    np.fill_diagonal(off_diagonal, 0.0)
    scale = np.sqrt(diagonal[:, None] * diagonal[None, :])
    correlations = np.zeros_like(gram)
    valid = scale > 0
    correlations[valid] = np.abs(off_diagonal[valid]) / scale[valid]
    correlations[(~valid) & (off_diagonal != 0)] = np.inf
    correlation_tolerance = (
        32.0 * np.finfo(float).eps * max(1, problem.n_features)
    )
    max_correlation = float(np.max(correlations, initial=0.0))
    return {
        "normalization": "X.T @ X / n",
        "empirical_measure": "tree-fitting rows with uniform weights",
        "expected_identity": (
            "off-diagonal entries are zero; each diagonal is the node "
            "training-mass fraction for unnormalized local stumps"
        ),
        "max_abs_centered_column_mean": float(
            np.max(np.abs(np.mean(problem.X, axis=0)), initial=0.0)
        ),
        "max_abs_off_diagonal": float(
            np.max(np.abs(off_diagonal), initial=0.0)
        ),
        "max_abs_off_diagonal_correlation": max_correlation,
        "minimum_diagonal": float(np.min(diagonal)),
        "maximum_diagonal": float(np.max(diagonal, initial=0.0)),
        "diagonal_correlation_tolerance": float(correlation_tolerance),
        "empirically_diagonal": bool(max_correlation <= correlation_tolerance),
    }


def _path_statistics(path: Any) -> dict[str, Any]:
    diagnostics = [
        dict(item)
        for item in path.diagnostics
        if isinstance(item, Mapping)
    ]
    iterations = [
        int(item["n_iter"])
        for item in diagnostics
        if item.get("n_iter") is not None
    ]
    convergence = [
        bool(item["converged"])
        for item in diagnostics
        if item.get("converged") is not None
    ]
    optimizer_calls = sum(
        float(item.get("lambda", 0.0)) > 0
        and int(item.get("n_iter", 0)) > 0
        for item in diagnostics
    )
    metadata = dict(path.metadata)
    raw_relative_kkt_residuals = [
        float(item["kkt_relative_residual"])
        for item in diagnostics
        if item.get("kkt_relative_residual") is not None
    ]
    relative_kkt_residuals = [
        value for value in raw_relative_kkt_residuals if np.isfinite(value)
    ]
    raw_relative_relaxed_kkt_residuals = [
        float(item["kkt_relaxed_relative_residual"])
        for item in diagnostics
        if item.get("kkt_relaxed_relative_residual") is not None
    ]
    relative_relaxed_kkt_residuals = [
        value
        for value in raw_relative_relaxed_kkt_residuals
        if np.isfinite(value)
    ]
    unresolved = metadata.get("unresolved_intervals", ())
    unresolved_reasons: dict[str, int] = {}
    for interval in unresolved:
        reason = str(interval.get("reason", "unknown"))
        unresolved_reasons[reason] = unresolved_reasons.get(reason, 0) + 1
    return {
        "status": str(path.status),
        "exact": bool(path.exact),
        "native_points": int(path.n_points),
        "positive_optimizer_calls": int(optimizer_calls),
        "total_iterations": int(sum(iterations)) if iterations else None,
        "converged_point_fraction": (
            float(np.mean(convergence)) if convergence else None
        ),
        "cached_sufficient_statistics": metadata.get(
            "cached_sufficient_statistics"
        ),
        "quadratic_backend": metadata.get("quadratic_backend"),
        "assumed_diagonal_gram": bool(
            metadata.get("assumed_diagonal_gram", False)
        ),
        "adaptive_tolerance_met": metadata.get("adaptive_tolerance_met"),
        "point_solutions_certified": metadata.get(
            "point_solutions_certified"
        ),
        "point_solutions_kkt_accepted": metadata.get(
            "point_solutions_kkt_accepted"
        ),
        "kkt_acceptance_mode": metadata.get("kkt_acceptance_mode"),
        "kkt_face_relaxation_used": metadata.get(
            "kkt_face_relaxation_used"
        ),
        "max_relative_kkt_residual": (
            float(np.max(relative_kkt_residuals))
            if raw_relative_kkt_residuals
            and len(relative_kkt_residuals) == len(raw_relative_kkt_residuals)
            else None
        ),
        "n_nonfinite_kkt_residuals": int(
            len(raw_relative_kkt_residuals) - len(relative_kkt_residuals)
        ),
        "max_relative_relaxed_kkt_residual": (
            float(np.max(relative_relaxed_kkt_residuals))
            if raw_relative_relaxed_kkt_residuals
            and len(relative_relaxed_kkt_residuals)
            == len(raw_relative_relaxed_kkt_residuals)
            else None
        ),
        "n_nonfinite_relaxed_kkt_residuals": int(
            len(raw_relative_relaxed_kkt_residuals)
            - len(relative_relaxed_kkt_residuals)
        ),
        "solver_stopping_tests_met": metadata.get(
            "solver_stopping_tests_met"
        ),
        "n_initial_points": metadata.get("n_initial_points"),
        "n_refinement_points": metadata.get("n_refinement_points"),
        "n_unresolved_intervals": int(len(unresolved)),
        "unresolved_reasons": unresolved_reasons,
    }


def _dense_comparison(
    problem: scaling.TreePathProblem,
    reference_path: Any,
    candidate_path: Any,
    lambdas: np.ndarray,
    support_tolerance: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    reference = _evaluate_path(reference_path, lambdas)
    candidate = _evaluate_path(candidate_path, lambdas)
    differences = np.linalg.norm(candidate - reference, axis=1)
    # A single path-wide normalizer makes this metric invariant to rescaling y
    # without exploding at the exactly-zero lambda_max endpoint.
    coefficient_scale = max(
        float(np.max(np.linalg.norm(reference, axis=1))),
        np.finfo(float).tiny,
    )
    coefficient_error = differences / coefficient_scale

    reference_objective = scaling.objective_values(
        problem, lambdas, reference
    )
    candidate_objective = scaling.objective_values(
        problem, lambdas, candidate
    )
    objective_scale = max(
        float(np.max(np.abs(reference_objective))),
        np.finfo(float).tiny,
    )
    objective_excess = (
        candidate_objective - reference_objective
    ) / objective_scale
    objective_gap = np.abs(objective_excess)

    reference_support = np.abs(reference) > support_tolerance
    candidate_support = np.abs(candidate) > support_tolerance
    support_match = np.all(reference_support == candidate_support, axis=1)
    reference_topology = [
        scaling.topology_signature(
            beta, problem.parent_indices, support_tolerance
        )
        for beta in reference
    ]
    candidate_topology = [
        scaling.topology_signature(
            beta, problem.parent_indices, support_tolerance
        )
        for beta in candidate
    ]
    topology_match = np.asarray(
        [left == right for left, right in zip(reference_topology, candidate_topology)]
    )
    summary = {
        "coefficient_relative_error_median": float(
            np.median(coefficient_error)
        ),
        "coefficient_relative_error_max": float(np.max(coefficient_error)),
        "objective_relative_gap_median": float(np.median(objective_gap)),
        "objective_relative_gap_max": float(np.max(objective_gap)),
        "objective_relative_excess_median": float(
            np.median(objective_excess)
        ),
        "objective_relative_excess_max": float(np.max(objective_excess)),
        "support_agreement_fraction": float(np.mean(support_match)),
        "topology_agreement_fraction": float(np.mean(topology_match)),
        "reference_unique_supports": len(
            {tuple(np.flatnonzero(row).tolist()) for row in reference_support}
        ),
        "candidate_unique_supports": len(
            {tuple(np.flatnonzero(row).tolist()) for row in candidate_support}
        ),
        "reference_unique_topologies": len(set(reference_topology)),
        "candidate_unique_topologies": len(set(candidate_topology)),
        "coefficient_error_normalizer": coefficient_scale,
        "objective_error_normalizer": objective_scale,
    }
    dense = {
        "lambdas": lambdas,
        "coefficient_relative_error": coefficient_error,
        "objective_relative_excess": objective_excess,
        "objective_relative_gap": objective_gap,
        "support_match": support_match,
        "topology_match": topology_match,
        "coefficient_error_normalizer": coefficient_scale,
        "objective_error_normalizer": objective_scale,
    }
    return summary, dense


def _exact_dense_comparison(
    lambdas: np.ndarray,
) -> tuple[dict[str, Any], dict[str, Any]]:
    zeros = np.zeros(lambdas.size, dtype=float)
    matches = np.ones(lambdas.size, dtype=bool)
    summary = {
        "coefficient_relative_error_median": 0.0,
        "coefficient_relative_error_max": 0.0,
        "objective_relative_gap_median": 0.0,
        "objective_relative_gap_max": 0.0,
        "objective_relative_excess_median": 0.0,
        "objective_relative_excess_max": 0.0,
        "support_agreement_fraction": 1.0,
        "topology_agreement_fraction": 1.0,
    }
    dense = {
        "lambdas": lambdas,
        "coefficient_relative_error": zeros,
        "objective_relative_excess": zeros,
        "objective_relative_gap": zeros,
        "support_match": matches,
        "topology_match": matches,
    }
    return summary, dense


def _json_compatible(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_compatible(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_compatible(value.tolist())
    if isinstance(value, np.generic):
        return _json_compatible(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _summary_fields() -> list[str]:
    return [
        "schema_version",
        "repeat",
        "seed",
        "dataset_name",
        "dataset_digest",
        "n_samples",
        "raw_feature_count",
        "requested_internal_nodes",
        "n_internal_nodes",
        "tree_depth",
        "design_rank",
        "method",
        "outcome",
        "status",
        "exact",
        "wall_seconds",
        "lambda_setup_seconds",
        "lambda_max",
        "method_lambda_max",
        "zero_solution_lambda",
        "lambda_threshold_source",
        "minimum_ratio",
        "dense_points",
        "native_points",
        "positive_optimizer_calls",
        "total_iterations",
        "converged_point_fraction",
        "cached_sufficient_statistics",
        "quadratic_backend",
        "assumed_diagonal_gram",
        "gram_empirically_diagonal",
        "gram_max_abs_off_diagonal",
        "gram_max_abs_off_diagonal_correlation",
        "adaptive_tolerance_met",
        "point_solutions_certified",
        "point_solutions_kkt_accepted",
        "kkt_acceptance_mode",
        "kkt_face_relaxation_used",
        "max_relative_kkt_residual",
        "n_nonfinite_kkt_residuals",
        "max_relative_relaxed_kkt_residual",
        "n_nonfinite_relaxed_kkt_residuals",
        "solver_stopping_tests_met",
        "n_initial_points",
        "n_refinement_points",
        "n_unresolved_intervals",
        "coefficient_relative_error_median",
        "coefficient_relative_error_max",
        "objective_relative_gap_median",
        "objective_relative_gap_max",
        "objective_relative_excess_median",
        "objective_relative_excess_max",
        "support_agreement_fraction",
        "topology_agreement_fraction",
        "reference_unique_supports",
        "candidate_unique_supports",
        "reference_unique_topologies",
        "candidate_unique_topologies",
        "coefficient_error_normalizer",
        "objective_error_normalizer",
        "error_type",
        "error_message",
    ]


def _write_outputs(
    document: dict[str, Any], output_directory: Path
) -> tuple[Path, Path]:
    output_directory.mkdir(parents=True, exist_ok=True)
    json_path = output_directory / "results.json"
    csv_path = output_directory / "results.csv"
    serializable = _json_compatible(document)
    _atomic_write(json_path, json.dumps(serializable, indent=2, sort_keys=True) + "\n")

    stream = io.StringIO(newline="")
    fields = _summary_fields()
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for run in serializable["runs"]:
        writer.writerow({name: run.get(name) for name in fields})
    _atomic_write(csv_path, stream.getvalue())
    return json_path.resolve(), csv_path.resolve()


def _base_summary(
    *,
    args: argparse.Namespace,
    dataset: scaling.RegressionDataset,
    problem: scaling.TreePathProblem,
    repeat: int,
    seed: int,
    method: str,
    result: _TimedPath,
    lambda_max: float,
    lambda_setup_seconds: float,
    gram: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "repeat": repeat,
        "seed": seed,
        "dataset_name": dataset.name,
        "dataset_digest": dataset.digest,
        "n_samples": problem.n_samples,
        "raw_feature_count": problem.raw_feature_count,
        "requested_internal_nodes": int(args.nodes),
        "n_internal_nodes": problem.n_features,
        "tree_depth": problem.tree_depth,
        "design_rank": problem.design_rank,
        "method": method,
        "outcome": "complete" if result.path is not None else "failed",
        "status": "failed" if result.path is None else str(result.path.status),
        "exact": False if result.path is None else bool(result.path.exact),
        "wall_seconds": result.wall_seconds,
        "lambda_setup_seconds": lambda_setup_seconds,
        "lambda_max": lambda_max,
        "method_lambda_max": lambda_max,
        "zero_solution_lambda": lambda_max,
        "lambda_threshold_source": {
            METHOD_EXACT: "solver_internal",
            METHOD_FIXED: "shared_benchmark_setup",
            METHOD_ADAPTIVE: "solver_internal",
        }[method],
        "minimum_ratio": float(args.lambda_min_ratio),
        "dense_points": int(args.dense_points),
        "gram_empirically_diagonal": gram["empirically_diagonal"],
        "gram_max_abs_off_diagonal": gram["max_abs_off_diagonal"],
        "gram_max_abs_off_diagonal_correlation": gram[
            "max_abs_off_diagonal_correlation"
        ],
        "error_type": result.error_type,
        "error_message": result.error_message,
    }


def _run_repeat(
    args: argparse.Namespace,
    dataset: scaling.RegressionDataset,
    repeat: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    seed = int(args.seed) + repeat
    if int(args.samples) == 0 or int(args.samples) == dataset.y.size:
        subset = dataset
    else:
        subset = scaling.subset_regression_dataset(
            dataset, int(args.samples), seed=seed
        )
    problem = scaling.build_tree_path_problem(
        subset.X,
        subset.y,
        int(args.nodes),
        seed=seed,
        dataset_name=subset.name,
        dataset_digest=subset.digest,
        min_samples_leaf=int(args.min_samples_leaf),
    )
    gram = _gram_diagnostics(problem)
    if bool(args.assume_diagonal_gram) and not gram["empirically_diagonal"]:
        raise RuntimeError(
            "the local-stump Gram matrix failed the empirical diagonal check; "
            "rerun with --no-assume-diagonal-gram"
        )

    lambda_started = time.perf_counter()
    lambda_max = scaling.zero_solution_lambda_max(problem)
    lambda_setup_seconds = float(time.perf_counter() - lambda_started)
    fixed_lambdas = _dense_lambda_grid(
        lambda_max, float(args.lambda_min_ratio), int(args.fixed_points)
    )
    dense_lambdas = _dense_lambda_grid(
        lambda_max, float(args.lambda_min_ratio), int(args.dense_points)
    )

    exact = _timed(
        METHOD_EXACT,
        lambda: hicap_regression_path(
            problem.X,
            problem.y,
            problem.groups,
            fit_intercept=False,
            tolerance=float(args.tolerance),
            max_events=int(args.max_events),
            oracle_max_iter=int(args.max_iter),
        ),
    )
    fixed = _timed(
        METHOD_FIXED,
        lambda: apa_apg_regression_path(
            problem.X,
            problem.y,
            problem.groups,
            fixed_lambdas,
            max_iter=int(args.max_iter),
            tol=float(args.tolerance),
            ord="inf",
            cache_quadratic=True,
            assume_diagonal_gram=bool(args.assume_diagonal_gram),
        ),
    )
    adaptive_support_tolerance = (
        float(args.adaptive_support_tolerance)
        if bool(args.adaptive_support_check)
        else None
    )
    adaptive_kkt_tolerance = (
        float(args.adaptive_kkt_tolerance)
        if bool(args.adaptive_kkt_check)
        else None
    )
    def solve_adaptive() -> Any:
        # Let the adaptive path calculate its own threshold.  Besides making
        # its wall time genuinely end-to-end, this activates its analytic zero
        # solution at lambda_max instead of spending an APA iteration budget
        # on an endpoint whose exact coefficient is already known.
        path = apa_apg_adaptive_regression_path(
            problem.X,
            problem.y,
            problem.groups,
            fit_intercept=False,
            minimum_ratio=float(args.lambda_min_ratio),
            initial_points=int(args.adaptive_initial_points),
            coefficient_tolerance=float(args.adaptive_coefficient_tolerance),
            objective_tolerance=float(args.adaptive_objective_tolerance),
            support_tolerance=adaptive_support_tolerance,
            max_points=int(args.adaptive_max_points),
            max_depth=int(args.adaptive_max_depth),
            predictor=str(args.adaptive_predictor),
            max_iter=int(args.max_iter),
            tol=float(args.tolerance),
            cache_quadratic=True,
            assume_diagonal_gram=bool(args.assume_diagonal_gram),
            kkt_tolerance=adaptive_kkt_tolerance,
            kkt_face_tolerance=float(args.adaptive_kkt_face_tolerance),
        )
        adaptive_lambda_max = float(path.metadata["lambda_max"])
        adaptive_zero_threshold = float(path.metadata["zero_solution_lambda"])
        threshold_scale = max(
            1.0,
            abs(lambda_max),
            abs(adaptive_lambda_max),
            abs(adaptive_zero_threshold),
        )
        if max(
            abs(adaptive_lambda_max - lambda_max),
            abs(adaptive_zero_threshold - lambda_max),
        ) > 1e-7 * threshold_scale:
            raise RuntimeError(
                "adaptive and shared zero-solution thresholds disagree: "
                f"{adaptive_zero_threshold:.16g} versus {lambda_max:.16g}"
            )
        return path

    adaptive = _timed(METHOD_ADAPTIVE, solve_adaptive)

    summaries: list[dict[str, Any]] = []
    details: dict[str, Any] = {
        "repeat": repeat,
        "seed": seed,
        "problem": {
            "n_samples": problem.n_samples,
            "raw_feature_count": problem.raw_feature_count,
            "requested_internal_nodes": int(args.nodes),
            "n_internal_nodes": problem.n_features,
            "tree_depth": problem.tree_depth,
            "tree_node_count": problem.tree_node_count,
            "design_rank": problem.design_rank,
            "gram_condition": problem.gram_condition,
            "n_groups": len(problem.groups),
            "group_memberships": problem.group_memberships,
            "tree_fit_seconds": problem.tree_fit_seconds,
            "transform_seconds": problem.transform_seconds,
            "lambda_setup_seconds": lambda_setup_seconds,
            "lambda_max": lambda_max,
            "gram": gram,
        },
        "methods": {},
    }

    for result in (exact, fixed, adaptive):
        summary = _base_summary(
            args=args,
            dataset=subset,
            problem=problem,
            repeat=repeat,
            seed=seed,
            method=result.method,
            result=result,
            lambda_max=lambda_max,
            lambda_setup_seconds=lambda_setup_seconds,
            gram=gram,
        )
        method_detail: dict[str, Any] = {
            "method": result.method,
            "wall_seconds": result.wall_seconds,
            "error_type": result.error_type,
            "error_message": result.error_message,
        }
        if result.path is not None:
            statistics = _path_statistics(result.path)
            summary.update(statistics)
            summary["method_lambda_max"] = float(
                result.path.metadata.get("lambda_max", lambda_max)
            )
            summary["zero_solution_lambda"] = float(
                result.path.metadata.get(
                    "zero_solution_lambda",
                    result.path.metadata.get("lambda_max", lambda_max),
                )
            )
            method_detail.update(statistics)
            method_detail.update(
                {
                    "lambdas": np.asarray(result.path.lambdas),
                    "coefficients": np.asarray(result.path.coefficients),
                    "events": tuple(str(event) for event in result.path.events),
                }
            )
            if result.method == METHOD_ADAPTIVE:
                metadata = dict(result.path.metadata)
                method_detail["adaptive_metadata"] = {
                    key: value
                    for key, value in metadata.items()
                    if key not in {"accepted_intervals", "unresolved_intervals"}
                }
                method_detail["unresolved_intervals"] = metadata.get(
                    "unresolved_intervals", ()
                )

        reference_available = bool(
            exact.path is not None
            and exact.path.exact
            and exact.path.status == "complete"
        )
        if reference_available and result.path is not None:
            if result.method == METHOD_EXACT:
                metric_summary, dense = _exact_dense_comparison(dense_lambdas)
            else:
                metric_summary, dense = _dense_comparison(
                    problem,
                    exact.path,
                    result.path,
                    dense_lambdas,
                    float(args.metric_support_tolerance),
                )
            summary.update(metric_summary)
            method_detail["dense_comparison"] = dense
        summaries.append(summary)
        details["methods"][result.method] = method_detail

    return summaries, details


def run_benchmark(args: argparse.Namespace) -> tuple[Path, Path, list[Path]]:
    dataset = scaling.load_regression_dataset(
        args.dataset,
        data_home=args.data_home,
        allow_download=bool(args.allow_download),
        fallback_to_diabetes=bool(args.fallback_to_diabetes),
    )
    if int(args.samples) < 0 or int(args.samples) > dataset.y.size:
        raise ValueError(
            f"samples must be zero or lie in [2, {dataset.y.size}]"
        )
    if int(args.samples) == 1:
        raise ValueError("samples must be zero or at least two")

    all_summaries: list[dict[str, Any]] = []
    repeats: list[dict[str, Any]] = []
    print(
        "repeat\tn\tp\tmethod\tstatus\tseconds\tpoints\tmax-coef-error",
        flush=True,
    )
    with threadpool_limits(limits=int(args.threads)):
        for repeat in range(int(args.repeats)):
            summaries, detail = _run_repeat(args, dataset, repeat)
            all_summaries.extend(summaries)
            repeats.append(detail)
            for summary in summaries:
                error = summary.get("coefficient_relative_error_max")
                error_text = "-" if error is None else f"{float(error):.3g}"
                print(
                    f"{repeat}\t{summary['n_samples']}\t"
                    f"{summary['n_internal_nodes']}\t{summary['method']}\t"
                    f"{summary['status']}\t{summary['wall_seconds']:.4g}\t"
                    f"{summary.get('native_points', '-')}\t{error_text}",
                    flush=True,
                )

    metadata = scaling.environment_metadata()
    metadata.update(
        {
            "schema_name": SCHEMA_NAME,
            "schema_version": SCHEMA_VERSION,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "command": list(sys.argv),
            "arguments": vars(args),
            "dataset_name": dataset.name,
            "dataset_requested": dataset.requested_name,
            "dataset_digest": dataset.digest,
            "dataset_rows": int(dataset.y.size),
            "dataset_raw_features": int(dataset.X.shape[1]),
            "dataset_fallback_reason": dataset.fallback_reason,
            "timing_scope": (
                "path solver call only; tree transform, shared lambda setup, "
                "dense evaluation, serialization, and plotting excluded. "
                "Exact hiCAP and adaptive APA each recompute lambda_max inside "
                "their timed solver calls; fixed-grid APA uses the shared "
                "precomputed grid and excludes that setup time."
            ),
            "dense_reference": "linear interpolation of certified exact hiCAP knots",
            "coefficient_error_scaling": (
                "L2 candidate-minus-reference error divided by the maximum "
                "exact coefficient L2 norm over the dense lambda grid"
            ),
            "objective_error_scaling": (
                "candidate-minus-reference objective difference divided by "
                "the maximum absolute exact objective over the dense lambda grid"
            ),
            "adaptive_kkt_semantics": (
                "strict residual/certification uses the exact floating-point "
                "active face; positive face tolerance is reported separately "
                "as relaxed near-face acceptance and is not a certificate"
            ),
            "loss_scaling": "0.5 * ||y - X beta||^2 / n",
            "penalty": "lambda * sum_g ||beta[g]||_inf",
            "diagonal_assumption_scope": (
                "valid for these tree-fitting rows and uniform weights; it "
                "must be rechecked for held-out/OOB rows or changed weights"
            ),
        }
    )
    document = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "metadata": metadata,
        "runs": all_summaries,
        "repeats": repeats,
    }
    output_directory = Path(args.output_dir).resolve()
    json_path, csv_path = _write_outputs(document, output_directory)

    figures: list[Path] = []
    if bool(args.plot):
        # Import only after timing so font discovery and backend initialization
        # cannot contaminate solver wall-clock measurements.
        from plot_sparse_pruning_adaptive_paths import save_benchmark_figure

        try:
            figures = save_benchmark_figure(
                _json_compatible(document),
                output_base=output_directory / "hicap_fixed_adaptive_paths",
                formats=tuple(args.formats),
            )
        except ValueError as exc:
            print(
                f"Plot skipped (JSON/CSV were retained): {exc}",
                file=sys.stderr,
                flush=True,
            )
    return json_path, csv_path, figures


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=("diabetes", "california_housing"),
        default="diabetes",
    )
    parser.add_argument("--data-home", type=Path, default=None)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument(
        "--fallback-to-diabetes",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=0,
        help="rows per repeat; zero uses the complete dataset",
    )
    parser.add_argument("--nodes", type=int, default=7)
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--lambda-min-ratio", type=float, default=1e-3)
    parser.add_argument("--fixed-points", type=int, default=21)
    parser.add_argument("--dense-points", type=int, default=401)
    parser.add_argument("--max-iter", type=int, default=2_000)
    parser.add_argument("--max-events", type=int, default=10_000)
    parser.add_argument("--tolerance", type=float, default=1e-7)
    parser.add_argument(
        "--metric-support-tolerance",
        type=float,
        default=1e-6,
        help="absolute coefficient threshold for support/topology comparisons",
    )
    parser.add_argument("--adaptive-initial-points", type=int, default=5)
    parser.add_argument("--adaptive-max-points", type=int, default=81)
    parser.add_argument("--adaptive-max-depth", type=int, default=10)
    parser.add_argument(
        "--adaptive-predictor", choices=("secant", "previous"), default="secant"
    )
    parser.add_argument(
        "--adaptive-coefficient-tolerance", type=float, default=1e-2
    )
    parser.add_argument(
        "--adaptive-objective-tolerance", type=float, default=5e-4
    )
    parser.add_argument(
        "--adaptive-support-tolerance", type=float, default=1e-3
    )
    parser.add_argument(
        "--adaptive-support-check",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="include thresholded-support equality in midpoint refinement",
    )
    parser.add_argument(
        "--adaptive-kkt-check",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "run independent strict KKT and optional relaxed near-face checks "
            "at every stored adaptive point"
        ),
    )
    parser.add_argument("--adaptive-kkt-tolerance", type=float, default=1e-3)
    parser.add_argument(
        "--adaptive-kkt-face-tolerance", type=float, default=1e-2
    )
    parser.add_argument(
        "--assume-diagonal-gram",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "use the O(p)-storage diagonal quadratic backend after an "
            "independent empirical Gram check"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BENCHMARK_DIRECTORY / "adaptive_path_comparison" / "diabetes",
    )
    parser.add_argument(
        "--plot", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--formats", nargs="+", choices=("png", "svg", "pdf"), default=("png", "svg")
    )
    return parser


def _validate_arguments(args: argparse.Namespace) -> None:
    integer_names = (
        "nodes",
        "min_samples_leaf",
        "repeats",
        "threads",
        "fixed_points",
        "dense_points",
        "max_iter",
        "max_events",
        "adaptive_initial_points",
        "adaptive_max_points",
        "adaptive_max_depth",
    )
    for name in integer_names:
        if int(getattr(args, name)) < 1:
            raise ValueError(f"{name} must be positive")
    if int(args.fixed_points) < 3 or int(args.dense_points) < 3:
        raise ValueError("fixed_points and dense_points must be at least three")
    if not 0 < float(args.lambda_min_ratio) < 1:
        raise ValueError("lambda_min_ratio must lie strictly between zero and one")
    for name in (
        "tolerance",
        "metric_support_tolerance",
        "adaptive_coefficient_tolerance",
        "adaptive_objective_tolerance",
        "adaptive_support_tolerance",
        "adaptive_kkt_tolerance",
    ):
        value = float(getattr(args, name))
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")
    kkt_face_tolerance = float(args.adaptive_kkt_face_tolerance)
    if not np.isfinite(kkt_face_tolerance) or kkt_face_tolerance < 0:
        raise ValueError(
            "adaptive_kkt_face_tolerance must be nonnegative and finite"
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        _validate_arguments(args)
        json_path, csv_path, figures = run_benchmark(args)
    except (ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    print(f"JSON: {json_path}")
    print(f"CSV:  {csv_path}")
    for figure in figures:
        print(f"Plot: {figure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
