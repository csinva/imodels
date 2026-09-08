#!/usr/bin/env python3
"""Benchmark the diagonal exact-prox and exact-topology tree paths.

The workload uses local stumps from real fitted regression trees.  It times:

* all exact topology knots directly from fitted-tree node statistics;
* all exact zero-support topology knots via tree isotonic regression; and
* a requested grid of exact coefficient points via the diagonal one-sweep
  laminar proximal map.

Small problems are additionally checked against the complete hiCAP homotopy.
The companion plotter can overlay the existing hiCAP/APA-APG2 scaling artifact
when both use the same dataset, seed, sample sizes, and tree sizes.
"""
from __future__ import annotations

import os
import tempfile

for _variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_variable, "1")
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "imodels-mpl-cache"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(tempfile.gettempdir(), "imodels-xdg-cache"))

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from threadpoolctl import threadpool_limits
from sklearn.tree import DecisionTreeRegressor


BENCHMARK_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY_ROOT = BENCHMARK_DIRECTORY.parent
for _directory in (REPOSITORY_ROOT, BENCHMARK_DIRECTORY):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

from _sparse_pruning_tree_scaling import (  # noqa: E402
    build_tree_path_problem,
    environment_metadata,
    load_regression_dataset,
    subset_regression_dataset,
)
from imodels.tree.sparse_pruning import (  # noqa: E402
    fitted_tree_linf_exact_topology_path,
)
from imodels.tree.sparse_pruning.optimization import (  # noqa: E402
    hicap_regression_path,
    laminar_group_linf_exact_topology_path,
    laminar_group_linf_regression_path,
)


SCHEMA_NAME = "imodels.sparse_pruning.diagonal_topology_benchmark"
SCHEMA_VERSION = 3


def _lambda_grid(lambda_max: float, n_points: int, minimum_ratio: float):
    positive = np.geomspace(
        1.05 * lambda_max,
        1.05 * lambda_max * minimum_ratio,
        num=n_points - 1,
    )
    return np.r_[positive, 0.0]


def _topology_from_beta(beta: np.ndarray, groups, tolerance: float):
    active = np.abs(beta) > tolerance
    return tuple(
        number for number, group in enumerate(groups) if np.any(active[group])
    )


def _validate_against_hicap(problem, topology, coefficient_path, tolerance):
    started = time.perf_counter()
    exact = hicap_regression_path(
        problem.X,
        problem.y,
        problem.groups,
        fit_intercept=False,
        tolerance=tolerance,
    )
    wall_seconds = time.perf_counter() - started
    if not exact.exact:
        return {
            "hicap_validation_status": exact.status,
            "hicap_validation_seconds": wall_seconds,
            "hicap_stored_coefficient_points": exact.n_points,
            "max_coefficient_error": None,
            "topology_states_match": False,
            "max_topology_activation_error": None,
        }

    reference = np.vstack(
        [
            (
                np.zeros(problem.n_features)
                if lam >= exact.lambdas[0]
                else exact.at(float(lam))[0]
            )
            for lam in coefficient_path.lambdas
        ]
    )
    maximum_error = float(
        np.max(np.abs(reference - coefficient_path.coefficients), initial=0.0)
    )
    positive = np.unique(topology.activation_lambdas)
    bounds = np.unique(np.r_[0.0, positive[positive > 0.0]])
    midpoints = 0.5 * (bounds[:-1] + bounds[1:])
    topology_matches = True
    coefficient_scale = max(
        float(np.max(np.abs(reference), initial=0.0)), 1.0
    )
    support_tolerance = 100.0 * tolerance * coefficient_scale
    for lam in midpoints:
        beta = exact.at(float(lam))[0]
        observed = _topology_from_beta(
            beta, problem.groups, support_tolerance
        )
        if observed != topology.topology_at(float(lam)):
            topology_matches = False
            break

    feature_activation = np.zeros(problem.n_features, dtype=float)
    for feature in range(problem.n_features):
        active_rows = np.flatnonzero(
            np.abs(exact.coefficients[:, feature]) > support_tolerance
        )
        if active_rows.size:
            first_active = int(active_rows[0])
            feature_activation[feature] = float(
                exact.lambdas[max(0, first_active - 1)]
            )
    reference_activation = np.asarray(
        [np.max(feature_activation[group]) for group in problem.groups]
    )
    maximum_activation_error = float(
        np.max(
            np.abs(reference_activation - topology.activation_lambdas),
            initial=0.0,
        )
    )
    return {
        "hicap_validation_status": exact.status,
        "hicap_validation_seconds": wall_seconds,
        "hicap_stored_coefficient_points": exact.n_points,
        "max_coefficient_error": maximum_error,
        "topology_states_match": bool(
            topology_matches
            and maximum_activation_error
            <= 100.0 * tolerance * max(1.0, float(topology.lambdas[0]))
        ),
        "max_topology_activation_error": maximum_activation_error,
    }


def _run_case(
    dataset,
    *,
    axis: str,
    n_samples: int,
    n_nodes: int,
    repeat: int,
    seed: int,
    min_samples_leaf: int,
    path_points: int,
    minimum_ratio: float,
    tolerance: float,
    validate_hicap_nodes: int,
):
    subset = subset_regression_dataset(dataset, n_samples, seed=seed)
    problem = build_tree_path_problem(
        subset.X,
        subset.y,
        n_nodes,
        seed=seed,
        dataset_name=subset.name,
        dataset_digest=subset.digest,
        min_samples_leaf=min_samples_leaf,
    )
    native_estimator = DecisionTreeRegressor(
        max_leaf_nodes=n_nodes + 1,
        min_samples_leaf=min_samples_leaf,
        random_state=seed,
    ).fit(subset.X, subset.y)
    diagonal_tolerance = (
        32.0 * np.finfo(float).eps * max(1, problem.n_features)
    )
    if problem.gram_max_off_diagonal_correlation > diagonal_tolerance:
        raise RuntimeError(
            "fitted local-stump Gram matrix failed numerical diagonality "
            "verification"
        )

    started = time.perf_counter()
    fitted_tree_topology = fitted_tree_linf_exact_topology_path(
        native_estimator
    )
    fitted_tree_topology_seconds = time.perf_counter() - started
    np.testing.assert_array_equal(fitted_tree_topology.node_ids, problem.node_ids)
    np.testing.assert_allclose(
        fitted_tree_topology.metadata["linear_scores"],
        problem.tree_linear_scores,
        atol=100.0 * tolerance,
        rtol=0.0,
    )

    started = time.perf_counter()
    topology = laminar_group_linf_exact_topology_path(
        problem.X,
        problem.y,
        problem.groups,
        fit_intercept=False,
        assume_diagonal_gram=True,
        include_coefficients=False,
        solver_tolerance=tolerance,
    )
    topology_seconds = time.perf_counter() - started
    np.testing.assert_allclose(
        fitted_tree_topology.activation_lambdas,
        topology.activation_lambdas,
        atol=100.0 * tolerance * max(1.0, float(topology.lambdas[0])),
        rtol=0.0,
    )
    lambda_max = float(topology.lambdas[0])
    lambdas = _lambda_grid(lambda_max, path_points, minimum_ratio)

    started = time.perf_counter()
    coefficient_path = laminar_group_linf_regression_path(
        problem.X,
        problem.y,
        problem.groups,
        lambdas,
        fit_intercept=False,
        assume_diagonal_gram=True,
        tol=tolerance,
    )
    coefficient_seconds = time.perf_counter() - started
    stationarity = [
        float(item.get("relative_stationarity_residual", 0.0))
        for item in coefficient_path.diagnostics
    ]

    validation: dict[str, Any] = {
        "hicap_validation_status": None,
        "hicap_validation_seconds": None,
        "hicap_stored_coefficient_points": None,
        "max_coefficient_error": None,
        "topology_states_match": None,
        "max_topology_activation_error": None,
    }
    if problem.n_features <= validate_hicap_nodes:
        validation = _validate_against_hicap(
            problem, topology, coefficient_path, tolerance
        )

    return {
        "axis": axis,
        "repeat": repeat,
        "seed": seed,
        "n_samples": problem.n_samples,
        "requested_nodes": n_nodes,
        "n_internal_nodes": problem.n_features,
        "tree_depth": problem.tree_depth,
        "group_memberships": problem.group_memberships,
        "gram_condition": problem.gram_condition,
        "gram_max_off_diagonal": problem.gram_max_off_diagonal,
        "gram_max_off_diagonal_correlation": (
            problem.gram_max_off_diagonal_correlation
        ),
        "numerical_diagonal_correlation_tolerance": diagonal_tolerance,
        "diagonality_verified": True,
        "tree_fit_seconds": problem.tree_fit_seconds,
        "transform_seconds": problem.transform_seconds,
        "lambda_max": lambda_max,
        "n_topology_knots": topology.n_knots,
        "n_topology_states": topology.n_states,
        "fitted_tree_topology_seconds": fitted_tree_topology_seconds,
        "topology_seconds": topology_seconds,
        "coefficient_grid_points": coefficient_path.n_points,
        "coefficient_grid_seconds": coefficient_seconds,
        "coefficient_points_per_second": (
            coefficient_path.n_points / coefficient_seconds
        ),
        "max_relative_stationarity_residual": max(stationarity, default=0.0),
        "point_solutions_certified": coefficient_path.metadata[
            "point_solutions_certified"
        ],
        **validation,
    }


def run_benchmark(arguments: argparse.Namespace) -> dict[str, Any]:
    dataset = load_regression_dataset(
        arguments.dataset,
        data_home=arguments.data_home,
        allow_download=arguments.allow_download,
        fallback_to_diabetes=arguments.fallback_to_diabetes,
    )
    records = []
    with threadpool_limits(limits=arguments.threads):
        if arguments.axis in {"nodes", "both"}:
            for requested_nodes in arguments.node_counts:
                for repeat in range(arguments.repeats):
                    records.append(
                        _run_case(
                            dataset,
                            axis="nodes",
                            n_samples=arguments.fixed_observations,
                            n_nodes=requested_nodes,
                            repeat=repeat,
                            seed=arguments.seed + repeat,
                            min_samples_leaf=arguments.min_samples_leaf,
                            path_points=arguments.path_points,
                            minimum_ratio=arguments.lambda_min_ratio,
                            tolerance=arguments.tolerance,
                            validate_hicap_nodes=arguments.validate_hicap_nodes,
                        )
                    )
        if arguments.axis in {"observations", "both"}:
            for observations in arguments.observation_counts:
                for repeat in range(arguments.repeats):
                    records.append(
                        _run_case(
                            dataset,
                            axis="observations",
                            n_samples=observations,
                            n_nodes=arguments.fixed_nodes,
                            repeat=repeat,
                            seed=arguments.seed + repeat,
                            min_samples_leaf=arguments.min_samples_leaf,
                            path_points=arguments.path_points,
                            minimum_ratio=arguments.lambda_min_ratio,
                            tolerance=arguments.tolerance,
                            validate_hicap_nodes=arguments.validate_hicap_nodes,
                        )
                    )
    return {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "metadata": {
            **environment_metadata(),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "dataset_name": dataset.name,
            "dataset_digest": dataset.digest,
            "dataset_rows": dataset.y.size,
            "dataset_raw_features": dataset.X.shape[1],
            "dataset_fallback_reason": dataset.fallback_reason,
            "arguments": {
                key: (str(value) if isinstance(value, Path) else value)
                for key, value in vars(arguments).items()
            },
            "timing_scope": (
                "complete path call; tree fit, stump transform, and group "
                "construction excluded"
            ),
        },
        "records": records,
    }


def _write_outputs(document: dict[str, Any], output_directory: Path):
    output_directory.mkdir(parents=True, exist_ok=True)
    json_path = output_directory / "results.json"
    csv_path = output_directory / "results.csv"
    json_path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    records = document["records"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    return json_path, csv_path


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="california_housing")
    parser.add_argument("--data-home", default=None)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument(
        "--fallback-to-diabetes",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--axis", choices=("nodes", "observations", "both"), default="both")
    parser.add_argument("--node-counts", nargs="+", type=int, default=[3, 7, 15, 31, 63, 127, 255, 511])
    parser.add_argument("--observation-counts", nargs="+", type=int, default=[128, 512, 2048, 8192, 16384])
    parser.add_argument("--fixed-observations", type=int, default=1024)
    parser.add_argument("--fixed-nodes", type=int, default=15)
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    parser.add_argument("--path-points", type=int, default=20)
    parser.add_argument("--lambda-min-ratio", type=float, default=1e-3)
    parser.add_argument("--tolerance", type=float, default=1e-8)
    parser.add_argument("--validate-hicap-nodes", type=int, default=31)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BENCHMARK_DIRECTORY / "diagonal_topology_scaling" / "california",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    document = run_benchmark(arguments)
    json_path, csv_path = _write_outputs(document, arguments.output_dir)
    print(json_path)
    print(csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
