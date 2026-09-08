"""Focused tests for the adaptive fitted-tree path benchmark helpers."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


BENCHMARK_PATH = (
    Path(__file__).resolve().parents[2]
    / "benchmarks"
    / "benchmark_sparse_pruning_adaptive_paths.py"
)
MODULE_NAME = "_benchmark_sparse_pruning_adaptive_paths_test"
SPEC = importlib.util.spec_from_file_location(MODULE_NAME, BENCHMARK_PATH)
assert SPEC is not None and SPEC.loader is not None
BENCHMARK = importlib.util.module_from_spec(SPEC)
sys.modules[MODULE_NAME] = BENCHMARK
SPEC.loader.exec_module(BENCHMARK)


def test_path_statistics_exposes_nonfinite_kkt_failures() -> None:
    path = SimpleNamespace(
        diagnostics=(
            {
                "lambda": 1.0,
                "n_iter": 3,
                "converged": False,
                "kkt_relative_residual": np.inf,
                "kkt_relaxed_relative_residual": 0.2,
            },
        ),
        metadata={},
        status="partial",
        exact=False,
        n_points=1,
    )

    result = BENCHMARK._path_statistics(path)

    assert result["max_relative_kkt_residual"] is None
    assert result["n_nonfinite_kkt_residuals"] == 1
    assert result["max_relative_relaxed_kkt_residual"] == 0.2
    assert result["n_nonfinite_relaxed_kkt_residuals"] == 0


def test_dense_path_errors_are_invariant_to_response_rescaling() -> None:
    X = np.asarray([[-1.0], [1.0]])
    y = np.asarray([-1.0, 1.0])
    lambdas = np.asarray([1.0, 0.5, 0.0])
    reference = SimpleNamespace(
        lambdas=np.asarray([1.0, 0.0]),
        coefficients=np.asarray([[0.0], [1.0]]),
    )
    candidate = SimpleNamespace(
        lambdas=np.asarray([1.0, 0.0]),
        coefficients=np.asarray([[0.0], [0.8]]),
    )
    problem = SimpleNamespace(
        X=X,
        y=y,
        groups=(np.asarray([0]),),
        parent_indices=np.asarray([-1]),
        n_samples=2,
    )
    baseline, _ = BENCHMARK._dense_comparison(
        problem, reference, candidate, lambdas, support_tolerance=1e-8
    )

    multiplier = 1e-7
    scaled_problem = SimpleNamespace(
        X=X,
        y=multiplier * y,
        groups=problem.groups,
        parent_indices=problem.parent_indices,
        n_samples=2,
    )
    scaled_reference = SimpleNamespace(
        lambdas=multiplier * reference.lambdas,
        coefficients=multiplier * reference.coefficients,
    )
    scaled_candidate = SimpleNamespace(
        lambdas=multiplier * candidate.lambdas,
        coefficients=multiplier * candidate.coefficients,
    )
    scaled, _ = BENCHMARK._dense_comparison(
        scaled_problem,
        scaled_reference,
        scaled_candidate,
        multiplier * lambdas,
        support_tolerance=multiplier * 1e-8,
    )

    for key in (
        "coefficient_relative_error_median",
        "coefficient_relative_error_max",
        "objective_relative_gap_median",
        "objective_relative_gap_max",
        "objective_relative_excess_median",
        "objective_relative_excess_max",
    ):
        np.testing.assert_allclose(scaled[key], baseline[key], rtol=1e-12)
