"""Focused tests for the real-tree sparse-pruning scaling benchmark."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

BENCHMARK_DIRECTORY = Path(__file__).resolve().parents[2] / "benchmarks"
if str(BENCHMARK_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIRECTORY))

import plot_sparse_pruning_tree_scaling as scaling_plot
import plot_sparse_pruning_diagonal_topology as diagonal_plot
import benchmark_sparse_pruning_diagonal_coefficient_path as coefficient_benchmark
from _sparse_pruning_tree_scaling import (
    METHOD_APA_WARM,
    METHOD_EXACT,
    ScalingConfig,
    ScalingRecord,
    SolverPayload,
    build_tree_path_problem,
    load_regression_dataset,
    make_lambda_grid,
    run_solver_payload,
    subset_regression_dataset,
)


def test_full_coefficient_benchmark_accepts_root_only_tree(monkeypatch, tmp_path):
    def unnecessary_oracle(*args, **kwargs):
        raise AssertionError("a root-only tree needs no coefficient oracle")

    monkeypatch.setattr(coefficient_benchmark, "LaminarGroupLinfProx", unnecessary_oracle)
    monkeypatch.setattr(coefficient_benchmark, "hicap_regression_path", unnecessary_oracle)
    args = SimpleNamespace(seed=0, min_samples_leaf=2, max_events=100,
                           legacy_max_nodes=7, legacy_tolerance=1e-9)
    row = coefficient_benchmark._run_case(np.zeros((8, 2)), np.arange(8.), 3, 0, args)
    assert row["n_internal_nodes"] == 0
    assert row["full_exact"] is True
    assert row["full_status"] == "complete"
    assert row["full_stored_points"] == row["structural_stored_points"] == 1
    assert row["full_coefficient_storage_bytes"] == 0
    assert row["darkblue_status"] == row["legacy_status"] == "not_needed_no_splits"
    assert row["darkblue_total_seconds"] is None
    assert row["darkblue_max_relative_coefficient_error"] is None
    assert row["legacy_exact"] is None
    report = {"configuration": {"repeats": 1, "threads": 1},
              "dataset": {"n_observations": 8}, "records": [row]}
    coefficient_benchmark._write_artifacts(report, tmp_path)
    coefficient_benchmark._plot(report, tmp_path)
    assert (tmp_path / "benchmark.json").is_file()
    assert (tmp_path / "diagonal_coefficient_scaling.png").is_file()


def _small_diabetes_problem(internal_nodes: int):
    dataset = subset_regression_dataset(
        load_regression_dataset("diabetes"),
        80,
        seed=11,
    )
    return build_tree_path_problem(
        dataset.X,
        dataset.y,
        requested_internal_nodes=internal_nodes,
        seed=11,
        dataset_name=dataset.name,
        dataset_digest=dataset.digest,
        min_samples_leaf=2,
    )


def test_fitted_tree_problem_is_full_rank_with_rooted_laminar_groups():
    problem = _small_diabetes_problem(internal_nodes=7)

    assert problem.actual_internal_nodes == 7
    assert problem.X.shape == (80, 7)
    assert problem.design_rank == problem.actual_internal_nodes
    assert np.linalg.matrix_rank(problem.X) == problem.actual_internal_nodes
    assert problem.gram_max_off_diagonal_correlation < 1e-12
    assert np.allclose(problem.X.mean(axis=0), 0.0, atol=1e-12)
    assert np.isclose(problem.y.mean(), 0.0, atol=1e-12)

    expected_root = set(range(problem.actual_internal_nodes))
    group_sets = []
    for node, group in enumerate(problem.groups):
        assert group.dtype.kind in "iu"
        assert group.ndim == 1
        assert group.size == np.unique(group).size
        assert np.all(group >= 0)
        assert np.all(group < problem.actual_internal_nodes)
        assert int(group[0]) == node
        group_sets.append(set(int(index) for index in group))

    assert group_sets[0] == expected_root
    assert sum(group == expected_root for group in group_sets) == 1
    for left, left_group in enumerate(group_sets):
        for right_group in group_sets[left + 1 :]:
            assert (
                left_group.isdisjoint(right_group)
                or left_group <= right_group
                or right_group <= left_group
            )


def test_tiny_real_tree_solver_benchmark_smoke():
    problem = _small_diabetes_problem(internal_nodes=1)
    grid = make_lambda_grid(problem, n_points=2, minimum_ratio=0.1)
    config = ScalingConfig(max_iter=5, tolerance=1e-7)

    exact = run_solver_payload(
        SolverPayload(problem, METHOD_EXACT, grid, config)
    )
    apa = run_solver_payload(
        SolverPayload(problem, METHOD_APA_WARM, grid, config)
    )

    assert exact.outcome == "complete"
    assert exact.status == "complete"
    assert exact.exact
    assert apa.outcome == "complete"
    assert apa.status == "partial"
    assert apa.all_points_converged is False
    assert not apa.exact
    for result in (exact, apa):
        assert result.wall_seconds >= 0.0
        assert result.coefficients.shape[1] == problem.actual_internal_nodes
        assert np.all(np.isfinite(result.coefficients))


def test_real_tree_scaling_plot_writes_png(tmp_path):
    common = {
        "dataset_name": "diabetes",
        "dataset_digest": "test-fixture",
        "requested_samples": 40,
        "n_samples": 40,
        "requested_internal_nodes": 3,
        "n_internal_nodes": 3,
        "n_groups": 3,
        "design_rank": 3,
        "requested_path_points": 2,
        "max_iter": 5,
        "outcome": "complete",
        "solver_status": "complete",
    }
    rows = []
    for axis in ("nodes", "observations"):
        rows.append(
            asdict(
                ScalingRecord(
                    **common,
                    scaling_axis=axis,
                    method=METHOD_EXACT,
                    exact=True,
                    solver_wall_seconds=0.01,
                )
            )
        )
        rows.append(
            asdict(
                ScalingRecord(
                    **{**common, "solver_status": "partial"},
                    scaling_axis=axis,
                    method=METHOD_APA_WARM,
                    solver_wall_seconds=0.02,
                    all_points_converged=False,
                    coefficient_relative_error_max=1e-3,
                    objective_relative_excess_max=1e-5,
                )
            )
        )

    x_values, timings, _, _ = scaling_plot._timing_summary(
        rows, "nodes", METHOD_APA_WARM
    )
    np.testing.assert_array_equal(x_values, [3])
    np.testing.assert_allclose(timings, [0.02])
    invalid_x, _ = scaling_plot._uncertified_summary(
        rows, "nodes", METHOD_APA_WARM
    )
    assert invalid_x.size == 0

    figure = scaling_plot.make_figure(rows)
    output = tmp_path / "tree-scaling-smoke.png"
    figure.savefig(output, dpi=40)
    scaling_plot.plt.close(figure)

    assert output.is_file()
    assert output.stat().st_size > 0


def test_real_tree_scaling_cli_spawn_smoke(tmp_path):
    script = BENCHMARK_DIRECTORY / "benchmark_sparse_pruning_tree_scaling.py"
    output_directory = tmp_path / "scaling-results"
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--dataset",
            "diabetes",
            "--axis",
            "nodes",
            "--node-counts",
            "1",
            "--fixed-observations",
            "40",
            "--path-points",
            "3",
            "--metric-points",
            "3",
            "--max-iter",
            "20",
            "--repeats",
            "1",
            "--timeout-seconds",
            "20",
            "--threads",
            "1",
            "--seed",
            "17",
            "--output-dir",
            str(output_directory),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert completed.returncode == 0, completed.stderr
    csv_path = output_directory / "results.csv"
    json_path = output_directory / "results.json"
    assert csv_path.is_file() and csv_path.stat().st_size > 0
    assert json_path.is_file() and json_path.stat().st_size > 0

    document = json.loads(json_path.read_text(encoding="utf-8"))
    records = document["records"]
    assert document["schema_version"] == 2
    assert len(records) == 2
    assert {record["method"] for record in records} == {
        METHOD_EXACT,
        METHOD_APA_WARM,
    }
    assert {record["scaling_axis"] for record in records} == {"nodes"}
    assert all(record["n_samples"] == 40 for record in records)
    assert all(record["n_internal_nodes"] == 1 for record in records)
    assert all(record["design_rank"] == 1 for record in records)
    assert all(record["outcome"] == "complete" for record in records)
    assert all(record["schema_version"] == 2 for record in records)
    assert all(record["metric_points"] == 3 for record in records)
    assert all(record["max_events"] == 10_000 for record in records)
    assert all(record["min_samples_leaf"] == 2 for record in records)
    assert all(record["threads"] == 1 for record in records)


def test_diagonal_overlay_uses_requested_sizes_and_rejects_failed_hicap():
    new_document = {
        "records": [
            {
                "axis": "nodes",
                "requested_nodes": 511,
                "n_internal_nodes": actual,
                "topology_seconds": elapsed,
            }
            for actual, elapsed in ((430, 1.0), (438, 2.0), (438, 3.0))
        ]
    }
    x, median, lower, upper = diagonal_plot._new_series(
        new_document, "nodes", "topology_seconds"
    )
    np.testing.assert_array_equal(x, [511])
    np.testing.assert_allclose([median[0], lower[0], upper[0]], [2.0, 1.5, 2.5])

    legacy = {
        "records": [
            {
                "scaling_axis": "nodes",
                "method": "hicap_exact",
                "requested_internal_nodes": 31,
                "outcome": "complete",
                "solver_status": "complete",
                "exact": True,
                "solver_wall_seconds": 2.0,
            },
            {
                "scaling_axis": "nodes",
                "method": "hicap_exact",
                "requested_internal_nodes": 63,
                "outcome": "complete",
                "solver_status": "numerical_failure",
                "exact": False,
                "solver_wall_seconds": 30.0,
            },
        ]
    }
    old_x, old_median, _, _ = diagonal_plot._old_series(
        legacy, "nodes", "hicap_exact"
    )
    np.testing.assert_array_equal(old_x, [31])
    np.testing.assert_allclose(old_median, [2.0])


def test_diagonal_overlay_rejects_unknown_schema_versions():
    new = {
        "schema_name": diagonal_plot.NEW_SCHEMA,
        "schema_version": diagonal_plot.NEW_SCHEMA_VERSION,
        "metadata": {"dataset_digest": "same", "arguments": {}},
    }
    legacy = {
        "schema_version": diagonal_plot.LEGACY_SCHEMA_VERSION,
        "metadata": {"dataset_digest": "same", "arguments": {}},
    }
    diagonal_plot._validate_compatibility(new, legacy)

    with pytest.raises(ValueError, match="schema version"):
        diagonal_plot._validate_compatibility(
            {**new, "schema_version": 999}, legacy
        )
    with pytest.raises(ValueError, match="legacy benchmark schema version"):
        diagonal_plot._validate_compatibility(
            new, {**legacy, "schema_version": 999}
        )
