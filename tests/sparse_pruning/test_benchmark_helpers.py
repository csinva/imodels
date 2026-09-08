"""Tests for the real-tree path-scaling benchmark helpers."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


BENCHMARK_DIRECTORY = Path(__file__).resolve().parents[2] / "benchmarks"
if str(BENCHMARK_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIRECTORY))

import _sparse_pruning_tree_scaling as scaling


def test_source_provenance_tracks_uncommitted_source_not_artifacts(tmp_path, monkeypatch):
    runtime = tmp_path / "imodels"
    benchmark = tmp_path / "benchmarks"
    runtime.mkdir()
    benchmark.mkdir()
    module = runtime / "example.py"
    module.write_text("VALUE = 1\n")
    (benchmark / "run.py").write_text("pass\n")
    (tmp_path / "pyproject.toml").write_text("[project]\nname='example'\n")

    def git_result(command, **kwargs):
        assert command[:3] == ["git", "-C", str(tmp_path)]
        return SimpleNamespace(stdout="a" * 40 if command[3] == "rev-parse" else " M imodels/example.py\n")

    monkeypatch.setattr(scaling.subprocess, "run", git_result)
    first = scaling.source_provenance(tmp_path)
    assert first["git_commit_sha"] == "a" * 40
    assert first["git_dirty"] is True
    assert set(first["source_file_sha256"]) == {
        "imodels/example.py", "benchmarks/run.py", "pyproject.toml",
    }
    (benchmark / "results.json").write_text('{"seconds": 1}')
    assert scaling.source_provenance(tmp_path) == first
    module.write_text("VALUE = 2\n")
    changed = scaling.source_provenance(tmp_path)
    assert changed["git_commit_sha"] == first["git_commit_sha"]
    assert changed["source_sha256"] != first["source_sha256"]
    assert changed["source_file_sha256"]["benchmarks/run.py"] == first["source_file_sha256"]["benchmarks/run.py"]


def test_source_provenance_works_without_git(tmp_path, monkeypatch):
    def unavailable(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(scaling.subprocess, "run", unavailable)
    provenance = scaling.source_provenance(tmp_path)
    assert provenance["git_commit_sha"] is None
    assert provenance["git_dirty"] is None
    assert len(provenance["source_sha256"]) == 64
    assert provenance["source_file_sha256"] == {}


def test_dataset_subsets_are_deterministic_and_nested() -> None:
    dataset = scaling.load_regression_dataset("diabetes")
    small = scaling.subset_regression_dataset(dataset, 32, seed=7)
    repeated = scaling.subset_regression_dataset(dataset, 32, seed=7)
    large = scaling.subset_regression_dataset(dataset, 64, seed=7)

    assert dataset.name == "diabetes"
    assert dataset.X.shape == (442, 10)
    np.testing.assert_array_equal(small.X, repeated.X)
    assert small.digest == repeated.digest
    assert {tuple(row) for row in small.X} <= {tuple(row) for row in large.X}


def test_california_cache_failure_has_explicit_diabetes_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable(**kwargs):
        raise OSError("not cached")

    monkeypatch.setattr(scaling, "fetch_california_housing", unavailable)
    dataset = scaling.load_regression_dataset(
        "california", allow_download=False, fallback_to_diabetes=True
    )

    assert dataset.requested_name == "california"
    assert dataset.name == "diabetes"
    assert "not cached" in dataset.fallback_reason
    with pytest.raises(RuntimeError, match="not cached"):
        scaling.load_regression_dataset(
            "california", allow_download=False, fallback_to_diabetes=False
        )


def test_descendant_groups_follow_preorder_parents() -> None:
    groups = scaling.descendant_groups([-1, 0, 1, 1, 0])
    assert [group.tolist() for group in groups] == [
        [0, 1, 2, 3, 4],
        [1, 2, 3],
        [2],
        [3],
        [4],
    ]


def test_tree_problem_and_lambda_grid_match_exact_threshold() -> None:
    dataset = scaling.subset_regression_dataset(
        scaling.load_regression_dataset("diabetes"), 64, seed=11
    )
    problem = scaling.build_tree_path_problem(
        dataset.X,
        dataset.y,
        3,
        seed=11,
        dataset_name=dataset.name,
        dataset_digest=dataset.digest,
    )
    grid = scaling.make_lambda_grid(problem, n_points=4, minimum_ratio=1e-2)
    result = scaling.run_solver_payload(
        scaling.SolverPayload(
            problem,
            scaling.METHOD_EXACT,
            grid,
            scaling.ScalingConfig(max_iter=500),
        )
    )

    assert problem.actual_internal_nodes == 3
    assert problem.design_rank == problem.actual_internal_nodes
    assert problem.gram_max_off_diagonal_correlation < 1e-12
    assert np.max(np.abs(problem.X.mean(axis=0))) < 1e-12
    assert abs(problem.y.mean()) < 1e-12
    np.testing.assert_array_equal(
        problem.groups[0], np.arange(problem.actual_internal_nodes)
    )
    assert result.outcome == result.status == "complete"
    assert result.exact
    assert result.lambdas[-1] == 0
    assert result.lambdas[0] == pytest.approx(grid.lambda_max, rel=2e-9)
    assert grid.lambda_upper == pytest.approx(1.05 * grid.lambda_max)


def test_solver_comparison_and_result_serialization(tmp_path: Path) -> None:
    dataset = scaling.subset_regression_dataset(
        scaling.load_regression_dataset("diabetes"), 40, seed=13
    )
    problem = scaling.build_tree_path_problem(
        dataset.X,
        dataset.y,
        2,
        seed=13,
        dataset_name=dataset.name,
        dataset_digest=dataset.digest,
    )
    grid = scaling.make_lambda_grid(problem, n_points=3)
    config = scaling.ScalingConfig(max_iter=10)
    exact = scaling.run_solver_payload(
        scaling.SolverPayload(problem, scaling.METHOD_EXACT, grid, config)
    )
    apa = scaling.run_solver_payload(
        scaling.SolverPayload(problem, scaling.METHOD_APA_WARM, grid, config)
    )
    comparison = scaling.compare_paths(
        problem,
        exact,
        apa,
        grid.lambdas,
        support_tolerance=config.support_tolerance,
    )
    records = [
        scaling.make_scaling_record(
            problem,
            grid,
            config,
            result,
            scaling_axis="nodes",
            requested_samples=40,
            repeat=0,
            comparison=comparison if result.method == scaling.METHOD_APA_WARM else None,
        )
        for result in (exact, apa)
    ]
    csv_path, json_path = scaling.write_results(
        records,
        csv_path=tmp_path / "result.csv",
        json_path=tmp_path / "result.json",
        metadata={"purpose": "test"},
    )
    document = scaling.read_json_results(json_path)

    assert csv_path.stat().st_size > 0
    assert len(document["records"]) == 2
    assert len(document["aggregates"]) == 2
    assert comparison["coefficient_relative_error_max"] >= 0
    assert 0 <= comparison["support_agreement_fraction"] <= 1


def test_result_schema_rejects_partial_or_mismatched_records(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "result.csv"
    json_path = tmp_path / "result.json"
    with pytest.raises(ValueError, match="missing ScalingRecord fields"):
        scaling.write_results(
            [{"schema_version": scaling.SCHEMA_VERSION}],
            csv_path=csv_path,
            json_path=json_path,
        )

    record = scaling.ScalingRecord()
    scaling.write_results(
        [record],
        csv_path=csv_path,
        json_path=json_path,
        metadata={"schema_version": -1},
    )
    document = json.loads(json_path.read_text(encoding="utf-8"))
    assert document["metadata"]["schema_version"] == scaling.SCHEMA_VERSION

    document["records"][0]["schema_version"] = -1
    json_path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported schema"):
        scaling.read_json_results(json_path)


def test_aggregation_keeps_timeouts_out_of_completed_timing() -> None:
    base = {
        "scaling_axis": "nodes",
        "method": scaling.METHOD_EXACT,
        "requested_samples": 64,
        "requested_internal_nodes": 7,
    }
    rows = [
        {**base, "outcome": "complete", "solver_wall_seconds": 1.0},
        {**base, "outcome": "complete", "solver_wall_seconds": 3.0},
        {
            **base,
            "outcome": "timeout",
            "solver_wall_seconds": None,
            "censored": True,
        },
    ]
    summary = scaling.aggregate_records(rows)[0]

    assert summary["completed_runs"] == 2
    assert summary["timeout_runs"] == summary["censored_runs"] == 1
    assert summary["wall_seconds_median"] == pytest.approx(2.0)


def test_aggregation_times_usable_fixed_budget_apa_paths() -> None:
    rows = [
        {
            "scaling_axis": "nodes",
            "method": scaling.METHOD_APA_WARM,
            "requested_samples": 64,
            "requested_internal_nodes": 7,
            "outcome": "complete",
            "solver_status": "partial",
            "solver_wall_seconds": 2.5,
            "all_points_converged": False,
            "sample_coefficient_relative_error_median": 1e-3,
        }
    ]

    summary = scaling.aggregate_records(rows)[0]

    assert summary["executed_runs"] == summary["usable_runs"] == 1
    assert summary["completed_runs"] == 0
    assert summary["incomplete_path_runs"] == 1
    assert summary["iteration_limited_runs"] == 1
    assert summary["wall_seconds_median"] == pytest.approx(2.5)
    assert summary["sample_coefficient_relative_error_median"] == pytest.approx(
        1e-3
    )
