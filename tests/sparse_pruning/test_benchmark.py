"""Smoke tests for the standalone sparse-pruning path benchmark."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


BENCHMARK_PATH = (
    Path(__file__).resolve().parents[2]
    / "benchmarks"
    / "benchmark_sparse_pruning_paths.py"
)
MODULE_NAME = "_benchmark_sparse_pruning_paths"
SPEC = importlib.util.spec_from_file_location(MODULE_NAME, BENCHMARK_PATH)
assert SPEC is not None and SPEC.loader is not None
BENCHMARK = importlib.util.module_from_spec(SPEC)
sys.modules[MODULE_NAME] = BENCHMARK
SPEC.loader.exec_module(BENCHMARK)


def test_path_benchmark_json_smoke(capsys):
    exit_code = BENCHMARK.main(
        [
            "--sizes",
            "20x3",
            "--path-points",
            "2",
            "--repeats",
            "1",
            "--max-iter",
            "5",
            "--seed",
            "7",
            "--json",
        ]
    )

    report = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert len(report) == 1
    assert report[0]["samples"] == 20
    assert report[0]["features"] == 3
    assert report[0]["path_points"] == 2
    assert [method["method"] for method in report[0]["methods"]] == [
        "apa_apg2_cold",
        "apa_apg2_warm",
        "hicap_exact",
    ]
    assert report[0]["methods"][0]["available"]
    assert report[0]["methods"][0]["wall_seconds_median"] >= 0.0
