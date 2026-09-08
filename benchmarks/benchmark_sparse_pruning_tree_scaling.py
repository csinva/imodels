#!/usr/bin/env python3
"""Benchmark exact hiCAP and warm APA-APG2 on fitted tree features.

Two one-factor-at-a-time sweeps are run on a real regression dataset:

* ``nodes`` changes the number of internal CART nodes at fixed sample size;
* ``observations`` changes sample size at a fixed number of internal nodes.

Each tree is converted to the same centered local-stump basis used by sparse
pruning in imodels.  Both solvers therefore receive exactly the same design,
response, descendant groups, objective scaling, and lambda grid.  Solver calls
run sequentially in fresh spawned processes so native numerical routines can
be stopped at a hard wall-clock timeout.

Example
-------
python \
    benchmarks/benchmark_sparse_pruning_tree_scaling.py \
    --dataset california_housing \
    --data-home ~/cache_imodels_data/sklearn_data \
    --node-counts 3 7 15 31 63 \
    --observation-counts 128 256 512 1024 2048 4096 8192
"""
from __future__ import annotations

# Apply reproducible defaults before NumPy/SciPy are imported in this process
# or in a spawned child. threadpoolctl below enforces the requested setting at
# runtime as well.
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
import multiprocessing as mp
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from threadpoolctl import threadpool_limits


BENCHMARK_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY_ROOT = BENCHMARK_DIRECTORY.parent
for source_directory in (REPOSITORY_ROOT, BENCHMARK_DIRECTORY):
    if str(source_directory) not in sys.path:
        sys.path.insert(0, str(source_directory))

import _sparse_pruning_tree_scaling as scaling


def _terminal_result(
    payload: scaling.SolverPayload,
    *,
    outcome: str,
    status: str,
    wall_seconds: float,
    error_type: str,
    error_message: str,
) -> scaling.SolverResult:
    return scaling.SolverResult(
        method=payload.method,
        outcome=outcome,
        status=status,
        exact=False,
        wall_seconds=float(wall_seconds),
        lambdas=np.empty(0),
        coefficients=np.empty((0, payload.problem.n_features)),
        error_type=error_type,
        error_message=error_message,
    )


def _solver_worker(
    sender: Any,
    payload: scaling.SolverPayload,
    threads: int,
) -> None:
    """Spawn target that always attempts to return a SolverResult."""

    try:
        with threadpool_limits(limits=threads):
            result = scaling.run_solver_payload(payload)
        sender.send(result)
    except BaseException as exc:
        # A failure outside run_solver_payload should still become a durable
        # record instead of silently killing the benchmark child.
        result = _terminal_result(
            payload,
            outcome="failed",
            status="failed",
            wall_seconds=0.0,
            error_type=type(exc).__name__,
            error_message=str(exc),
        )
        try:
            sender.send(result)
        except BaseException:
            pass
    finally:
        sender.close()


def run_with_timeout(
    payload: scaling.SolverPayload,
    *,
    timeout_seconds: float,
    threads: int,
    context: mp.context.BaseContext | None = None,
) -> tuple[scaling.SolverResult, float, bool]:
    """Run one solver in isolation and return result, parent time, censoring."""

    if not np.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive and finite")
    if threads < 1:
        raise ValueError("threads must be positive")
    if context is None:
        context = mp.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(
        target=_solver_worker,
        args=(sender, payload, int(threads)),
    )
    started = time.perf_counter()
    process.start()
    sender.close()
    result: scaling.SolverResult | None = None
    deadline = started + float(timeout_seconds)
    try:
        while True:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                break
            if receiver.poll(min(0.05, remaining)):
                result = receiver.recv()
                break
            if not process.is_alive():
                if receiver.poll(0.1):
                    result = receiver.recv()
                break
    finally:
        parent_seconds = time.perf_counter() - started

    if result is None and process.is_alive():
        process.terminate()
        process.join(timeout=2.0)
        if process.is_alive() and hasattr(process, "kill"):
            process.kill()
            process.join(timeout=2.0)
        receiver.close()
        return (
            _terminal_result(
                payload,
                outcome="timeout",
                status="timeout",
                wall_seconds=timeout_seconds,
                error_type="TimeoutError",
                error_message=f"exceeded {timeout_seconds:g} seconds",
            ),
            float(parent_seconds),
            True,
        )

    process.join(timeout=2.0)
    if process.is_alive():
        process.terminate()
        process.join(timeout=2.0)
    exit_code = process.exitcode
    receiver.close()
    if result is None:
        result = _terminal_result(
            payload,
            outcome="failed",
            status="worker_crash",
            wall_seconds=parent_seconds,
            error_type="WorkerCrash",
            error_message=f"solver child exited with code {exit_code}",
        )
    return result, float(parent_seconds), False


def _default_observation_counts(n_available: int) -> list[int]:
    candidates = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    values = [value for value in candidates if value <= n_available]
    if n_available not in values and (not values or n_available < 2 * values[-1]):
        values.append(n_available)
    return sorted(set(values))


def _case_specs(
    args: argparse.Namespace,
    n_available: int,
) -> list[tuple[str, int, int]]:
    node_counts = (
        [3, 7, 15, 31, 63]
        if args.node_counts is None
        else [int(value) for value in args.node_counts]
    )
    observation_counts = (
        _default_observation_counts(n_available)
        if args.observation_counts is None
        else [int(value) for value in args.observation_counts]
    )
    fixed_observations = (
        min(2048, n_available)
        if args.fixed_observations is None
        else int(args.fixed_observations)
    )
    for value in [fixed_observations, *observation_counts]:
        if value < 2 or value > n_available:
            raise ValueError(
                f"observation counts must lie in [2, {n_available}]; got {value}"
            )
    for value in [int(args.fixed_nodes), *node_counts]:
        if value < 1:
            raise ValueError("node counts must be positive")

    cases: list[tuple[str, int, int]] = []
    if args.axis in {"both", "nodes"}:
        cases.extend(
            ("nodes", fixed_observations, value)
            for value in sorted(set(node_counts))
        )
    if args.axis in {"both", "observations"}:
        cases.extend(
            ("observations", value, int(args.fixed_nodes))
            for value in sorted(set(observation_counts))
        )
    return cases


def _write_current_results(
    records: list[scaling.ScalingRecord],
    *,
    args: argparse.Namespace,
    dataset: scaling.RegressionDataset,
    csv_path: Path,
    json_path: Path,
) -> None:
    metadata = scaling.environment_metadata()
    metadata.update(
        {
            "dataset_name": dataset.name,
            "dataset_requested": dataset.requested_name,
            "dataset_digest": dataset.digest,
            "dataset_rows": int(dataset.y.size),
            "dataset_raw_features": int(dataset.X.shape[1]),
            "dataset_fallback_reason": dataset.fallback_reason,
            "command": list(sys.argv),
            "arguments": vars(args),
            "timing_scope": "solver call only; spawned-process overhead separate",
            "exact_workload": "all certified breakpoints from lambda_max to zero",
            "apa_workload": "requested warm-started lambda grid including zero",
        }
    )
    scaling.write_results(
        records,
        csv_path=csv_path,
        json_path=json_path,
        metadata=metadata,
    )


def _format_seconds(result: scaling.SolverResult) -> str:
    if result.outcome == "timeout":
        return f">={result.wall_seconds:.3g}s"
    if result.outcome != "complete":
        return "-"
    return f"{result.wall_seconds:.4g}s"


def run_benchmark(args: argparse.Namespace) -> tuple[Path, Path]:
    dataset = scaling.load_regression_dataset(
        args.dataset,
        data_home=args.data_home,
        allow_download=bool(args.allow_download),
        fallback_to_diabetes=bool(args.fallback_to_diabetes),
    )
    if dataset.fallback_reason:
        print(
            f"Requested {dataset.requested_name!r}, using {dataset.name!r}: "
            f"{dataset.fallback_reason}",
            flush=True,
        )
    cases = _case_specs(args, dataset.y.size)
    output_directory = args.output_dir.resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    csv_path = output_directory / "results.csv"
    json_path = output_directory / "results.json"
    config = scaling.ScalingConfig(
        max_iter=int(args.max_iter),
        tolerance=float(args.tolerance),
        support_tolerance=float(args.support_tolerance),
        max_events=int(args.max_events),
    )
    context = mp.get_context("spawn")
    records: list[scaling.ScalingRecord] = []
    print(
        "axis\trepeat\tn\tnodes\tmethod\toutcome/status\ttime\tpoints\tmax-coef-error",
        flush=True,
    )

    for scaling_axis, requested_samples, requested_nodes in cases:
        for repeat in range(int(args.repeats)):
            seed = int(args.seed) + repeat
            subset = scaling.subset_regression_dataset(
                dataset, requested_samples, seed=seed
            )
            problem = scaling.build_tree_path_problem(
                subset.X,
                subset.y,
                requested_nodes,
                seed=seed,
                dataset_name=subset.name,
                dataset_digest=subset.digest,
                min_samples_leaf=int(args.min_samples_leaf),
            )
            if problem.design_rank != problem.actual_internal_nodes:
                print(
                    f"warning: rank-deficient design at {scaling_axis}, "
                    f"n={problem.n_samples}, p={problem.n_features}",
                    flush=True,
                )
            grid = scaling.make_lambda_grid(
                problem,
                n_points=int(args.path_points),
                minimum_ratio=float(args.lambda_min_ratio),
                include_zero=True,
            )
            results: dict[str, tuple[scaling.SolverResult, float, bool]] = {}
            method_order = list(args.methods)
            if (repeat + requested_nodes + requested_samples) % 2:
                method_order.reverse()
            for method in method_order:
                payload = scaling.SolverPayload(problem, method, grid, config)
                results[method] = run_with_timeout(
                    payload,
                    timeout_seconds=float(args.timeout_seconds),
                    threads=int(args.threads),
                    context=context,
                )

            comparison: dict[str, float | int] | None = None
            if (
                scaling.METHOD_EXACT in results
                and scaling.METHOD_APA_WARM in results
            ):
                exact_result = results[scaling.METHOD_EXACT][0]
                apa_result = results[scaling.METHOD_APA_WARM][0]
                if (
                    exact_result.outcome == "complete"
                    and exact_result.status == "complete"
                    and exact_result.exact
                    and apa_result.outcome == "complete"
                    and apa_result.status in {"complete", "partial"}
                ):
                    evaluation_grid = scaling.make_lambda_grid(
                        problem,
                        n_points=int(args.metric_points),
                        minimum_ratio=float(args.lambda_min_ratio),
                        # Match the APA grid's full interval exactly. Values
                        # above the true lambda_max have the exact zero
                        # solution and are safely clamped by path evaluation.
                        lambda_max=grid.lambda_upper,
                        upper_factor=1.0,
                        include_zero=True,
                    )
                    comparison = scaling.compare_paths(
                        problem,
                        exact_result,
                        apa_result,
                        evaluation_grid.lambdas,
                        support_tolerance=config.support_tolerance,
                    )
                    # Separate point-solver error from the additional error
                    # incurred when a finite APA grid is treated as a path.
                    sample_comparison = scaling.compare_paths(
                        problem,
                        exact_result,
                        apa_result,
                        grid.lambdas,
                        support_tolerance=config.support_tolerance,
                    )
                    for key in (
                        "coefficient_relative_error_median",
                        "coefficient_relative_error_max",
                        "objective_relative_excess_median",
                        "objective_relative_excess_max",
                        "support_agreement_fraction",
                        "topology_agreement_fraction",
                    ):
                        comparison[f"sample_{key}"] = sample_comparison[key]

            for method in args.methods:
                result, parent_seconds, censored = results[method]
                record = scaling.make_scaling_record(
                    problem,
                    grid,
                    config,
                    result,
                    scaling_axis=scaling_axis,
                    requested_samples=requested_samples,
                    repeat=repeat,
                    timeout_seconds=float(args.timeout_seconds),
                    parent_wall_seconds=parent_seconds,
                    censored=censored,
                    comparison=(
                        comparison
                        if method == scaling.METHOD_APA_WARM
                        else None
                    ),
                    metric_points=int(args.metric_points),
                    min_samples_leaf=int(args.min_samples_leaf),
                    threads=int(args.threads),
                )
                records.append(record)
                error = (
                    "-"
                    if record.coefficient_relative_error_max is None
                    else f"{record.coefficient_relative_error_max:.3g}"
                )
                print(
                    f"{scaling_axis}\t{repeat}\t{problem.n_samples}\t"
                    f"{problem.n_features}\t{method}\t"
                    f"{result.outcome}/{result.status}\t"
                    f"{_format_seconds(result)}\t{result.native_points}\t{error}",
                    flush=True,
                )
            _write_current_results(
                records,
                args=args,
                dataset=dataset,
                csv_path=csv_path,
                json_path=json_path,
            )

    print(f"Wrote {csv_path}", flush=True)
    print(f"Wrote {json_path}", flush=True)
    return csv_path, json_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=("diabetes", "california_housing", "california"),
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
        "--axis", choices=("both", "nodes", "observations"), default="both"
    )
    parser.add_argument("--node-counts", nargs="+", type=int, default=None)
    parser.add_argument(
        "--observation-counts", nargs="+", type=int, default=None
    )
    parser.add_argument("--fixed-observations", type=int, default=None)
    parser.add_argument("--fixed-nodes", type=int, default=15)
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=scaling.SUPPORTED_METHODS,
        default=list(scaling.SUPPORTED_METHODS),
    )
    parser.add_argument("--path-points", type=int, default=20)
    parser.add_argument("--metric-points", type=int, default=101)
    parser.add_argument("--lambda-min-ratio", type=float, default=1e-3)
    parser.add_argument("--max-iter", type=int, default=2_000)
    parser.add_argument("--max-events", type=int, default=10_000)
    parser.add_argument("--tolerance", type=float, default=1e-7)
    parser.add_argument("--support-tolerance", type=float, default=1e-3)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/tree_path_scaling"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.repeats < 1:
        raise ValueError("repeats must be positive")
    if args.path_points < 2 or args.metric_points < 2:
        raise ValueError("path and metric point counts must be at least two")
    if args.threads < 1:
        raise ValueError("threads must be positive")
    run_benchmark(args)
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
