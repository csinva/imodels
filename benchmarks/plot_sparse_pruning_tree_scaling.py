#!/usr/bin/env python3
"""Plot real-tree scaling results for exact hiCAP and warm APA-APG2.

The benchmark runner writes durable JSON/CSV records.  This script only reads
those records, so figure styling never changes the measured solver times.

Example
-------
python benchmarks/plot_sparse_pruning_tree_scaling.py \
    --results benchmarks/tree_path_scaling/results.json
"""
from __future__ import annotations

import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "imodels-mpl-cache"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(tempfile.gettempdir(), "imodels-xdg-cache"))

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


BENCHMARK_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY_ROOT = BENCHMARK_DIRECTORY.parent
for source_directory in (REPOSITORY_ROOT, BENCHMARK_DIRECTORY):
    if str(source_directory) not in sys.path:
        sys.path.insert(0, str(source_directory))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import _sparse_pruning_tree_scaling as scaling


METHOD_LABELS = {
    scaling.METHOD_EXACT: "Exact hiCAP (complete path)",
    scaling.METHOD_APA_WARM: "Warm APA-APG2 (fixed-budget grid)",
}
METHOD_COLORS = {
    scaling.METHOD_EXACT: "#1f77b4",
    scaling.METHOD_APA_WARM: "#d95f02",
}
METHOD_MARKERS = {
    scaling.METHOD_EXACT: "o",
    scaling.METHOD_APA_WARM: "s",
}
AXES = ("nodes", "observations")
AXIS_LABELS = {
    "nodes": "Internal tree nodes / stump features",
    "observations": "Observations",
}
AXIS_FIELDS = {
    "nodes": "n_internal_nodes",
    "observations": "n_samples",
}


def _parse_formats(values: Sequence[str]) -> tuple[str, ...]:
    formats: list[str] = []
    for raw_value in values:
        value = raw_value.lower().lstrip(".")
        if value not in {"png", "svg", "pdf"}:
            raise argparse.ArgumentTypeError(
                "--formats accepts only png, svg, and pdf"
            )
        if value not in formats:
            formats.append(value)
    return tuple(formats)


def _finite(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _completed(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return usable outputs, including APA paths stopped at their iteration cap."""

    completed = []
    for row in rows:
        if row.get("outcome") != "complete" or _finite(
            row.get("solver_wall_seconds")
        ) is None:
            continue
        if row.get("method") == scaling.METHOD_EXACT:
            if row.get("solver_status") != "complete" or not bool(
                row.get("exact")
            ):
                continue
        elif row.get("method") == scaling.METHOD_APA_WARM:
            if row.get("solver_status") not in {"complete", "partial"}:
                continue
        else:
            continue
        completed.append(row)
    return completed


def _group_by_x(
    rows: Iterable[dict[str, Any]], axis_name: str
) -> dict[int, list[dict[str, Any]]]:
    field = AXIS_FIELDS[axis_name]
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("scaling_axis") != axis_name:
            continue
        value = row.get(field)
        if value is not None:
            grouped[int(value)].append(row)
    return dict(grouped)


def _timing_summary(
    rows: Iterable[dict[str, Any]], axis_name: str, method: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    selected = [row for row in rows if row.get("method") == method]
    grouped = _group_by_x(_completed(selected), axis_name)
    x_values: list[int] = []
    medians: list[float] = []
    q1_values: list[float] = []
    q3_values: list[float] = []
    for x_value in sorted(grouped):
        timings = np.asarray(
            [float(row["solver_wall_seconds"]) for row in grouped[x_value]]
        )
        x_values.append(x_value)
        medians.append(float(np.median(timings)))
        q1_values.append(float(np.percentile(timings, 25.0)))
        q3_values.append(float(np.percentile(timings, 75.0)))
    return tuple(
        np.asarray(values, dtype=float)
        for values in (x_values, medians, q1_values, q3_values)
    )


def _timeout_summary(
    rows: Iterable[dict[str, Any]], axis_name: str, method: str
) -> tuple[np.ndarray, np.ndarray]:
    field = AXIS_FIELDS[axis_name]
    points: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        if (
            row.get("scaling_axis") != axis_name
            or row.get("method") != method
            or row.get("outcome") != "timeout"
        ):
            continue
        timeout = _finite(row.get("timeout_seconds"))
        if timeout is not None:
            points[int(row[field])].append(timeout)
    x_values = np.asarray(sorted(points), dtype=float)
    limits = np.asarray(
        [float(np.median(points[int(value)])) for value in x_values], dtype=float
    )
    return x_values, limits


def _uncertified_summary(
    rows: Iterable[dict[str, Any]], axis_name: str, method: str
) -> tuple[np.ndarray, np.ndarray]:
    """Return solver calls that ended but did not produce a usable path."""

    field = AXIS_FIELDS[axis_name]
    points: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        if (
            row.get("scaling_axis") != axis_name
            or row.get("method") != method
            or row.get("outcome") != "complete"
        ):
            continue
        if method == scaling.METHOD_EXACT:
            usable = row.get("solver_status") == "complete" and bool(
                row.get("exact")
            )
        else:
            usable = row.get("solver_status") in {"complete", "partial"}
        timing = _finite(row.get("solver_wall_seconds"))
        if not usable and timing is not None:
            points[int(row[field])].append(timing)
    x_values = np.asarray(sorted(points), dtype=float)
    timings = np.asarray(
        [float(np.median(points[int(value)])) for value in x_values], dtype=float
    )
    return x_values, timings


def _accuracy_summary(
    rows: Iterable[dict[str, Any]], axis_name: str, metric: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    selected = [
        row
        for row in _completed(rows)
        if row.get("method") == scaling.METHOD_APA_WARM
        and _finite(row.get(metric)) is not None
    ]
    grouped = _group_by_x(selected, axis_name)
    x_values: list[int] = []
    medians: list[float] = []
    q1_values: list[float] = []
    q3_values: list[float] = []
    for x_value in sorted(grouped):
        values = np.asarray(
            [abs(float(row[metric])) for row in grouped[x_value]], dtype=float
        )
        values = np.maximum(values, np.finfo(float).tiny)
        x_values.append(x_value)
        medians.append(float(np.median(values)))
        q1_values.append(float(np.percentile(values, 25.0)))
        q3_values.append(float(np.percentile(values, 75.0)))
    return tuple(
        np.asarray(values, dtype=float)
        for values in (x_values, medians, q1_values, q3_values)
    )


def _agreement_summary(
    rows: Iterable[dict[str, Any]], axis_name: str, metric: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    selected = [
        row
        for row in _completed(rows)
        if row.get("method") == scaling.METHOD_APA_WARM
        and _finite(row.get(metric)) is not None
    ]
    grouped = _group_by_x(selected, axis_name)
    x_values: list[int] = []
    medians: list[float] = []
    q1_values: list[float] = []
    q3_values: list[float] = []
    for x_value in sorted(grouped):
        values = np.asarray(
            [float(row[metric]) for row in grouped[x_value]], dtype=float
        )
        x_values.append(x_value)
        medians.append(float(np.median(values)))
        q1_values.append(float(np.percentile(values, 25.0)))
        q3_values.append(float(np.percentile(values, 75.0)))
    return tuple(
        np.asarray(values, dtype=float)
        for values in (x_values, medians, q1_values, q3_values)
    )


def _set_discrete_log_x(axis: plt.Axes, values: Sequence[float]) -> None:
    positive = sorted({int(value) for value in values if value > 0})
    if not positive:
        return
    if len(positive) > 1 and positive[-1] / positive[0] >= 4:
        axis.set_xscale("log", base=2)
    axis.set_xticks(positive)
    axis.set_xticklabels([str(value) for value in positive])


def make_figure(
    rows: list[dict[str, Any]],
    *,
    coefficient_target: float = 1e-2,
    objective_target: float = 1e-4,
) -> plt.Figure:
    """Build a two-axis timing and path-accuracy figure."""

    style = {
        "font.size": 10,
        "axes.labelsize": 10.5,
        "axes.titlesize": 11.5,
        "axes.titleweight": "semibold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.22,
        "grid.linewidth": 0.7,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "svg.fonttype": "none",
    }
    with plt.rc_context(style):
        figure, plot_axes = plt.subplots(
            3,
            2,
            figsize=(12.2, 10.2),
            sharex="col",
            gridspec_kw={"height_ratios": (1.35, 1.0, 0.72)},
        )
        figure.subplots_adjust(
            left=0.085,
            right=0.97,
            bottom=0.10,
            top=0.83,
            hspace=0.16,
            wspace=0.22,
        )

        all_x: dict[str, set[int]] = {name: set() for name in AXES}
        for column, axis_name in enumerate(AXES):
            timing_axis = plot_axes[0, column]
            accuracy_axis = plot_axes[1, column]
            topology_axis = plot_axes[2, column]
            for method in scaling.SUPPORTED_METHODS:
                x_values, medians, q1_values, q3_values = _timing_summary(
                    rows, axis_name, method
                )
                all_x[axis_name].update(int(value) for value in x_values)
                if x_values.size:
                    color = METHOD_COLORS[method]
                    timing_axis.plot(
                        x_values,
                        medians,
                        color=color,
                        marker=METHOD_MARKERS[method],
                        linewidth=2.0,
                        markersize=6,
                        label=METHOD_LABELS[method],
                    )
                    if np.any(q3_values > q1_values):
                        timing_axis.fill_between(
                            x_values,
                            q1_values,
                            q3_values,
                            color=color,
                            alpha=0.14,
                            linewidth=0,
                        )
                timeout_x, timeout_limits = _timeout_summary(
                    rows, axis_name, method
                )
                all_x[axis_name].update(int(value) for value in timeout_x)
                if timeout_x.size:
                    displayed_x = timeout_x * (
                        0.975 if method == scaling.METHOD_EXACT else 1.025
                    )
                    timing_axis.scatter(
                        displayed_x,
                        timeout_limits,
                        marker="^",
                        s=65,
                        facecolors="none",
                        edgecolors=METHOD_COLORS[method],
                        linewidths=1.6,
                        zorder=5,
                    )
                    for x_value, limit in zip(displayed_x, timeout_limits):
                        timing_axis.annotate(
                            f"≥{limit:g}s",
                            (x_value, limit),
                            xytext=(0, 7),
                            textcoords="offset points",
                            ha="center",
                            va="bottom",
                            fontsize=8,
                            color=METHOD_COLORS[method],
                        )
                invalid_x, invalid_timings = _uncertified_summary(
                    rows, axis_name, method
                )
                all_x[axis_name].update(int(value) for value in invalid_x)
                if invalid_x.size:
                    timing_axis.scatter(
                        invalid_x,
                        invalid_timings,
                        marker="X",
                        s=62,
                        color=METHOD_COLORS[method],
                        linewidths=0.8,
                        zorder=5,
                    )

            metric_styles = (
                (
                    "coefficient_relative_error_max",
                    "coefficient error: interpolated path",
                    "#6a3d9a",
                    "o",
                    coefficient_target,
                ),
                (
                    "sample_coefficient_relative_error_max",
                    "coefficient error: solved grid points",
                    "#e7298a",
                    "^",
                    coefficient_target,
                ),
                (
                    "objective_relative_excess_max",
                    "objective excess: interpolated path",
                    "#1b9e77",
                    "D",
                    objective_target,
                ),
            )
            for metric, label, color, marker, target in metric_styles:
                x_values, medians, q1_values, q3_values = _accuracy_summary(
                    rows, axis_name, metric
                )
                all_x[axis_name].update(int(value) for value in x_values)
                if x_values.size:
                    accuracy_axis.plot(
                        x_values,
                        medians,
                        color=color,
                        marker=marker,
                        linewidth=1.8,
                        markersize=5.5,
                        label=label,
                    )
                    if np.any(q3_values > q1_values):
                        accuracy_axis.fill_between(
                            x_values,
                            q1_values,
                            q3_values,
                            color=color,
                            alpha=0.12,
                            linewidth=0,
                        )
                    accuracy_axis.axhline(
                        target,
                        color=color,
                        linestyle=(0, (2, 2)),
                        linewidth=0.9,
                        alpha=0.65,
                    )

            timing_axis.set_title(
                "Scale tree size" if axis_name == "nodes" else "Scale data size"
            )
            timing_axis.set_yscale("log")
            timing_axis.set_ylabel("Solver time (seconds)" if column == 0 else "")
            accuracy_axis.set_yscale("log")
            accuracy_axis.set_ylabel(
                "APA discrepancy vs exact hiCAP" if column == 0 else ""
            )
            topology_styles = (
                (
                    "topology_agreement_fraction",
                    "topology agreement: interpolated path",
                    "#7570b3",
                    "o",
                ),
                (
                    "sample_topology_agreement_fraction",
                    "topology agreement: solved grid points",
                    "#e7298a",
                    "^",
                ),
            )
            for metric, label, color, marker in topology_styles:
                x_values, medians, q1_values, q3_values = _agreement_summary(
                    rows, axis_name, metric
                )
                if x_values.size:
                    topology_axis.plot(
                        x_values,
                        medians,
                        color=color,
                        marker=marker,
                        linewidth=1.8,
                        markersize=5.5,
                        label=label,
                    )
                    if np.any(q3_values > q1_values):
                        topology_axis.fill_between(
                            x_values,
                            q1_values,
                            q3_values,
                            color=color,
                            alpha=0.12,
                            linewidth=0,
                        )
            topology_axis.axhline(
                0.95,
                color="0.35",
                linestyle=(0, (2, 2)),
                linewidth=0.9,
                alpha=0.65,
            )
            topology_axis.set_ylim(-0.03, 1.03)
            topology_axis.set_ylabel(
                "Exact topology agreement" if column == 0 else ""
            )
            topology_axis.set_xlabel(AXIS_LABELS[axis_name])
            _set_discrete_log_x(timing_axis, all_x[axis_name])
            _set_discrete_log_x(accuracy_axis, all_x[axis_name])
            _set_discrete_log_x(topology_axis, all_x[axis_name])

        method_handles = [
            Line2D(
                [0],
                [0],
                color=METHOD_COLORS[method],
                marker=METHOD_MARKERS[method],
                linewidth=2,
                label=METHOD_LABELS[method],
            )
            for method in scaling.SUPPORTED_METHODS
        ]
        method_handles.append(
            Line2D(
                [0],
                [0],
                color="0.35",
                marker="^",
                markerfacecolor="none",
                linestyle="none",
                label="timeout (right-censored)",
            )
        )
        method_handles.append(
            Line2D(
                [0],
                [0],
                color="0.35",
                marker="X",
                linestyle="none",
                label="returned, but not certified/complete",
            )
        )
        figure.legend(
            handles=method_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.925),
            ncol=4,
        )
        accuracy_handles, accuracy_labels = plot_axes[1, 0].get_legend_handles_labels()
        if accuracy_handles:
            plot_axes[1, 0].legend(accuracy_handles, accuracy_labels, loc="best")
        topology_handles, topology_labels = plot_axes[2, 0].get_legend_handles_labels()
        if topology_handles:
            plot_axes[2, 0].legend(
                topology_handles, topology_labels, loc="lower left"
            )

        datasets = sorted({str(row.get("dataset_name")) for row in rows})
        path_points = sorted(
            {
                int(row["requested_path_points"])
                for row in rows
                if row.get("requested_path_points") is not None
            }
        )
        iterations = sorted(
            {int(row["max_iter"]) for row in rows if row.get("max_iter") is not None}
        )
        repetitions = 1 + max(
            (int(row.get("repeat", 0)) for row in rows), default=0
        )
        apa_finished = [
            row
            for row in rows
            if row.get("method") == scaling.METHOD_APA_WARM
            and row.get("outcome") == "complete"
        ]
        apa_converged = sum(
            row.get("all_points_converged") is True for row in apa_finished
        )
        figure.suptitle(
            "Exact hiCAP vs warm APA-APG2 on fitted regression-tree features",
            fontsize=15,
            fontweight="bold",
            y=0.985,
        )
        configuration = (
            f"data: {', '.join(datasets)}   •   APA grid points: "
            f"{', '.join(map(str, path_points))}   •   iterations/point: "
            f"{', '.join(map(str, iterations))}   •   repetitions: {repetitions}"
        )
        figure.text(0.5, 0.948, configuration, ha="center", va="top", fontsize=9.5)
        figure.text(
            0.5,
            0.018,
            "Solver-only medians; bands are IQR. Exact hiCAP follows every knot to λ=0; "
            f"APA is shown at its fixed budget ({apa_converged}/{len(apa_finished)} runs "
            "met every point's stopping test). "
            "Grid setup and tree transformation are excluded. X marks an unusable exact path.",
            ha="center",
            va="bottom",
            fontsize=8.6,
            color="0.32",
        )
        return figure


def _median_by_case(
    rows: list[dict[str, Any]], axis_name: str, method: str
) -> dict[int, float]:
    x_values, medians, _, _ = _timing_summary(rows, axis_name, method)
    return {int(x): float(value) for x, value in zip(x_values, medians)}


def print_summary(rows: list[dict[str, Any]]) -> None:
    """Print paired median timings and accuracy for quick inspection."""

    for axis_name in AXES:
        exact = _median_by_case(rows, axis_name, scaling.METHOD_EXACT)
        apa = _median_by_case(rows, axis_name, scaling.METHOD_APA_WARM)
        field = AXIS_FIELDS[axis_name]
        accuracy = _group_by_x(
            [
                row
                for row in _completed(rows)
                if row.get("method") == scaling.METHOD_APA_WARM
                and _finite(row.get("coefficient_relative_error_max")) is not None
            ],
            axis_name,
        )
        sample_accuracy = _group_by_x(
            [
                row
                for row in _completed(rows)
                if row.get("method") == scaling.METHOD_APA_WARM
                and _finite(row.get("sample_coefficient_relative_error_max"))
                is not None
            ],
            axis_name,
        )
        print(f"\n{axis_name}")
        print(
            "x\thicap_s\tapa_s\tapa/hicap\t"
            "APA_path_error\tAPA_gridpoint_error"
        )
        all_values = sorted(
            {int(row[field]) for row in rows if row.get("scaling_axis") == axis_name}
        )
        for x_value in all_values:
            ratio = (
                apa[x_value] / exact[x_value]
                if x_value in apa and x_value in exact and exact[x_value] > 0
                else None
            )
            errors = [
                float(row["coefficient_relative_error_max"])
                for row in accuracy.get(x_value, [])
            ]
            sample_errors = [
                float(row["sample_coefficient_relative_error_max"])
                for row in sample_accuracy.get(x_value, [])
            ]
            values = (
                str(x_value),
                "-" if x_value not in exact else f"{exact[x_value]:.4g}",
                "-" if x_value not in apa else f"{apa[x_value]:.4g}",
                "-" if ratio is None else f"{ratio:.3g}",
                "-" if not errors else f"{np.median(errors):.3g}",
                (
                    "-"
                    if not sample_errors
                    else f"{np.median(sample_errors):.3g}"
                ),
            )
            print("\t".join(values))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("benchmarks/tree_path_scaling/results.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--prefix", default="tree_path_scaling")
    parser.add_argument("--formats", nargs="+", default=("png", "svg"))
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--coefficient-target", type=float, default=1e-2)
    parser.add_argument("--objective-target", type=float, default=1e-4)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    formats = _parse_formats(args.formats)
    document = scaling.read_json_results(args.results)
    rows = [dict(row) for row in document["records"]]
    if not rows:
        raise RuntimeError(f"no benchmark records found in {args.results}")
    output_directory = args.output_dir or args.results.resolve().parent
    output_directory.mkdir(parents=True, exist_ok=True)
    figure = make_figure(
        rows,
        coefficient_target=float(args.coefficient_target),
        objective_target=float(args.objective_target),
    )
    outputs: list[Path] = []
    for output_format in formats:
        output = output_directory / f"{args.prefix}.{output_format}"
        save_kwargs = {"bbox_inches": "tight"}
        if output_format == "png":
            save_kwargs["dpi"] = int(args.dpi)
        figure.savefig(output, **save_kwargs)
        outputs.append(output.resolve())
    plt.close(figure)
    print_summary(rows)
    print("\nWrote:")
    for output in outputs:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
