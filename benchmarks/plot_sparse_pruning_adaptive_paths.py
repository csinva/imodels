#!/usr/bin/env python3
"""Plot exact, fixed-grid, and adaptive sparse-pruning path benchmarks.

This module is both a standalone plot command and the plotting backend used by
``benchmark_sparse_pruning_adaptive_paths.py``.  It reads only the durable JSON
artifact, so figures can be restyled without rerunning timed solver calls.
"""
from __future__ import annotations

import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "imodels-mpl-cache"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(tempfile.gettempdir(), "imodels-xdg-cache"))

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


SCHEMA_NAME = "imodels.sparse_pruning.adaptive_path_benchmark"
SCHEMA_VERSION = 1
METHOD_EXACT = "hicap_exact"
METHOD_FIXED = "apa_apg2_fixed"
METHOD_ADAPTIVE = "apa_apg2_adaptive"
METHOD_LABELS = {
    METHOD_EXACT: "Exact hiCAP",
    METHOD_FIXED: "Fixed-grid APA-APG2",
    METHOD_ADAPTIVE: "Adaptive APA-APG2",
}
METHOD_COLORS = {
    METHOD_EXACT: "#2474B5",
    METHOD_FIXED: "#D55E00",
    METHOD_ADAPTIVE: "#009E73",
}
METHOD_STYLES = {
    METHOD_EXACT: "-",
    METHOD_FIXED: "--",
    METHOD_ADAPTIVE: ":",
}
METHOD_MARKERS = {
    METHOD_EXACT: "o",
    METHOD_FIXED: "s",
    METHOD_ADAPTIVE: "D",
}


def _validate_document(document: Mapping[str, Any]) -> None:
    if document.get("schema_name") != SCHEMA_NAME:
        raise ValueError(
            f"expected schema_name={SCHEMA_NAME!r}; got "
            f"{document.get('schema_name')!r}"
        )
    if document.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"expected schema_version={SCHEMA_VERSION}; got "
            f"{document.get('schema_version')!r}"
        )
    if not isinstance(document.get("runs"), list) or not isinstance(
        document.get("repeats"), list
    ):
        raise ValueError("benchmark JSON must contain runs and repeats lists")


def load_benchmark(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        document = json.load(handle)
    _validate_document(document)
    return document


def _path_position(
    lambdas: Sequence[float], lambda_max: float, minimum_ratio: float
) -> np.ndarray:
    """Map lambda to a finite decreasing-regularization coordinate."""

    values = np.asarray(lambdas, dtype=float)
    epsilon = minimum_ratio / 10.0
    return -np.log10(np.maximum(values / lambda_max, 0.0) + epsilon)


def _representative_repeat(document: Mapping[str, Any]) -> Mapping[str, Any]:
    for repeat in document["repeats"]:
        methods = repeat.get("methods", {})
        if all(
            methods.get(method, {}).get("lambdas") is not None
            for method in (METHOD_EXACT, METHOD_FIXED, METHOD_ADAPTIVE)
        ) and (
            methods[METHOD_EXACT].get("exact") is True
            and methods[METHOD_EXACT].get("status") == "complete"
        ):
            return repeat
    raise ValueError(
        "no repeat contains all three native paths and a complete exact reference"
    )


def _native_path_panel(
    axis: plt.Axes,
    repeat: Mapping[str, Any],
    minimum_ratio: float,
) -> None:
    methods = repeat["methods"]
    exact_coefficients = np.asarray(
        methods[METHOD_EXACT]["coefficients"], dtype=float
    )
    p = exact_coefficients.shape[1]
    amplitudes = np.max(np.abs(exact_coefficients), axis=0)
    selected = np.argsort(amplitudes)[::-1][: min(6, p)]
    feature_colors = plt.get_cmap("tab10")(
        np.linspace(0.0, 0.8, max(1, selected.size))
    )
    lambda_max = float(repeat["problem"]["lambda_max"])

    for method in (METHOD_EXACT, METHOD_FIXED, METHOD_ADAPTIVE):
        detail = methods[method]
        lambdas = np.asarray(detail["lambdas"], dtype=float)
        coefficients = np.asarray(detail["coefficients"], dtype=float)
        x_values = _path_position(lambdas, lambda_max, minimum_ratio)
        for position, feature in enumerate(selected):
            marker = None if method == METHOD_EXACT else METHOD_MARKERS[method]
            axis.plot(
                x_values,
                coefficients[:, feature],
                color=feature_colors[position],
                linestyle=METHOD_STYLES[method],
                linewidth=1.75 if method == METHOD_EXACT else 1.15,
                marker=marker,
                markersize=2.8,
                alpha=0.92 if method == METHOD_EXACT else 0.68,
            )

    method_handles = [
        Line2D(
            [0],
            [0],
            color="#333333",
            linestyle=METHOD_STYLES[method],
            marker=(None if method == METHOD_EXACT else METHOD_MARKERS[method]),
            markersize=4,
            label=METHOD_LABELS[method],
        )
        for method in (METHOD_EXACT, METHOD_FIXED, METHOD_ADAPTIVE)
    ]
    feature_handles = [
        Line2D(
            [0], [0], color=feature_colors[position], label=rf"$\beta_{{{feature}}}$"
        )
        for position, feature in enumerate(selected)
    ]
    method_legend = axis.legend(
        handles=method_handles, loc="upper left", ncol=1, fontsize=8.0
    )
    axis.add_artist(method_legend)
    axis.legend(
        handles=feature_handles,
        title="largest exact coefficients",
        loc="lower right",
        ncol=min(3, len(feature_handles)),
        fontsize=8,
        title_fontsize=8,
    )
    axis.axhline(0.0, color="#888888", linewidth=0.65, alpha=0.5)
    axis.set_ylabel("Coefficient")
    axis.set_title(
        "A  Native solution paths (markers are solved APA points)", loc="left"
    )


def _error_panel(
    axis: plt.Axes,
    repeat: Mapping[str, Any],
    minimum_ratio: float,
) -> None:
    lambda_max = float(repeat["problem"]["lambda_max"])
    tiny = 1e-16
    for method in (METHOD_FIXED, METHOD_ADAPTIVE):
        dense = repeat["methods"][method].get("dense_comparison")
        if dense is None:
            continue
        x_values = _path_position(
            dense["lambdas"], lambda_max, minimum_ratio
        )
        coefficient_error = np.maximum(
            np.asarray(dense["coefficient_relative_error"], dtype=float), tiny
        )
        objective_gap = np.maximum(
            np.asarray(dense["objective_relative_gap"], dtype=float), tiny
        )
        axis.plot(
            x_values,
            coefficient_error,
            color=METHOD_COLORS[method],
            linewidth=1.7,
            label=f"{METHOD_LABELS[method]}: coefficient",
        )
        axis.plot(
            x_values,
            objective_gap,
            color=METHOD_COLORS[method],
            linewidth=1.25,
            linestyle="--",
            alpha=0.8,
            label=f"{METHOD_LABELS[method]}: objective",
        )
    axis.set_yscale("log")
    axis.set_ylabel("Relative error / gap")
    axis.set_title("B  Dense-grid error against exact hiCAP", loc="left")
    axis.legend(loc="best", fontsize=7.8)


def _agreement_panel(
    axis: plt.Axes,
    repeat: Mapping[str, Any],
    minimum_ratio: float,
) -> None:
    lambda_max = float(repeat["problem"]["lambda_max"])
    offsets = {METHOD_FIXED: -0.018, METHOD_ADAPTIVE: 0.018}
    for method in (METHOD_FIXED, METHOD_ADAPTIVE):
        dense = repeat["methods"][method].get("dense_comparison")
        if dense is None:
            continue
        x_values = _path_position(
            dense["lambdas"], lambda_max, minimum_ratio
        )
        support = np.asarray(dense["support_match"], dtype=float)
        topology = np.asarray(dense["topology_match"], dtype=float)
        axis.plot(
            x_values,
            support + offsets[method],
            color=METHOD_COLORS[method],
            linewidth=1.25,
            label=f"{METHOD_LABELS[method]}: support",
        )
        axis.plot(
            x_values,
            topology + offsets[method] - 0.045,
            color=METHOD_COLORS[method],
            linewidth=1.0,
            linestyle="--",
            alpha=0.78,
            label=f"{METHOD_LABELS[method]}: topology",
        )
    axis.set_ylim(-0.09, 1.09)
    axis.set_yticks((0, 1), labels=("different", "equal"))
    axis.set_ylabel("Agreement")
    axis.set_title("C  Dense-grid support and tree-topology agreement", loc="left")
    axis.legend(loc="lower left", fontsize=7.7)


def _frontier_panel(axis: plt.Axes, document: Mapping[str, Any]) -> None:
    runs = [
        run
        for run in document["runs"]
        if run.get("outcome") == "complete"
        and run.get("coefficient_relative_error_max") is not None
    ]
    positive_errors = [
        float(run["coefficient_relative_error_max"])
        for run in runs
        if float(run["coefficient_relative_error_max"]) > 0
    ]
    floor = max(min(positive_errors, default=1e-12) / 8.0, 1e-16)
    seen: set[str] = set()
    for run in runs:
        method = str(run["method"])
        error = max(float(run["coefficient_relative_error_max"]), floor)
        seconds = max(float(run["wall_seconds"]), np.finfo(float).tiny)
        points = int(run.get("native_points") or 0)
        axis.scatter(
            seconds,
            error,
            s=45 + 3.5 * np.sqrt(max(points, 1)),
            marker=METHOD_MARKERS[method],
            color=METHOD_COLORS[method],
            edgecolor="white",
            linewidth=0.6,
            alpha=0.8,
            label=None if method in seen else METHOD_LABELS[method],
            zorder=3,
        )
        seen.add(method)

    # Annotate median time/error and native-point count for each method.
    for method in (METHOD_EXACT, METHOD_FIXED, METHOD_ADAPTIVE):
        selected = [run for run in runs if run["method"] == method]
        if not selected:
            continue
        seconds = float(np.median([float(run["wall_seconds"]) for run in selected]))
        error = max(
            float(
                np.median(
                    [float(run["coefficient_relative_error_max"]) for run in selected]
                )
            ),
            floor,
        )
        points = int(
            np.median(
                [int(run.get("native_points") or 0) for run in selected]
            )
        )
        axis.annotate(
            f"{points} native points",
            (seconds, error),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=7.2,
            color=METHOD_COLORS[method],
        )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("Solver wall time (seconds; lower is better)")
    axis.set_ylabel("Maximum relative coefficient error")
    axis.set_title(
        "D  Runtime–quality frontier (marker size reflects path size)",
        loc="left",
    )
    axis.legend(loc="best", fontsize=8)


def make_benchmark_figure(document: Mapping[str, Any]) -> plt.Figure:
    _validate_document(document)
    repeat = _representative_repeat(document)
    arguments = document["metadata"].get("arguments", {})
    minimum_ratio = float(arguments.get("lambda_min_ratio", 1e-3))
    problem = repeat["problem"]
    gram = problem["gram"]

    style = {
        "font.size": 9.5,
        "axes.labelsize": 9.5,
        "axes.titlesize": 10.5,
        "axes.titleweight": "semibold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.2,
        "grid.linewidth": 0.65,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "svg.fonttype": "none",
    }
    with plt.rc_context(style):
        figure, axes = plt.subplots(
            2,
            2,
            figsize=(12.4, 8.8),
            gridspec_kw={"height_ratios": (1.15, 1.0)},
        )
        _native_path_panel(axes[0, 0], repeat, minimum_ratio)
        _error_panel(axes[0, 1], repeat, minimum_ratio)
        _agreement_panel(axes[1, 0], repeat, minimum_ratio)
        _frontier_panel(axes[1, 1], document)

        for axis in (axes[0, 0], axes[0, 1], axes[1, 0]):
            axis.set_xlabel(
                r"Path position $-\log_{10}(\lambda/\lambda_{max}+\epsilon)$ "
                "(regularization decreases →)"
            )

        dataset = document["metadata"].get("dataset_name", "dataset")
        figure.suptitle(
            "Exact hiCAP vs fixed-grid and adaptive APA-APG2\n"
            f"{dataset}; n={problem['n_samples']}, "
            f"p={problem['n_internal_nodes']} fitted-tree stumps",
            x=0.06,
            y=0.985,
            ha="left",
            fontsize=14,
            fontweight="bold",
        )
        diagonal_text = (
            "verified diagonal" if gram["empirically_diagonal"] else "not diagonal"
        )
        adaptive_metadata = repeat["methods"][METHOD_ADAPTIVE].get(
            "adaptive_metadata", {}
        )
        kkt_mode = adaptive_metadata.get("kkt_acceptance_mode", "not_run")
        if kkt_mode == "relaxed_face":
            kkt_text = "Adaptive point check: relaxed near-face (not strict KKT).  "
        elif kkt_mode == "strict":
            kkt_text = "Adaptive point check: strict KKT.  "
        else:
            kkt_text = ""
        figure.text(
            0.06,
            0.012,
            f"Gram check on fitted rows: {diagonal_text}; max |offdiag correlation| "
            f"= {float(gram['max_abs_off_diagonal_correlation']):.2e}.  "
            f"{kkt_text}Errors use a dense shared lambda grid. Timings exclude tree fitting, "
            "evaluation, and plotting; exact/adaptive include their internal "
            "threshold LP, while fixed-grid excludes shared grid setup.",
            fontsize=8.2,
            color="#444444",
        )
        figure.subplots_adjust(
            left=0.075,
            right=0.975,
            bottom=0.12,
            top=0.88,
            wspace=0.25,
            hspace=0.34,
        )
    return figure


def save_benchmark_figure(
    document: Mapping[str, Any],
    *,
    output_base: str | Path,
    formats: Sequence[str] = ("png", "svg"),
) -> list[Path]:
    figure = make_benchmark_figure(document)
    base = Path(output_base)
    base.parent.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    try:
        for raw_format in formats:
            image_format = str(raw_format).lower().lstrip(".")
            if image_format not in {"png", "svg", "pdf"}:
                raise ValueError("formats may contain only png, svg, or pdf")
            output = base.with_suffix(f".{image_format}")
            figure.savefig(
                output,
                dpi=180 if image_format == "png" else None,
                bbox_inches="tight",
            )
            outputs.append(output.resolve())
    finally:
        plt.close(figure)
    return outputs


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument(
        "--output-base",
        type=Path,
        default=None,
        help="path without extension; defaults beside results.json",
    )
    parser.add_argument(
        "--formats", nargs="+", choices=("png", "svg", "pdf"), default=("png", "svg")
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        document = load_benchmark(args.results)
    except ValueError as exc:
        parser.error(str(exc))
    output_base = (
        args.results.parent / "hicap_fixed_adaptive_paths"
        if args.output_base is None
        else args.output_base
    )
    try:
        outputs = save_benchmark_figure(
            document, output_base=output_base, formats=args.formats
        )
    except ValueError as exc:
        parser.error(str(exc))
    for output in outputs:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
