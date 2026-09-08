#!/usr/bin/env python3
"""Plot warm APA-APG2 and exact hiCAP regularization paths.

This script uses the deterministic nested-tree regression problem from
``benchmark_sparse_pruning_paths.py`` and writes a publication-ready comparison
of

* exact, piecewise-linear hiCAP coefficient trajectories;
* the finite lambda samples returned by warm-started APA-APG2;
* thresholded support and ancestor-closed tree-topology sizes; and
* the coefficient discrepancy at the sampled lambda values.

Run from the repository root, for example::

    python benchmarks/plot_sparse_pruning_paths.py \
        --size 128x7 --path-points 40 --max-iter 2000 \
        --output-dir benchmarks/path_figures

Both PNG and SVG are produced by default.  Matplotlib is only needed to run
this visualization script; it is not required by the path solver APIs.
"""
from __future__ import annotations

import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "imodels-mpl-cache"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(tempfile.gettempdir(), "imodels-xdg-cache"))

import argparse
import sys
import time
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


# Make both the checkout and the adjacent benchmark helper importable when the
# file is invoked directly.
BENCHMARK_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY_ROOT = BENCHMARK_DIRECTORY.parent
for source_directory in (REPOSITORY_ROOT, BENCHMARK_DIRECTORY):
    if str(source_directory) not in sys.path:
        sys.path.insert(0, str(source_directory))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

import benchmark_sparse_pruning_paths as benchmark
from imodels.tree.sparse_pruning.optimization.apa import apa_apg_regression_path
from imodels.tree.sparse_pruning.optimization.hicap import hicap_regression_path


def _parse_formats(values: Sequence[str]) -> tuple[str, ...]:
    formats: list[str] = []
    for raw_value in values:
        value = raw_value.lower().lstrip(".")
        if value not in {"png", "svg"}:
            raise argparse.ArgumentTypeError(
                "--formats accepts only 'png' and 'svg'"
            )
        if value not in formats:
            formats.append(value)
    return tuple(formats)


def _evaluate_path(path: object, lambdas: np.ndarray) -> np.ndarray:
    """Evaluate a descending path by its piecewise-linear interpolation."""

    source_lambdas = np.asarray(getattr(path, "lambdas"), dtype=float)
    source_coefficients = np.asarray(getattr(path, "coefficients"), dtype=float)
    order = np.argsort(source_lambdas, kind="stable")
    increasing_lambdas = source_lambdas[order]
    increasing_coefficients = source_coefficients[order]

    # Keep the same right-continuous convention as RegularizationPath.at for
    # simultaneous/repeated homotopy events.
    unique_lambdas, unique_indices = np.unique(
        increasing_lambdas, return_index=True
    )
    increasing_coefficients = increasing_coefficients[unique_indices]
    return np.column_stack(
        [
            np.interp(lambdas, unique_lambdas, increasing_coefficients[:, feature])
            for feature in range(increasing_coefficients.shape[1])
        ]
    )


def _support_sizes(coefficients: np.ndarray, tolerance: float) -> np.ndarray:
    return np.count_nonzero(np.abs(coefficients) > tolerance, axis=1)


def _topology_sizes(coefficients: np.ndarray, tolerance: float) -> np.ndarray:
    return np.asarray(
        [
            len(benchmark.topology_signature(beta, tolerance))
            for beta in coefficients
        ],
        dtype=int,
    )


def _comparison_grid(
    exact_path: object,
    lambda_max: float,
    lambda_min: float,
    dense_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Combine a dense log grid with every exact knot in the plotted range."""

    exact_lambdas = np.asarray(getattr(exact_path, "lambdas"), dtype=float)
    tolerance = np.finfo(float).eps * max(1.0, lambda_max) * 32.0
    knots = exact_lambdas[
        (exact_lambdas <= lambda_max + tolerance)
        & (exact_lambdas >= lambda_min - tolerance)
    ]
    grid = np.unique(
        np.concatenate(
            (np.geomspace(lambda_max, lambda_min, dense_points), knots)
        )
    )[::-1]
    return grid, np.unique(knots)[::-1]


def _discrete_feature_ticks(n_features: int) -> list[int]:
    if n_features <= 8:
        return list(range(n_features))
    return sorted(
        set(
            int(value)
            for value in np.linspace(0, n_features - 1, num=min(6, n_features))
        )
    )


def make_path_figure(
    *,
    n_samples: int,
    n_features: int,
    path_points: int,
    lambda_min_ratio: float,
    seed: int,
    max_iter: int,
    tolerance: float,
    support_tolerance: float,
    dense_points: int,
    annotate_coefficient_knots: bool = False,
    lambda_axis: str = "log",
) -> tuple[plt.Figure, dict[str, float | int | str]]:
    """Solve both paths and build the comparison figure."""

    problem = benchmark.make_problem(
        n_samples=n_samples,
        n_features=n_features,
        n_path_points=path_points,
        lambda_min_ratio=lambda_min_ratio,
        seed=seed,
    )

    exact_start = time.perf_counter()
    exact_path = hicap_regression_path(
        problem.X,
        problem.y,
        problem.groups,
        fit_intercept=False,
        tolerance=tolerance,
    )
    exact_seconds = time.perf_counter() - exact_start
    if not exact_path.exact or exact_path.status != "complete":
        raise RuntimeError(
            "hiCAP did not return a certified complete path: "
            f"status={exact_path.status!r}, exact={exact_path.exact}"
        )

    apa_start = time.perf_counter()
    apa_path = apa_apg_regression_path(
        problem.X,
        problem.y,
        problem.groups,
        problem.lambdas,
        max_iter=max_iter,
        tol=tolerance,
        ord="inf",
    )
    apa_seconds = time.perf_counter() - apa_start

    lambda_max = float(problem.lambdas[0])
    lambda_min = float(problem.lambdas[-1])
    exact_lambdas, visible_knots = _comparison_grid(
        exact_path,
        lambda_max,
        lambda_min,
        dense_points,
    )
    exact_coefficients = _evaluate_path(exact_path, exact_lambdas)
    exact_at_samples = _evaluate_path(exact_path, problem.lambdas)
    apa_coefficients = np.asarray(apa_path.coefficients, dtype=float)

    exact_support = _support_sizes(exact_coefficients, support_tolerance)
    apa_support = _support_sizes(apa_coefficients, support_tolerance)
    exact_sample_support = _support_sizes(exact_at_samples, support_tolerance)
    exact_topology = _topology_sizes(exact_coefficients, support_tolerance)
    apa_topology = _topology_sizes(apa_coefficients, support_tolerance)
    exact_sample_topology = _topology_sizes(exact_at_samples, support_tolerance)

    differences = apa_coefficients - exact_at_samples
    max_abs_difference = np.max(np.abs(differences), axis=1)
    relative_difference = np.linalg.norm(differences, axis=1) / np.maximum(
        np.linalg.norm(exact_at_samples, axis=1), 1.0
    )
    support_disagreement = float(np.mean(apa_support != exact_sample_support))
    topology_disagreement = float(np.mean(apa_topology != exact_sample_topology))

    lambda_scale = lambda_max
    exact_x = exact_lambdas / lambda_scale
    sample_x = problem.lambdas / lambda_scale
    knot_x = visible_knots / lambda_scale

    # A coefficient knot need not change the selected variables.  Classify
    # each visible knot by comparing the exact path on the open segments just
    # above and below it.  This is used only by the optional explanatory
    # overlay; the underlying path and benchmark metrics are unchanged.
    structural_knot = np.zeros(knot_x.size, dtype=bool)
    if annotate_coefficient_knots and knot_x.size:
        path_lambdas = np.asarray(exact_path.lambdas, dtype=float)
        path_coefficients = np.asarray(exact_path.coefficients, dtype=float)
        for visible_index, knot in enumerate(visible_knots):
            knot_index = int(np.argmin(np.abs(path_lambdas - knot)))
            if knot_index == 0:
                coefficient_above = path_coefficients[knot_index]
            else:
                lambda_above = 0.5 * (
                    path_lambdas[knot_index - 1] + path_lambdas[knot_index]
                )
                coefficient_above = _evaluate_path(
                    exact_path, np.asarray([lambda_above])
                )[0]
            if knot_index + 1 == path_lambdas.size:
                coefficient_below = path_coefficients[knot_index]
            else:
                lambda_below = 0.5 * (
                    path_lambdas[knot_index] + path_lambdas[knot_index + 1]
                )
                coefficient_below = _evaluate_path(
                    exact_path, np.asarray([lambda_below])
                )[0]

            support_above = tuple(
                np.flatnonzero(np.abs(coefficient_above) > support_tolerance)
            )
            support_below = tuple(
                np.flatnonzero(np.abs(coefficient_below) > support_tolerance)
            )
            topology_above = benchmark.topology_signature(
                coefficient_above, support_tolerance
            )
            topology_below = benchmark.topology_signature(
                coefficient_below, support_tolerance
            )
            structural_knot[visible_index] = (
                support_above != support_below
                or topology_above != topology_below
            )

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
        figure = plt.figure(figsize=(12.4, 8.1))
        grid_spec = figure.add_gridspec(
            2,
            3,
            height_ratios=(2.15, 1.0),
            left=0.075,
            right=0.96,
            bottom=0.13,
            top=0.88,
            hspace=0.42,
            wspace=0.32,
        )
        coefficient_axis = figure.add_subplot(grid_spec[0, :])
        support_axis = figure.add_subplot(grid_spec[1, 0])
        topology_axis = figure.add_subplot(grid_spec[1, 1], sharex=support_axis)
        difference_axis = figure.add_subplot(grid_spec[1, 2], sharex=support_axis)

        color_map = plt.get_cmap("turbo")
        color_normalization = Normalize(vmin=0, vmax=max(1, n_features - 1))
        for feature in range(n_features):
            color = color_map(color_normalization(feature))
            coefficient_axis.plot(
                exact_x,
                exact_coefficients[:, feature],
                color=color,
                linewidth=1.65,
                alpha=0.9,
                zorder=2,
            )
            coefficient_axis.plot(
                sample_x,
                apa_coefficients[:, feature],
                color=color,
                linewidth=0.8,
                linestyle=(0, (2.0, 2.2)),
                marker="o",
                markersize=3.5,
                markerfacecolor="white",
                markeredgewidth=0.8,
                alpha=0.82,
                zorder=3,
            )
        coefficient_axis.axhline(0.0, color="0.25", linewidth=0.7, zorder=1)
        if knot_x.size:
            coefficient_axis.plot(
                knot_x,
                np.full(knot_x.size, 1.012),
                linestyle="none",
                marker="|",
                markersize=5,
                color="0.4",
                alpha=0.72,
                transform=coefficient_axis.get_xaxis_transform(),
                clip_on=False,
            )
        if annotate_coefficient_knots and knot_x.size:
            knot_color = "#c2185b"
            structural_color = "#168f45"
            for x_value in knot_x:
                coefficient_axis.axvline(
                    x_value,
                    color=knot_color,
                    linewidth=1.0,
                    linestyle=(0, (2.0, 2.2)),
                    alpha=0.48,
                    zorder=1,
                )
            coefficient_axis.plot(
                knot_x,
                np.full(knot_x.size, 1.012),
                linestyle="none",
                marker="D",
                markersize=5.2,
                markerfacecolor=knot_color,
                markeredgecolor="white",
                markeredgewidth=0.65,
                transform=coefficient_axis.get_xaxis_transform(),
                clip_on=False,
                zorder=7,
            )
            if np.any(structural_knot):
                coefficient_axis.plot(
                    knot_x[structural_knot],
                    np.full(np.count_nonzero(structural_knot), 1.012),
                    linestyle="none",
                    marker="o",
                    markersize=10.0,
                    markerfacecolor="none",
                    markeredgecolor=structural_color,
                    markeredgewidth=1.5,
                    transform=coefficient_axis.get_xaxis_transform(),
                    clip_on=False,
                    zorder=8,
                )
            coefficient_axis.text(
                0.015,
                0.965,
                "◆  coefficient-path knot\n○  green ring: also structural",
                transform=coefficient_axis.transAxes,
                ha="left",
                va="top",
                fontsize=8.7,
                color=knot_color,
                bbox={
                    "boxstyle": "round,pad=0.3",
                    "facecolor": "white",
                    "edgecolor": "0.82",
                    "alpha": 0.92,
                },
                zorder=9,
            )
            coefficient_only = np.flatnonzero(~structural_knot)
            for knot_index in coefficient_only:
                annotation_offset = (
                    (-82, -62) if lambda_axis == "linear" else (-38, -62)
                )
                annotation_alignment = (
                    "right" if lambda_axis == "linear" else "center"
                )
                coefficient_axis.annotate(
                    "coefficient-only\n(slope changes; support does not)",
                    xy=(knot_x[knot_index], 1.0),
                    xycoords=coefficient_axis.get_xaxis_transform(),
                    xytext=annotation_offset,
                    textcoords="offset points",
                    ha=annotation_alignment,
                    va="top",
                    fontsize=8.5,
                    color=knot_color,
                    arrowprops={
                        "arrowstyle": "-|>",
                        "color": knot_color,
                        "linewidth": 1.0,
                        "shrinkA": 2,
                        "shrinkB": 3,
                    },
                    bbox={
                        "boxstyle": "round,pad=0.25",
                        "facecolor": "white",
                        "edgecolor": knot_color,
                        "alpha": 0.9,
                    },
                    zorder=9,
                )
        coefficient_title = "Coefficient trajectories"
        if lambda_axis == "linear":
            coefficient_title += " (linear lambda axis)"
        coefficient_axis.set_title(coefficient_title)
        coefficient_axis.set_ylabel(r"Coefficient $\beta_j$")
        coefficient_axis.set_xscale(lambda_axis)
        coefficient_axis.set_xlim(1.0, lambda_min_ratio)
        coefficient_axis.tick_params(axis="x", labelbottom=False)
        method_handles = [
            Line2D(
                [0],
                [0],
                color="0.15",
                linewidth=1.8,
                label="Exact hiCAP (continuous)",
            ),
            Line2D(
                [0],
                [0],
                color="0.15",
                linewidth=0.9,
                linestyle=(0, (2.0, 2.2)),
                marker="o",
                markersize=4,
                markerfacecolor="white",
                label=f"Warm APA-APG2 ({path_points} samples)",
            ),
            Line2D(
                [0],
                [0],
                color="0.4",
                linestyle="none",
                marker="|",
                markersize=7,
                label=f"Exact knots in view ({knot_x.size})",
            ),
        ]
        coefficient_axis.legend(handles=method_handles, loc="best", ncol=3)
        scalar_map = plt.cm.ScalarMappable(
            norm=color_normalization, cmap=color_map
        )
        colorbar = figure.colorbar(
            scalar_map,
            ax=coefficient_axis,
            pad=0.012,
            fraction=0.022,
            aspect=28,
            ticks=_discrete_feature_ticks(n_features),
        )
        colorbar.set_label("Feature index $j$")

        method_colors = {"exact": "#2166ac", "apa": "#b2182b"}
        support_axis.plot(
            exact_x,
            exact_support,
            color=method_colors["exact"],
            linewidth=2.0,
            drawstyle="steps-post",
            label="Exact hiCAP",
        )
        support_axis.plot(
            sample_x,
            apa_support,
            color=method_colors["apa"],
            linewidth=1.15,
            linestyle="--",
            marker="o",
            markersize=3.8,
            drawstyle="steps-post",
            label="Warm APA-APG2",
        )
        support_axis.set_title("Thresholded support size")
        support_axis.set_ylabel("Selected coefficients")
        support_axis.set_ylim(-0.35, n_features + 1.5)
        support_axis.legend(loc="best", fontsize=8.5)
        support_axis.text(
            0.03,
            0.96,
            rf"selected if $|\beta_j|>{support_tolerance:.0e}$",
            transform=support_axis.transAxes,
            ha="left",
            va="top",
            fontsize=8.5,
            color="0.25",
        )

        topology_axis.plot(
            exact_x,
            exact_topology,
            color=method_colors["exact"],
            linewidth=2.0,
            drawstyle="steps-post",
        )
        topology_axis.plot(
            sample_x,
            apa_topology,
            color=method_colors["apa"],
            linewidth=1.15,
            linestyle="--",
            marker="o",
            markersize=3.8,
            drawstyle="steps-post",
        )
        topology_axis.set_title("Ancestor-closed topology size")
        topology_axis.set_ylabel("Nodes in topology")
        topology_axis.set_ylim(-0.35, n_features + 1.5)

        positive_floor = max(np.finfo(float).eps, support_tolerance * 1e-4)
        difference_axis.plot(
            sample_x,
            np.maximum(max_abs_difference, positive_floor),
            color="#6a3d9a",
            linewidth=1.6,
            marker="o",
            markersize=3.8,
            label=r"$\max_j |\beta_j^{APA}-\beta_j^{exact}|$",
        )
        difference_axis.axhline(
            support_tolerance,
            color="0.25",
            linewidth=1.0,
            linestyle=":",
            label="support threshold",
        )
        difference_axis.set_yscale("log")
        difference_axis.set_title("Coefficient discrepancy")
        difference_axis.set_ylabel("Maximum absolute error")
        difference_axis.legend(loc="upper right", fontsize=8.1)
        difference_axis.text(
            0.03,
            0.04,
            "relative error\n"
            f"median {np.median(relative_difference):.2e}\n"
            f"max {np.max(relative_difference):.2e}",
            transform=difference_axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=8.2,
            color="0.25",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78},
        )

        for axis in (support_axis, topology_axis, difference_axis):
            axis.set_xscale(lambda_axis)
            axis.set_xlim(1.0, lambda_min_ratio)
            axis.set_xlabel(
                r"$\lambda/\lambda_{\mathrm{grid,max}}$ (decreases →)"
            )

        figure.suptitle(
            "Exact hiCAP path versus warm-started APA-APG2",
            fontsize=15,
            fontweight="semibold",
            y=0.97,
        )
        figure.text(
            0.5,
            0.925,
            (
                f"deterministic nested tree · n={n_samples}, p={n_features} · "
                f"hiCAP {exact_seconds:.3f} s / {exact_path.n_points} stored knots · "
                f"APA {apa_seconds:.3f} s / {path_points} grid points / "
                f"{max_iter} iterations/point"
            ),
            ha="center",
            va="center",
            fontsize=10,
            color="0.3",
        )
        figure.text(
            0.075,
            0.035,
            (
                "Support and topology depend on the displayed numerical threshold: "
                "visually close coefficient curves need not select the same nodes. "
                f"Grid-point disagreement: support {support_disagreement:.0%}, "
                f"topology {topology_disagreement:.0%}."
            ),
            ha="left",
            va="bottom",
            fontsize=9,
            color="0.3",
        )

    metrics: dict[str, float | int | str] = {
        "exact_seconds": exact_seconds,
        "apa_seconds": apa_seconds,
        "exact_points": exact_path.n_points,
        "visible_exact_knots": int(visible_knots.size),
        "apa_points": apa_path.n_points,
        "relative_error_median": float(np.median(relative_difference)),
        "relative_error_max": float(np.max(relative_difference)),
        "support_disagreement_fraction": support_disagreement,
        "topology_disagreement_fraction": topology_disagreement,
        "exact_status": exact_path.status,
        "apa_status": apa_path.status,
    }
    return figure, metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--size",
        type=benchmark.parse_size,
        default=(128, 7),
        metavar="NXP",
        help="sample-by-feature problem size (default: 128x7)",
    )
    parser.add_argument("--path-points", type=int, default=40)
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--lambda-min-ratio", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--tol", type=float, default=1e-7)
    parser.add_argument("--support-tol", type=float, default=1e-6)
    parser.add_argument("--dense-points", type=int, default=600)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument(
        "--annotate-coefficient-knots",
        action="store_true",
        help=(
            "overlay exact coefficient-knot guides and distinguish knots that "
            "also change support/tree topology"
        ),
    )
    parser.add_argument(
        "--lambda-axis",
        choices=("log", "linear"),
        default="log",
        help="horizontal lambda scale used by every panel (default: log)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BENCHMARK_DIRECTORY / "path_figures",
    )
    parser.add_argument(
        "--output-name",
        help="filename stem (default describes the problem and grid size)",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=("png", "svg"),
        metavar="FORMAT",
        help="one or both of: png svg (default: png svg)",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    n_samples, n_features = args.size
    if args.path_points < 2:
        raise SystemExit("--path-points must be at least 2")
    if args.max_iter < 1:
        raise SystemExit("--max-iter must be positive")
    if not 0 < args.lambda_min_ratio < 1:
        raise SystemExit("--lambda-min-ratio must lie strictly between 0 and 1")
    if args.tol <= 0 or args.support_tol <= 0:
        raise SystemExit("--tol and --support-tol must be positive")
    if args.dense_points < 20:
        raise SystemExit("--dense-points must be at least 20")
    if args.dpi < 1:
        raise SystemExit("--dpi must be positive")
    try:
        output_formats = _parse_formats(args.formats)
    except argparse.ArgumentTypeError as exc:
        raise SystemExit(str(exc)) from exc

    figure, metrics = make_path_figure(
        n_samples=n_samples,
        n_features=n_features,
        path_points=args.path_points,
        lambda_min_ratio=args.lambda_min_ratio,
        seed=args.seed,
        max_iter=args.max_iter,
        tolerance=args.tol,
        support_tolerance=args.support_tol,
        dense_points=args.dense_points,
        annotate_coefficient_knots=args.annotate_coefficient_knots,
        lambda_axis=args.lambda_axis,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_name = args.output_name or (
        f"apa_apg2_vs_hicap_{n_samples}x{n_features}_{args.path_points}points"
    )
    outputs: list[Path] = []
    for output_format in output_formats:
        output_path = args.output_dir / f"{output_name}.{output_format}"
        save_options: dict[str, object] = {
            "bbox_inches": "tight",
            "format": output_format,
        }
        if output_format == "png":
            save_options["dpi"] = args.dpi
        figure.savefig(output_path, **save_options)
        outputs.append(output_path.resolve())
    plt.close(figure)

    print("Generated path comparison:")
    for output_path in outputs:
        print(f"  {output_path}")
    print(
        "Solvers: "
        f"exact hiCAP={metrics['exact_seconds']:.3f}s/"
        f"{metrics['exact_points']} points ({metrics['exact_status']}), "
        f"warm APA-APG2={metrics['apa_seconds']:.3f}s/"
        f"{metrics['apa_points']} points ({metrics['apa_status']})"
    )
    print(
        "Grid comparison: "
        f"relative coefficient error median={metrics['relative_error_median']:.3e}, "
        f"max={metrics['relative_error_max']:.3e}; "
        f"support disagreement={metrics['support_disagreement_fraction']:.1%}; "
        f"topology disagreement={metrics['topology_disagreement_fraction']:.1%}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
