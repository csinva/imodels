#!/usr/bin/env python3
"""Time complete diagonal-tree hiCAP paths on bundled real diabetes data.

Fits CART once per case, extracts its unnormalized local-stump sufficient
statistics once, then compares four different outputs/workloads:

* new diagonal continuation: discovers every coefficient knot;
* green topology solver: discovers every structural knot only;
* dark-blue exact prox: solves the new path's already-known knot grid;
* generic hiCAP: discovers every coefficient knot (small trees only).

The dark-blue grid timing does not include discovering that grid.  Tree fitting,
statistic extraction and proximal-operator compilation are recorded separately.
No downloads or optional datasets are needed.  JSON/CSV retain individual runs;
the standalone plots show medians and min/max ranges across repeats.
"""
from __future__ import annotations

import os
import tempfile

for _thread_variable in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_thread_variable, "1")
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "imodels-mpl-cache"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(tempfile.gettempdir(), "imodels-xdg-cache"))

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.tree import DecisionTreeRegressor
from threadpoolctl import threadpool_limits

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from benchmarks._sparse_pruning_tree_scaling import environment_metadata  # noqa: E402
from imodels.tree.sparse_pruning.optimization.diagonal_homotopy import (  # noqa: E402
    tree_group_linf_exact_coefficient_path,
)
from imodels.tree.sparse_pruning.fitted_tree import (  # noqa: E402
    fitted_tree_linf_exact_topology_path,
)
from imodels.tree.sparse_pruning.optimization.hicap import hicap_regression_path  # noqa: E402
from imodels.tree.sparse_pruning.optimization.topology import (  # noqa: E402
    tree_group_linf_exact_topology_path,
)
from imodels.tree.sparse_pruning.optimization.tree_prox import LaminarGroupLinfProx  # noqa: E402


def _descendant_groups(parents: np.ndarray) -> tuple[np.ndarray, ...]:
    """Materialize oracle-only groups; the new solver receives just parents."""
    memberships: list[list[int]] = [[] for _ in parents]
    for node in range(parents.size):
        ancestor = node
        while ancestor >= 0:
            memberships[ancestor].append(node)
            ancestor = int(parents[ancestor])
    return tuple(np.asarray(group, dtype=np.intp) for group in memberships)


def _canonical_knots(path, relative_tolerance: float) -> np.ndarray:
    """Remove only numerically collinear interior points from a reference."""
    lambdas = np.asarray(path.lambdas)
    beta = np.asarray(path.coefficients)
    if len(lambdas) < 3:
        return lambdas
    unique = np.r_[True, np.diff(lambdas) < 0.0]
    lambdas, beta = lambdas[unique], beta[unique]
    if len(lambdas) < 3:
        return lambdas
    slope = np.diff(beta, axis=0) / np.diff(lambdas)[:, None]
    difference = np.max(np.abs(np.diff(slope, axis=0)), axis=1)
    scale = np.maximum(
        1.0,
        np.maximum(np.max(np.abs(slope[:-1]), axis=1),
                   np.max(np.abs(slope[1:]), axis=1)),
    )
    return lambdas[np.r_[True, difference > relative_tolerance * scale, True]]


def _relative_knot_distance(first: np.ndarray, second: np.ndarray) -> float:
    distance = np.abs(first[:, None] - second[None, :])
    return float(max(distance.min(axis=0).max(), distance.min(axis=1).max())
                 / max(float(first.max()), float(second.max()), 1e-300))


def _legacy_comparison(h, diagonal, groups, full, args) -> dict[str, Any]:
    p = len(h)
    # These pseudo-observations have exactly X.T X / p = D and X.T y / p = h.
    # The original real-data pruning objective differs only by a constant.
    design = np.diag(np.sqrt(p * diagonal))
    response = np.sqrt(p) * h / np.sqrt(diagonal)
    started = time.perf_counter()
    try:
        reference = hicap_regression_path(
            design, response, groups, fit_intercept=False,
            tolerance=args.legacy_tolerance,
            tie_tolerance=10.0 * args.legacy_tolerance,
            max_events=args.max_events,
        )
    except Exception as exc:
        return {
            "legacy_seconds": time.perf_counter() - started,
            "legacy_status": f"{type(exc).__name__}: {exc}",
            "legacy_exact": False,
        }
    record: dict[str, Any] = {
        "legacy_seconds": time.perf_counter() - started,
        "legacy_status": reference.status,
        "legacy_exact": bool(reference.exact),
        "legacy_stored_points": int(reference.n_points),
    }
    if not reference.exact or not full.exact:
        return record
    full_knots = _canonical_knots(full, 1e-7)
    legacy_knots = _canonical_knots(reference, 1e-7)
    record.update({
        "legacy_canonical_knots": int(len(legacy_knots)),
        "full_canonical_knots": int(len(full_knots)),
        "legacy_relative_knot_distance": _relative_knot_distance(
            full_knots, legacy_knots
        ),
    })
    probes = np.unique(np.r_[full.lambdas, reference.lambdas])
    probes = np.unique(np.r_[probes, 0.5 * (probes[:-1] + probes[1:])])
    upper = min(float(full.lambdas[0]), float(reference.lambdas[0]))
    probes = probes[probes <= upper]
    record["legacy_max_coefficient_error"] = float(max(
        (np.max(np.abs(full.at(float(lam))[0] - reference.at(float(lam))[0]))
         for lam in probes), default=0.0
    ))
    record["legacy_knots_match"] = bool(
        len(full_knots) == len(legacy_knots)
        and record["legacy_relative_knot_distance"] <= 100 * args.legacy_tolerance
    )
    return record


def _run_case(X, y, requested_nodes: int, repeat: int, args) -> dict[str, Any]:
    seed = args.seed + repeat
    started = time.perf_counter()
    tree = DecisionTreeRegressor(
        max_leaf_nodes=requested_nodes + 1,
        min_samples_leaf=args.min_samples_leaf,
        random_state=seed,
    ).fit(X, y)
    fit_seconds = time.perf_counter() - started

    started = time.perf_counter()
    native = fitted_tree_linf_exact_topology_path(tree)
    h = np.asarray(native.metadata["linear_scores"])
    diagonal = np.asarray(native.metadata["gram_diagonal"])
    parents = np.asarray(native.metadata["parent_indices"], dtype=np.intp)
    extraction_seconds = time.perf_counter() - started
    p = len(h)
    started = time.perf_counter()
    structural = tree_group_linf_exact_topology_path(h, parents)
    structural_seconds = time.perf_counter() - started

    started = time.perf_counter()
    full = tree_group_linf_exact_coefficient_path(
        h, diagonal, parents, max_events=args.max_events,
    )
    full_seconds = time.perf_counter() - started

    # A constant/root-only CART is valid. Its empty path needs no nonempty
    # proximal operator or generic design oracle. Nulls mean "not run", not a
    # fabricated zero-second timing or independently measured zero error.
    point_result = {
        "darkblue_status": "not_needed_no_splits",
        "darkblue_setup_seconds": None,
        "darkblue_same_knot_grid_seconds": None,
        "darkblue_total_seconds": None,
        "darkblue_grid_points": 0,
        "darkblue_max_knot_coefficient_error": None,
        "darkblue_max_midpoint_coefficient_error": None,
        "darkblue_max_relative_coefficient_error": None,
    }
    if p:
        started = time.perf_counter()
        groups = _descendant_groups(parents)
        sqrt_diagonal = np.sqrt(diagonal)
        prox = LaminarGroupLinfProx(
            groups, p, coordinate_weights=1.0 / sqrt_diagonal,
        )
        center = h / sqrt_diagonal
        prox_setup_seconds = time.perf_counter() - started

        started = time.perf_counter()
        point_coefficients = np.asarray([
            prox(center, float(lam)) / sqrt_diagonal for lam in full.lambdas
        ])
        prox_seconds = time.perf_counter() - started
        knot_error = float(np.max(np.abs(point_coefficients - full.coefficients), initial=0))
        midpoints = (full.lambdas[:-1] + full.lambdas[1:]) / 2.0
        midpoint_error = float(max((
            np.max(np.abs(full.at(float(lam))[0]
                          - prox(center, float(lam)) / sqrt_diagonal))
            for lam in midpoints
        ), default=0.0))
        coefficient_scale = max(float(np.max(np.abs(h / diagonal), initial=0)), 1e-300)
        point_result.update({
            "darkblue_status": "complete",
            "darkblue_setup_seconds": prox_setup_seconds,
            "darkblue_same_knot_grid_seconds": prox_seconds,
            "darkblue_total_seconds": prox_setup_seconds + prox_seconds,
            "darkblue_grid_points": int(full.n_points),
            "darkblue_max_knot_coefficient_error": knot_error,
            "darkblue_max_midpoint_coefficient_error": midpoint_error,
            "darkblue_max_relative_coefficient_error": max(knot_error, midpoint_error) / coefficient_scale,
        })
    structural_distances = np.min(
        np.abs(structural.lambdas[:, None] - full.lambdas[None, :]), axis=1
    )
    structural_error = float(np.max(structural_distances, initial=0)
                             / max(float(structural.lambdas[0]), 1e-300))
    record: dict[str, Any] = {
        "requested_nodes": requested_nodes,
        "n_internal_nodes": p,
        "n_observations": len(y),
        "tree_depth": tree.get_depth(),
        "repeat": repeat,
        "seed": seed,
        "tree_fit_seconds": fit_seconds,
        "stats_and_native_topology_seconds": extraction_seconds,
        "structural_seconds": structural_seconds,
        "structural_status": structural.status,
        "structural_stored_points": int(len(structural.lambdas)),
        "full_path_seconds": full_seconds,
        "full_status": full.status,
        "full_exact": bool(full.exact),
        "full_stored_points": int(full.n_points),
        "full_positive_knots": int(np.count_nonzero(full.lambdas > 0)),
        "full_interval_count": max(0, int(full.n_points) - 1),
        "full_coefficient_storage_bytes": int(full.coefficients.nbytes),
        "full_metadata": dict(full.metadata),
        **point_result,
        "structural_max_relative_knot_distance": structural_error,
        "legacy_status": "skipped_size_limit" if p else "not_needed_no_splits",
        "legacy_exact": None,
    }
    if 0 < p <= args.legacy_max_nodes:
        record.update(_legacy_comparison(h, diagonal, groups, full, args))
    return record


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _write_artifacts(report: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "benchmark.json").write_text(
        json.dumps(report, indent=2, default=_json_default) + "\n"
    )
    records = report["records"]
    fields = sorted({key for row in records for key in row if key != "full_metadata"})
    with (output_dir / "benchmark.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def _plot(report: dict, output_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    records = report["records"]
    sizes = sorted({row["requested_nodes"] for row in records})
    grouped = [[row for row in records if row["requested_nodes"] == size] for size in sizes]
    x = [np.median([row["n_internal_nodes"] for row in group]) for group in grouped]
    colors = {"full": "#d06b14", "structural": "#2c994b", "prox": "#15467e", "legacy": "#bf4c94"}

    def series(ax, field, label, color, *, predicate=lambda row: True, floor=None):
        values = [[row[field] for row in group
                   if predicate(row) and row.get(field) is not None
                   and np.isfinite(row[field])] for group in grouped]
        # An all-NaN fill_between collection gives Matplotlib invalid log-axis
        # limits. More importantly, an empty series must not imply successful
        # measurements when every path stopped before reaching lambda zero.
        if not any(values):
            return False
        center = [np.median(value) if value else np.nan for value in values]
        low = [min(value) if value else np.nan for value in values]
        high = [max(value) if value else np.nan for value in values]
        if floor is not None:
            center, low, high = [np.maximum(value, floor) for value in (center, low, high)]
        ax.plot(x, center, "o-", color=color, label=label, lw=2, ms=5)
        ax.fill_between(x, low, high, color=color, alpha=0.14)
        return True

    def no_measurements(ax, message):
        ax.set_yscale("linear")
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_ylabel("")
        ax.text(0.5, 0.5, message, transform=ax.transAxes, ha="center",
                va="center", fontsize=10, color="#a32b2b")

    with plt.rc_context({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}):
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.9))
        complete = lambda row: row.get("full_exact") is True
        timing_available = series(axes[0], "full_path_seconds", "New: discover all coefficient knots", colors["full"], predicate=complete)
        timing_available |= series(axes[0], "structural_seconds", "Green: structural knots only", colors["structural"])
        timing_available |= series(axes[0], "darkblue_same_knot_grid_seconds", "Dark blue: supplied knot grid", colors["prox"], predicate=complete)
        timing_available |= series(axes[0], "legacy_seconds", "Legacy hiCAP: full path", colors["legacy"],
                                   predicate=lambda row: row.get("legacy_exact") is True)
        if timing_available:
            axes[0].set_yscale("log")
            axes[0].set_ylabel("Wall time (seconds; log scale)")
            axes[0].legend(fontsize=8, loc="best")
        else:
            no_measurements(axes[0], "No complete path timings available")
        axes[0].set_title("Path construction / point evaluation")

        series(axes[1], "full_stored_points", "All coefficient knots + zero", colors["full"], predicate=complete)
        series(axes[1], "structural_stored_points", "Structural knots + zero", colors["structural"])
        axes[1].set_ylabel("Stored lambda values")
        axes[1].set_title("How much path is returned?")
        if axes[1].get_legend_handles_labels()[0]:
            axes[1].legend(fontsize=8)

        errors_available = series(axes[2], "darkblue_max_relative_coefficient_error",
                                  "Knot + midpoint error vs dark blue", colors["prox"], predicate=complete, floor=1e-16)
        errors_available |= series(axes[2], "legacy_relative_knot_distance",
                                   "Knot distance vs legacy hiCAP", colors["legacy"],
                                   predicate=lambda row: complete(row) and row.get("legacy_exact") is True,
                                   floor=1e-16)
        if errors_available:
            axes[2].set_yscale("log")
            axes[2].set_ylabel("Relative error (values <1e−16 floored)")
            axes[2].legend(fontsize=8, loc="best")
        else:
            no_measurements(axes[2], "No complete coefficient-path comparisons\n\nErrors on partial paths are omitted.")
        axes[2].set_title("Independent numerical agreement")
        for ax in axes:
            ax.set_xlabel("Actual internal tree nodes / coefficients")
            ax.grid(alpha=0.2)
            ax.set_xticks(x)
            ax.set_xticklabels([str(int(value)) for value in x])
        config = report["configuration"]
        fig.suptitle(
            f"Full hiCAP coefficient paths on fitted diabetes trees (n={report['dataset']['n_observations']})\n"
            f"{config['repeats']} repeat(s), {config['threads']} BLAS thread(s); median with min–max bands",
            fontsize=13,
        )
        failures = sum(row.get("full_exact") is not True for row in records)
        legacy_failures = sum(row.get("legacy_exact") is False for row in records)
        if failures:
            failed_sizes = [
                f"{int(position)}: {sum(not complete(row) for row in group)}/{len(group)}"
                for position, group in zip(x, grouped) if any(not complete(row) for row in group)
            ]
            fig.text(0.5, 0.83,
                     f"New solver incomplete: {failures}/{len(records)} runs. "
                     "Partial-path timings and point counts are omitted.\n"
                     "Failures / runs by node count: " + "; ".join(failed_sizes),
                     ha="center", fontsize=9, color="#a32b2b")
        fig.text(0.5, 0.02,
                 "Fit/statistics excluded. Dark-blue compilation excluded and recorded separately; its lambda grid is supplied by the new solver.\n"
                 f"Only certified complete paths enter comparisons. New incomplete runs: {failures}; legacy incomplete runs: {legacy_failures}. "
                 "Green does not return coefficient-only knots.",
                 ha="center", fontsize=8)
        fig.tight_layout(rect=(0, 0.11, 1, 0.82 if failures else 0.89))
        for extension in ("png", "svg"):
            fig.savefig(output_dir / f"diagonal_coefficient_scaling.{extension}", dpi=180)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node-counts", type=int, nargs="+", default=[7, 15, 31, 63, 127])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--observations", type=int, default=442)
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    parser.add_argument("--legacy-max-nodes", type=int, default=31)
    parser.add_argument("--legacy-tolerance", type=float, default=1e-9)
    parser.add_argument("--max-events", type=int, default=10000)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output-dir", type=Path,
                        default=REPOSITORY_ROOT / "benchmarks/diagonal_coefficient_scaling")
    args = parser.parse_args()
    if (args.repeats < 1 or args.threads < 1 or args.max_events < 1
            or args.min_samples_leaf < 1 or min(args.node_counts) < 1
            or not 2 <= args.observations <= 442):
        parser.error("counts must be positive; observations must be between 2 and 442")
    bunch = load_diabetes()
    indices = np.random.default_rng(args.seed).permutation(len(bunch.target))[:args.observations]
    X = np.asarray(bunch.data)[indices]
    raw_y = np.asarray(bunch.target)[indices]
    target_scale = float(raw_y.std()) or 1.0
    y = (raw_y - raw_y.mean()) / target_scale
    report: dict[str, Any] = {
        "schema": "imodels.sparse_pruning.diagonal_coefficient_benchmark",
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": vars(args),
        "dataset": {
            "name": "sklearn.load_diabetes", "n_observations": len(y),
            "target_center": float(raw_y.mean()), "target_scale": target_scale,
            "digest": hashlib.sha256(X.tobytes() + raw_y.tobytes()).hexdigest(),
            "path_objective": "0.5 beta.T D beta - h.T beta + lambda sum_v ||beta[subtree(v)]||_inf",
            "legacy_design": "p pseudo-observations matching the fitted-tree D and h exactly",
        },
        "environment": environment_metadata(),
        "records": [],
    }
    with threadpool_limits(limits=args.threads):
        for requested_nodes in args.node_counts:
            for repeat in range(args.repeats):
                row = _run_case(X, y, requested_nodes, repeat, args)
                report["records"].append(row)
                _write_artifacts(report, args.output_dir)
                error = row["darkblue_max_relative_coefficient_error"]
                error_label = "not_run" if error is None else f"{error:.3g}"
                print(f"p={row['n_internal_nodes']:4d} repeat={repeat} "
                      f"K={row['full_stored_points']:4d} full={row['full_path_seconds']:.4g}s "
                      f"status={row['full_status']} "
                      f"blue_error={error_label} "
                      f"legacy={row['legacy_status']}", flush=True)
    _plot(report, args.output_dir)
    print(f"Saved benchmark artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
