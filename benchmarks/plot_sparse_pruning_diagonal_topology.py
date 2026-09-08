#!/usr/bin/env python3
"""Plot exact diagonal topology/point paths against legacy hiCAP and APA."""
from __future__ import annotations

import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "imodels-mpl-cache"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(tempfile.gettempdir(), "imodels-xdg-cache"))

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


NEW_SCHEMA = "imodels.sparse_pruning.diagonal_topology_benchmark"
NEW_SCHEMA_VERSION = 3
LEGACY_SCHEMA_VERSION = 2
COLORS = {
    "topology": "#009E73",
    "topology_design": "#56B4E9",
    "direct": "#0072B2",
    "hicap": "#CC79A7",
    "apa": "#D55E00",
}


def _load(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _summary(values):
    values = np.asarray(values, dtype=float)
    return (
        float(np.median(values)),
        float(np.quantile(values, 0.25)),
        float(np.quantile(values, 0.75)),
    )


def _new_x(record, axis):
    return record["requested_nodes"] if axis == "nodes" else record["n_samples"]


def _old_x(record, axis):
    return (
        record["requested_internal_nodes"]
        if axis == "nodes"
        else record["n_samples"]
    )


def _usable_legacy_record(record, method, field):
    if record.get("outcome") != "complete" or record.get(field) is None:
        return False
    if method == "hicap_exact":
        return bool(
            record.get("exact") is True
            and record.get("solver_status") == "complete"
        )
    return True


def _new_series(document, axis, field):
    grouped = defaultdict(list)
    for record in document["records"]:
        if record["axis"] != axis:
            continue
        x = _new_x(record, axis)
        grouped[int(x)].append(float(record[field]))
    x_values = np.asarray(sorted(grouped))
    if x_values.size == 0:
        empty = np.asarray([], dtype=float)
        return empty, empty, empty, empty
    summaries = np.asarray([_summary(grouped[x]) for x in x_values])
    return x_values, summaries[:, 0], summaries[:, 1], summaries[:, 2]


def _old_series(document, axis, method, field="solver_wall_seconds"):
    grouped = defaultdict(list)
    for record in document["records"]:
        if record["scaling_axis"] != axis or record["method"] != method:
            continue
        if not _usable_legacy_record(record, method, field):
            continue
        x = _old_x(record, axis)
        grouped[int(x)].append(float(record[field]))
    x_values = np.asarray(sorted(grouped))
    if x_values.size == 0:
        empty = np.asarray([], dtype=float)
        return empty, empty, empty, empty
    summaries = np.asarray([_summary(grouped[x]) for x in x_values])
    return x_values, summaries[:, 0], summaries[:, 1], summaries[:, 2]


def _plot_timing(axis, document, legacy, scaling_axis):
    path_points = int(document["metadata"]["arguments"]["path_points"])
    apa_partial = any(
        record.get("method") == "apa_apg2_warm"
        and record.get("solver_status") != "complete"
        for record in legacy["records"]
    )
    specifications = [
        (
            _new_series(
                document, scaling_axis, "fitted_tree_topology_seconds"
            ),
            "Tree-native exact topology: all knots",
            COLORS["topology"],
            "o",
        ),
        (
            _new_series(document, scaling_axis, "topology_seconds"),
            "Design-input exact topology: all knots",
            COLORS["topology_design"],
            "v",
        ),
        (
            _new_series(document, scaling_axis, "coefficient_grid_seconds"),
            f"Exact diagonal prox: {path_points} points",
            COLORS["direct"],
            "s",
        ),
        (
            _old_series(legacy, scaling_axis, "hicap_exact"),
            "Legacy hiCAP: all coefficient knots",
            COLORS["hicap"],
            "^",
        ),
        (
            _old_series(legacy, scaling_axis, "apa_apg2_warm"),
            (
                f"APA-APG2: {path_points} warm points"
                + (" (partial)" if apa_partial else "")
            ),
            COLORS["apa"],
            "D",
        ),
    ]
    for (x, median, lower, upper), label, color, marker in specifications:
        axis.plot(x, median, marker=marker, color=color, label=label, linewidth=1.8)
        axis.fill_between(x, lower, upper, color=color, alpha=0.13)
    axis.set_xscale("log", base=2)
    axis.set_yscale("log")
    axis.grid(True, which="both", alpha=0.2)
    axis.set_ylabel("Wall time (seconds, median)")
    axis.set_xlabel(
        "Requested internal tree nodes"
        if scaling_axis == "nodes"
        else "Observations"
    )


def _lookup_medians(document, axis, field):
    x, median, _, _ = _new_series(document, axis, field)
    return dict(zip(x.tolist(), median.tolist()))


def _lookup_old_medians(document, axis, method, field="solver_wall_seconds"):
    x, median, _, _ = _old_series(document, axis, method, field)
    return dict(zip(x.tolist(), median.tolist()))


def make_figure(document, legacy):
    _validate_compatibility(document, legacy)
    arguments = document["metadata"]["arguments"]
    dataset_label = document["metadata"]["dataset_name"].replace("_", " ").title()
    fixed_observations = int(arguments["fixed_observations"])
    fixed_nodes = int(arguments["fixed_nodes"])
    path_points = int(arguments["path_points"])
    repeats = int(arguments["repeats"])
    figure, axes = plt.subplots(2, 2, figsize=(12.0, 8.5), constrained_layout=True)
    _plot_timing(axes[0, 0], document, legacy, "nodes")
    axes[0, 0].set_title(
        f"A  Scaling with tree size ({dataset_label}, n={fixed_observations:,})",
        loc="left",
    )
    axes[0, 0].legend(fontsize=7.7, loc="upper left")

    _plot_timing(axes[0, 1], document, legacy, "observations")
    axes[0, 1].set_title(
        f"B  Scaling with observations ({fixed_nodes}-node target)", loc="left"
    )

    x, topology_knots, _, _ = _new_series(
        document, "nodes", "n_topology_knots"
    )
    axes[1, 0].plot(
        x,
        topology_knots,
        color=COLORS["topology"],
        marker="o",
        linewidth=1.8,
        label="Exact structural knots",
    )
    old_x, old_y, _, _ = _old_series(
        legacy, "nodes", "hicap_exact", field="native_points"
    )
    axes[1, 0].plot(
        old_x,
        old_y,
        color=COLORS["hicap"],
        marker="^",
        linewidth=1.8,
        label="Stored coefficient-path points (hiCAP)",
    )
    apa_x, apa_points, _, _ = _old_series(
        legacy,
        "nodes",
        "apa_apg2_warm",
        field="requested_path_points",
    )
    axes[1, 0].plot(
        apa_x,
        apa_points,
        color=COLORS["apa"],
        linestyle="--",
        label="APA requested points (run range)",
    )
    axes[1, 0].set_xscale("log", base=2)
    axes[1, 0].set_yscale("log")
    axes[1, 0].grid(True, which="both", alpha=0.2)
    axes[1, 0].set_xlabel("Requested internal tree nodes")
    axes[1, 0].set_ylabel("Stored knots or points")
    axes[1, 0].set_title("C  Structural events are a smaller path target", loc="left")
    axes[1, 0].legend(fontsize=8)

    direct = _lookup_medians(document, "nodes", "coefficient_grid_seconds")
    topology = _lookup_medians(
        document, "nodes", "fitted_tree_topology_seconds"
    )
    apa = _lookup_old_medians(legacy, "nodes", "apa_apg2_warm")
    hicap = _lookup_old_medians(legacy, "nodes", "hicap_exact")
    common_direct = sorted(set(direct) & set(apa))
    common_topology = sorted(set(topology) & set(hicap))
    axes[1, 1].plot(
        common_direct,
        [apa[value] / direct[value] for value in common_direct],
        color=COLORS["direct"],
        marker="s",
        linewidth=1.8,
        label=f"APA {path_points}-point / exact-prox {path_points}-point",
    )
    axes[1, 1].plot(
        common_topology,
        [hicap[value] / topology[value] for value in common_topology],
        color=COLORS["topology"],
        marker="o",
        linewidth=1.8,
        label="hiCAP coefficient path / tree-native topology path",
    )
    axes[1, 1].axhline(1.0, color="#555555", linewidth=0.8)
    axes[1, 1].set_xscale("log", base=2)
    axes[1, 1].set_yscale("log")
    axes[1, 1].grid(True, which="both", alpha=0.2)
    axes[1, 1].set_xlabel("Requested internal tree nodes")
    axes[1, 1].set_ylabel("Ratio of median wall times")
    axes[1, 1].set_title(
        "D  Ratios of displayed medians (not equal-accuracy)", loc="left"
    )
    axes[1, 1].legend(fontsize=8)

    figure.suptitle(
        "Exact diagonal hiCAP point solves and exact structural pruning path\n"
        f"Real fitted CART local-stump designs; medians over {repeats} seeded problems",
        fontsize=13,
    )
    figure.text(
        0.5,
        -0.035,
        "Tree-native topology uses fitted-node sufficient statistics; design-input topology also "
        "forms h and D from X. Both enumerate every zero-support event without coefficient vectors.\n"
        f"Direct prox and APA use the same {path_points}-point grid; APA runs are capped/partial, "
        "and only certified-complete legacy hiCAP paths are shown.",
        ha="center",
        fontsize=8.2,
    )
    return figure


def _validate_compatibility(document, legacy):
    if document.get("schema_name") != NEW_SCHEMA:
        raise ValueError("unexpected diagonal-topology benchmark schema")
    if document.get("schema_version") != NEW_SCHEMA_VERSION:
        raise ValueError("unexpected diagonal-topology benchmark schema version")
    if legacy.get("schema_version") != LEGACY_SCHEMA_VERSION:
        raise ValueError("unexpected legacy benchmark schema version")
    new_metadata = document.get("metadata", {})
    old_metadata = legacy.get("metadata", {})
    for key in (
        "dataset_digest",
        "dataset_name",
        "dataset_rows",
        "dataset_raw_features",
        "numpy",
        "platform",
        "python",
        "scikit_learn",
        "scipy",
    ):
        if new_metadata.get(key) != old_metadata.get(key):
            raise ValueError(f"benchmark artifacts disagree on {key}")
    new_arguments = new_metadata.get("arguments", {})
    old_arguments = old_metadata.get("arguments", {})
    for key in (
        "fixed_nodes",
        "fixed_observations",
        "lambda_min_ratio",
        "min_samples_leaf",
        "path_points",
        "repeats",
        "seed",
        "threads",
    ):
        if new_arguments.get(key) != old_arguments.get(key):
            raise ValueError(f"benchmark artifacts disagree on {key}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("legacy", type=Path)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    document = _load(arguments.results)
    legacy = _load(arguments.legacy)
    figure = make_figure(document, legacy)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(arguments.output, dpi=180, bbox_inches="tight")
    if arguments.output.suffix.lower() != ".svg":
        figure.savefig(arguments.output.with_suffix(".svg"), bbox_inches="tight")
    plt.close(figure)
    print(arguments.output)


if __name__ == "__main__":
    main()
