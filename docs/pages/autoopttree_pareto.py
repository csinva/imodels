"""Regenerate Fig 1 on the autoopttree page.

Reads the autoresearch run and the baseline benchmark from an agentic-imodels
checkout, and rewrites the ``var PARETO = {...};`` line in autoopttree.html that
the figure draws from.

    uv run python autoopttree_pareto.py --root /path/to/agentic-imodels/evolve_optimal_tree

Every point is one solver on the 70-problem suite (14 datasets x 5 penalties, 30 s
cap each). Its position is two geometric means over the problems:

* time: seconds spent, floored at 1 ms and capped at the 30 s limit, exactly as
  the leaderboard computes it, so an unsolved problem counts at the cap;
* criterion: the objective of the returned tree, misclassification rate plus the
  penalty times the number of leaves. A solver that returned no tree on a problem
  (GOSDT stopped for memory) is charged the objective of a single leaf there,
  which is what predicting the majority class costs and so the least any tree
  could have scored.

A solver run three times is plotted at the mean of its three runs with one
standard deviation either way; everything else is a single run. Criterion and
time both come from runs made at the 30 s cap: the baselines' first run is the
600 s benchmark cut down to 30 s, and its time can be cut down that way but the
tree it returned cannot, since it had twenty times longer to find it, so for
those baselines the criterion is averaged over the two runs made at 30 s.
"""

import argparse
import json
import math
import os
import re
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PAGE = os.path.join(HERE, "autoopttree.html")
CAP = 30.0
FLOOR = 1e-3

# baselines, and the label the figure gives each; their first run comes from the benchmark
BASELINES = {
    "gosdt": "GOSDT",
    "gosdt_mc8": "GOSDT, 8 threads",
    "pygosdt_v1": "pygosdt",
    "streed": "STreeD",
    "gosdt_guesses": "gosdt-guesses",
    "gosdt_guesses_guided": "gosdt-guesses, guided",
    "split": "SPLIT",
}
MULTICORE_BASELINES = {"gosdt_mc8"}
HEURISTIC_BASELINES = {"gosdt_guesses_guided", "split"}
# names the repeat script used for evolved solvers, mapped to their run names
REPEAT_ALIASES = {"autoopttree": "v23_word_compaction",
                  "autoopttree_v40": "v40_sequential",
                  "autoopttree_v46": "v46_topk_pairs"}
# the points the figure names directly; everything else is in the tooltip and the table
LABELLED = {
    "gosdt": "GOSDT",
    "streed": "STreeD",
    "pygosdt_v1": "pygosdt (start)",
    "split": "SPLIT",
    "gosdt_guesses_guided": "gosdt-guesses, guided",
    "v23_word_compaction": "v23",
    "v40_sequential": "v40 (shipped)",
    "v46_topk_pairs": "v46, 8 threads",
    "v46_anytime_1ms": "v46 anytime, 1 ms",
    "v46_anytime_100ms": "v46 anytime, 100 ms",
}
# where each direct label sits relative to its point, in pixels (x right, y down),
# chosen by rendering the page and moving labels off each other and off the marks
OFFSETS = {
    "gosdt": [40, -40],
    "streed": [40, -30],
    "pygosdt_v1": [-10, -45],
    "split": [-40, -40],
    "gosdt_guesses_guided": [95, 5],
    "v23_word_compaction": [55, -30],
    "v40_sequential": [-10, -45],
    "v46_topk_pairs": [10, 55],
    "v46_anytime_1ms": [60, -35],
    "v46_anytime_100ms": [-60, 45],
}


def leaf_objectives(suite):
    """Objective of predicting the majority class with one leaf, per problem."""
    out = {}
    for name, _ in suite.DATASETS:
        y = suite.load_dataset(name).iloc[:, -1]
        wrong = 1.0 - y.value_counts().max() / len(y)
        for lam in suite.LAMBDAS:
            out[(name, float(lam))] = wrong + lam
    return out


def metrics(rows, leaf):
    """Geometric-mean time and criterion, and counts, for one run's 70 rows."""
    df = pd.DataFrame(rows)
    secs = pd.to_numeric(df["seconds"], errors="coerce").fillna(CAP).clip(lower=FLOOR, upper=CAP)
    obj = pd.to_numeric(df["objective"], errors="coerce")
    missing = obj.isna()
    obj = obj.where(~missing, [leaf[(d, float(l))] for d, l in zip(df["dataset"], df["lam"])])
    return {
        "t": float(np.exp(np.log(secs).mean())),
        "c": float(np.exp(np.log(obj).mean())),
        "solved": int((df["status"] == "optimal").sum()),
        "wrong": int((df.get("verdict") == "WRONG").sum()),
        "notree": int(missing.sum()),
        "pairs": len(df),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.environ.get("AGENTIC_IMODELS_TREE"),
                    help="agentic-imodels/evolve_optimal_tree checkout")
    ap.add_argument("--run", default="sep15-run1")
    args = ap.parse_args()
    if not args.root:
        sys.exit("pass --root or set AGENTIC_IMODELS_TREE")
    root = os.path.abspath(os.path.expanduser(args.root))
    sys.path.insert(0, os.path.join(root, "src"))
    import suite  # noqa: E402
    from evaluate import rows_from_benchmark  # noqa: E402

    run_dir = os.path.join(root, "runs", args.run)
    results = os.path.join(root, "baselines", "benchmarks", "results")
    pairs = pd.read_csv(os.path.join(run_dir, "results", "pair_results.csv"))
    overall = pd.read_csv(os.path.join(run_dir, "results", "overall_results.csv"))
    overall = overall.drop_duplicates("model_name", keep="last").set_index("model_name")
    leaf = leaf_objectives(suite)

    # run number -> rows, per model; run 0 is the recorded one
    runs = {}
    for model in pairs["model"].unique():
        if model in BASELINES:
            continue
        runs.setdefault(model, {})[0] = pairs[pairs["model"] == model].to_dict("records")
    for model in BASELINES:
        bench = rows_from_benchmark(model)
        if bench is not None:
            runs.setdefault(model, {})[0] = bench
        elif model in set(pairs["model"]):
            runs.setdefault(model, {})[0] = pairs[pairs["model"] == model].to_dict("records")
    for fname in ("repeats.csv", "repeats_all.csv"):
        path = os.path.join(results, fname)
        if not os.path.exists(path):
            continue
        reps = pd.read_csv(path)
        for (model, rep), d in reps.groupby(["model", "repeat"]):
            model = REPEAT_ALIASES.get(model, model)
            if model in runs and len(d) == 70:
                runs[model].setdefault(int(rep), d.to_dict("records"))

    points = []
    for model, by_run in runs.items():
        per = {r: metrics(rows, leaf) for r, rows in sorted(by_run.items())}
        from_benchmark = model in BASELINES and rows_from_benchmark(model) is not None
        # see the module docstring: a 600 s run's trees do not stand for a 30 s run's
        crit_runs = [r for r in per if not (from_benchmark and r == 0)] or list(per)
        t = [per[r]["t"] for r in per]
        c = [per[r]["c"] for r in crit_runs]
        row = overall.loc[model] if model in overall.index else None
        if model in BASELINES:
            group = "baseline"
            multicore = model in MULTICORE_BASELINES
            label = BASELINES[model]
        else:
            group = "anytime" if "anytime" in model else "exact"
            multicore = bool(row is not None and str(row["multicore"]).lower() == "true")
            label = model
        proof_gap = (group == "exact" and row is not None and str(row["exact"]) == "approximate")
        points.append({
            "model": model,
            "label": label,
            "direct": LABELLED.get(model, ""),
            "group": group,
            "heuristic": model in HEURISTIC_BASELINES,
            "multicore": multicore,
            "runs": len(per),
            "t": float(np.mean(t)), "t_sd": float(np.std(t, ddof=1)) if len(t) > 1 else 0.0,
            "c": float(np.mean(c)), "c_sd": float(np.std(c, ddof=1)) if len(c) > 1 else 0.0,
            "solved": float(np.mean([per[r]["solved"] for r in per])),
            "wrong": max(per[r]["wrong"] for r in per),
            "notree": max(per[r]["notree"] for r in per),
            "proof_gap": proof_gap,
        })

    # Pareto frontier over the plotted positions: faster, or better than everything faster
    front, best = [], math.inf
    for p in sorted(points, key=lambda p: (p["t"], p["c"])):
        if p["c"] < best - 1e-12:
            front.append({"t": p["t"], "c": p["c"], "model": p["model"]})
            best = p["c"]

    points.sort(key=lambda p: (p["group"] != "baseline", p["t"]))
    data = {"points": points, "frontier": front, "cap": CAP, "offsets": OFFSETS}
    blob = json.dumps(data, separators=(",", ":"))

    page = open(PAGE).read()
    page, n = re.subn(r"var PARETO = \{.*?\};", lambda _: f"var PARETO = {blob};", page, flags=re.S)
    assert n == 1, "expected one `var PARETO = {...};` line in the page"

    open(PAGE, "w").write(page)
    print(f"{len(points)} points, {len(front)} on the frontier, "
          f"{sum(p['runs'] > 1 for p in points)} with repeats -> {PAGE}")


if __name__ == "__main__":
    main()
