"""Regenerate Fig 1 on the fastsmalltree page.

Reads the autoresearch run, the baseline benchmark and the TabArena-14 sweep from an
agentic-imodels checkout, and rewrites the ``var PARETO = {...};`` line in
fastsmalltree.html that the figure's panels draw from: ``dev`` is the development
suite, ``external`` the held-out TabArena-14 built by
baselines/benchmarks/external/build_tabarena14.py, ``full`` the full-size datasets at
30 minutes, and ``nolimit`` the held-out problems at 4 hours each
(baselines/benchmarks/external/run_external_nolimit.py).

    uv run python fastsmalltree_pareto.py --root /path/to/agentic-imodels/evolve_optimal_tree

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
PAGE = os.path.join(HERE, "fastsmalltree.html")
CAP = 30.0
FULL_CAP = 1800.0      # the full-size hidden run: 30 minutes a problem
FULL_PAIRS = 14        # one penalty (0.05) on each of the 14 full-size datasets
FULL_OFFSETS = {"fastsmalltree_v46_anytime100": [64, 24], "fastsmalltree_v49": [-54, 42],
                "fastsmalltree_v40": [34, -36], "streed": [-52, -30], "gosdt": [-62, -30],
                "split": [-26, 44], "gosdt_guesses": [-88, -8],
                "gosdt_guesses_guided": [36, -34], "gg_guided_e60d2": [74, 30]}
NOLIMIT_WALL = 4 * 3600   # the hidden run without the 30 s cap: 4 h a problem, 6 GB
FLOOR = 1e-3


def audit_gap(model):
    """Versions the Methods audit found not to preserve complete optimality, drawn as
    approximate: v20 to v39 (the depth-3 floor for seven or more leaves). The external sweeps name v23 by the package name alone."""
    if model in ("autoopttree", "fastsmalltree"):
        return True
    m = re.match(r"(?:autoopttree_|fastsmalltree_)?v(\d+)", model)
    if not m:
        return False
    n = int(m.group(1))
    return 20 <= n <= 39

# solvers whose first run comes from the 600 s benchmark, and the label the figure gives each
BASELINES = {
    "gosdt": "GOSDT",
    "gosdt_mc8": "GOSDT, 8 threads",
    "pygosdt_v1": "pygosdt",
    "streed": "STreeD",
    "gosdt_guesses": "gosdt-guesses",
    "gosdt_guesses_guided": "gosdt-guesses, guided",
    "split": "SPLIT",
    # the same baseline at settings found by sweeping its hyperparameters rather than
    # taking the paper's headline recipe, so it is judged at its best on this suite
    "gg_guided_e60d2": "gosdt-guesses, guided (tuned)",
    "gg_guided_db5": "gosdt-guesses, guided (depth 5)",
    "gg_exact_simsup": "gosdt-guesses (similar support)",
}
MULTICORE_BASELINES = {"gosdt_mc8"}
# pygosdt is version 1 of the line the loop evolved, so it is drawn with the loop's versions
# rather than as an outside baseline, even though its rows come from the benchmark
V1 = {"pygosdt_v1"}
# baselines drawn as approximate: the heuristics
APPROX_BASELINES = {"split", "gosdt_guesses_guided", "gg_guided_e60d2", "gg_guided_db5"}
HEURISTIC_BASELINES = {"gosdt_guesses_guided", "gg_guided_e60d2", "gg_guided_db5", "split"}
# the evolved solvers as the external sweep names them
EXTERNAL_EVOLVED = {"fastsmalltree": ("v23", False, "exact"),
                    "fastsmalltree_v40": ("FastSmallTree (v40)", False, "exact"),
                    "fastsmalltree_v46": ("v46, 8 threads", True, "exact"),
                    "fastsmalltree_v46_anytime100": ("v46 anytime, 100 ms", True, "approximate"),
                    "fastsmalltree_v49": ("v49, 8 threads", True, "exact")}
# names the repeat script used for evolved solvers, mapped to their run names
REPEAT_ALIASES = {"fastsmalltree": "v23_word_compaction",
                  "fastsmalltree_v40": "v40_sequential",
                  "fastsmalltree_v46": "v46_topk_pairs",
                  "fastsmalltree_v46_anytime100": "v46_anytime_100ms",
                  "fastsmalltree_v49": "v49_cands_pairs_lazy_ws"}
# The points the figure names directly; everything else is in the tooltip. The anytime
# versions are left unlabelled: at half the figure's width their labels collide with the
# exact ones, and the text and tooltips carry which cap is which.
LABELLED = {
    "gosdt": "GOSDT",
    "streed": "STreeD",
    "pygosdt_v1": "pygosdt (v1)",
    "split": "SPLIT",
    # the depth-5 run lands exactly on the published guided point, so one label names both
    "gosdt_guesses_guided": "gosdt-guesses, guided (and depth 5)",
    "gg_guided_e60d2": "gosdt-guesses, guided (tuned)",
    "gosdt_mc8": "GOSDT, 8 threads",
    "gosdt_guesses": "gosdt-guesses",
    "gg_exact_simsup": "gosdt-guesses (similar support)",
    "v23_word_compaction": "v23",
    "v40_sequential": "FastSmallTree",
    "v49_cands_pairs_lazy_ws": "v49, 8 threads",
}
EXTERNAL_LABELLED = {"gosdt": "GOSDT", "streed": "STreeD", "pygosdt_v1": "pygosdt (v1)", "split": "SPLIT",
                     "gosdt_guesses": "gosdt-guesses", "gosdt_guesses_guided": "gosdt-guesses, guided",
                     "gg_guided_e60d2": "gosdt-guesses, guided (tuned)", "fastsmalltree": "v23", "fastsmalltree_v40": "FastSmallTree",
                     "fastsmalltree_v46_anytime100": "v46 anytime, 100 ms", "fastsmalltree_v49": "v49, 8 threads"}
EXTERNAL_OFFSETS = {"gosdt": [-58, -40], "streed": [-55, -32], "pygosdt_v1": [-5, -42], "split": [-45, 38],
                    "gosdt_guesses": [-62, 34], "gosdt_guesses_guided": [82, 10],
                    "gg_guided_e60d2": [92, -30], "fastsmalltree": [46, -26],
                    "fastsmalltree_v40": [-56, -30], 
                    "fastsmalltree_v46_anytime100": [66, 26], "fastsmalltree_v49": [-64, 54]}
NOLIMIT_OFFSETS = {"gosdt": [60, 30], "streed": [-55, -32], "pygosdt_v1": [-5, -42], "split": [-45, 38],
                   "gosdt_guesses": [-62, 34], "gosdt_guesses_guided": [82, 10],
                   "gg_guided_e60d2": [92, -30], "fastsmalltree": [50, 20],
                   "fastsmalltree_v40": [10, 70],
                   "fastsmalltree_v46_anytime100": [66, 26], "fastsmalltree_v49": [60, 40]}
# where each direct label sits relative to its point, in pixels (x right, y down),
# chosen by rendering the page and moving labels off each other and off the marks
OFFSETS = {
    # the four baselines near 0.5 s fan out up and to the left, in the order of their points
    "gg_exact_simsup": [-110, -127],
    "gosdt_guesses": [-70, -104],
    "gosdt_mc8": [-60, -84],
    "gosdt": [-40, -66],

    "streed": [-48, -34],
    "pygosdt_v1": [-15, -42],
    "split": [0, -34],
    "gosdt_guesses_guided": [120, 8],
    "v23_word_compaction": [58, -26],
    "v40_sequential": [-5, -46],
    "v49_cands_pairs_lazy_ws": [22, 52],
    "gg_guided_e60d2": [30, -80],
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


def metrics(rows, leaf, cap=CAP):
    """Geometric-mean time and criterion, and counts, for one run's rows."""
    df = pd.DataFrame(rows)
    secs = pd.to_numeric(df["seconds"], errors="coerce").fillna(cap).clip(lower=FLOOR, upper=cap)
    obj = pd.to_numeric(df["objective"], errors="coerce")
    missing = obj.isna()
    obj = obj.where(~missing, [leaf[(d, float(l))] for d, l in zip(df["dataset"], df["lam"])])
    return {
        "t": float(np.exp(np.log(secs).mean())),
        "c": float(np.exp(np.log(obj).mean())),
        "solved": int((df["status"] == "optimal").sum()),
        "wrong": int((df.get("verdict") == "WRONG").sum()),
        "notree": int(missing.sum()),
        "memory": int((df["status"] == "memory").sum()),
        # a run that used the whole budget: the solver stopped itself on time, or was killed
        "unfinished": int((((df["status"] == "time") & (secs >= 0.99 * cap)) | (df["status"] == "unfinished")).sum()),
        "pairs": len(df),
    }


def model_ids(df):
    """The benchmark harness recorded this model under its earlier name; read it as the new one."""
    df["model"] = df["model"].astype(str).str.replace("autoopttree", "fastsmalltree", regex=False)
    return df


def external_points(root, data="data", pairs="external_pairs.csv", npairs=70, cap=CAP,
                    offsets=None):
    """Points for a held-out panel: every solver's runs on a TabArena-14 build, judged against
    the best tree any run returned for the problem, since these datasets have no certified
    optima. `data` is the dataset folder and `pairs` the sweep that ran over it."""
    ext = os.path.join(root, "baselines", "benchmarks", "external")
    path = os.path.join(ext, "results", pairs)
    if not os.path.exists(path):
        return None
    d = model_ids(pd.read_csv(path))
    leaf = {}
    for name in d["dataset"].unique():
        y = pd.read_csv(os.path.join(ext, data, f"{name}.csv")).iloc[:, -1]
        wrong = 1.0 - y.value_counts().max() / len(y)
        for lam in d.loc[d["dataset"] == name, "lam"].unique():
            leaf[(name, float(lam))] = wrong + lam
    best = d.groupby(["dataset", "lam"])["objective"].min().rename("best")
    d = d.join(best, on=["dataset", "lam"])
    d["verdict"] = np.where((d["status"] == "optimal") & (d["objective"] > d["best"] + 1e-9), "WRONG", "ok")
    d.loc[d["status"] == "crash", "verdict"] = "WRONG"
    points = []
    for model, g in d.groupby("model"):
        if "anytime" in model:
            continue
        per = {r: metrics(rows.to_dict("records"), leaf, cap) for r, rows in g.groupby("repeat")
               if len(rows) == npairs}
        if not per:
            continue
        t = [v["t"] for v in per.values()]
        c = [v["c"] for v in per.values()]
        if model in EXTERNAL_EVOLVED:
            label, multicore, group = EXTERNAL_EVOLVED[model]
            if audit_gap(model):
                group = "approximate"
        else:
            label, multicore = BASELINES.get(model, model), model in MULTICORE_BASELINES
            group = "exact" if model in V1 else ("baseline_approx" if model in APPROX_BASELINES else "baseline_exact")
        points.append({
            "model": model, "label": label, "direct": EXTERNAL_LABELLED.get(model, ""), "group": group,
            "heuristic": model in HEURISTIC_BASELINES, "multicore": multicore, "runs": len(per),
            "t": float(np.mean(t)), "t_sd": float(np.std(t, ddof=1)) if len(t) > 1 else 0.0,
            "c": float(np.mean(c)), "c_sd": float(np.std(c, ddof=1)) if len(c) > 1 else 0.0,
            "solved": float(np.mean([v["solved"] for v in per.values()])),
            "wrong": max(v["wrong"] for v in per.values()),
            "notree": max(v["notree"] for v in per.values()),
            "memory": max(v["memory"] for v in per.values()),
            "unfinished": max(v["unfinished"] for v in per.values()),
            "pairs": npairs, "proof_gap": False,
        })
    return {"points": points, "offsets": EXTERNAL_OFFSETS if offsets is None else offsets}


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
        reps = model_ids(pd.read_csv(path))
        for (model, rep), d in reps.groupby(["model", "repeat"]):
            model = REPEAT_ALIASES.get(model, model)
            # a baseline measured only here (the tuned gosdt-guesses settings) has no run 0
            # from the 600 s benchmark, so its repeat rows are all it has
            if len(d) == 70 and (model in runs or model in BASELINES):
                runs.setdefault(model, {}).setdefault(int(rep), d.to_dict("records"))

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
            group = "exact" if model in V1 else ("baseline_approx" if model in APPROX_BASELINES else "baseline_exact")
            multicore = model in MULTICORE_BASELINES
            label = BASELINES[model]
        else:
            # the anytime rows (the exact solver stopped at a cap) are not plotted; the one
            # approximate version left is the round that guessed its bounds from a reference model
            if "anytime" in model:
                continue
            approximate = row is not None and str(row["exact"]) == "approximate"
            group = "approximate" if (approximate or audit_gap(model)) else "exact"
            multicore = bool(row is not None and str(row["multicore"]).lower() == "true")
            label = "FastSmallTree (v40)" if model == "v40_sequential" else model
        proof_gap = False
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

    points.sort(key=lambda p: (not p["group"].startswith("baseline"), p["t"]))
    data = {"dev": {"points": points, "offsets": OFFSETS},
            "external": external_points(root), "cap": CAP,
            "full": external_points(root, data="data_full", pairs="external_pairs_full30min.csv",
                                    npairs=FULL_PAIRS, cap=FULL_CAP, offsets=FULL_OFFSETS),
            # no time limit: seconds is the solver's clock, or the wall time at a memory stop, so
            # the cap only fills the rows that had neither
            "nolimit": external_points(root, pairs="external_pairs_nolimit.csv", cap=NOLIMIT_WALL,
                                       offsets=NOLIMIT_OFFSETS)}
    blob = json.dumps(data, separators=(",", ":"))

    page = open(PAGE).read()
    page, n = re.subn(r"var PARETO = \{.*?\};", lambda _: f"var PARETO = {blob};", page, flags=re.S)
    assert n == 1, "expected one `var PARETO = {...};` line in the page"

    open(PAGE, "w").write(page)
    ext = data["external"]
    print(f"dev: {len(points)} points, {sum(p['runs'] > 1 for p in points)} with repeats; "
          f"external: {len(ext['points']) if ext else 0} points, "
          f"full: {len(data['full']['points']) if data['full'] else 0} points, "
          f"nolimit: {len(data['nolimit']['points']) if data['nolimit'] else 0} points -> {PAGE}")


if __name__ == "__main__":
    main()
