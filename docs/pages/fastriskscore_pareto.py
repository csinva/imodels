"""Regenerate Fig 1 and Table 1 on the fastriskscore page.

Reads the autoresearch runs and the benchmark results from an agentic-imodels checkout and
rewrites, in fastriskscore.html, the ``var PARETO = {...};`` line that Fig 1 draws from and
the rows between ``<!-- TABLE1 -->`` and ``<!-- /TABLE1 -->``.

    uv run python fastriskscore_pareto.py --root /path/to/agentic-imodels/evolve_slim

Panels: ``dev`` is the visible suite (14 datasets x 5 values of k, 60 s a problem), ``hidden``
the 27 held-out TabArena datasets under the same protocol, ``full`` the same 27 at full size
(k = 5, 10 minutes a problem). A point is one solver: its geometric-mean fit time over the
panel's problems and its mean excess training loss over the best known loss of each problem
(src/best_known.csv: the lowest calibrated log loss any integer solver reached). A solver run
three times is drawn at the mean of its runs.
"""

import argparse
import json
import math
import os
import re

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PAGE = os.path.join(HERE, "fastriskscore.html")
SHIP = "v35_scratch"
BASELINES = {
    "fasterrisk": "FasterRisk",
    "fasterrisk_wide": "FasterRisk, wide search",
    "riskslim": "RiskSLIM (CPLEX CE)",
    "slim_milp": "SLIM (HiGHS)",
    "rounded_lr": "Rounded L1 logistic",
    "imodels_slim": "imodels SLIMClassifier",
}
DIRECT = {"fasterrisk": "FasterRisk", "fasterrisk_wide": "FasterRisk, wide", "riskslim": "RiskSLIM",
          "slim_milp": "SLIM", "rounded_lr": "Rounded L1-LR", "imodels_slim": "imodels SLIM"}
FLOOR = 1e-5


def read(path):
    return pd.read_csv(path, dtype={"model_name": str}) if os.path.exists(path) else pd.DataFrame()


def problem_regret(root, prefix, models):
    """Mean regret per model recomputed from the per-problem rows against the current best known."""
    best = pd.read_csv(os.path.join(root, "src", "best_known.csv"))
    key = {(r.suite, r.dataset, int(r.k)): r.loss for r in best.itertuples()}
    out = {}
    frames = [read(os.path.join(root, "results", f"{p}problem_results.csv")) for p in prefix]
    frames += [read(os.path.join(root, "runs", r, "results", "problem_results.csv"))
               for r in ("sep26-run1", "sep26-run2") if "" in prefix]
    for df in frames:
        if df.empty:
            continue
        for m, g in df.groupby("model"):
            if m in models:
                reg = g["loss"] - [key[(s, d, int(k))] for s, d, k in zip(g["suite"], g["dataset"], g["k"])]
                out[m] = (float(reg.mean()), float(np.exp(np.mean(np.log(np.maximum(g["seconds"], 1e-3))))),
                          float(g["auc_test"].mean()), float(g["loss"].mean()), int(len(g)))
    return out


def point(model, label, group, vals, runs, direct=None):
    reg = [v[0] for v in vals]
    t = [v[1] for v in vals]
    return {"model": model, "label": label, "group": group, "runs": len(vals),
            "c": max(float(np.mean(reg)), FLOOR), "c_sd": float(np.std(reg)), "c_raw": float(np.mean(reg)),
            "t": float(np.mean(t)), "t_sd": float(np.std(t)), "auc": float(np.mean([v[2] for v in vals])),
            "loss": float(np.mean([v[3] for v in vals])), "direct": direct}


def panel(root, prefix, version_names, ship_names, fr_names, extra_prefix=None):
    names = set(version_names) | set(ship_names) | set(fr_names) | set(BASELINES)
    stats = problem_regret(root, [prefix] + ([extra_prefix] if extra_prefix else []), names)
    pts = []
    for m, (label, grp) in version_names.items():
        if m in stats:
            pts.append(point(m, label, grp, [stats[m]], 1))
    ship = [stats[m] for m in ship_names if m in stats]
    if ship:
        pts.append(point(SHIP, "FastRiskScore (shipped, run 2 v35)", "run2", ship, len(ship), "FastRiskScore"))
    fr = [stats[m] for m in fr_names if m in stats]
    if fr:
        pts.append(point("fasterrisk", BASELINES["fasterrisk"], "baseline", fr, len(fr), DIRECT["fasterrisk"]))
    for m, label in BASELINES.items():
        if m != "fasterrisk" and m in stats:
            pts.append(point(m, label, "baseline", [stats[m]], 1, DIRECT[m]))
    return {"points": pts}


def versions(root, run, tag=""):
    ov = read(os.path.join(root, "runs", run, "results", "overall_results.csv"))
    ov = ov[ov["status"].isin(["keep", "discard"]) & (ov["model_name"] != "pyfasterrisk_v1")]
    grp = "run1" if run.endswith("1") else "run2"
    return {m + tag: (f"run {grp[-1]}, {m}" + ("" if s == "keep" else " (discarded)"), grp if s == "keep" else grp + "_discard")
            for m, s in zip(ov["model_name"], ov["status"])}


def table_rows(panels):
    order = [("fastriskscore", SHIP, "FastRiskScore (ours)"), ("fasterrisk", "fasterrisk", "FasterRisk"),
             ("fasterrisk_wide", "fasterrisk_wide", "FasterRisk, wide search"),
             ("rounded_lr", "rounded_lr", "Rounded L1 logistic"),
             ("imodels_slim", "imodels_slim", "imodels SLIMClassifier (before)"),
             ("slim_milp", "slim_milp", "SLIM (HiGHS)"), ("riskslim", "riskslim", "RiskSLIM (CPLEX CE)"),
             ("continuous_beam", "continuous_beam", "Real-valued k-sparse (not integer)")]
    rows = []
    for _, key, label in order:
        cells = []
        for P in panels:
            p = next((q for q in P["points"] if q["model"] == key), None) if P else None
            if p is None:
                cells += ["&ndash;"] * 3
            else:
                cells += [f"{p['loss']:.4f}", f"{p['auc']:.3f}", f"{p['t']:.3f}" if p['t'] < 1 else f"{p['t']:.3g}"]
        b = " class=\"ours\"" if key == SHIP else ""
        rows.append(f"<tr{b}><td>{label}</td>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
    return "\n".join("                          " + r for r in rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    root = ap.parse_args().root
    dev_versions = {**versions(root, "sep26-run1"), **versions(root, "sep26-run2")}
    dev_versions.pop(SHIP, None)
    dev = panel(root, "", dev_versions, [SHIP, SHIP + "_rep2", SHIP + "_rep3"],
                ["fasterrisk", "fasterrisk_rep2", "fasterrisk_rep3"], extra_prefix="t1200_")
    hid_versions = {}
    for run, tag in (("sep26-run1", "_run1"), ("sep26-run2", "_run2")):
        for m, (label, grp) in versions(root, run).items():
            if not grp.endswith("_discard"):
                hid_versions[m + tag] = (label, grp)
    hid_versions.pop(SHIP + "_run2", None)
    hidden = panel(root, "hidden_", hid_versions, [SHIP + "_run2", SHIP + "_run2_rep2", SHIP + "_run2_rep3"],
                   ["fasterrisk", "fasterrisk_rep2", "fasterrisk_rep3"], extra_prefix="hidden_t1200_")
    full = panel(root, "hidden_full_t600_", {}, [SHIP], ["fasterrisk"])
    # the real-valued reference goes in the table only: its excess loss is below zero
    for P, pre in ((dev, ""), (hidden, "hidden_"), (full, "hidden_full_t600_")):
        s = problem_regret(root, [pre], {"continuous_beam"})
        if "continuous_beam" in s:
            P.setdefault("table_only", []).append(point("continuous_beam", "real-valued", "reference",
                                                          [s["continuous_beam"]], 1))
    data = {"dev": dev, "hidden": hidden, "full": full if full["points"] else None}
    html = open(PAGE).read()
    html = re.sub(r"var PARETO = .*?;\n", "var PARETO = " + json.dumps(data, separators=(",", ":")) + ";\n", html,
                  count=1, flags=re.S)
    tbl_panels = [dict(points=P["points"] + P.get("table_only", [])) if P else None for P in (dev, hidden, full)]
    html = re.sub(r"(<!-- TABLE1 -->).*?(<!-- /TABLE1 -->)",
                  lambda m: m.group(1) + "\n" + table_rows(tbl_panels) + "\n                          " + m.group(2),
                  html, count=1, flags=re.S)
    open(PAGE, "w").write(html)
    for name, P in data.items():
        if P:
            for p in sorted(P["points"], key=lambda q: q["t"])[:4] + [q for q in P["points"] if q["model"] in (SHIP, "fasterrisk")]:
                print(f"{name:6s} {p['label'][:40]:40s} t={p['t']:.4f} regret={p['c_raw']:.5f} auc={p['auc']:.4f} runs={p['runs']}")
