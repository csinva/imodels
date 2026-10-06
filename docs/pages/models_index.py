"""Write docs/pages/models.html: an index of every class in imodels.__all__, grouped by category.

Categories are read off the sub-package a class is defined in, as the docs sidebar does
(html.mako), so the page and the sidebar always agree. Classifier, Regressor and CV variants
of one model share a row. Each family has a one-line description below; a class that is not
listed here still appears, with the first line of its docstring.

    uv run python docs/pages/models_index.py      # build_docs.sh runs this before build_pages.py
"""

import html
import inspect
import os
import re
import warnings

warnings.simplefilter("ignore")
import imodels  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

# (key, label, description); the order is the page's order
CATEGORIES = [
    ("tree", "Trees", "Single trees and sums of trees, grown greedily, searched for optimality, or regularized after fitting."),
    ("rule_set", "Rule sets", "Unordered sets of if-then rules, combined by a vote or a sparse linear model."),
    ("rule_list", "Rule lists", "Ordered if-then-else lists, read from the top until a rule applies."),
    ("algebraic", "Algebraic models", "Linear, additive and integer-point models."),
    ("util", "Utilities", "Wrappers that select or distill interpretable models."),
    ("discretization", "Discretization", "Turn numeric features into bins, as input for the rule-based models."),
    ("clustering", "Clustering", "Clustering helpers."),
    ("experimental", "Experimental", "Models that are less tested."),
]
MISC = {"discretization", "clustering", "experimental"}  # the sidebar's "Misc classes"

# family -> (display name, description, badges)
FAMILIES = {
    "FIGS": ("FIGS", "Fast interpretable greedy-tree sums: a sum of small trees with few splits in total.", []),
    "FastSmallTree": ("FastSmallTree", "The certifiably optimal small tree for error plus a penalty per leaf.", ["autoresearch", "numba"]),
    "HSTree": ("Hierarchical shrinkage", "Post-hoc regularization of any tree or tree ensemble, shrinking each node towards its parent.", []),
    "HSDecisionTreeCCP": ("Hierarchical shrinkage + CCP", "Hierarchical shrinkage of a cost-complexity-pruned tree, both tuned by cross-validation.", []),
    "DecisionTreeCCP": ("CCP-pruned tree", "A CART tree pruned to a target complexity by cost-complexity pruning.", []),
    "GreedyTree": ("Greedy tree (CART)", "scikit-learn's CART tree with imodels' rule extraction and printing.", []),
    "C45Tree": ("C4.5 tree", "A C4.5 decision tree, split by information-gain ratio.", []),
    "TaoTree": ("TAO tree", "Tree alternating optimization: refines the splits of a grown tree, one node at a time.", []),
    "IRF": ("Iterative random forest", "Random forests refit with feature weights, to find stable feature interactions.", []),
    "RuleFit": ("RuleFit", "A sparse linear model on rules extracted from a tree ensemble.", []),
    "SkopeRules": ("Skope-rules", "Rules from bagged trees, kept by precision and recall and deduplicated.", []),
    "BoostedRules": ("Boosted rules", "A boosted sum of short rules.", []),
    "Slipper": ("SLIPPER", "Boosted rules fit with the confidence-rated SLIPPER algorithm.", []),
    "BayesianRuleSet": ("Bayesian rule set", "A Bayesian or-of-ands rule set.", []),
    "FPLasso": ("FP-Lasso", "A lasso over rules mined by frequent-pattern growth.", []),
    "FPSkope": ("FP-Skope", "Skope-rules over rules mined by frequent-pattern growth.", []),
    "BayesianRuleList": ("Bayesian rule list", "An if-then-else list chosen by a posterior over lists.", []),
    "GreedyRuleList": ("Greedy rule list", "A rule list grown greedily, one split at a time.", []),
    "OneR": ("OneR", "A rule list on the single most predictive feature.", []),
    "FastFrugalTree": ("Fast-and-frugal tree", "One cue per level, each with an exit.", []),
    "FastRiskScore": ("FastRiskScore", "A sparse integer risk score with a calibrated risk for every total.", ["autoresearch", "numba"]),
    "GPGam": ("GPGam", "An additive Gaussian-process model with pairwise interactions and posterior bands.", ["autoresearch"]),
    "TreeGAM": ("Tree GAM", "A GAM whose shape functions are boosted small trees.", []),
    "MarginalShrinkageLinear": ("Marginal shrinkage linear model", "A linear model shrunk towards each feature's marginal effect.", []),
    "SLIM": ("SLIM", "A sparse linear model with integer coefficients. The classifier is deprecated in favour of FastRiskScore.", []),
    "AutoInterpretable": ("AutoInterpretable", "Fits and selects among interpretable models automatically.", []),
    "Distilled": ("Distillation", "Distills a black-box model into an interpretable one.", []),
    "BasicDiscretizer": ("Basic discretizer", "Equal-width or quantile bins (scikit-learn's KBinsDiscretizer).", []),
    "MDLPDiscretizer": ("MDLP discretizer", "Supervised bins by Fayyad and Irani's MDLP criterion.", []),
    "BRLDiscretizer": ("BRL discretizer", "The MDLP discretization used by the Bayesian rule list.", []),
    "RFDiscretizer": ("RF discretizer", "Bins from the split points of a random forest.", []),
    "StableClustering": ("Stable clustering", "Picks the number of clusters by how stable the clustering is across repeated runs.", []),
    "BART": ("BART", "Bayesian additive regression trees (slow).", ["experimental"]),
}
DEPRECATED = {"SLIMClassifier"}
POSTS = {"FIGS": "figs.html", "FastSmallTree": "fastsmalltree.html", "HSTree": "shrinkage.html",
         "FastRiskScore": "fastriskscore.html", "GPGam": "gpgam.html"}


def family_of(name):
    return re.sub(r"(Classifier|Regressor)?(CV)?$", "", name) or name


def kind_of(name):
    m = re.search(r"(Classifier|Regressor)(CV)?$", name)
    return (m.group(1) + (" CV" if m.group(2) else "")) if m else name


def api_href(cls):
    mod = cls.__module__.split(".")[1:]
    return "/".join(mod) + f".html#{cls.__module__}.{cls.__name__}"


def first_line(cls):
    doc = (inspect.getdoc(cls) or "").strip().split("\n\n")[0].replace("\n", " ")
    return doc if doc and "scikit-learn" not in doc else ""


def collect():
    rows = {}
    for name in imodels.__all__:
        cls = getattr(imodels, name, None)
        if not inspect.isclass(cls) or not cls.__module__.startswith("imodels."):
            continue
        cat = cls.__module__.split(".")[1]
        fam = family_of(name)
        rows.setdefault((cat, fam), []).append(cls)
    return rows


def badge(b):
    return f'<span class="mi-badge mi-{b}">{b}</span>'


def render():
    rows = collect()
    sections, nav = [], []
    for misc_part in (False, True):
        if misc_part:
            sections.append('<h2 class="mi-part" id="misc">Misc classes</h2>')
        for key, label, desc in CATEGORIES:
            if (key in MISC) != misc_part:
                continue
            fams = sorted([f for (c, f) in rows if c == key],
                          key=lambda f: list(FAMILIES).index(f) if f in FAMILIES else 999)
            if not fams:
                continue
            nav.append(f'<a class="mi-chip" href="#cat-{key}"><span class="cat-dot cat-{key}"></span>{label}</a>')
            trs = []
            for f in fams:
                classes = sorted(rows[(key, f)], key=lambda c: ("Regressor" in c.__name__, "CV" in c.__name__))
                disp, text, badges = FAMILIES.get(f, (f, first_line(classes[0]), []))
                links = " ".join(
                    f'<a class="mi-cls{" mi-dep" if c.__name__ in DEPRECATED else ""}" href="{api_href(c)}" '
                    f'title="{c.__name__}">{html.escape(kind_of(c.__name__))}</a>' for c in classes)
                post = f' <a class="mi-post" href="{POSTS[f]}">post&nbsp;&rarr;</a>' if f in POSTS else ""
                trs.append(
                    f'<tr><td class="mi-name">{html.escape(disp)}{"".join(badge(b) for b in badges)}</td>'
                    f'<td class="mi-desc">{html.escape(text)}{post}</td><td class="mi-links">{links}</td></tr>')
            sections.append(
                f'<section class="mi-cat" id="cat-{key}">\n<h3><span class="cat-dot cat-{key}"></span>{label}</h3>\n'
                f'<p class="mi-catdesc">{html.escape(desc)}</p>\n<table class="mi-table">\n<tbody>\n'
                + "\n".join(trs) + "\n</tbody>\n</table>\n</section>")
    n = sum(len(v) for v in rows.values())
    return f'''<section id="section-intro">
<div class="article" style="padding-right: 2%; padding-left: 2%;">
<h1 style="padding-bottom: 0px;">Prediction classes</h1>
<p class="mi-lede">All {n} classes in <code>imodels</code>, grouped by the kind of model they fit. Every model follows the
scikit-learn API (<code>fit</code>, <code>predict</code>, <code>predict_proba</code>); the variant links open its API page.</p>
<nav class="mi-nav">{"".join(nav)}</nav>
<style>
  .mi-lede {{ color: var(--ink-soft); max-width: 46rem; }}
  .mi-nav {{ display: flex; flex-wrap: wrap; gap: .5rem; margin: 1.2rem 0 1.6rem; }}
  .mi-chip {{ display: inline-flex; align-items: center; gap: .45rem; padding: .3rem .8rem; border: 1px solid var(--line);
    border-radius: 999px; font-size: .85rem; color: var(--ink); text-decoration: none; background: var(--surface); }}
  .mi-chip:hover {{ border-color: var(--accent); color: var(--accent); }}
  .mi-part {{ margin-top: 2.4rem; font-size: 1.15rem; color: var(--muted); text-transform: uppercase; letter-spacing: .06em; }}
  .mi-cat {{ margin: 1.6rem 0 2rem; scroll-margin-top: 4.5rem; }}
  .mi-cat h3 {{ display: flex; align-items: center; gap: .55rem; margin: 0 0 .2rem; font-size: 1.2rem; }}
  .mi-cat h3 .cat-dot {{ width: .75rem; height: .75rem; }}
  .mi-catdesc {{ margin: 0 0 .7rem; color: var(--muted); font-size: .9rem; }}
  .mi-table {{ width: 100%; table-layout: fixed; border-collapse: collapse; background: var(--surface); border: 1px solid var(--line);
    border-radius: 10px; overflow: hidden; font-size: .9rem; }}
  #content .mi-table td, #content .mi-table tr {{ padding: .55rem .8rem; border: none; border-top: 1px solid var(--line-soft); vertical-align: top; }}
  #content .mi-table tr:first-child td {{ border-top: none; }}
  .mi-name {{ font-weight: 600; width: 13rem; }}
  .mi-desc {{ color: var(--ink-soft); }}
  .mi-links {{ text-align: right; width: 12.5rem; }}
  .mi-cls {{ display: inline-block; margin: 0 0 .2rem .3rem; padding: .05rem .55rem; border: 1px solid var(--line);
    border-radius: 6px; font-size: .78rem; text-decoration: none; background: var(--surface-alt); }}
  .mi-cls:hover {{ border-color: var(--accent); }}
  .mi-dep {{ text-decoration: line-through; color: var(--muted); }}
  .mi-post {{ white-space: nowrap; font-size: .82rem; margin-left: .3rem; }}
  .mi-badge {{ display: inline-block; margin-left: .4rem; font-size: .62rem; font-weight: 500; letter-spacing: .05em;
    text-transform: uppercase; border-radius: 999px; padding: .02em .5em; vertical-align: .15em; border: 1px solid var(--line); color: var(--muted); }}
  .mi-autoresearch {{ color: var(--research); background: var(--research-bg); border-color: var(--research-line); }}
  @media (max-width: 640px) {{
    #content .mi-table tr {{ display: block; padding: .5rem .2rem; border-top: 1px solid var(--line-soft); }}
    #content .mi-table td {{ display: block; border: none; padding: .15rem .6rem; width: auto; text-align: left; }}
    .mi-links {{ white-space: normal; }}
    .mi-cls {{ margin: .15rem .3rem 0 0; }}
  }}
</style>
{chr(10).join(sections)}
</div>
</section>
'''


if __name__ == "__main__":
    out = os.path.join(HERE, "models.html")
    with open(out, "w") as f:
        f.write(render())
    print("  wrote pages/models.html")
