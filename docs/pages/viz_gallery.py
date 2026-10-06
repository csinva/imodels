"""Build the imodels.viz post (docs/viz.html): its figures, interactive pages and article body.

Each example's code string is executed verbatim, so the code shown under a figure is exactly what
produced it. Figures go to docs/viz_gallery/{static,interactive}; the article body goes to
docs/pages/viz.html, which build_pages.py wraps in the site shell. Run from docs/:

    uv run python pages/viz_gallery.py               # figures, interactive pages and the article
    uv run python pages/viz_gallery.py --post-only   # only the article, from the existing figures
    uv run python build_pages.py
"""

import html
import re
import os
import sys
import textwrap
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))  # docs/pages
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
OUT = os.path.join(os.path.dirname(HERE), "viz_gallery")

SETUP = """
import numpy as np, pandas as pd
import imodels.viz as viz
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
import imodels
cancer = datasets.load_breast_cancer(as_frame=True)   # target 0 = malignant, 1 = benign
diabetes = datasets.load_diabetes(as_frame=True)
"""


def loan_data(n=1200, seed=0):
    """Synthetic credit data with binary, integer and continuous features."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "credit_score": rng.integers(480, 851, n),
        "late_payments": rng.poisson(1.2, n),
        "income_k": np.round(rng.lognormal(4.1, 0.45, n), 1),
        "debt_to_income": np.round(rng.beta(2, 5, n), 3),
        "employed": rng.binomial(1, 0.85, n),
        "has_mortgage": rng.binomial(1, 0.4, n),
    })
    logit = (0.018 * (X.credit_score - 650) - 0.7 * X.late_payments + 0.02 * (X.income_k - 60)
             - 6 * (X.debt_to_income - 0.3) + 1.6 * X.employed + 0.4 * X.has_mortgage)
    y = pd.Series(rng.random(n) < 1 / (1 + np.exp(-logit)), name="approved").astype(int)
    return X, y


EXAMPLES = [
    dict(slug="iris", title="Classification with training data",
         note="Pass X and y and every split shows a class-stacked histogram of its feature, with the "
              "threshold marked on the axis. Each ribbon is split by class, with widths proportional to the "
              "samples of each class flowing down that branch.",
         code="""
d = datasets.load_iris(as_frame=True)
X, y = d.data, d.target
clf = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X, y)
fig = viz.draw(clf, X, y, class_names=d.target_names, title="Iris species")
""", interactive="viz.interactive(clf, X, y, class_names=d.target_names, title='Iris species')"),
    dict(slug="iris_simple", title="Simple mode",
         note="simple=True replaces each card with a box filled by the class proportions entering it. "
              "Ribbons split by class, so the flow shows which classes take each branch.",
         code="""
d = datasets.load_iris(as_frame=True)
X, y = d.data, d.target
clf = DecisionTreeClassifier(max_depth=4, random_state=0).fit(X, y)
fig = viz.draw(clf, X, y, class_names=d.target_names, simple=True, title="Iris species")
""", interactive="viz.interactive(clf, X, y, class_names=d.target_names, simple=True, title='Iris species')"),
    dict(slug="california_big", crop=1250, title="A bigger tree",
         note="A depth-8 tree with about 300 nodes on all 20,640 California districts. The static view shows "
              "the top three levels. The interactive page holds the full tree and opens collapsed in simple mode "
              "(the Simple button switches to charts); its feature panel shows where each feature is used.",
         code="""
d = datasets.fetch_california_housing(as_frame=True)
X = d.data
y = pd.qcut(d.target, 3, labels=False)   # value tier: low / mid / high
clf = DecisionTreeClassifier(max_depth=8, min_samples_leaf=50, random_state=0).fit(X, y)
fig = viz.draw(clf, X, y, class_names=["low", "mid", "high"], max_depth=3,
               title="California house value tier")
""", interactive="viz.interactive(clf, X, y, class_names=['low', 'mid', 'high'], title='California house value tier')"),
    dict(slug="loan", title="Binary and integer features",
         note="Thresholds on 0/1 features read as “= 0 / = 1” and thresholds on integer "
              "features as “≤ k / ≥ k+1”, inferred from the data. Synthetic credit data.",
         code="""
X, y = loan_data()   # synthetic credit data, see docs/pages/viz_gallery.py
clf = DecisionTreeClassifier(max_depth=3, min_samples_leaf=20, random_state=0).fit(X, y)
fig = viz.draw(clf, X, y, class_names={0: "denied", 1: "approved"}, title="Loan approval")
""", interactive="viz.interactive(clf, X, y, class_names={0: 'denied', 1: 'approved'}, title='Loan approval')"),
    dict(slug="diabetes_path", title="Regression and one sample's decision path",
         note="Regression splits show a feature/target scatter with the mean of each side drawn as a "
              "line. Passing x highlights one sample's path and marks its value in each chart.",
         code="""
d = datasets.load_diabetes(as_frame=True)
X, y = d.data, d.target
reg = DecisionTreeRegressor(max_depth=3, random_state=0).fit(X, y)
fig = viz.draw(reg, X, y, x=X.iloc[7], target_name="progression", title="Diabetes progression")
""", interactive="viz.interactive(reg, X, y, target_name='progression', title='Diabetes progression')"),
    dict(slug="iris_dark", title="Dark theme",
         note="Both themes use their own validated color steps rather than an inverted palette.",
         code="""
d = datasets.load_iris(as_frame=True)
clf = DecisionTreeClassifier(max_depth=2, random_state=0).fit(d.data, d.target)
fig = viz.draw(clf, d.data, d.target, class_names=d.target_names, theme="dark",
               x=d.data.iloc[120], title="Iris species")
"""),
    dict(slug="california_lr", focus=(0, 0.45), title="Left-to-right layout",
         note="orientation=\"LR\" suits deep, narrow trees and wide screens. Leaves show the distribution "
              "of the target in that leaf against the full target range.",
         code="""
d = datasets.fetch_california_housing(as_frame=True)
X, y = d.data.iloc[:4000], d.target.iloc[:4000]
reg = DecisionTreeRegressor(max_depth=3, min_samples_leaf=50, random_state=0).fit(X, y)
fig = viz.draw(reg, X, y, orientation="LR", target_name="value ($100k)", title="California house values")
"""),
    dict(slug="wine_nodata", title="Model only, no data",
         note="Without data the cards show the class mix from the fitted tree itself. Useful when the "
              "training set is not at hand.",
         code="""
d = datasets.load_wine()
clf = DecisionTreeClassifier(max_depth=3, random_state=0).fit(d.data, d.target)
fig = viz.draw(clf, feature_names=d.feature_names, class_names=["barolo", "grignolino", "barbera"],
               title="Wine cultivar")
"""),
    dict(slug="cancer_truncated", title="Deep tree, truncated view",
         note="max_depth limits what is drawn. Subtrees below the cut become stacked cards that count "
              "their hidden nodes. The interactive page holds the whole tree.",
         code="""
d = datasets.load_breast_cancer(as_frame=True)
X, y = d.data, d.target
clf = DecisionTreeClassifier(random_state=0).fit(X, y)   # unrestricted depth
fig = viz.draw(clf, X, y, class_names=d.target_names, max_depth=2, title="Breast cancer diagnosis")
""", interactive="viz.interactive(clf, X, y, class_names=d.target_names, title='Breast cancer diagnosis (full tree)')"),
    dict(slug="digits_compact", crop=1300, focus=(0.5, 0), title="Many classes, compact style",
         note="Trees with more than 24 leaves switch to compact cards automatically. Ten classes use the "
              "eight validated hues plus four extras, so the legend carries identity.",
         code="""
d = datasets.load_digits()
clf = DecisionTreeClassifier(max_depth=5, random_state=0).fit(d.data, d.target)
fig = viz.draw(clf, feature_names=[f"px{i}" for i in range(64)], title="Handwritten digits")
""", interactive="viz.interactive(clf, d.data, d.target, feature_names=[f'px{i}' for i in range(64)], title='Handwritten digits')"),
    dict(slug="forest_member", title="One tree from a random forest",
         note="Any single tree of an ensemble works: forest.estimators_[i] or gbm.estimators_[i, 0].",
         code="""
d = datasets.load_breast_cancer(as_frame=True)
X, y = d.data, d.target
rf = RandomForestClassifier(n_estimators=50, max_depth=4, random_state=0).fit(X, y)
fig = viz.draw(rf.estimators_[0], X, y, feature_names=list(X.columns),
               class_names=d.target_names, max_depth=3, title="Random forest, tree 0")
"""),
    dict(slug="gbm_stage", title="A gradient boosting stage",
         note="Boosting trees fit residuals, so pass the residual target to see what the stage learned.",
         code="""
d = datasets.load_diabetes(as_frame=True)
X, y = d.data, d.target
gbm = GradientBoostingRegressor(n_estimators=20, max_depth=2, random_state=0).fit(X, y)
resid = y - y.mean()   # what the first stage is fit to
fig = viz.draw(gbm.estimators_[0, 0], X, resid, feature_names=list(X.columns),
               target_name="residual", title="Gradient boosting, stage 1")
"""),
]


def loans_with_categories(n=1500, seed=0):
    """Synthetic credit data with a categorical column and missing incomes (for FastRiskScore)."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "credit_score": rng.integers(480, 851, n).astype(float),
        "late_payments": rng.poisson(1.2, n).astype(float),
        "income_k": np.round(rng.lognormal(4.1, 0.45, n), 1),
        "housing": rng.choice(["own", "rent", "mortgage", "other"], n, p=[.3, .35, .3, .05]),
        "employed": rng.binomial(1, 0.85, n),
    })
    X.loc[rng.random(n) < 0.12, "income_k"] = np.nan
    logit = (0.02 * (X.credit_score - 650) - 0.8 * X.late_payments + 0.02 * (X.income_k.fillna(30) - 60)
             + X.housing.map({"own": 1.0, "mortgage": 0.6, "rent": -0.4, "other": -1.2}) + 1.5 * X.employed
             - 1.0 * X.income_k.isna())
    y = pd.Series((rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int), name="approved")
    return X, y


IMODELS = [
    dict(group="skmore", slug="sk_forest", crop=860, focus=(0, 0), title="Random forest",
         note="Forests draw their first few trees in a grid; predictions average every tree. In interactive "
              "mode Predict lists each tree's vote.",
         code="""
from sklearn.ensemble import RandomForestClassifier
X, y = cancer.data, cancer.target
model = RandomForestClassifier(n_estimators=50, max_depth=3, random_state=0).fit(X, y)
kw = dict(class_names=["malignant", "benign"], title="Random forest, breast cancer", max_trees=3)
fig = viz.draw(model, X, y, **kw)
"""),
    dict(group="skmore", slug="sk_gbm", crop=760, focus=(0, 0), title="Gradient boosting",
         note="Boosted trees add up: each leaf shows its contribution and the waterfall in Predict sums them.",
         code="""
X, y = diabetes.data, diabetes.target
model = GradientBoostingRegressor(n_estimators=60, max_depth=2, random_state=0).fit(X, y)
kw = dict(title="Gradient boosting, diabetes progression", max_trees=3)
fig = viz.draw(model, X, y, **kw)
"""),
    dict(group="skmore", slug="sk_hgb", crop=780, focus=(0, 0), title="Histogram gradient boosting",
         note="HistGradientBoosting trees, including where each split sends missing values.",
         code="""
from sklearn.ensemble import HistGradientBoostingClassifier
X, y = cancer.data, cancer.target
model = HistGradientBoostingClassifier(max_iter=40, max_leaf_nodes=5, random_state=0).fit(X, y)
kw = dict(class_names=["malignant", "benign"], title="HistGradientBoosting, breast cancer", max_trees=3)
fig = viz.draw(model, X, y, **kw)
"""),
    dict(group="skmore", slug="sk_logreg", title="Logistic regression in a pipeline",
         note="The pipeline's scaler is folded in, so coefficients read per raw unit. Bars show each feature's "
              "typical effect on the log-odds.",
         code="""
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
X, y = cancer.data, cancer.target
model = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=2000)).fit(X, y)
kw = dict(class_names=["malignant", "benign"], title="Logistic regression, breast cancer")
fig = viz.draw(model, X, y, **kw)
"""),
    dict(group="skmore", slug="sk_isotonic", title="Isotonic regression",
         note="A monotone fit of one feature, drawn as a shape function over the data's distribution.",
         code="""
from sklearn.isotonic import IsotonicRegression
X, y = diabetes.data[["bmi"]], diabetes.target
model = IsotonicRegression(out_of_bounds="clip").fit(X["bmi"], y)
kw = dict(title="Isotonic regression, diabetes progression by bmi")
fig = viz.draw(model, X, y, **kw)
"""),
    dict(group="trees", slug="im_hstree", title="Hierarchical shrinkage (HSTree)",
         note="HSTree shrinks each node's prediction toward its ancestors. The cards show the shrunk values the "
              "model predicts with; histograms still come from the training data.",
         code="""
X, y = cancer.data, cancer.target
model = imodels.HSTreeClassifier(max_leaf_nodes=8, reg_param=10).fit(X, y)
kw = dict(class_names=["malignant", "benign"], title="HSTree, breast cancer")
fig = viz.draw(model, X, y, **kw)
"""),
    dict(group="trees", slug="im_figs", focus=(0, 0), title="FIGS: a sum of trees",
         note="FIGS grows several small trees whose leaf values add up. Trees sit side by side with a + between "
              "them; Predict shows each tree's contribution.",
         code="""
X, y = diabetes.data, diabetes.target
model = imodels.FIGSRegressor(max_rules=10).fit(X, y)
kw = dict(title="FIGS, diabetes progression")
fig = viz.draw(model, X, y, **kw)
"""),
    dict(group="trees", slug="im_c45", title="C4.5 tree",
         note="C4.5 splits are strict (x < t). Leaves store only a label, so their class mix is counted from the data.",
         code="""
X, y = cancer.data, cancer.target
model = imodels.C45TreeClassifier(max_rules=6).fit(X, y)
kw = dict(class_names=["malignant", "benign"], title="C4.5, breast cancer")
fig = viz.draw(model, X, y, **kw)
"""),
    dict(group="trees", slug="im_irf", crop=930, focus=(0, 0), title="Iterative random forest (IRF)",
         note="IRF reweights features over several forests; its final forest averages its trees like any forest.",
         code="""
X, y = cancer.data, cancer.target
model = imodels.IRFClassifier(n_estimators=20, max_depth=3, random_state=0).fit(X, y)
kw = dict(class_names=["malignant", "benign"], title="IRF, breast cancer", max_trees=3)
fig = viz.draw(model, X, y, **kw)
"""),
    dict(group="lists", slug="im_greedyrl", title="Greedy rule list",
         note="Rule lists read top to bottom: the first rule that holds decides. Each rule's outcome sits to its "
              "right and the flow narrows as rules capture samples.",
         code="""
X, y = cancer.data, cancer.target
model = imodels.GreedyRuleListClassifier(max_depth=4).fit(X, y)
kw = dict(class_names=["malignant", "benign"], title="Greedy rule list, breast cancer")
fig = viz.draw(model, X, y, **kw)
"""),
    dict(group="lists", slug="im_brl", title="Bayesian rule list",
         note="Rules with several conditions list them inside the card and branch yes / no. BRL needs 0/1 "
              "features, so this example uses median splits of eight measurements.",
         code="""
X = (cancer.data.iloc[:, :8] > cancer.data.iloc[:, :8].median()).astype(int)
X.columns = [c.replace(" ", "_") + "_high" for c in X.columns]
y = cancer.target
model = imodels.BayesianRuleListClassifier(max_iter=2000, n_chains=2, random_state=0).fit(X, y)
kw = dict(class_names=["malignant", "benign"], title="Bayesian rule list")
fig = viz.draw(model, X, y, **kw)
"""),
    dict(group="sets", slug="im_rulefit", title="RuleFit",
         note="Rule sets add up: each row is a rule (or linear term) with its effect, the share of samples it "
              "covers and their classes. Rows are ranked by how much they move predictions.",
         code="""
X, y = cancer.data, cancer.target
model = imodels.RuleFitClassifier(max_rules=12, random_state=0).fit(X, y)
kw = dict(class_names=["malignant", "benign"], title="RuleFit, breast cancer")
fig = viz.draw(model, X, y, **kw)
"""),
    dict(group="sets", slug="im_skope", title="Skope rules",
         note="Skope scores a sample by the precision-weighted share of its rules that hold.",
         code="""
X, y = cancer.data, cancer.target
model = imodels.SkopeRulesClassifier(n_estimators=5, max_depth=2, random_state=0).fit(X, y)
kw = dict(class_names=["malignant", "benign"], title="Skope rules, breast cancer")
fig = viz.draw(model, X, y, **kw)
"""),
    dict(group="sets", slug="im_boosted", title="Boosted rules",
         note="Boosted stumps become one rule each plus a share of the baseline, so the table adds up to the "
              "model's decision function exactly.",
         code="""
X, y = cancer.data, cancer.target
model = imodels.BoostedRulesClassifier(n_estimators=8, random_state=0).fit(X, y)
kw = dict(class_names=["malignant", "benign"], title="Boosted rules, breast cancer")
fig = viz.draw(model, X, y, **kw)
"""),
    dict(group="scores", slug="im_riskscore", title="FastRiskScore",
         note="A points table next to the risk each total score maps to, with the training samples at each score "
              "underneath. In interactive mode, click an item to switch it on or off and watch the score move along the curve.",
         code="""
X, y = cancer.data, cancer.target
model = imodels.FastRiskScoreClassifier(k=5, time_limit=20).fit(X, y)
kw = dict(class_names=["malignant", "benign"], title="FastRiskScore, breast cancer")
fig = viz.draw(model, X, y, **kw)
"""),
    dict(group="scores", slug="im_riskscore_loans", title="FastRiskScore with categories and missing values",
         note="FastRiskScore turns string columns into level indicators and gives missing values their own item. "
              "Predict offers a dropdown per categorical feature and a missing switch.",
         code="""
X, y = loans_with_categories()   # credit data with a 'housing' column and missing incomes
model = imodels.FastRiskScoreClassifier(k=6, time_limit=30).fit(X, y)
kw = dict(class_names=["denied", "approved"], title="FastRiskScore, loan approval")
fig = viz.draw(model, X, y, **kw)
"""),
    dict(group="scores", slug="im_slim", title="SLIM",
         note="Sparse integer points on 0/1 features, drawn as a scorecard. The intercept moves into the risk "
              "formula so the score is a plain sum of points.",
         code="""
X = (cancer.data.iloc[:, :10] > cancer.data.iloc[:, :10].median()).astype(int)
X.columns = [c.replace(" ", "_") + "_high" for c in X.columns]
y = cancer.target
model = imodels.SLIMClassifier(alpha=0.5).fit(X, y)
kw = dict(class_names=["malignant", "benign"], title="SLIM, breast cancer")
fig = viz.draw(model, X, y, **kw)
"""),
    dict(group="additive", slug="im_treegam", title="TreeGAM",
         note="One shape function per feature on a shared scale, with the data's distribution underneath. "
              "Panels are ordered by how much each feature moves predictions.",
         code="""
X, y = diabetes.data, diabetes.target
model = imodels.TreeGAMRegressor(n_boosting_rounds=20).fit(X, y)
kw = dict(title="TreeGAM, diabetes progression")
fig = viz.draw(model, X, y, **kw)
"""),
    dict(group="additive", slug="im_treegam_clf", title="TreeGAM classifier",
         note="For classifiers the shape functions add up to a probability that is clipped to [0, 1].",
         code="""
X, y = cancer.data, cancer.target
model = imodels.TreeGAMClassifier(n_boosting_rounds=20).fit(X, y)
kw = dict(class_names=["malignant", "benign"], title="TreeGAM, breast cancer")
fig = viz.draw(model, X, y, **kw)
"""),
    dict(group="additive", slug="im_linear", title="Linear model with marginal shrinkage",
         note="Linear terms are drawn by their typical effect (coefficient times the feature's spread); "
              "labels give the raw coefficient per unit.",
         code="""
X, y = diabetes.data, diabetes.target
model = imodels.MarginalShrinkageLinearModelRegressor().fit(X, y)
kw = dict(title="Marginal shrinkage linear model, diabetes")
fig = viz.draw(model, X, y, **kw)
"""),
]
for ex in IMODELS:
    ex["interactive"] = "viz.interactive(model, X, y, **kw)"
    ex["code"] = ex["code"].rstrip() + "\npage = viz.interactive(model, X, y, **kw)\n"

GROUPS = [
    ("skmore", "More scikit-learn models", "Forests, boosting, linear models and isotonic regression."),
    ("trees", "Trees and sums of trees", "imodels trees reuse the decision-tree views; FIGS adds trees side by side."),
    ("lists", "Rule lists", "An ordered list where the first rule that holds decides."),
    ("sets", "Rule sets", "Weighted rules (and linear terms) that add up to a score."),
    ("scores", "Scoring systems", "Integer points and the risk each total maps to."),
    ("additive", "Additive and linear models", "One shape function or coefficient per feature."),
]

# every supported model, by the view it gets: (view, what it shows, [(model, source)])
SK, IM = "scikit-learn", "imodels"
SUPPORTED = [
    ("Decision trees", "Data-aware cards, class-split flows, fold and unfold", [
        ("DecisionTreeClassifier", SK), ("DecisionTreeRegressor", SK), ("ExtraTreeClassifier", SK),
        ("ExtraTreeRegressor", SK), ("GreedyTree", IM), ("DecisionTreeCCP", IM), ("HSTree", IM),
        ("TaoTree", IM), ("C45Tree", IM), ("FastSmallTree", IM)]),
    ("Tree ensembles", "A grid of trees that average or add up", [
        ("RandomForest", SK), ("ExtraTrees", SK), ("GradientBoosting", SK), ("HistGradientBoosting", SK),
        ("FIGS", IM), ("IRF", IM)]),
    ("Rule lists", "Rules in order, each outcome beside its rule", [
        ("GreedyRuleList", IM), ("OneR", IM), ("FastFrugalTree", IM), ("BayesianRuleList", IM)]),
    ("Rule sets", "Ranked rules with effect, coverage and classes", [
        ("RuleFit", IM), ("FPLasso", IM), ("SkopeRules", IM), ("FPSkope", IM), ("BoostedRules", IM),
        ("Slipper", IM), ("BayesianRuleSet", IM)]),
    ("Scoring systems", "Points table beside the risk for every score", [
        ("FastRiskScore", IM), ("SLIM", IM)]),
    ("Additive and linear models", "Shape functions or coefficient rows", [
        ("LinearRegression", SK), ("Ridge / Lasso / ElasticNet", SK), ("LogisticRegression", SK),
        ("SGD", SK), ("LinearSVC / LinearSVR", SK), ("Huber / RANSAC", SK), ("Poisson / Gamma / Tweedie", SK),
        ("RidgeClassifier / Perceptron", SK), ("IsotonicRegression", SK), ("TreeGAM", IM), ("GPGam", IM),
        ("MarginalShrinkageLinear", IM)]),
]
SUPPORT_NOTE = ("Pipelines whose earlier steps only scale features (Standard, MinMax, MaxAbs, Robust) are drawn in raw "
                "units. AutoInterpretable draws the model it selected. Not supported: kNN, kernel SVMs, MLPs and "
                "other models without readable structure; multiclass linear models, boosting and FIGS.")


def run_examples():
    os.makedirs(os.path.join(OUT, "static"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "interactive"), exist_ok=True)
    ns = {"loan_data": loan_data, "loans_with_categories": loans_with_categories}
    exec(SETUP, ns)
    for ex in EXAMPLES + IMODELS:
        t = time.time()
        exec(textwrap.dedent(ex["code"]), ns)
        ns["fig"].save(os.path.join(OUT, "static", ex["slug"] + ".svg"))
        if ex.get("interactive"):
            eval(ex["interactive"], ns).save(os.path.join(OUT, "interactive", ex["slug"] + ".html"))
        print(f"{ex['slug']:<18} {time.time() - t:5.2f}s")
    # sklearn's own plot_tree for comparison
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn import datasets
    from sklearn.tree import DecisionTreeClassifier, plot_tree

    d = datasets.load_iris(as_frame=True)
    clf = DecisionTreeClassifier(max_depth=3, random_state=0).fit(d.data, d.target)
    fig, ax = plt.subplots(figsize=(11, 6))
    plot_tree(clf, feature_names=list(d.data.columns), class_names=list(d.target_names), filled=True, ax=ax)
    fig.savefig(os.path.join(OUT, "static", "sklearn_plot_tree.svg"), bbox_inches="tight")
    plt.close(fig)


POST = """<section id="section-intro">

                <div class="article" style="padding-right: 2%; padding-left: 2%;">
                    <h1 style="padding-bottom: 0px;">imodels.viz: figures you can read, pages you can explore</h1>
                    <h3 class="post-authors" style="color:gray;padding-top:0px;">Chandan Singh &middot; October 2026</h3>
                    <p class="post-links"><a href="viz/index.html">🗂 Doc</a>, <a href="https://github.com/csinva/imodels/tree/master/imodels/viz">💻 Code</a></p>
                    <hr>

                    <p class="abstract">An interpretable model is only as useful as the picture you can make of it.
                        <code>imodels.viz</code> draws a fitted model two ways: as a static figure (SVG, PNG, PDF) for a
                        paper or slide, and as a single offline HTML page where you can fold the model, run a sample
                        through it and ask what would change its prediction. It covers the models in <code>imodels</code>
                        (trees, sums of trees, rule lists, rule sets, scoring systems and additive models) and
                        scikit-learn's trees, forests, boosting and linear models, __NMODELS__+ model classes in all.</p>

                    <nav class="toc-main">
                      <a href="#quickstart"><span>1</span> Quickstart</a>
                      <a href="#supported"><span>2</span> Supported models</a>
                      <a href="#interactive"><span>3</span> Interactive mode</a>
                      <a href="#gallery"><span>4</span> Gallery</a>
                      <a href="#export"><span>5</span> Every tree is a scikit-learn tree</a>
                      <a href="#compare"><span>6</span> Compared with plot_tree</a>
                    </nav>

                    <style>
                      .vz-filters { display: flex; flex-wrap: wrap; gap: 0.45rem; margin: 1.2rem 0 0.2rem; }
                      .vz-pill { font: inherit; font-size: 0.84rem; color: var(--ink-soft); background: var(--surface);
                        border: 1px solid var(--line); border-radius: 999px; padding: 0.3rem 0.8rem; cursor: pointer; }
                      .vz-pill span { color: var(--muted); margin-left: 0.3rem; font-size: 0.78rem; }
                      .vz-pill:hover { border-color: var(--accent); color: var(--accent); }
                      .vz-pill[aria-pressed="true"] { background: var(--ink); border-color: var(--ink); color: #fff; }
                      .vz-pill[aria-pressed="true"] span { color: rgba(255,255,255,.7); }
                      .vz-filter-note { min-height: 1.4em; margin: 0.5rem 0 0; font-size: 0.88rem; color: var(--muted); }
                      .vz-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(15.5rem, 1fr)); gap: 1.1rem; margin: 0.6rem 0 2.4rem; }
                      .vz-card { display: flex; flex-direction: column; margin: 0; background: var(--surface); border: 1px solid var(--line);
                        border-radius: 12px; overflow: hidden; box-shadow: 0 1px 2px rgba(27,31,35,.04);
                        transition: transform .15s ease, box-shadow .15s ease, border-color .15s ease; min-width: 0; }
                      .vz-card[hidden] { display: none; }
                      .vz-card:hover { transform: translateY(-2px); border-color: #cfd6db; box-shadow: 0 2px 4px rgba(27,31,35,.06), 0 12px 28px rgba(27,31,35,.10); }
                      .vz-thumb { position: relative; display: block; aspect-ratio: 4 / 3; overflow: hidden; background: #ffffff;
                        border-bottom: 1px solid var(--line-soft); cursor: zoom-in; }
                      .vz-card.dark .vz-thumb { background: #1a1a19; }
                      .vz-thumb img { position: absolute; max-width: none; height: auto; transition: transform .25s ease; transform-origin: 50% 0; }
                      .vz-card:hover .vz-thumb img { transform: scale(1.04); }
                      .vz-body { padding: 0.75rem 0.9rem 0.85rem; display: flex; flex-direction: column; gap: 0.3rem; flex: 1; min-width: 0; }
                      .vz-top { display: flex; justify-content: space-between; align-items: baseline; gap: 0.5rem; }
                      .vz-card h3 { font-size: 0.95rem; margin: 0; line-height: 1.3; border: none; padding: 0; }
                      .vz-num { white-space: nowrap; flex-shrink: 0; font: 500 0.7rem ui-monospace, SFMono-Regular, Menlo, monospace; color: var(--muted); }
                      .vz-meta { display: flex; flex-wrap: wrap; align-items: center; gap: 0.4rem; margin-top: auto; padding-top: 0.2rem; }
                      .vz-model { font: 0.72rem ui-monospace, SFMono-Regular, Menlo, monospace; color: var(--ink-soft); background: var(--code-bg);
                        border-radius: 999px; padding: 0.15rem 0.55rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 100%; }
                      .vz-live { display: inline-flex; align-items: center; gap: 0.3rem; font-size: 0.7rem; letter-spacing: .05em; text-transform: uppercase;
                        color: #2f7d5b; border: 1px solid #b9dccb; background: #f0f8f4; border-radius: 999px; padding: 0.05rem 0.5rem; }
                      .vz-live:before { content: ""; width: 0.4rem; height: 0.4rem; border-radius: 50%; background: #2f8f62; }
                      .vz-demo { border: 1px solid var(--line); border-radius: 12px; overflow: hidden; margin: 1rem 0 0.4rem; background: var(--surface); }
                      .vz-demo iframe { display: block; width: 100%; height: min(78vh, 720px); border: 0; }
                      .vz-tabs { display: flex; flex-wrap: wrap; gap: 0.2rem; padding: 0.45rem 0.5rem 0; border-bottom: 1px solid var(--line); background: var(--surface-alt); }
                      .vz-tab { font: inherit; font-size: 0.86rem; color: var(--muted); background: none; border: 1px solid transparent;
                        border-bottom: none; border-radius: 8px 8px 0 0; padding: 0.4rem 0.85rem; cursor: pointer; margin-bottom: -1px; }
                      .vz-tab:hover { color: var(--ink); }
                      .vz-tab[aria-selected="true"] { color: var(--ink); background: var(--surface); border-color: var(--line); font-weight: 600; }
                      .vz-lb { width: min(1180px, 94vw); max-height: 92vh; padding: 0; border: 1px solid var(--line); border-radius: 14px;
                        box-shadow: 0 24px 60px rgba(0,0,0,.25); color: var(--ink); background: var(--surface); }
                      .vz-lb[open] { display: flex; flex-direction: column; }
                      .vz-lb::backdrop { background: rgba(20,24,28,.55); backdrop-filter: blur(2px); }
                      .vz-lb-head { display: flex; justify-content: space-between; align-items: center; gap: 1rem; padding: 0.8rem 1.1rem;
                        border-bottom: 1px solid var(--line); }
                      .vz-lb-head h3 { margin: 0 0.6rem 0 0; font-size: 1.05rem; border: none; padding: 0; display: inline; }
                      .vz-lb-model { font: 0.75rem ui-monospace, SFMono-Regular, Menlo, monospace; color: var(--muted); }
                      .vz-lb-nav { display: flex; gap: 0.35rem; flex-shrink: 0; }
                      .vz-lb-nav button { font: inherit; width: 2.1rem; height: 2.1rem; border-radius: 8px; border: 1px solid var(--line);
                        background: var(--surface); color: var(--ink-soft); cursor: pointer; }
                      .vz-lb-nav button:hover { border-color: var(--accent); color: var(--accent); }
                      .vz-lb-fig { overflow: auto; background: #ffffff; padding: 1rem; text-align: center; cursor: zoom-in; flex: 1 1 auto; min-height: 12rem; }
                      .vz-lb-fig.dark { background: #1a1a19; }
                      .vz-lb-fig img { max-width: 100%; max-height: 62vh; }
                      .vz-lb-fig.zoomed { cursor: zoom-out; text-align: left; }
                      .vz-lb-fig.zoomed img { max-width: none; max-height: none; }
                      .vz-lb-info { padding: 0.8rem 1.1rem 1rem; border-top: 1px solid var(--line); overflow: auto; flex: 0 0 auto; max-height: 34vh; }
                      .vz-lb-info p { margin: 0 0 0.6rem; color: var(--ink-soft); font-size: 0.92rem; }
                      .vz-lb-links { display: flex; flex-wrap: wrap; align-items: center; gap: 1rem; margin-bottom: 0.7rem; font-size: 0.88rem; }
                      .vz-btn { background: var(--accent); color: #fff !important; padding: 0.35rem 0.8rem; border-radius: 8px; text-decoration: none !important; }
                      .vz-btn:hover { background: var(--accent-dark); }
                      .vz-lb-info pre { margin: 0; font-size: 0.78rem; }
                      .vz-models { display: grid; grid-template-columns: repeat(auto-fill, minmax(17rem, 1fr)); gap: 1rem; margin: 1rem 0 0.6rem; }
                      .vz-mcard { border: 1px solid var(--line); border-radius: 12px; padding: 0.85rem 1rem 1rem; background: var(--surface); }
                      .vz-mcard h3 { font-size: 0.98rem; margin: 0 0 0.15rem; border: none; padding: 0; display: flex; justify-content: space-between; }
                      .vz-mcard h3 span { font-weight: 400; color: var(--muted); font-size: 0.8rem; }
                      .vz-mcard p { margin: 0 0 0.6rem; font-size: 0.84rem; color: var(--ink-soft); }
                      .vz-chips { display: flex; flex-wrap: wrap; gap: 0.35rem; }
                      .vz-chip { display: inline-flex; align-items: center; gap: 0.35rem; font: 0.74rem ui-monospace, SFMono-Regular, Menlo, monospace;
                        background: var(--code-bg); border-radius: 999px; padding: 0.22rem 0.6rem 0.22rem 0.45rem; white-space: nowrap; }
                      .vz-chip:before, .vz-key i { content: ""; display: inline-block; width: 0.45rem; height: 0.45rem; border-radius: 50%; }
                      .vz-chip.sk:before, .vz-key i.sk { background: var(--accent); }
                      .vz-chip.im:before, .vz-key i.im { background: var(--cat-rule-set); }
                      .vz-key { color: var(--muted); font-size: 0.85rem; }
                      .vz-key i { margin: 0 0.3rem 0 0.7rem; }
                      .vz-cmp { margin: 0; border: 1px solid var(--line); border-radius: 12px; overflow: hidden; background: #ffffff; display: flex; flex-direction: column; }
                      .vz-cmp a { display: flex; align-items: center; justify-content: center; padding: 0.8rem; flex: 1; }
                      .vz-cmp img { max-width: 100%; max-height: 24rem; }
                      .vz-cmp figcaption { padding: 0.6rem 0.9rem; border-top: 1px solid var(--line-soft); background: var(--surface);
                        font: 600 0.9rem ui-monospace, SFMono-Regular, Menlo, monospace; }
                      .vz-compare { display: grid; grid-template-columns: repeat(auto-fit, minmax(20rem, 1fr)); gap: 1.4rem; margin-top: 1rem; }
                    </style>

                    <h2 id="quickstart">1. Quickstart</h2>

                    <pre><code class="language-python">from imodels import FIGSClassifier
from imodels import viz
from sklearn.datasets import load_breast_cancer

X, y = load_breast_cancer(return_X_y=True, as_frame=True)
model = FIGSClassifier(max_rules=8).fit(X, y)

viz.draw(model, X, y).save("figs.svg")             # static: .svg .png .pdf .html
viz.interactive(model, X, y).save("figs.html")     # one offline page; also renders inline in Jupyter</code></pre>

                    <p>Both calls take the fitted model and, optionally, the training data. With data, every split shows
                        the distribution of its feature and every leaf its class mix or target range; without it, the
                        figure falls back to what the model stores. Other arguments set names
                        (<code>feature_names</code>, <code>class_names</code>, <code>target_name</code>), highlight one
                        sample's path (<code>x=</code>), limit the depth drawn (<code>max_depth</code>), and switch
                        theme, orientation and style. See the <a href="viz/index.html">API docs</a>.</p>

                    <h2 id="supported">2. Supported models</h2>

                    <p>Models are grouped by the view they get.
                        <span class="vz-key"><i class="sk"></i>scikit-learn<i class="im"></i>imodels</span></p>
                    <div class="vz-models">__SUPPORTED__</div>
                    <p class="caption" style="text-align:left">__SUPPORT_NOTE__</p>

                    <h2 id="interactive">3. Interactive mode</h2>

                    <p><code>viz.interactive</code> writes one self-contained HTML file with no server and no network
                        requests. Across all models the page offers the same tools:</p>
                    <ul>
                      <li><b>Try a sample.</b> Type feature values or load a random training row. The path it takes
                        lights up, and a waterfall shows how the prediction is built (leaf values, rule weights,
                        points or shape-function terms).</li>
                      <li><b>What would change it.</b> The smallest single-feature and two-feature changes that flip a
                        class, or move a regression prediction the most, with round values just past each threshold.</li>
                      <li><b>Features panel.</b> Each feature's importance and marginal distribution; click one to
                        highlight every split or rule that uses it.</li>
                      <li><b>Fold and simplify.</b> Click a split to fold its subtree; the Simple toggle swaps the charts
                        for boxes filled by class proportions. Export the current view as SVG.</li>
                    </ul>

                    <span class="fig-anchor" id="fig1"></span>
                    <div class="vz-demo">
                      <div class="vz-tabs" role="tablist">__TABS__</div>
                      <iframe id="vz-demo-frame" src="viz_gallery/interactive/__TAB0__.html" title="Interactive model" loading="lazy"></iframe>
                    </div>
                    <p class="caption"><b>Fig 1.</b> <span id="vz-demo-cap">__CAP0__</span>
                        <a id="vz-demo-open" href="viz_gallery/interactive/__TAB0__.html">Open full page</a>.</p>

                    <h2 id="gallery">4. Gallery</h2>

                    <p>__NFIGS__ examples, each made by the code in its card. Click a card to see the whole
                        figure, its code and, for <span class="vz-live">live</span> examples, the interactive page.</p>

                    <div class="vz-filters" role="toolbar" aria-label="Filter examples">__PILLS__</div>
                    <p class="vz-filter-note" id="vz-filter-note"></p>
                    <div class="vz-grid" id="vz-grid">
__CARDS__
                    </div>

                    <dialog class="vz-lb" id="vz-lb" aria-label="Example">
                      <div class="vz-lb-head">
                        <div><h3 id="vz-lb-title"></h3><span class="vz-lb-model" id="vz-lb-model"></span></div>
                        <div class="vz-lb-nav">
                          <button type="button" id="vz-lb-prev" aria-label="Previous example">&#8592;</button>
                          <button type="button" id="vz-lb-next" aria-label="Next example">&#8594;</button>
                          <button type="button" id="vz-lb-close" aria-label="Close">&#10005;</button>
                        </div>
                      </div>
                      <div class="vz-lb-fig" id="vz-lb-fig"><img id="vz-lb-img" alt=""></div>
                      <div class="vz-lb-info">
                        <p id="vz-lb-note"></p>
                        <div class="vz-lb-links" id="vz-lb-links"></div>
                        <pre><code class="language-python" id="vz-lb-code"></code></pre>
                      </div>
                    </dialog>

                    <script>
                    (function () {
                      // demo tabs
                      var frame = document.getElementById('vz-demo-frame'), cap = document.getElementById('vz-demo-cap'),
                          open = document.getElementById('vz-demo-open');
                      document.querySelectorAll('.vz-tab').forEach(function (t) {
                        t.addEventListener('click', function () {
                          document.querySelectorAll('.vz-tab').forEach(function (u) { u.setAttribute('aria-selected', u === t); });
                          frame.src = open.href = 'viz_gallery/interactive/' + t.dataset.slug + '.html';
                          cap.textContent = t.dataset.cap;
                        });
                      });
                      // filters
                      var cards = Array.prototype.slice.call(document.querySelectorAll('.vz-card[data-group]'));
                      var note = document.getElementById('vz-filter-note');
                      document.querySelectorAll('.vz-pill').forEach(function (b) {
                        b.addEventListener('click', function () {
                          document.querySelectorAll('.vz-pill').forEach(function (c) { c.setAttribute('aria-pressed', c === b); });
                          cards.forEach(function (c) { c.hidden = b.dataset.group !== 'all' && c.dataset.group !== b.dataset.group; });
                          note.textContent = b.dataset.blurb || '';
                        });
                      });
                      // lightbox
                      var lb = document.getElementById('vz-lb'), cur = 0;
                      function $(id) { return document.getElementById(id); }
                      function shown() { return cards.filter(function (c) { return !c.hidden; }); }
                      function show(card) {
                        var d = card.dataset;
                        cur = shown().indexOf(card);
                        $('vz-lb-title').textContent = d.title;
                        $('vz-lb-model').textContent = d.model || '';
                        $('vz-lb-note').textContent = d.note;
                        $('vz-lb-img').src = d.svg; $('vz-lb-img').alt = d.title;
                        $('vz-lb-fig').classList.toggle('dark', card.classList.contains('dark'));
                        $('vz-lb-fig').classList.remove('zoomed');
                        $('vz-lb-code').textContent = card.querySelector('template').content.textContent.trim();
                        $('vz-lb-links').innerHTML = (d.live ? '<a class="vz-btn" href="viz_gallery/interactive/' + d.slug +
                          '.html">Open interactive page &#8599;</a>' : '') + '<a href="' + d.svg + '">Full-size SVG</a>';
                        if (!lb.open) lb.showModal();
                        history.replaceState(null, '', '#' + d.slug);
                      }
                      function step(k) { var s = shown(); show(s[(cur + k + s.length) % s.length]); }
                      cards.forEach(function (c) {
                        c.querySelector('a.vz-thumb').addEventListener('click', function (e) { e.preventDefault(); show(c); });
                      });
                      $('vz-lb-prev').onclick = function () { step(-1); };
                      $('vz-lb-next').onclick = function () { step(1); };
                      $('vz-lb-close').onclick = function () { lb.close(); };
                      $('vz-lb-fig').onclick = function () { this.classList.toggle('zoomed'); };
                      lb.addEventListener('click', function (e) { if (e.target === lb) lb.close(); });
                      lb.addEventListener('close', function () { history.replaceState(null, '', location.pathname + location.search); });
                      lb.addEventListener('keydown', function (e) {
                        if (e.key === 'ArrowRight') step(1);
                        if (e.key === 'ArrowLeft') step(-1);
                      });
                      var start = cards.filter(function (c) { return '#' + c.dataset.slug === location.hash; })[0];
                      if (start) show(start);
                    })();
                    </script>

                    <h2 id="export">5. Every tree is a scikit-learn tree</h2>

                    <p><code>imodels.viz</code> has no drawing code for any particular imodels tree. Every
                        tree-based model is first exported to the scikit-learn estimator that makes the same
                        predictions, with <code>imodels.to_sklearn</code>, and then drawn like any scikit-learn
                        model: single trees (CART variants, HSTree, TAO, C4.5, FastSmallTree) become a
                        <code>DecisionTreeClassifier</code> or <code>DecisionTreeRegressor</code>, IRF becomes a
                        <code>RandomForestClassifier</code>, and FIGS becomes a list of regression trees whose
                        predictions add up, the same view as gradient boosting. The tests check every export
                        against the model's own <code>predict</code> / <code>predict_proba</code>.</p>

                    <pre><code class="language-python">import imodels
tree = imodels.to_sklearn(model, X)    # X (optional) recounts each node's samples

# anything that reads scikit-learn trees now reads imodels trees, e.g. dtreeviz
import dtreeviz
dtreeviz.model(tree, X, y, feature_names=list(X.columns)).view()
sklearn.tree.plot_tree(tree)</code></pre>

                    <p>So the export also makes every tree-based imodels model work with
                        <a href="https://github.com/parrt/dtreeviz">dtreeviz</a>, including ones dtreeviz could
                        not read before, such as C4.5, FastSmallTree and IRF.</p>

                    <h2 id="compare">6. Compared with plot_tree</h2>

                    <span class="fig-anchor" id="fig3"></span>
                    <div class="vz-compare">
                      <figure class="vz-cmp"><a href="viz_gallery/static/sklearn_plot_tree.svg"><img src="viz_gallery/static/sklearn_plot_tree.svg" alt="sklearn plot_tree of an iris tree" loading="lazy"></a>
                        <figcaption>sklearn.tree.plot_tree</figcaption></figure>
                      <figure class="vz-cmp"><a href="viz_gallery/static/iris.svg"><img src="viz_gallery/static/iris.svg" alt="imodels.viz drawing of the same iris tree" loading="lazy"></a>
                        <figcaption>imodels.viz.draw</figcaption></figure>
                    </div>
                    <p class="caption"><b>Fig 3.</b> The same depth-3 iris tree. <code>plot_tree</code> prints each
                        node's impurity and counts; <code>viz.draw</code> shows each split's feature distribution with
                        its threshold, edges sized by the samples (split by class) that take them, and leaves summarised
                        by class and purity.</p>

                    <p style="margin-top:2rem;color:var(--muted);font-size:0.85rem">Generated by
                        <code>docs/pages/viz_gallery.py</code> on __DATE__.</p>

                </div>
            </section>
            <section>
            </section>
"""


CROP_W = 720  # figure units shown across a card thumbnail, so card text stays readable
DEMOS = [("iris", "Decision tree", "A depth-3 tree on iris. Open Predict and change petal length to watch the path move."),
         ("im_figs", "FIGS", "A sum of two trees: each tree adds its leaf value, and Predict shows the waterfall."),
         ("im_riskscore_loans", "Risk score", "FastRiskScore with a categorical column and missing values: "
          "points add to a total, and the curve maps the total to a risk."),
         ("im_treegam", "GAM", "TreeGAM: one shape function per feature, ordered by how much each moves predictions."),
         ("im_rulefit", "Rule set", "RuleFit: weighted rules and linear terms that add up to the prediction.")]
ORDER = ["sktrees", "trees", "lists", "sets", "scores", "additive", "skmore"]


def _svg_size(path):
    head = open(path, encoding="utf-8").read(2000)
    vb = re.search(r'viewBox="([^"]+)"', head).group(1).split()
    return float(vb[2]), float(vb[3])


def _thumb_style(path, focus, crop=None):
    """CSS placing the full SVG inside a 4:3 card window: whole if small, else a readable crop."""
    w, h = _svg_size(path)
    if w <= CROP_W * 1.15 and h <= CROP_W * 0.75 * 1.15:  # small: the whole figure, centered
        cw = max(w, h / 0.75)
        x0, y0 = (w - cw) / 2, (h - cw * 0.75) / 2
    else:
        cw = min(w, crop or min(max(CROP_W, 0.75 * w), 1100))
        fx, fy = focus
        x0 = min(max(fx * w - cw / 2, 0), w - cw)
        ch = cw * 0.75
        y0 = min(max(fy * h - ch / 2, 0), h - ch) if h > ch else (h - ch) / 2
    ch = cw * 0.75
    return f"width:{w / cw * 100:.2f}%;left:{-x0 / cw * 100:.2f}%;top:{-y0 / ch * 100:.2f}%"


def _default_focus(ex):
    """Trees crop around their root (top center); tables and panels from their labels (top left)."""
    table = ex["group"] in ("sets", "scores", "additive") or ex["slug"] in ("sk_logreg", "sk_isotonic")
    return (0.0, 0.0) if table else (0.5, 0.0)


def _model_class(code):
    m = re.search(r"\b([A-Z][A-Za-z0-9]*(?:Classifier|Regressor|Regression)(?:CV)?)\(", code)
    return m.group(1) if m else ""


def write_post():
    groups = [("sktrees", "Decision trees", "scikit-learn decision trees: classification and regression, with and "
               "without data, both themes and orientations.")] + GROUPS
    blurb = {k: b for k, _, b in groups}
    names = {k: nm for k, nm, _ in groups}
    exs = [dict(ex, group="sktrees") for ex in EXAMPLES] + IMODELS
    exs = sorted(exs, key=lambda ex: ORDER.index(ex["group"]))

    def card(i, ex):
        dark = 'theme="dark"' in ex["code"]
        svg = f"viz_gallery/static/{ex['slug']}.svg"
        code = textwrap.dedent(ex["code"]).strip()
        model = _model_class(code)
        style = _thumb_style(os.path.join(OUT, "static", ex["slug"] + ".svg"), ex.get("focus", _default_focus(ex)), ex.get("crop"))
        live = '<span class="vz-live">live</span>' if ex.get("interactive") else ""
        attrs = {"data-group": ex["group"], "data-slug": ex["slug"], "data-title": ex["title"], "data-note": ex["note"],
                 "data-svg": svg, "data-model": model}
        if ex.get("interactive"):
            attrs["data-live"] = "1"
        attr = " ".join(f'{k}="{html.escape(v)}"' for k, v in attrs.items())
        return f"""                      <figure class="vz-card{' dark' if dark else ''}" id="{ex['slug']}" {attr}>
                        <a class="vz-thumb" href="{svg}" aria-label="{html.escape(ex['title'])}: enlarge"><img src="{svg}" alt="{html.escape(ex['title'])}" loading="lazy" style="{style}"></a>
                        <div class="vz-body">
                          <div class="vz-top"><h3>{html.escape(ex['title'])}</h3><span class="vz-num">{i:02d}</span></div>
                          <div class="vz-meta">{f'<span class="vz-model">{html.escape(model)}</span>' if model else ''}{live}</div>
                        </div>
                        <template>{html.escape(code)}</template>
                      </figure>"""

    cards = [card(i, ex) for i, ex in enumerate(exs, 1)]
    counts = {k: sum(ex["group"] == k for ex in exs) for k in ORDER}
    pills = [f'<button type="button" class="vz-pill" data-group="all" aria-pressed="true">All<span>{len(exs)}</span></button>']
    pills += [f'<button type="button" class="vz-pill" data-group="{k}" data-blurb="{html.escape(blurb[k])}" '
              f'aria-pressed="false">{html.escape(names[k])}<span>{counts[k]}</span></button>' for k in ORDER]
    tabs = [f'<button type="button" class="vz-tab" role="tab" data-slug="{s}" data-cap="{html.escape(c)}" '
            f'aria-selected="{"true" if k == 0 else "false"}">{html.escape(t)}</button>' for k, (s, t, c) in enumerate(DEMOS)]
    mcards = []
    for view, vblurb, models in SUPPORTED:
        chips = "".join(f'<span class="vz-chip {"sk" if src == SK else "im"}" title="{src}">{html.escape(m)}</span>'
                        for m, src in models)
        mcards.append(f'<div class="vz-mcard"><h3>{html.escape(view)}<span>{len(models)}</span></h3>'
                      f'<p>{html.escape(vblurb)}</p><div class="vz-chips">{chips}</div></div>')
    n_models = sum(len(m) for _, _, m in SUPPORTED)
    post = (POST.replace("__CARDS__", "\n".join(cards)).replace("__PILLS__", "".join(pills))
            .replace("__TABS__", "".join(tabs)).replace("__TAB0__", DEMOS[0][0]).replace("__CAP0__", html.escape(DEMOS[0][2]))
            .replace("__SUPPORTED__", "".join(mcards)).replace("__NMODELS__", str(n_models // 10 * 10))
            .replace("__SUPPORT_NOTE__", html.escape(SUPPORT_NOTE)).replace("__NFIGS__", str(len(exs)))
            .replace("__DATE__", time.strftime("%Y-%m-%d")))
    with open(os.path.join(HERE, "viz.html"), "w", encoding="utf-8") as f:
        f.write(post)


if __name__ == "__main__":
    if "--post-only" not in sys.argv:  # --post-only: rewrite the article from the existing figures
        run_examples()
    write_post()
    print("wrote", os.path.join(HERE, "viz.html"))
