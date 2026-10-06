"""Build the imodels.viz post (docs/viz.html): its figures, interactive pages and article body.

Each example's code string is executed verbatim, so the code shown under a figure is exactly what
produced it. Figures go to docs/viz_gallery/{static,interactive}; the article body goes to
docs/pages/viz.html, which build_pages.py wraps in the site shell. Run from docs/:

    uv run python pages/viz_gallery.py   # needs imodels[viz] extras: matplotlib for the plot_tree reference
    uv run python build_pages.py
"""

import html
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
    dict(slug="california_big", title="A bigger tree",
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
    dict(slug="california_lr", title="Left-to-right layout",
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
    dict(slug="digits_compact", title="Many classes, compact style",
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
    dict(group="skmore", slug="sk_forest", title="Random forest",
         note="Forests draw their first few trees in a grid; predictions average every tree. In interactive "
              "mode Predict lists each tree's vote.",
         code="""
from sklearn.ensemble import RandomForestClassifier
X, y = cancer.data, cancer.target
model = RandomForestClassifier(n_estimators=50, max_depth=3, random_state=0).fit(X, y)
kw = dict(class_names=["malignant", "benign"], title="Random forest, breast cancer", max_trees=3)
fig = viz.draw(model, X, y, **kw)
"""),
    dict(group="skmore", slug="sk_gbm", title="Gradient boosting",
         note="Boosted trees add up: each leaf shows its contribution and the waterfall in Predict sums them.",
         code="""
X, y = diabetes.data, diabetes.target
model = GradientBoostingRegressor(n_estimators=60, max_depth=2, random_state=0).fit(X, y)
kw = dict(title="Gradient boosting, diabetes progression", max_trees=3)
fig = viz.draw(model, X, y, **kw)
"""),
    dict(group="skmore", slug="sk_hgb", title="Histogram gradient boosting",
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
    dict(group="trees", slug="im_figs", title="FIGS: a sum of trees",
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
    dict(group="trees", slug="im_irf", title="Iterative random forest (IRF)",
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
                    <p class="post-links"><a href="viz/index.html">🗂 Doc</a>, <a href="https://github.com/csinva/imodels/tree/master/imodels/viz">💻 Code</a>, <a href="#gallery">🖼 Gallery</a></p>
                    <hr>

                    <p class="abstract">An interpretable model is only as useful as the picture you can make of it.
                        <code>imodels.viz</code> draws a fitted model two ways: as a static figure (SVG, PNG, PDF) for a
                        paper or slide, and as a single offline HTML page where you can fold the model, run a sample
                        through it and ask what would change its prediction. It covers the models in <code>imodels</code>
                        (trees, sums of trees, rule lists, rule sets, scoring systems and additive models) and
                        scikit-learn's trees, forests, boosting and linear models, __NMODELS__+ model classes in all.
                        Every view is tested to reproduce its model's <code>predict</code> / <code>predict_proba</code>
                        to machine precision, so the figure shows what the model computes.
                        Each of the __NFIGS__ figures below was made by the one call shown under it.</p>

                    <nav class="toc-main">
                      <a href="#quickstart"><span>1</span> Quickstart</a>
                      <a href="#supported"><span>2</span> Supported models</a>
                      <a href="#interactive"><span>3</span> Interactive mode</a>
                      <a href="#gallery"><span>4</span> Gallery</a>
                      <a href="#export"><span>5</span> Every tree is a scikit-learn tree</a>
                      <a href="#compare"><span>6</span> Compared with plot_tree</a>
                    </nav>

                    <style>
                      .vz-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(21rem, 1fr)); gap: 1.4rem; margin: 1.2rem 0 2.2rem; }
                      .vz-card { display: flex; flex-direction: column; margin: 0; background: var(--surface); border: 1px solid var(--line);
                        border-radius: 12px; overflow: hidden; box-shadow: 0 1px 2px rgba(27,31,35,.04), 0 6px 18px rgba(27,31,35,.05);
                        transition: transform .15s ease, box-shadow .15s ease; min-width: 0; }
                      .vz-card:hover { transform: translateY(-2px); box-shadow: 0 2px 4px rgba(27,31,35,.06), 0 12px 28px rgba(27,31,35,.09); }
                      .vz-thumb { display: flex; align-items: center; justify-content: center; height: 17rem; padding: 0.8rem;
                        background: #fff; border-bottom: 1px solid var(--line-soft); }
                      .vz-thumb.dark { background: #1a1a19; }
                      .vz-thumb img { max-width: 100%; max-height: 100%; object-fit: contain; }
                      .vz-body { padding: 0.9rem 1.1rem 1rem; display: flex; flex-direction: column; gap: 0.45rem; flex: 1; min-width: 0; }
                      .vz-top { display: flex; justify-content: space-between; align-items: baseline; gap: 0.6rem; }
                      .vz-card h3 { font-size: 1.02rem; margin: 0; line-height: 1.3; border: none; padding: 0; }
                      .vz-num { white-space: nowrap; flex-shrink: 0; font: 500 0.72rem ui-monospace, SFMono-Regular, Menlo, monospace; color: var(--muted); }
                      .vz-card p { margin: 0; font-size: 0.88rem; line-height: 1.5; color: var(--ink-soft); }
                      .vz-links { margin-top: auto; padding-top: 0.3rem; display: flex; gap: 1rem; font-size: 0.85rem; }
                      .vz-card details summary { cursor: pointer; font-size: 0.82rem; color: var(--muted); }
                      .vz-card details pre { font-size: 0.78rem; margin: 0.5rem 0 0; }
                      .vz-group-blurb { color: var(--ink-soft); margin-top: -0.3rem; }
                      .vz-demo { border: 1px solid var(--line); border-radius: 12px; overflow: hidden; margin: 1rem 0 0.4rem; background: var(--surface); }
                      .vz-demo iframe { display: block; width: 100%; height: min(78vh, 720px); border: 0; }
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
                    <div class="vz-demo"><iframe src="viz_gallery/interactive/iris.html" title="Interactive iris decision tree" loading="lazy"></iframe></div>
                    <p class="caption"><b>Fig 1.</b> A depth-3 tree on iris, live. Open the Predict panel and change
                        petal length to watch the path move. <a href="viz_gallery/interactive/iris.html">Open full page</a>.</p>

                    <span class="fig-anchor" id="fig2"></span>
                    <div class="vz-demo"><iframe src="viz_gallery/interactive/im_riskscore_loans.html" title="Interactive FastRiskScore" loading="lazy"></iframe></div>
                    <p class="caption"><b>Fig 2.</b> A FastRiskScore scorecard with a categorical column and missing
                        values, live. Each row's points add to a total, and the curve maps the total to a risk.
                        <a href="viz_gallery/interactive/im_riskscore_loans.html">Open full page</a>.</p>

                    <h2 id="gallery">4. Gallery</h2>

                    <p>Click a figure for the full-size SVG, or <i>Interactive</i> for its page. Code is under each card.</p>

                    <h3 id="g-trees">4.1 Decision trees</h3>
                    <p class="vz-group-blurb">scikit-learn trees: classification and regression, with and without data, both themes and orientations.</p>
                    <div class="vz-grid">
__FIGS__
                    </div>

__GROUPS__

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
                      <div class="vz-card"><a class="vz-thumb" href="viz_gallery/static/sklearn_plot_tree.svg"><img src="viz_gallery/static/sklearn_plot_tree.svg" alt="sklearn plot_tree of an iris tree" loading="lazy"></a>
                        <div class="vz-body"><h3>sklearn.tree.plot_tree</h3></div></div>
                      <div class="vz-card"><a class="vz-thumb" href="viz_gallery/static/iris.svg"><img src="viz_gallery/static/iris.svg" alt="imodels.viz drawing of the same iris tree" loading="lazy"></a>
                        <div class="vz-body"><h3>imodels.viz.draw</h3></div></div>
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


def write_post():
    def card(i, ex):
        dark = " dark" if 'theme="dark"' in ex["code"] else ""
        svg = f"viz_gallery/static/{ex['slug']}.svg"
        links = [f'<a href="{svg}">SVG</a>']
        if ex.get("interactive"):
            links.insert(0, f'<a href="viz_gallery/interactive/{ex["slug"]}.html">Interactive &#8599;</a>')
        code = html.escape(textwrap.dedent(ex["code"]).strip())
        return f"""                      <figure class="vz-card" id="{ex['slug']}">
                        <a class="vz-thumb{dark}" href="{svg}"><img src="{svg}" alt="{html.escape(ex['title'])}" loading="lazy"></a>
                        <div class="vz-body">
                          <div class="vz-top"><h3>{html.escape(ex['title'])}</h3><span class="vz-num">{i:02d}</span></div>
                          <p>{html.escape(ex['note'])}</p>
                          <div class="vz-links">{''.join(links)}</div>
                          <details><summary>Code</summary><pre><code class="language-python">{code}</code></pre></details>
                        </div>
                      </figure>"""

    figs = [card(i, ex) for i, ex in enumerate(EXAMPLES, 1)]
    groups, n = [], len(EXAMPLES)
    for k, (key, name, blurb) in enumerate(GROUPS, 2):
        cards = []
        for ex in (ex for ex in IMODELS if ex["group"] == key):
            n += 1
            cards.append(card(n, ex))
        groups.append(f'                    <h3 id="g-{key}">4.{k} {html.escape(name)}</h3>\n'
                      f'                    <p class="vz-group-blurb">{html.escape(blurb)}</p>\n'
                      f'                    <div class="vz-grid">\n' + "\n".join(cards) + "\n                    </div>\n")
    mcards = []
    for view, blurb, models in SUPPORTED:
        chips = "".join(f'<span class="vz-chip {"sk" if src == SK else "im"}" title="{src}">{html.escape(m)}</span>'
                        for m, src in models)
        mcards.append(f'<div class="vz-mcard"><h3>{html.escape(view)}<span>{len(models)}</span></h3>'
                      f'<p>{html.escape(blurb)}</p><div class="vz-chips">{chips}</div></div>')
    n_models = sum(len(m) for _, _, m in SUPPORTED)
    post = (POST.replace("__FIGS__", "\n".join(figs)).replace("__GROUPS__", "\n".join(groups))
            .replace("__SUPPORTED__", "".join(mcards)).replace("__NMODELS__", str(n_models // 10 * 10))
            .replace("__SUPPORT_NOTE__", html.escape(SUPPORT_NOTE)).replace("__NFIGS__", str(n))
            .replace("__DATE__", time.strftime("%Y-%m-%d")))
    with open(os.path.join(HERE, "viz.html"), "w", encoding="utf-8") as f:
        f.write(post)


if __name__ == "__main__":
    run_examples()
    write_post()
    print("wrote", os.path.join(HERE, "viz.html"))
