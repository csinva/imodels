"""Every supported imodels estimator: the view must reproduce the model's own predictions,
and both rendering modes must produce valid output."""

import warnings
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

im = pytest.importorskip("imodels")

import imodels.viz as dti  # noqa: E402
from imodels.viz._adapt import to_view  # noqa: E402
from imodels.viz._views import AdditiveView  # noqa: E402

warnings.filterwarnings("ignore")
_bc = load_breast_cancer(as_frame=True)
_db = load_diabetes(as_frame=True)
Xb, yb = _bc.data, _bc.target
Xd, yd = _db.data, _db.target
Xbin = (Xb.iloc[:, :8] > Xb.iloc[:, :8].median()).astype(int)
Xbin.columns = [c.replace(" ", "_") + "_hi" for c in Xbin.columns]
DATA = {"b": (Xb, yb), "d": (Xd, yd), "bin": (Xbin, yb)}

CASES = {
    "GreedyTreeClassifier": (lambda: im.GreedyTreeClassifier(max_depth=3), "b"),
    "DecisionTreeCCPClassifier": (lambda: im.DecisionTreeCCPClassifier(
        estimator_=DecisionTreeClassifier(random_state=0), desired_complexity=6), "b"),
    "DecisionTreeCCPRegressor": (lambda: im.DecisionTreeCCPRegressor(
        estimator_=DecisionTreeRegressor(random_state=0), desired_complexity=6), "d"),
    "HSTreeClassifier": (lambda: im.HSTreeClassifier(estimator_=DecisionTreeClassifier(max_leaf_nodes=8), reg_param=10), "b"),
    "HSTreeRegressor": (lambda: im.HSTreeRegressor(estimator_=DecisionTreeRegressor(max_leaf_nodes=8), reg_param=10), "d"),
    "TaoTreeClassifier": (lambda: im.TaoTreeClassifier(model_args={"max_leaf_nodes": 6}, n_iters=3), "b"),
    "FastSmallTreeClassifier": (lambda: im.FastSmallTreeClassifier(regularization=0.05, time_limit=10), "b"),
    "C45TreeClassifier": (lambda: im.C45TreeClassifier(max_rules=6), "b"),
    "FIGSClassifier": (lambda: im.FIGSClassifier(max_rules=8), "b"),
    "FIGSRegressor": (lambda: im.FIGSRegressor(max_rules=8), "d"),
    "GreedyRuleListClassifier": (lambda: im.GreedyRuleListClassifier(max_depth=4), "b"),
    "OneRClassifier": (lambda: im.OneRClassifier(max_depth=4), "b"),
    "FastFrugalTreeClassifier": (lambda: im.FastFrugalTreeClassifier(), "b"),
    "BayesianRuleListClassifier": (lambda: im.BayesianRuleListClassifier(max_iter=2000, n_chains=2, random_state=0), "bin"),
    "RuleFitClassifier": (lambda: im.RuleFitClassifier(max_rules=10, random_state=0), "b"),
    "RuleFitRegressor": (lambda: im.RuleFitRegressor(max_rules=10, random_state=0), "d"),
    "FPLassoClassifier": (lambda: im.FPLassoClassifier(max_rules=10, random_state=0), "bin"),
    "SkopeRulesClassifier": (lambda: im.SkopeRulesClassifier(n_estimators=5, max_depth=2, random_state=0), "b"),
    "FPSkopeClassifier": (lambda: im.FPSkopeClassifier(precision_min=0.3, recall_min=0.05), "bin"),
    "BoostedRulesClassifier": (lambda: im.BoostedRulesClassifier(n_estimators=5, random_state=0), "b"),
    "SlipperClassifier": (lambda: im.SlipperClassifier(n_estimators=2, random_state=0), "b"),
    "BayesianRuleSetClassifier": (lambda: im.BayesianRuleSetClassifier(n_rules=20, num_iterations=50, num_chains=1, maxlen=2), "bin"),
    "FastRiskScoreClassifier": (lambda: im.FastRiskScoreClassifier(k=5, time_limit=20), "b"),
    "SLIMClassifier": (lambda: im.SLIMClassifier(), "b"),
    "SLIMRegressor": (lambda: im.SLIMRegressor(), "d"),
    "SLIMClassifier_binary": (lambda: im.SLIMClassifier(alpha=0.5), "bin"),
    "MarginalShrinkageLinearRegressor": (lambda: im.MarginalShrinkageLinearRegressor(), "d"),
    "TreeGAMClassifier": (lambda: im.TreeGAMClassifier(n_boosting_rounds=20), "b"),
    "TreeGAMRegressor": (lambda: im.TreeGAMRegressor(n_boosting_rounds=20), "d"),
    "GPGamRegressor": (lambda: im.GPGamRegressor(schedule=False, n_bins=16, n_pairs=0, n_steps=50), "d"),
}
_fitted = {}


def fitted(name):
    if name not in _fitted:
        make, data = CASES[name]
        X, y = DATA[data]
        _fitted[name] = (make().fit(X, y), X, y, data)
    return _fitted[name]


@pytest.mark.parametrize("name", sorted(CASES))
def test_view_reproduces_predictions(name):
    m, X, y, data = fitted(name)
    v = to_view(m, X, y)
    Xa = np.asarray(X, float)
    ours = v.output(v.score(Xa)) if isinstance(v, AdditiveView) else v.output(Xa)
    if data != "d" and name != "BayesianRuleSetClassifier":
        theirs = m.predict_proba(X)
        if np.ndim(ours) == 1:
            theirs = theirs[:, 1]
    else:
        theirs = np.asarray(m.predict(X), float)
        if np.ndim(ours) == 2:
            ours = ours[:, 1]
    np.testing.assert_allclose(np.asarray(ours, float), np.asarray(theirs, float), atol=1e-8)


@pytest.mark.parametrize("name", sorted(CASES))
def test_renders(name):
    m, X, y, _ = fitted(name)
    ET.fromstring(dti.draw(m, X, y).svg)
    ET.fromstring(dti.draw(m).svg)  # without data
    page = dti.interactive(m, X, y).html
    assert "__DATA__" not in page and "__SVG__" not in page


def test_slim_on_binary_features_is_a_scorecard():
    m, X, y, _ = fitted("SLIMClassifier_binary")
    v = to_view(m, X, y)
    assert v.family == "scorecard" and all(t.kind == "rule" for t in v.terms)
    assert all(float(t.weight).is_integer() for t in v.terms)
    assert v.risk and all(float(s).is_integer() for s, _ in v.risk)


def test_unsupported_models_say_why():
    m = im.BoostedRulesRegressor(n_estimators=3).fit(Xd, yd)
    with pytest.raises(NotImplementedError, match="weighted median"):
        dti.draw(m)


def test_fastriskscore_categorical_and_missing():
    """imodels >= 3.0.3: string columns become level indicators and missing values get their own item."""
    import pandas as pd

    rng = np.random.default_rng(0)
    n = 800
    X = pd.DataFrame({"score": rng.integers(480, 851, n).astype(float),
                      "income": np.round(rng.lognormal(4.1, 0.45, n), 1),
                      "housing": rng.choice(["own", "rent", "mortgage"], n)})
    X.loc[rng.random(n) < 0.15, "income"] = np.nan
    logit = 0.02 * (X.score - 650) + X.housing.map({"own": 1.0, "rent": -1.0, "mortgage": 0.3}) - 1.5 * X.income.isna()
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    m = im.FastRiskScoreClassifier(k=5, time_limit=30).fit(X, y)
    v = to_view(m, X, y)
    np.testing.assert_allclose(v.output(v.score(v.X)), m.predict_proba(X)[:, 1], atol=1e-12)
    texts = [v.term_text(t) for t in v.terms]
    assert any("housing =" in t for t in texts) or any("missing" in t for t in texts), texts
    ET.fromstring(dti.draw(m, X, y).svg)
    page = dti.interactive(m, X, y).html
    assert '"levels"' in page
