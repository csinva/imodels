"""scikit-learn models beyond single trees: every view must reproduce the model's predictions,
and both rendering modes must produce valid output."""

import warnings
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from sklearn import ensemble as en
from sklearn import isotonic
from sklearn import linear_model as lm
from sklearn import svm
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.tree import DecisionTreeClassifier

import imodels.viz as dti
from imodels.viz._adapt import to_view
from imodels.viz._views import AdditiveView

warnings.filterwarnings("ignore")
_bc, _db = load_breast_cancer(as_frame=True), load_diabetes(as_frame=True)
Xb, yb, Xd, yd = _bc.data, _bc.target, _db.data, _db.target
Xn = Xb.copy()
Xn.iloc[::7, 0] = np.nan  # missing values
DATA = {"b": (Xb, yb), "d": (Xd, yd), "p": (Xd, np.round(yd / 30)), "n": (Xn, yb), "iso": (Xd[["bmi"]], yd)}

CASES = {
    "LinearRegression": (lambda: lm.LinearRegression(), "d"),
    "Ridge": (lambda: lm.Ridge(1.0), "d"),
    "LassoCV": (lambda: lm.LassoCV(cv=3), "d"),
    "HuberRegressor": (lambda: lm.HuberRegressor(), "d"),
    "PoissonRegressor": (lambda: lm.PoissonRegressor(alpha=0.01), "p"),
    "LinearSVR": (lambda: make_pipeline(StandardScaler(), svm.LinearSVR(random_state=0, max_iter=5000)), "d"),
    "RANSACRegressor": (lambda: lm.RANSACRegressor(random_state=0), "d"),
    "LogisticRegression": (lambda: make_pipeline(StandardScaler(), lm.LogisticRegression(max_iter=1000)), "b"),
    "LogisticRegression_two_scalers": (lambda: make_pipeline(MinMaxScaler(), StandardScaler(),
                                                             lm.LogisticRegression(max_iter=1000)), "b"),
    "SGDClassifier": (lambda: make_pipeline(RobustScaler(), lm.SGDClassifier(loss="log_loss", random_state=0)), "b"),
    "LinearSVC": (lambda: make_pipeline(StandardScaler(), svm.LinearSVC(random_state=0)), "b"),
    "RidgeClassifier": (lambda: lm.RidgeClassifier(), "b"),
    "RandomForestClassifier": (lambda: en.RandomForestClassifier(n_estimators=20, max_depth=4, random_state=0), "b"),
    "RandomForestRegressor": (lambda: en.RandomForestRegressor(n_estimators=20, max_depth=4, random_state=0), "d"),
    "ExtraTreesClassifier": (lambda: en.ExtraTreesClassifier(n_estimators=10, max_depth=4, random_state=0), "b"),
    "GradientBoostingClassifier": (lambda: en.GradientBoostingClassifier(n_estimators=30, max_depth=2, random_state=0), "b"),
    "GradientBoostingRegressor": (lambda: en.GradientBoostingRegressor(n_estimators=30, max_depth=2, random_state=0), "d"),
    "HistGradientBoostingClassifier": (lambda: en.HistGradientBoostingClassifier(max_iter=30, random_state=0), "b"),
    "HistGradientBoostingRegressor": (lambda: en.HistGradientBoostingRegressor(max_iter=30, random_state=0), "d"),
    "HistGradientBoosting_missing": (lambda: en.HistGradientBoostingClassifier(max_iter=20, random_state=0), "n"),
    "DecisionTree_missing": (lambda: DecisionTreeClassifier(max_depth=4, random_state=0), "n"),
    "DecisionTree_scaled": (lambda: make_pipeline(StandardScaler(), DecisionTreeClassifier(max_depth=3, random_state=0)), "b"),
    "IsotonicRegression": (lambda: isotonic.IsotonicRegression(out_of_bounds="clip"), "iso"),
}
_fitted = {}


def fitted(name):
    if name not in _fitted:
        make, data = CASES[name]
        X, y = DATA[data]
        _fitted[name] = (make().fit(X.values.ravel() if data == "iso" else X, y), X, y, data)
    return _fitted[name]


@pytest.mark.parametrize("name", sorted(CASES))
def test_view_reproduces_predictions(name):
    m, X, y, data = fitted(name)
    v = to_view(m, X, y)
    Xa = np.asarray(X, float)
    ours = v.output(v.score(Xa)) if isinstance(v, AdditiveView) else v.output(Xa)
    threshold = isinstance(v, AdditiveView) and v.link == "threshold"
    Xp = X.values.ravel() if data == "iso" else X
    if data in ("b", "n") and hasattr(m, "predict_proba") and not threshold:
        theirs = m.predict_proba(Xp)
        if np.ndim(ours) == 1:
            theirs = theirs[:, 1]
    else:
        theirs = np.asarray(m.predict(Xp))
        if threshold:
            theirs = (theirs == m.classes_[1]).astype(float)
        if np.ndim(ours) == 2:
            ours = ours[:, 1]
    np.testing.assert_allclose(np.asarray(ours, float), np.asarray(theirs, float), atol=1e-8)


@pytest.mark.parametrize("name", sorted(CASES))
def test_renders(name):
    m, X, y, _ = fitted(name)
    ET.fromstring(dti.draw(m, X, y).svg)
    page = dti.interactive(m, X, y).html
    assert "__DATA__" not in page


def test_ensembles_draw_a_few_trees():
    m, X, y, _ = fitted("RandomForestClassifier")
    info = to_view(m, X, y)
    assert len(info.roots) == 20 and len(info.shown_roots) == 6
    svg = dti.draw(m, X, y, max_trees=2).svg
    assert "tree 2 of 20" in svg and "tree 3 of 20" not in svg


def test_opaque_models_and_pipelines_say_why():
    with pytest.raises(NotImplementedError, match="no readable structure"):
        dti.draw(KNeighborsClassifier().fit(Xb, yb))
    with pytest.raises(NotImplementedError, match="only scaling steps"):
        dti.draw(make_pipeline(PCA(3), lm.LogisticRegression()).fit(Xb, yb))


@pytest.mark.parametrize("nan_like", [0, 1])  # missing rows resemble the left (0) or right (1) group
@pytest.mark.parametrize("make", [
    lambda: en.HistGradientBoostingClassifier(max_iter=1, max_leaf_nodes=2, min_samples_leaf=1,
                                              learning_rate=1.0, early_stopping=False),
    lambda: en.HistGradientBoostingRegressor(max_iter=1, max_leaf_nodes=2, min_samples_leaf=1,
                                             learning_rate=1.0, early_stopping=False),
    lambda: en.RandomForestClassifier(n_estimators=2, max_depth=1, bootstrap=False, random_state=0),
    lambda: en.RandomForestRegressor(n_estimators=2, max_depth=1, bootstrap=False, random_state=0),
], ids=["hgb_clf", "hgb_reg", "rf_clf", "rf_reg"])
def test_ensembles_attach_missing_values_to_their_branch(make, nan_like):
    X = np.array([[0.0], [0.0], [0.0], [1.0], [1.0], [1.0], [np.nan], [np.nan]])
    y = np.array([0, 0, 0, 1, 1, 1, nan_like, nan_like])
    m = make().fit(X, y)
    info = to_view(m, X, y)
    for r in info.roots:
        root = info.nodes[r]
        assert root.nan_left == (nan_like == 0)
        left, right = info.nodes[root.left], info.nodes[root.right]
        assert sorted(left.idx.tolist()) == ([0, 1, 2, 6, 7] if nan_like == 0 else [0, 1, 2])
        assert sorted(right.idx.tolist()) == ([3, 4, 5] if nan_like == 0 else [3, 4, 5, 6, 7])
        if info.is_clf and isinstance(m, en.HistGradientBoostingClassifier):  # counts come from the data
            assert left.counts.tolist() == ([5.0, 0.0] if nan_like == 0 else [3.0, 0.0])
            assert right.counts.tolist() == ([0.0, 3.0] if nan_like == 0 else [0.0, 5.0])
