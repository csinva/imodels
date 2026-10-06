import re
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_diabetes, load_iris
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, ExtraTreeClassifier

import imodels.viz as dti
from imodels.viz._extract import extract
from imodels.viz._layout import tidy
from imodels.viz._text import fmt


@pytest.fixture(scope="module")
def iris():
    d = load_iris(as_frame=True)
    return d.data, d.target, DecisionTreeClassifier(max_depth=3, random_state=0).fit(d.data, d.target)


@pytest.fixture(scope="module")
def diabetes():
    d = load_diabetes(as_frame=True)
    return d.data, d.target, DecisionTreeRegressor(max_depth=3, random_state=0).fit(d.data, d.target)


def parse(svg):
    return ET.fromstring(svg)  # raises on malformed XML


@pytest.mark.parametrize("kw", [{}, {"orientation": "LR"}, {"theme": "dark"}, {"style": "compact"}, {"max_depth": 1},
                                {"simple": True}, {"simple": True, "orientation": "LR", "max_depth": 1}])
def test_draw_variants_are_valid_svg(iris, kw):
    X, y, clf = iris
    parse(dti.draw(clf, X, y, **kw).svg)
    parse(dti.draw(clf, **kw).svg)


def test_regression_and_instance(diabetes):
    X, y, reg = diabetes
    svg = dti.draw(reg, X, y, x=X.iloc[0]).svg
    parse(svg)
    leaf = reg.apply(X.iloc[[0]])[0]
    pred = reg.tree_.value[leaf, 0, 0]
    assert fmt(pred, 4) in svg


def test_counts_match_sklearn(iris):
    X, y, clf = iris
    info = extract(clf, X, y)
    for nd in info.nodes:
        assert nd.n == clf.tree_.n_node_samples[nd.id]
        assert np.isclose(nd.counts.sum(), clf.tree_.weighted_n_node_samples[nd.id])
        assert len(nd.idx) == nd.n  # decision_path routing agrees with the tree
        assert np.allclose(np.bincount(y.values[nd.idx], minlength=3), nd.counts)


def test_rules_collapse_to_intervals(iris):
    X, y, clf = iris
    info = extract(clf, X, y)
    deepest = max(info.nodes, key=lambda n: n.depth)
    rules = info.rules(deepest.id)
    names = [r for r in rules if "petal width" in r]
    assert len(names) <= 1  # repeated splits on one feature merge


def test_integer_and_binary_labels():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"flag": rng.integers(0, 2, 400), "count": rng.integers(0, 10, 400)})
    y = (X["flag"] & (X["count"] > 4)).astype(int)
    clf = DecisionTreeClassifier(max_depth=2, random_state=0).fit(X, y)
    info = extract(clf, X, y)
    labels = {info.edge_label(n.id) for n in info.nodes if n.parent >= 0}
    assert labels & {"= 0", "= 1"}
    assert any(l.startswith("≥") for l in labels)


def test_ensembles_and_pipelines(iris, diabetes):
    X, y, _ = iris
    rf = RandomForestClassifier(n_estimators=3, max_depth=3, random_state=0).fit(X, y)
    parse(dti.draw(rf.estimators_[0], X, y, feature_names=list(X.columns)).svg)
    parse(dti.draw(rf, X, y).svg)  # whole forests draw as averaged trees
    pipe = make_pipeline(StandardScaler(), DecisionTreeClassifier(max_depth=2)).fit(X, y)
    parse(dti.draw(pipe).svg)
    parse(dti.draw(ExtraTreeClassifier(max_depth=3, random_state=0).fit(X, y)).svg)
    Xd, yd, _ = diabetes
    gbm = GradientBoostingRegressor(n_estimators=2, max_depth=2).fit(Xd, yd)
    parse(dti.draw(gbm.estimators_[0, 0], Xd, yd - yd.mean()).svg)


def test_layout_no_overlap(iris):
    X, y, _ = iris
    clf = DecisionTreeClassifier(random_state=0).fit(X, y)
    info = extract(clf)
    kids = {n.id: [n.left, n.right] for n in info.nodes if not n.is_leaf}
    rng = np.random.default_rng(1)
    size = {n.id: float(rng.uniform(40, 200)) for n in info.nodes}
    depth = {n.id: n.depth for n in info.nodes}
    pos, top, _, _ = tidy(0, kids, size, depth, {k: 50.0 for k in size}, gap=10)
    for d in set(depth.values()):
        row = sorted((pos[i] - size[i] / 2, pos[i] + size[i] / 2) for i in pos if depth[i] == d)
        for (_, a_hi), (b_lo, _) in zip(row, row[1:]):
            assert b_lo - a_hi >= 10 - 1e-6


def test_interactive(iris, diabetes, tmp_path):
    X, y, clf = iris
    page = dti.interactive(clf, X, y, class_names=["a", "b", "c"])
    assert "__DATA__" not in page.html and "__CSS_LIGHT__" not in page.html
    assert page._repr_html_().startswith("<iframe")
    page.save(tmp_path / "t.html")
    Xd, yd, reg = diabetes
    assert "regression" in dti.interactive(reg).html


def test_save_formats(iris, tmp_path):
    X, y, clf = iris
    fig = dti.draw(clf, X, y)
    try:
        import cairosvg  # noqa: F401  (optional: only PNG / PDF need it)
        exts = ("svg", "html", "png", "pdf")
    except ImportError:
        exts = ("svg", "html")
    for ext in exts:
        p = tmp_path / f"t.{ext}"
        fig.save(p)
        assert p.stat().st_size > 1000
    with pytest.raises(ValueError):
        fig.save(tmp_path / "t.jpg")


def test_fmt():
    assert fmt(2.45) == "2.45"
    assert fmt(1234.567) == "1,235"
    assert fmt(-1.75) == "−1.75"
    assert fmt(0) == "0"


def test_ribbons_split_by_class(iris):
    X, y, clf = iris
    svg = dti.draw(clf, X, y).svg
    root = parse(svg)
    ns = "{http://www.w3.org/2000/svg}"
    groups = [g for g in root.iter(ns + "g") if "fill-opacity" in (g.get("style") or "")]
    info = extract(clf)
    n_edges = sum(1 for n in info.nodes if n.parent >= 0)
    assert len(groups) == n_edges
    expected = sum(int((n.counts > 0).sum()) for n in info.nodes if n.parent >= 0)
    assert sum(len(g.findall(ns + "path")) for g in groups) == expected


def test_interactive_simple_payload(iris):
    X, y, clf = iris
    html = dti.interactive(clf, X, y, simple=True).html
    assert '"simple":true' in html and '"ssvg"' in html and '"mix"' in html


def test_interactive_hud_payload(iris):
    import json, re
    X, y, clf = iris
    html = dti.interactive(clf, X, y).html
    data = json.loads(re.search(r'id="dti-data">(.*?)</script>', html, re.S).group(1))
    imps = [f["imp"] for f in data["feats"]]
    assert imps == sorted(imps, reverse=True)
    for f in data["feats"]:
        assert f["nsplit"] == sum(1 for n in data["nodes"] if not n["leaf"] and n["f"] == f["i"])
        assert sum(sum(b) for b in f["hist"]) == len(X)  # every row lands in a bin
        assert all(f["hlo"] <= t <= f["hhi"] for t in f["thr"])
    assert 'id="hud"' in html


def test_interactive_simple_auto(iris):
    X, y, _ = iris
    small = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X, y)
    big = DecisionTreeClassifier(random_state=0).fit(np.random.default_rng(0).normal(size=(600, 4)),
                                                     np.random.default_rng(1).integers(0, 3, 600))
    assert big.tree_.n_leaves > 32
    assert '"simple":false' in dti.interactive(small).html
    assert '"simple":true' in dti.interactive(big).html
    assert '"simple":false' in dti.interactive(big, simple=False).html
