"""imodels.to_sklearn: tree-based models exported as fitted scikit-learn estimators.

Each export must predict exactly what the model predicts, so anything that reads
scikit-learn trees (imodels.viz, dtreeviz, sklearn.tree.plot_tree) shows the model itself.
"""

import shutil
import warnings

import numpy as np
import pytest
from scipy.special import softmax
from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text

import imodels
from imodels import to_sklearn

cancer = load_breast_cancer(return_X_y=True, as_frame=True)
diabetes = load_diabetes(return_X_y=True, as_frame=True)
iris = load_iris(return_X_y=True, as_frame=True)
_X, _y, _names = imodels.get_clean_dataset("csi_pecarn_pred")
csi = (_X, _y)

CASES = {
    "GreedyTreeClassifier": (lambda: imodels.GreedyTreeClassifier(max_depth=3), cancer),
    "DecisionTreeCCPClassifier": (lambda: imodels.DecisionTreeCCPClassifier(
        estimator_=DecisionTreeClassifier(max_depth=4, random_state=0)), cancer),
    "HSTreeClassifier": (lambda: imodels.HSTreeClassifier(max_leaf_nodes=8), cancer),
    "HSTreeRegressor": (lambda: imodels.HSTreeRegressor(max_leaf_nodes=8), diabetes),
    "HSTreeClassifierCV": (lambda: imodels.HSTreeClassifierCV(max_leaf_nodes=8), cancer),
    "TaoTreeClassifier": (lambda: imodels.TaoTreeClassifier(), cancer),
    "FastSmallTreeClassifier": (lambda: imodels.FastSmallTreeClassifier(time_limit=20), cancer),
    "C45TreeClassifier": (lambda: imodels.C45TreeClassifier(max_rules=10), cancer),
    "C45TreeClassifier_multiclass": (lambda: imodels.C45TreeClassifier(max_rules=10), iris),
    "FIGSClassifier": (lambda: imodels.FIGSClassifier(max_rules=10), cancer),
    "FIGSClassifier_multiclass": (lambda: imodels.FIGSClassifier(max_rules=10), iris),
    "FIGSRegressor": (lambda: imodels.FIGSRegressor(max_rules=10), diabetes),
    # several trees on 0/1 features: FIGS leaves later trees' roots without a value
    "FIGSClassifier_csi": (lambda: imodels.FIGSClassifier(max_rules=4), csi),
    "IRFClassifier": (lambda: imodels.IRFClassifier(n_estimators=10, max_depth=4, random_state=0), cancer),
    "IRFRegressor": (lambda: imodels.IRFRegressor(n_estimators=10, max_depth=4, random_state=0), diabetes),
}
_fitted = {}


def fitted(name):
    if name not in _fitted:
        make, (X, y) = CASES[name]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _fitted[name] = (make().fit(X, y), X, y)
    return _fitted[name]


@pytest.mark.parametrize("name", sorted(CASES))
def test_export_predicts_like_the_model(name):
    m, X, y = fitted(name)
    exported = to_sklearn(m, X)
    clf = hasattr(m, "classes_")
    if isinstance(exported, list):  # FIGS: the trees' predictions add up to the raw score
        assert all(isinstance(t, DecisionTreeRegressor) for t in exported)
        raw = sum(t.predict(X) for t in exported).reshape(len(X), -1)
        ours, theirs = (softmax(raw, axis=1), m.predict_proba(X)) if clf else (raw.ravel(), m.predict(X))
    else:
        kinds = (DecisionTreeClassifier, RandomForestClassifier) if clf else (DecisionTreeRegressor, RandomForestRegressor)
        assert isinstance(exported, kinds)
        ours, theirs = (exported.predict_proba(X), m.predict_proba(X)) if clf else (exported.predict(X), m.predict(X))
        np.testing.assert_array_equal(exported.predict(X), m.predict(X))
    np.testing.assert_allclose(ours, theirs, atol=1e-10)


@pytest.mark.parametrize("name", ["TaoTreeClassifier", "C45TreeClassifier"])
def test_counts_are_recounted_on_X(name):
    """TaoTree rewrites splits after counting and C4.5 stores no counts: X gives the real ones."""
    m, X, _ = fitted(name)
    tree = to_sklearn(m, X).tree_
    assert tree.n_node_samples[0] == len(X)
    left, right = tree.children_left, tree.children_right
    inner = left >= 0
    np.testing.assert_array_equal(tree.n_node_samples[inner],
                                  tree.n_node_samples[left[inner]] + tree.n_node_samples[right[inner]])


def test_export_is_a_copy():
    m, X, _ = fitted("HSTreeClassifier")
    before = m.estimator_.tree_.n_node_samples.copy()
    to_sklearn(m, X.iloc[:50])
    np.testing.assert_array_equal(m.estimator_.tree_.n_node_samples, before)


def test_export_text_works():
    m, X, _ = fitted("C45TreeClassifier")
    assert "class" in export_text(to_sklearn(m), feature_names=list(X.columns))


def test_not_a_tree_model():
    m = imodels.RuleFitRegressor(max_rules=5).fit(*diabetes)
    with pytest.raises(TypeError):
        to_sklearn(m)


@pytest.mark.parametrize("name", ["C45TreeClassifier", "FastSmallTreeClassifier", "TaoTreeClassifier",
                                  "IRFClassifier", "HSTreeRegressor", "FIGSClassifier"])
def test_dtreeviz_draws_every_tree_model(name):
    """The export makes every tree-based model drawable with dtreeviz."""
    dtreeviz = pytest.importorskip("dtreeviz")
    m, X, y = fitted(name)
    exported = to_sklearn(m, X)
    tree = exported[0] if isinstance(exported, list) else getattr(exported, "estimators_", [exported])[0]
    kw = dict(class_names=[str(c) for c in m.classes_]) if isinstance(tree, DecisionTreeClassifier) else {}
    viz = dtreeviz.model(tree, X, y, feature_names=list(X.columns), target_name="y", **kw)
    viz.leaf_sizes()  # matplotlib only
    if shutil.which("dot"):  # the full tree drawing needs the Graphviz binaries
        assert "<svg" in viz.view(depth_range_to_display=(0, 2)).svg()
