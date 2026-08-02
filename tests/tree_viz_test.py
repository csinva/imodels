"""Plotting imodels trees with dtreeviz (issue #135).

dtreeviz is not a dependency of imodels, so these tests skip when it isn't installed.
"""

import numpy as np
import pandas as pd
import pytest

import imodels

pytest.importorskip('dtreeviz')

FEATURE_NAMES = ['age', 'bmi', 'bp', 'chol']


def _data():
    rng = np.random.RandomState(0)
    X = pd.DataFrame(rng.randn(300, 4), columns=FEATURE_NAMES)
    y = ((X['age'] + 1.5 * rng.randn(300)) > 0).astype(int)
    return X, y


@pytest.mark.parametrize('model_name,kwargs', [
    ('FIGSClassifier', dict(max_rules=6)),
    ('HSTreeClassifier', {}),
    ('GreedyTreeClassifier', dict(max_leaf_nodes=5)),
    ('TaoTreeClassifier', {}),
])
def test_shadow_tree_from_classifiers(model_name, kwargs):
    """A ShadowSKDTree can be built from each tree-based classifier"""
    from dtreeviz.models.sklearn_decision_trees import ShadowSKDTree

    X, y = _data()
    model = getattr(imodels, model_name)(**kwargs).fit(X, y)

    shadow = imodels.shadow_tree(model, X, y)
    assert isinstance(shadow, ShadowSKDTree)
    assert shadow.nnodes() >= 1
    assert len(shadow.leaves) >= 1
    # the model's own column names are used
    assert shadow.feature_names == FEATURE_NAMES


def test_each_tree_of_an_ensemble_can_be_drawn():
    """Models made of several trees expose them individually via tree_num"""
    X, y = _data()
    model = imodels.BoostedRulesClassifier(n_estimators=3).fit(X, y)
    assert len(model.estimators_) == 3

    for tree_num in range(3):
        shadow = imodels.shadow_tree(model, X, y, tree_num=tree_num)
        assert shadow.nnodes() >= 1


def test_shadow_tree_for_a_regressor():
    X, y = _data()
    model = imodels.FIGSRegressor(max_rules=4).fit(X, X['age'])

    shadow = imodels.shadow_tree(model, X, X['age'])
    assert shadow.nnodes() >= 1
    assert not shadow.is_classifier()


def test_feature_and_class_names_can_be_overridden():
    X, y = _data()
    model = imodels.GreedyTreeClassifier(max_leaf_nodes=4).fit(X, y)

    shadow = imodels.shadow_tree(
        model, X, y, feature_names=list('abcd'),
        target_name='outcome', class_names=['no', 'yes'])
    assert shadow.feature_names == list('abcd')
    assert shadow.target_name == 'outcome'


def test_errors_are_clear():
    X, y = _data()

    with pytest.raises(ValueError, match='tree-based'):
        imodels.shadow_tree(imodels.SLIMClassifier().fit(X.values, y), X, y)

    model = imodels.FIGSClassifier(max_rules=4).fit(X, y)
    with pytest.raises(ValueError, match='tree_num'):
        imodels.shadow_tree(model, X, y, tree_num=99)


def test_renders_an_svg():
    """The shadow tree actually renders, not just constructs

    Needs the graphviz binaries, so skipped where `dot` isn't on PATH.
    """
    import shutil

    import dtreeviz

    if shutil.which('dot') is None:
        pytest.skip('graphviz (dot) is not installed')

    X, y = _data()
    model = imodels.FIGSClassifier(max_rules=6).fit(X, y)

    viz = dtreeviz.trees.DTreeVizAPI(imodels.shadow_tree(model, X, y))
    svg = viz.view().svg()

    assert svg.lstrip().startswith('<svg')
    assert len(svg) > 1000, 'expected a non-trivial drawing'
    # the split features the model actually used appear in the drawing
    used = {rule.split()[0] for rule in model.get_rules()['rule']
            if rule != 'else'}
    assert any(feature in svg for feature in used), used
