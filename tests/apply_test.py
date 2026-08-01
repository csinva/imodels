"""Tests for the apply() method, which reports leaf membership like sklearn's."""

import numpy as np
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeClassifier

import imodels
from imodels.util.apply import apply_leaves

N_SAMPLES = 300


def _data():
    rng = np.random.RandomState(0)
    X = pd.DataFrame(rng.randn(N_SAMPLES, 4), columns=list('abcd'))
    # additive structure, so FIGS grows more than one tree
    y = ((X['a'] > 0).astype(int) + (X['c'] > 0.5).astype(int) +
         (X['d'] < -0.5).astype(int) >= 2).astype(int)
    return X, y


SINGLE_TREE_MODELS = {
    'GreedyTreeClassifier': dict(max_leaf_nodes=4, random_state=0),
    'HSTreeClassifier': {},
    'TaoTreeClassifier': {},
}
MULTI_TREE_MODELS = {
    'FIGSClassifier': dict(max_rules=8),
    'BoostedRulesClassifier': dict(n_estimators=3),
    'TreeGAMClassifier': dict(n_boosting_rounds=3, random_state=0),
}


@pytest.mark.parametrize('model_name', sorted(SINGLE_TREE_MODELS))
def test_apply_single_tree(model_name):
    """A single-tree model returns one leaf index per sample"""
    X, y = _data()
    model = getattr(imodels, model_name)(**SINGLE_TREE_MODELS[model_name]).fit(X, y)

    leaves = model.apply(X)
    assert leaves.shape == (N_SAMPLES,)
    assert len(np.unique(leaves)) > 1, 'every sample landed in the same leaf'

    # the module-level function agrees
    assert np.array_equal(leaves, apply_leaves(model, X))


@pytest.mark.parametrize('model_name', sorted(MULTI_TREE_MODELS))
def test_apply_multiple_trees(model_name):
    """A model made of several trees returns one column per tree"""
    X, y = _data()
    model = getattr(imodels, model_name)(**MULTI_TREE_MODELS[model_name]).fit(X, y)

    leaves = model.apply(X)
    assert leaves.ndim == 2
    assert leaves.shape[0] == N_SAMPLES
    assert leaves.shape[1] > 1, 'expected this model to hold several trees'


def test_matches_sklearn_for_a_plain_tree():
    """GreedyTree is a CART wrapper, so its leaves match sklearn's exactly"""
    X, y = _data()
    ours = imodels.GreedyTreeClassifier(
        max_leaf_nodes=4, random_state=0).fit(X, y)
    theirs = DecisionTreeClassifier(max_leaf_nodes=4, random_state=0).fit(X, y)

    assert np.array_equal(ours.apply(X), theirs.apply(X))


def test_samples_in_a_leaf_share_a_prediction():
    """Leaf membership lines up with what the model predicts"""
    X, y = _data()
    model = imodels.GreedyTreeClassifier(max_leaf_nodes=4).fit(X, y)

    leaves = model.apply(X)
    probs = model.predict_proba(X)[:, 1]
    for leaf in np.unique(leaves):
        assert len(np.unique(probs[leaves == leaf])) == 1


def test_shrinkage_keeps_the_underlying_leaves():
    """Shrinkage changes leaf values, not which leaf a sample reaches"""
    X, y = _data()
    tree = DecisionTreeClassifier(max_leaf_nodes=6, random_state=0)
    shrunk = imodels.HSTreeClassifier(tree, reg_param=100).fit(X, y)

    unshrunk = DecisionTreeClassifier(
        max_leaf_nodes=6, random_state=0).fit(X, y)
    assert np.array_equal(shrunk.apply(X), unshrunk.apply(X))


def test_unsupported_model_raises_a_clear_error():
    X, y = _data()
    model = imodels.SLIMClassifier().fit(X.values, y)
    with pytest.raises(ValueError, match='SLIMClassifier'):
        apply_leaves(model, X.values)
