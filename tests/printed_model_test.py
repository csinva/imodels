"""Printing a model should name the caller's features."""

import numpy as np
import pandas as pd

import imodels

FEATURE_NAMES = ['age', 'bmi', 'bp', 'chol']


def _frame(binary=False):
    rng = np.random.RandomState(0)
    X = pd.DataFrame(rng.randn(200, 4), columns=FEATURE_NAMES)
    if binary:
        X = (X > 0).astype(int)
    return X, (X['age'] > (0.5 if binary else 0)).astype(int)


def test_greedy_tree_prints_feature_names():
    """The classifier printed feature_0 despite knowing the column names

    The regressor stored self.feature_names and passed it to export_text; the
    classifier never did, so export_text generated its own placeholders.
    """
    X, y = _frame()
    printed = str(imodels.GreedyTreeClassifier(max_leaf_nodes=3).fit(X, y))

    assert 'age' in printed
    assert 'feature_0' not in printed

    # numpy input still prints something sensible
    numpy_printed = str(imodels.GreedyTreeClassifier(
        max_leaf_nodes=3).fit(X.to_numpy(), y))
    assert 'X0' in numpy_printed or 'feature_0' in numpy_printed


def test_bayesian_rule_list_prints_feature_names():
    """BRL discarded DataFrame columns and printed X_0, X_1, ..."""
    X, y = _frame(binary=True)
    model = imodels.BayesianRuleListClassifier(
        max_iter=2000, n_chains=1, random_state=0).fit(X, y)

    printed = str(model)
    assert 'age' in printed
    assert 'X_0' not in printed
    assert any('age' in str(rule) for rule in model.rules_)


def test_explicit_feature_names_still_win():
    """Names passed to fit take precedence over the frame's columns"""
    X, y = _frame(binary=True)
    names = ['a1', 'a2', 'a3', 'a4']
    model = imodels.BayesianRuleListClassifier(
        max_iter=2000, n_chains=1, random_state=0).fit(X, y, feature_names=names)
    assert 'a1' in str(model)
