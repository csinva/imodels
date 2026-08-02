"""Tests for converting fitted trees into rule strings."""

import numpy as np
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeClassifier

from imodels.util.convert import tree_to_rules


def _rows_selected(X, rule):
    return int(X.eval(rule.replace(' and ', ' & ')).sum())


@pytest.mark.parametrize('scale', [1e-6, 1e-3, 1.0, 1e3, 1e6],
                         ids=['1e-6', '1e-3', '1', '1e3', '1e6'])
def test_rounded_rules_select_the_same_rows(scale):
    """Shortening a threshold must not move it onto the data

    Thresholds were rounded to 5 decimal places, so on small-scale features
    they collapsed to 0.0 and the rule selected entirely different rows -- 134
    instead of 9 in one case.
    """
    rng = np.random.RandomState(0)
    values = rng.randn(300, 2) * scale
    y = (values[:, 0] > 0).astype(int)
    tree = DecisionTreeClassifier(max_depth=2, random_state=0).fit(values, y)

    X = pd.DataFrame(values, columns=['a', 'b'])
    rounded = tree_to_rules(tree, ['a', 'b'], round_thresholds=True)
    exact = tree_to_rules(tree, ['a', 'b'], round_thresholds=False)

    assert len(rounded) == len(exact)
    for short, full in zip(rounded, exact):
        assert _rows_selected(X, short) == _rows_selected(X, full), (
            f'{short!r} and {full!r} disagree')


def test_thresholds_stay_short_at_ordinary_scales():
    """The shortening should still shorten: no full float repr for normal data"""
    rng = np.random.RandomState(0)
    values = rng.randn(300, 2)
    y = (values[:, 0] > 0).astype(int)
    tree = DecisionTreeClassifier(max_depth=2, random_state=0).fit(values, y)

    for rule in tree_to_rules(tree, ['a', 'b'], round_thresholds=True):
        for token in rule.replace(' and ', ' ').split():
            if token.replace('.', '').replace('-', '').isdigit() and '.' in token:
                decimals = len(token.split('.')[1])
                assert decimals <= 8, f'{token} in {rule!r} was not shortened'


def test_tree_with_no_splits():
    """A tree that never splits yields the catch-all rule, not a crash"""
    rng = np.random.RandomState(0)
    X = rng.randn(200, 3)
    tree = DecisionTreeClassifier(max_depth=3).fit(X, np.zeros(200, dtype=int))

    rules = tree_to_rules(tree, ['a', 'b', 'c'])
    assert rules == ['a == a']
