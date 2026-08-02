"""Tests for the C4.5 tree, whose tree is stored as XML."""

import numpy as np
import pandas as pd

from imodels import C45TreeClassifier


def test_feature_names_that_reduce_to_the_same_string():
    """Distinct columns must stay distinct internally

    The tree is XML, so names are stripped to valid element names. Two
    different columns could reduce to the same name, and the tree then read
    whichever came first -- splitting on one column but testing another.
    """
    rng = np.random.RandomState(0)
    X = pd.DataFrame(rng.randn(200, 4),
                     columns=['age (years)', 'age-years', 'other', 'last'])
    X['age (years)'] = rng.randn(200)
    X['age-years'] = rng.randn(200) * 5 + 10
    y = (X['age-years'] > 10).astype(int)

    model = C45TreeClassifier(max_rules=4).fit(X, y)

    assert len(set(model.feature_names)) == X.shape[1], 'names collided'
    # with the columns confused this scored 0.445, below chance
    assert np.mean(model.predict(X) == y) > 0.9


def test_rules_report_the_original_feature_names():
    """Reported rules should name the caller's columns, not internal ones"""
    rng = np.random.RandomState(0)
    X = pd.DataFrame(rng.randn(200, 3),
                     columns=['age (years)', 'blood pressure', 'weight-kg'])
    y = (X['age (years)'] > 0).astype(int)

    rules = C45TreeClassifier(max_rules=3).fit(X, y).get_rules()
    text = ' '.join(rules['rule'])
    assert 'age (years)' in text
    assert 'ageyears' not in text.replace('age (years)', '')


def test_awkward_feature_names_are_accepted():
    """Names starting with a digit, or with spaces, still work"""
    rng = np.random.RandomState(0)
    X = pd.DataFrame(rng.randn(150, 3), columns=['a b', '2bad', 'fine'])
    y = (X['a b'] > 0).astype(int)

    model = C45TreeClassifier(max_rules=3).fit(X, y)
    assert model.predict(X).shape == (150,)
    assert np.mean(model.predict(X) == y) > 0.9
