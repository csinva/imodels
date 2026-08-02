"""Fast-and-frugal trees (issue #56)."""

import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from imodels import FastFrugalTreeClassifier


def _data():
    X, y = load_breast_cancer(return_X_y=True)
    return train_test_split(X, y, random_state=0)


def test_shape_of_the_tree():
    """Every cue has an exit, and the list ends with a catch-all"""
    X_train, X_test, y_train, y_test = _data()
    model = FastFrugalTreeClassifier(max_depth=4).fit(X_train, y_train)

    assert len(model.rules_) <= 4
    # all but the last rule test a cue and exit to a class
    for rule in model.rules_[:-1]:
        assert {'col', 'index_col', 'cutoff', 'flip', 'val_right'} <= set(rule)
        assert rule['val_right'] in (0.0, 1.0)
    # the last is the catch-all
    assert 'col' not in model.rules_[-1]


@pytest.mark.parametrize('max_depth', [1, 2, 3, 4, 6])
def test_respects_max_depth(max_depth):
    """It never asks more questions than allowed"""
    X_train, X_test, y_train, y_test = _data()
    model = FastFrugalTreeClassifier(max_depth=max_depth).fit(X_train, y_train)

    assert len(model.rules_) <= max_depth
    assert model.predict(X_test).shape == (len(X_test),)


def test_accuracy_is_competitive():
    """A handful of questions should still classify well"""
    X_train, X_test, y_train, y_test = _data()
    model = FastFrugalTreeClassifier(max_depth=4).fit(X_train, y_train)

    assert accuracy_score(y_test, model.predict(X_test)) > 0.9


def test_predictions_follow_the_rules_in_order():
    """A case is decided by the first cue that catches it"""
    X_train, X_test, y_train, y_test = _data()
    model = FastFrugalTreeClassifier(max_depth=4).fit(X_train, y_train)

    probs = model.predict_proba(X_test)
    assert probs.shape == (len(X_test), 2)
    assert np.allclose(probs.sum(axis=1), 1)
    # exits are decisions, so the probabilities are 0/1
    assert set(np.unique(probs)) <= {0.0, 1.0}

    # reproduce the walk by hand
    expected = np.zeros(len(X_test))
    undecided = np.ones(len(X_test), dtype=bool)
    for rule in model.rules_:
        if 'col' not in rule:
            expected[undecided] = rule['val']
            break
        above = X_test[:, rule['index_col']] > rule['cutoff']
        exits = (~above if rule['flip'] else above) & undecided
        expected[exits] = rule['val_right']
        undecided &= ~exits
    assert np.allclose(expected, probs[:, 1])


def test_feature_names_and_printing():
    X_train, X_test, y_train, y_test = _data()
    names = list(load_breast_cancer().feature_names)
    model = FastFrugalTreeClassifier(max_depth=3).fit(
        X_train, y_train, feature_names=names)

    printed = str(model)
    assert 'Fast-and-frugal tree' in printed
    assert any(name in printed for name in names)

    # get_rules works through the shared rule-list API
    rules = model.get_rules()
    assert list(rules.columns[:2]) == ['rule', 'prediction']
    assert len(rules) == len(model.rules_)


def test_string_labels():
    X_train, X_test, y_train, y_test = _data()
    labels = np.where(y_train == 1, 'benign', 'malignant')

    model = FastFrugalTreeClassifier(max_depth=3).fit(X_train, labels)
    assert set(model.classes_) == {'benign', 'malignant'}
    assert set(np.unique(model.predict(X_test))) <= {'benign', 'malignant'}
