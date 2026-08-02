"""Tests for explain_classification_errors."""

import contextlib
import io

import numpy as np

from imodels import GreedyTreeClassifier, explain_classification_errors


def _case(seed, signal_column):
    rng = np.random.RandomState(seed)
    X = rng.randn(200, 3)
    y = (X[:, signal_column] > 0).astype(int)
    predictions = (X[:, signal_column] + 0.5 * rng.randn(200) > 0).astype(int)
    return X, predictions, y


def _explain(X, predictions, y, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return explain_classification_errors(X, predictions, y, **kwargs)


def test_each_call_gets_its_own_classifier():
    """Two explanations must not share one model

    The default classifier was constructed in the function signature, so it was
    created once at import and every call refitted the same object: the first
    explanation silently changed when the second one was made.
    """
    X1, p1, y1 = _case(0, 0)
    X2, p2, y2 = _case(1, 1)

    first, _ = _explain(X1, p1, y1)
    before = first.predict(np.c_[X1, y1]).copy()

    second, _ = _explain(X2, p2, y2)
    after = first.predict(np.c_[X1, y1])

    assert first is not second
    assert np.array_equal(before, after), 'the first explanation was overwritten'


def test_explicit_classifier_is_used():
    X, predictions, y = _case(0, 0)
    mine = GreedyTreeClassifier(max_depth=2)
    returned, _ = _explain(X, predictions, y, classifier=mine)
    assert returned is mine


def test_returns_feature_and_target_names():
    X, predictions, y = _case(0, 0)
    _, columns = _explain(X, predictions, y)
    assert list(columns) == ['X1', 'X2', 'X3', 'target']
