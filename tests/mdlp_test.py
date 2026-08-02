"""Tests for the MDLP discretizer."""

import contextlib
import io

import numpy as np
import pandas as pd

from imodels.discretization.mdlp import MDLPDiscretizer


def _frame(with_text_column=False):
    rng = np.random.RandomState(0)
    X = pd.DataFrame(rng.randn(200, 3), columns=list('abc'))
    X['target'] = (X['a'] > 0).astype(int)
    if with_text_column:
        X['label'] = ['x', 'y'] * 100
    return X


def _build(frame):
    with contextlib.redirect_stdout(io.StringIO()):  # it logs while discretizing
        return MDLPDiscretizer(dataset=frame, class_label='target')


def test_selects_numeric_features_without_private_pandas_apis():
    """Numeric columns were found via DataFrame._data

    That is a pandas internal: deprecated in pandas 2 and removed in pandas 3,
    where constructing the discretizer failed outright with
    AttributeError: 'DataFrame' object has no attribute '_data'.
    """
    discretizer = _build(_frame())
    assert sorted(discretizer._features) == ['a', 'b', 'c']


def test_non_numeric_columns_are_left_alone():
    discretizer = _build(_frame(with_text_column=True))
    assert sorted(discretizer._features) == ['a', 'b', 'c']
    assert 'label' in discretizer._ignored_features
    assert 'target' in discretizer._ignored_features


def test_explicit_features_still_honored():
    discretizer = _build(_frame())
    explicit = MDLPDiscretizer.__new__(MDLPDiscretizer)
    with contextlib.redirect_stdout(io.StringIO()):
        explicit.__init__(dataset=_frame(), class_label='target', features=['a'])
    assert list(explicit._features) == ['a']
    assert 'b' in explicit._ignored_features
