"""Checks that models advertising sample_weight actually use it.

A model that accepts sample_weight and quietly discards it is worse than one
that refuses it: the caller believes the data has been reweighted. The test
below makes the weights impossible to ignore -- half the labels are corrupted
and then given zero weight -- so a model that honors them recovers the clean
rule and one that doesn't cannot.
"""

import contextlib
import inspect
import io

import numpy as np
import pandas as pd
import pytest

import imodels
from tests.model_configs import (BINARY_INPUT_MODELS, EXCLUDED_MODELS,
                                 MODEL_KWARGS)

N = 300

WEIGHTED_MODELS = [
    m for m in imodels.ESTIMATORS
    if m.__name__ not in EXCLUDED_MODELS
    and 'sample_weight' in inspect.signature(m.fit).parameters
]


def _data(model_type):
    rng = np.random.RandomState(0)
    X = pd.DataFrame(rng.randn(N, 4), columns=list('abcd'))
    clean = (X['a'] > 0).astype(int)
    y = clean.copy()
    y.iloc[N // 2:] = 1 - y.iloc[N // 2:]  # second half is garbage
    weight = np.concatenate([np.ones(N // 2), np.zeros(N // 2)])
    if model_type.__name__ in BINARY_INPUT_MODELS:
        X = (X > 0).astype(int)
    if model_type not in imodels.CLASSIFIERS:
        y = y.astype(float)
    return X, y, clean, weight


def _accuracy(model, X, clean):
    """Agreement with the uncorrupted rule, on the half that was trusted"""
    predictions = np.asarray(model.predict(X), dtype=float).round()
    return (predictions[:N // 2] == clean[:N // 2]).mean()


@pytest.mark.parametrize('model_type', WEIGHTED_MODELS,
                         ids=[m.__name__ for m in WEIGHTED_MODELS])
def test_sample_weight_is_not_ignored(model_type):
    X, y, clean, weight = _data(model_type)
    kwargs = MODEL_KWARGS.get(model_type.__name__, {})

    with contextlib.redirect_stdout(io.StringIO()):
        unweighted = model_type(**kwargs).fit(X, y)
        weighted = model_type(**kwargs).fit(X, y, sample_weight=weight)

    # zeroing out the corrupted half must not leave the model untouched
    assert not np.array_equal(
        np.asarray(unweighted.predict(X), dtype=float),
        np.asarray(weighted.predict(X), dtype=float)), \
        f'{model_type.__name__} accepts sample_weight but ignores it'
    assert _accuracy(weighted, X, clean) >= _accuracy(unweighted, X, clean)


def test_fpskope_selects_rules_by_weighted_precision():
    """FPSkope mines itemsets unsupervised, so weights only reach it in scoring

    It accepted sample_weight, stored it, and used it nowhere: neither its
    itemset mining nor its rule scoring looked at it.
    """
    X, y, clean, weight = _data(imodels.FPSkopeClassifier)

    with contextlib.redirect_stdout(io.StringIO()):
        unweighted = imodels.FPSkopeClassifier(random_state=0).fit(X, y)
        weighted = imodels.FPSkopeClassifier(
            random_state=0).fit(X, y, sample_weight=weight)

    assert _accuracy(weighted, X, clean) > _accuracy(unweighted, X, clean)


@pytest.mark.parametrize('model_name',
                         ['SkopeRulesClassifier', 'FPSkopeClassifier'])
def test_uniform_weights_match_no_weights_in_scoring(model_name):
    """Weighting rule scoring must not perturb the unweighted result"""
    from imodels.util.score import _eval_rule_perf

    rng = np.random.RandomState(0)
    for _ in range(25):
        X = pd.DataFrame(rng.randn(60, 2), columns=['a', 'b'])
        y = np.array(rng.rand(60) > 0.5)
        rule = f'a > {rng.randn():.2f}'
        assert np.allclose(_eval_rule_perf(rule, X, y),
                           _eval_rule_perf(rule, X, y, np.ones(60)))


def test_zero_weight_samples_do_not_count_toward_a_rule():
    """A rule covering only zero-weight samples has no weighted precision"""
    from imodels.util.score import _eval_rule_perf

    X = pd.DataFrame({'a': [1.0, 2.0, 3.0, 4.0]})
    y = np.array([True, True, False, False])

    precision, recall = _eval_rule_perf('a > 0', X, y,
                                        np.array([0.0, 0.0, 1.0, 1.0]))
    assert (precision, recall) == (0, 0)  # both positives were weighted out
