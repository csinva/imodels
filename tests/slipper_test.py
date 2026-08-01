"""Tests for SLIPPER, which boosts single-rule estimators."""

import random

import numpy as np
import pytest

from imodels import SlipperClassifier
from imodels.rule_set.slipper_util import SlipperBaseEstimator

N_SAMPLES = 300


def _data(seed=0):
    rng = np.random.RandomState(seed)
    X = rng.randn(N_SAMPLES, 4)
    return X, (X[:, 0] > 0).astype(int)


def test_random_state_makes_it_reproducible():
    """random_state must control every random draw

    The rule learner drew from python's `random` module and from an unseeded
    train_test_split, neither of which random_state reaches, so the same
    random_state gave different models.
    """
    X, y = _data()

    first = SlipperClassifier(n_estimators=5, random_state=0).fit(X, y)

    # a different global random state must not change the result
    random.seed(123)
    np.random.seed(123)
    second = SlipperClassifier(n_estimators=5, random_state=0).fit(X, y)
    assert np.array_equal(first.predict(X), second.predict(X))

    # but a different random_state should
    other = SlipperClassifier(n_estimators=5, random_state=3).fit(X, y)
    assert not np.array_equal(first.predict(X), other.predict(X))


def test_refitting_starts_over():
    """Refitting on new data gives the same model as fitting it fresh"""
    X, y = _data()
    X_other = np.random.RandomState(7).randn(N_SAMPLES, 4)
    y_other = (X_other[:, 1] > 0).astype(int)

    refit = SlipperClassifier(n_estimators=5, random_state=0).fit(X, y)
    refit.fit(X_other, y_other)
    fresh = SlipperClassifier(n_estimators=5, random_state=0).fit(X_other, y_other)

    assert np.array_equal(refit.predict(X_other), fresh.predict(X_other))


def test_base_estimator_probabilities_are_a_distribution():
    """Each rule's predict_proba must be a valid distribution

    It returned [0, prediction], so rows did not sum to 1 and the first column
    was always zero. Boosting derives its sample weights from these, so the
    later estimators were fitted against corrupted weights.
    """
    X, y = _data()
    model = SlipperClassifier(n_estimators=3, random_state=0).fit(X, y)

    for estimator in model.estimators_:
        proba = estimator.predict_proba(X)
        assert proba.shape == (N_SAMPLES, 2)
        assert np.allclose(proba.sum(axis=1), 1)
        assert proba.min() >= 0 and proba.max() <= 1


def test_adding_estimators_does_not_destroy_accuracy():
    """More boosting rounds should not make the model worse than one rule

    Accuracy used to collapse from 0.99 with a single rule to below chance with
    three, because of the invalid probabilities above.
    """
    X, y = _data()
    single = SlipperClassifier(n_estimators=1, random_state=0).fit(X, y)
    several = SlipperClassifier(n_estimators=10, random_state=0).fit(X, y)

    baseline = np.mean(single.predict(X) == y)
    assert baseline > 0.9, 'a single rule should already fit this easy task'
    assert np.mean(several.predict(X) == y) > 0.9

    proba = several.predict_proba(X)
    assert np.allclose(proba.sum(axis=1), 1)


def test_base_estimator_works_without_sample_weight():
    """The rule learner defaults to uniform weights instead of failing"""
    X, y = _data()
    estimator = SlipperBaseEstimator(random_state=0).fit(X, y)
    assert np.mean(estimator.predict(X) == y) > 0.9
