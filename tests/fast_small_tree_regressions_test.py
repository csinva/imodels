"""Regression tests for FastSmallTreeClassifier bugs found in the 3.0.3 maintenance sweep.

Each test reproduces one confirmed bug on tiny data; the numbers in the names refer
to the findings of that sweep.
"""

import time
import warnings

import numpy as np
import pandas as pd
import pytest
import scipy.sparse

pytest.importorskip("numba")

from imodels import FastSmallTreeClassifier
from imodels.tree.optimal_tree.solver import TreeClassifier


def certified_codes(model, X):
    """Class codes from the solver's own traversal of the certified rules."""
    frame = pd.DataFrame(np.asarray(X, dtype=float))
    return TreeClassifier(model.tree_).predict_fast(frame).astype(int)


# 1 ---------------------------------------------------------------------------

def test_1_time_limit_interrupts_a_long_root_expansion():
    """The limit used to be checked only after 500 expansions, and one expansion of
    the root of wide continuous data takes seconds, so fits overshot by minutes."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(1000, 8))
    y = (X[:, 0] * X[:, 1] + 0.5 * rng.normal(size=1000) > 0).astype(int)
    model = FastSmallTreeClassifier(regularization=0.002, time_limit=0.2)
    with pytest.warns(RuntimeWarning, match="time limit reached"):
        model.fit(X, y)
    assert model.stop_reason_ == "time"
    assert model.time_ < 0.2 + 0.5
    assert not model.optimal_
    assert model.lowerbound_ <= model.upperbound_ + 1e-12
    assert model.objective_ == pytest.approx(model.upperbound_)
    assert np.isin(model.predict(X), [0, 1]).all()


def test_1_memory_limit_stops_the_search():
    """memory_limit is checked before the first expansion and when the store grows,
    not only between chunks of 500 expansions."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 4))
    y = (X[:, 0] * X[:, 1] > 0).astype(int)
    start = time.perf_counter()
    with pytest.warns(RuntimeWarning, match="memory limit reached"):
        model = FastSmallTreeClassifier(regularization=0.002, time_limit=30,
                                        memory_limit=1).fit(X, y)
    assert model.stop_reason_ == "memory"
    assert time.perf_counter() - start < 10
    assert model.n_leaves_ >= 1


def test_1_time_limit_leaves_easy_fits_certified():
    """A small problem still finishes, and is certified, well within its limit."""
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]] * 8)
    y = (X[:, 0] == X[:, 1]).astype(int)
    model = FastSmallTreeClassifier(regularization=0.01, time_limit=5).fit(X, y)
    assert model.optimal_ and model.stop_reason_ == "optimal"
    assert model.n_leaves_ == 4


# 2 ---------------------------------------------------------------------------

@pytest.mark.parametrize("lam", [0.0, 0.01, 0.1])
def test_2_balance_exact_tie_does_not_crash(lam):
    """A leaf with class counts (1, 2, 11) under balance weighs classes 1 and 2 at
    2/6 and 11/33, an exact tie that the solver and the sklearn leaf value used
    to round to different classes, failing fit with an AssertionError."""
    y = np.array([0, 0, 1, 1] + [2] * 11)
    x = np.array([1, 0, 1, 1] + [1] * 11)
    model = FastSmallTreeClassifier(balance=True, regularization=lam).fit(np.c_[x], y)
    assert (model.target_encoder_.transform(model.predict(np.c_[x]))
            == certified_codes(model, np.c_[x])).all()
    assert (model.estimator_.predict(np.c_[x]) == model.predict(np.c_[x])).all()


# 3 ---------------------------------------------------------------------------

@pytest.mark.parametrize("X, y", [
    (np.array([[1e8], [1e8 + 1], [1e8 + 2], [1e8 + 3]]), np.array([0, 1, 0, 1])),
    ((1.7e9 + np.arange(0, 2000, 10.0))[:, None],
     (1.7e9 + np.arange(0, 2000, 10.0) >= 1.700001005e9).astype(int)),
], ids=["1e8", "timestamps"])
def test_3_predict_matches_certified_tree_beyond_float32(X, y):
    """predict went through sklearn's float32 comparison, so on large-magnitude
    columns it disagreed with the certified tree on the training rows."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # the estimator_ float32 notice
        model = FastSmallTreeClassifier(regularization=0.01).fit(X, y)
    assert (model.target_encoder_.transform(model.predict(X)) == certified_codes(model, X)).all()
    assert (model.predict(X) == y).all()
    proba = model.predict_proba(X)
    assert (model.classes_[proba.argmax(axis=1)] == model.predict(X)).all()


def test_3_estimator_float32_mismatch_is_warned_and_predict_unaffected():
    X = np.array([[1e8], [1e8 + 1], [1e8 + 2], [1e8 + 3]])
    y = np.array([0, 1, 0, 1])
    with pytest.warns(RuntimeWarning, match="estimator_.predict"):
        model = FastSmallTreeClassifier(regularization=0.01).fit(X, y)
    assert (model.predict(X) == y).all()


# 6 ---------------------------------------------------------------------------

def test_6_string_categorical_fails_before_the_search():
    x = np.random.default_rng(0).normal(size=40)
    frame = pd.DataFrame({"a": pd.Categorical(np.where(x > 0, "u", "v"))})
    with pytest.raises(ValueError, match="numeric features"):
        FastSmallTreeClassifier().fit(frame, (x > 0).astype(int))


def test_6_numeric_categorical_is_used_by_value():
    x = np.random.default_rng(0).normal(size=40)
    y = (x > 0).astype(int)
    frame = pd.DataFrame({"a": pd.Categorical(np.where(x > 0, 1, 2))})
    model = FastSmallTreeClassifier(regularization=0.05).fit(frame, y)
    assert (model.predict(frame) == y).all()


# 7 ---------------------------------------------------------------------------

def test_7_duplicate_feature_names():
    rng = np.random.default_rng(0)
    X = np.c_[rng.normal(size=40), rng.normal(size=40)]
    y = (X[:, 1] > 0).astype(int)
    model = FastSmallTreeClassifier(regularization=0.05).fit(X, y, feature_names=["a", "a"])
    assert (model.predict(X) == y).all()
    frame = pd.DataFrame(X, columns=["a", "a"])
    try:
        model = FastSmallTreeClassifier(regularization=0.05).fit(frame, y)
        pred = model.predict(frame)
    except Exception as err:  # newer sklearn rejects duplicate DataFrame columns itself
        assert "unique" in str(err).lower(), err
        return
    assert (pred == y).all()


# 8 ---------------------------------------------------------------------------

@pytest.mark.parametrize("kwargs", [
    {"time_limit": -1}, {"time_limit": np.nan}, {"time_limit": np.inf}, {"time_limit": "x"},
    {"memory_limit": -5}, {"memory_limit": np.nan},
])
def test_8_invalid_limits_are_rejected(kwargs):
    X = np.array([[0], [1], [0], [1]])
    y = np.array([0, 1, 0, 1])
    with pytest.raises(ValueError, match="_limit"):
        FastSmallTreeClassifier(**kwargs).fit(X, y)


@pytest.mark.parametrize("time_limit", [0, None])
def test_8_no_time_limit_still_works(time_limit):
    X = np.array([[0], [1], [0], [1]])
    y = np.array([0, 1, 0, 1])
    model = FastSmallTreeClassifier(time_limit=time_limit).fit(X, y)
    assert model.optimal_


def test_8_sparse_X_in_fit_and_predict():
    """fit densified sparse X, but predict failed on it with an unrelated error."""
    rng = np.random.default_rng(0)
    X = rng.integers(0, 2, size=(40, 3)).astype(float)
    y = X[:, 0].astype(int)
    model = FastSmallTreeClassifier(regularization=0.05).fit(scipy.sparse.csr_matrix(X), y)
    assert (model.predict(scipy.sparse.csr_matrix(X)) == model.predict(X)).all()
    assert np.allclose(model.predict_proba(scipy.sparse.csc_matrix(X)), model.predict_proba(X))
