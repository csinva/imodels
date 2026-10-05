import itertools

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

pytest.importorskip("numba")

from imodels import FastRiskScoreClassifier  # noqa: E402
from imodels.algebraic.risk_score.fast_risk_score import calibrate  # noqa: E402
from imodels.algebraic.risk_score.solver import solve  # noqa: E402


@pytest.fixture(scope="module")
def cancer():
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    return train_test_split(X, y, random_state=42)


def test_points_respect_the_constraints(cancer):
    X_train, X_test, y_train, y_test = cancer
    for k, max_points in [(1, 5), (3, 5), (5, 3), (7, 1)]:
        m = FastRiskScoreClassifier(k=k, max_points=max_points).fit(X_train, y_train)
        assert 1 <= len(m.points_) <= k
        assert np.all(np.abs(m.coef_) <= max_points)
        assert m.coef_.dtype.kind == "i"
        assert m.scale_ >= 0
        assert roc_auc_score(y_test, m.predict_proba(X_test)[:, 1]) > 0.9


def test_probabilities_and_labels(cancer):
    X_train, X_test, y_train, _ = cancer
    labels = np.where(y_train == 1, "benign", "malignant")
    m = FastRiskScoreClassifier(k=3).fit(X_train, labels)
    proba = m.predict_proba(X_test)
    assert proba.shape == (len(X_test), 2)
    np.testing.assert_allclose(proba.sum(1), 1)
    assert set(m.predict(X_test)) <= {"benign", "malignant"}
    np.testing.assert_allclose(m.risk(m.total_score(X_test)), proba[:, 1])
    assert "add the points" in str(m)


def test_training_loss_is_the_calibrated_loss(cancer):
    X_train, _, y_train, _ = cancer
    m = FastRiskScoreClassifier(k=4).fit(X_train, y_train)
    p = m.predict_proba(X_train)[:, 1]
    loss = -np.mean(np.where(y_train == 1, np.log(p), np.log(1 - p)))
    assert abs(loss - m.train_loss_) < 1e-8


def test_categorical_and_missing_columns():
    rng = np.random.default_rng(0)
    n = 600
    color = rng.choice(["red", "green", "blue"], n)
    age = rng.normal(50, 10, n)
    age[rng.random(n) < 0.1] = np.nan
    logit = 1.5 * (color == "red") + 0.08 * np.nan_to_num(age - 50) - 0.5
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    X = pd.DataFrame({"color": color, "age": age})
    m = FastRiskScoreClassifier(k=3).fit(X, y)
    assert any(name.startswith("color") for name in m.points_)
    assert m.predict_proba(X.iloc[:5]).shape == (5, 2)
    # unseen category and missing values at predict time
    Xnew = pd.DataFrame({"color": ["purple", "red"], "age": [np.nan, 70.0]})
    assert np.all(np.isfinite(m.predict_proba(Xnew)))


def test_binarize_false_uses_columns_as_given():
    rng = np.random.default_rng(1)
    X = (rng.random((400, 6)) < 0.5).astype(float)
    y = (X[:, 0] + X[:, 1] - X[:, 2] + 0.5 * rng.standard_normal(400) > 0.5).astype(int)
    m = FastRiskScoreClassifier(k=3, binarize=False).fit(X, y)
    assert m.features_ == [f"X{i}" for i in range(6)]
    assert set(m.points_) <= {"X0", "X1", "X2"}
    assert m.profile_ == "decile"


def test_n_thresholds_selects_the_solver_profile(cancer):
    X_train, X_test, y_train, y_test = cancer
    m = FastRiskScoreClassifier(k=3).fit(X_train, y_train)
    assert m.profile_ == "decile"
    n_decile_features = len(m.features_)
    m = FastRiskScoreClassifier(k=3, n_thresholds=99).fit(X_train, y_train)
    assert m.profile_ == "fine"
    assert len(m.features_) > 5 * n_decile_features  # 99 thresholds per numeric column instead of 9
    assert 1 <= len(m.points_) <= 3 and np.all(np.abs(m.coef_) <= 5)
    assert roc_auc_score(y_test, m.predict_proba(X_test)[:, 1]) > 0.9


def test_unknown_profile_raises():
    with pytest.raises(ValueError, match="profile"):
        solve(np.eye(4), [0, 1, 0, 1], 2, profile="coarse")


def _brute_force(X, y, k, bound):
    best = np.inf
    d = X.shape[1]
    for size in range(1, k + 1):
        for S in itertools.combinations(range(d), size):
            for vals in itertools.product([v for v in range(-bound, bound + 1) if v], repeat=size):
                w = np.zeros(d)
                w[list(S)] = vals
                best = min(best, calibrate(X @ w, y)[2])
    return best


@pytest.mark.parametrize("seed", range(6))
def test_close_to_exhaustive_search_on_small_problems(seed):
    """On problems small enough to enumerate every integer point vector, the search's loss is
    within 1e-3 of the optimum (it is a heuristic, so equality is not guaranteed)."""
    rng = np.random.default_rng(seed)
    n, d, k, bound = 300, 6, 2, 3
    X = (rng.random((n, d)) < rng.uniform(0.2, 0.7, d)).astype(float)
    logit = X @ rng.normal(0, 1.5, d) - 1
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    points, loss, _, _ = solve(X, y, k, bound=bound, time_limit=30)
    assert np.count_nonzero(points) <= k and np.all(np.abs(points) <= bound)
    assert abs(calibrate(X @ points, y)[2] - loss) < 1e-8
    assert loss <= _brute_force(X, y, k, bound) + 1e-3
