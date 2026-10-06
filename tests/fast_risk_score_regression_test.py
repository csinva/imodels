"""Regressions of FastRiskScoreClassifier found by the pre-release bug sweep (one test per finding)."""

import os
import pickle
import subprocess
import sys
import textwrap
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

pytest.importorskip("numba")

from imodels import FastRiskScoreClassifier  # noqa: E402
from imodels.algebraic.risk_score.fast_risk_score import _Binarizer, _threshold  # noqa: E402
from imodels.util import numba_compile  # noqa: E402


def _data(n=400, d=3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = (X[:, 0] - X[:, 1] + rng.normal(size=n) > 0).astype(int)
    return X, y


# 1. rule names are unique, and points_ matches coef_
def test_large_close_values_get_distinct_names():
    rng = np.random.default_rng(0)
    x = 1.7e9 + rng.random(2000) * 1000  # epoch seconds: equal at 6 significant digits
    y = (rng.random(2000) < 1 / (1 + np.exp(-(x - x.mean()) / 100))).astype(int)
    m = FastRiskScoreClassifier(k=5).fit(pd.DataFrame({"t": x}), y)
    assert len(set(m.features_)) == len(m.features_) == len(m.binarizer_.rules_)
    assert len(m.points_) == np.count_nonzero(m.coef_)
    for (_, kind, value, name) in m.binarizer_.rules_:
        assert kind == "le" and float(name.split("<= ")[1]) == value  # applied exactly as printed
    neg = sum(v for v in m.points_.values() if v < 0)
    printed = str(m)
    rules = [l.split(None, 1)[1] for l in printed.splitlines() if l.strip().startswith(("+", "-")) and " t " in f" {l} "]
    assert len(set(rules)) == len(rules) == len(m.points_)  # every rule printed, none alike
    assert f"{neg:+d}" in str(m)  # the score table spans the real totals


def test_threshold_text_separates_from_the_next_value():
    assert _threshold(906.6, 907.4) == (906.6, "906.6")  # short data values keep their usual form
    d, text = _threshold(1700000453.2, 1700000453.9)
    assert text == "1700000453.2" and d == 1700000453.2
    d, text = _threshold(0.1234564, 0.1234566)
    assert 0.1234564 <= d < 0.1234566 and float(text) == d
    d, text = _threshold(1.0, np.nextafter(1.0, 2))  # adjacent floats: repr
    assert d == 1.0 and float(text) == 1.0


def test_colliding_names_across_columns_are_suffixed():
    # level "b missing" of column "a" and the missing values of column "a = b" have the same name,
    # and so do the rules of two columns with the same name
    a = np.array(["b missing", "c"] * 50, dtype=object)
    ab = np.where(np.arange(100) % 3 == 0, np.nan, np.arange(100) % 2)
    X = pd.DataFrame([a, ab, np.arange(100) % 2, np.arange(100) % 2], index=["a", "a = b", "d", "d"]).T
    X = X.astype({"a = b": float})
    X.columns = ["a", "a = b", "d", "d"]
    names = _Binarizer().fit(X).names_
    assert len(set(names)) == len(names)
    assert {"a = b missing", "a = b missing (2)", "d = 1", "d = 1 (2)"} <= set(names)


# 2. the binarizer: boolean output, computed per column, only the used rules at predict time
def test_binarizer_is_boolean_and_predicts_from_used_rules_only(monkeypatch):
    X, y = _data(d=5)
    m = FastRiskScoreClassifier(k=3, n_thresholds=99).fit(X, y)
    full = m.binarizer_.transform(pd.DataFrame(X))
    assert full.dtype == bool and full.shape == (len(X), len(m.features_))
    used = np.flatnonzero(m.coef_)
    np.testing.assert_array_equal(m.binarizer_.transform(pd.DataFrame(X), used), full[:, used])
    seen = []
    transform = _Binarizer.transform
    monkeypatch.setattr(_Binarizer, "transform", lambda self, frame, rules=None: seen.append(rules) or
                        transform(self, frame, rules))
    np.testing.assert_allclose(m.total_score(X), full @ m.coef_)
    np.testing.assert_array_equal(seen[-1], used)
    assert "is not counted" in " ".join(FastRiskScoreClassifier.__doc__.split())


# 3. the first compilation is thread-safe, and a retry after a failed wrap works
def test_concurrent_first_fits_and_retry_after_a_failed_jit():
    script = textwrap.dedent("""
        import threading
        import numba
        import numpy as np
        from imodels.algebraic.risk_score import solver

        real_njit, calls = numba.njit, []
        def failing_njit(*args, **kwargs):
            calls.append(1)
            if len(calls) == 5:
                raise RuntimeError("simulated failure")
            return real_njit(*args, **kwargs)
        numba.njit = failing_njit
        try:
            solver._jit()
        except RuntimeError:
            pass
        numba.njit = real_njit
        assert solver.nb is None
        assert all(not hasattr(solver.__dict__[name], "py_func") for name, _, _ in solver._KERNELS)

        rng = np.random.default_rng(0)
        X = (rng.random((300, 8)) < 0.4).astype(float)
        y = (X[:, 0] + X[:, 1] + rng.random(300) > 1.2).astype(np.int64)
        results, errors = [], []
        def run():
            try:
                results.append(tuple(solver.solve(X, y, 3)[0]))
            except Exception as err:
                errors.append(repr(err))
        threads = [threading.Thread(target=run) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors, errors
        assert len(results) == 8 and len(set(results)) == 1
        for name, f, _ in solver._KERNELS:
            assert solver.__dict__[name].py_func is f
        print("OK")
    """)
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=900)
    assert out.returncode == 0 and "OK" in out.stdout, out.stderr[-3000:]


# 4. fit on an array, predict on a DataFrame: columns by position, with sklearn's warning
def test_numpy_fit_dataframe_predict_uses_columns_by_position():
    X, y = _data()
    m = FastRiskScoreClassifier(k=3).fit(X, y)
    with pytest.warns(UserWarning, match="fitted without feature names"):
        p = m.predict_proba(pd.DataFrame(X, columns=list("pqr")))
    np.testing.assert_allclose(p, m.predict_proba(X))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        m.predict_proba(pd.DataFrame(X))  # integer column names are no feature names: no warning


# 5. refit on an array after a DataFrame: no stale feature_names_in_
def test_refit_on_numpy_forgets_dataframe_names():
    X, y = _data()
    m = FastRiskScoreClassifier(k=3).fit(pd.DataFrame(X, columns=list("pqr")), y)
    m.fit(X, y)
    assert not hasattr(m, "feature_names_in_")
    with pytest.warns(UserWarning, match="fitted without feature names"):
        p = m.predict_proba(pd.DataFrame(X, columns=list("pqr")))
    np.testing.assert_allclose(p, m.predict_proba(X))


# 6. invalid hyperparameters raise a ValueError that names them
@pytest.mark.parametrize("params, name", [
    (dict(k=0), "k"), (dict(k=2.9), "k"), (dict(k=True), "k"), (dict(k="3"), "k"),
    (dict(max_points=0), "max_points"), (dict(max_points=False), "max_points"),
    (dict(n_thresholds=0), "n_thresholds"), (dict(n_thresholds=-3), "n_thresholds"),
    (dict(n_thresholds=9.0), "n_thresholds"),
    (dict(time_limit=-1), "time_limit"), (dict(time_limit=0), "time_limit"),
    (dict(time_limit=None), "time_limit"), (dict(time_limit=float("inf")), "time_limit"),
    (dict(time_limit=float("nan")), "time_limit"), (dict(time_limit=True), "time_limit"),
])
def test_invalid_hyperparameters_raise(params, name):
    X, y = _data()
    with pytest.raises(ValueError, match=name):
        FastRiskScoreClassifier(**params).fit(X, y)


def test_valid_hyperparameter_types_are_accepted():
    X, y = _data()
    m = FastRiskScoreClassifier(k=np.int64(2), max_points=np.int32(3), n_thresholds=np.int64(4),
                                time_limit=5).fit(X, y)
    assert 1 <= len(m.points_) <= 2


# 7. array-likes go through check_array before their shape is read; sparse X is refused clearly
def test_array_likes_and_sparse_input():
    from sklearn.utils.estimator_checks import check_classifier_data_not_an_array
    check_classifier_data_not_an_array("FastRiskScoreClassifier", FastRiskScoreClassifier(k=2))
    X, y = _data()
    m = FastRiskScoreClassifier(k=2).fit(X.tolist(), list(y))
    np.testing.assert_allclose(m.predict_proba(X.tolist()), m.predict_proba(X))
    with pytest.raises(TypeError, match="sparse"):
        FastRiskScoreClassifier(k=2).fit(sparse.csr_matrix(X), y)
    with pytest.raises(TypeError, match="sparse"):
        m.predict(sparse.csr_matrix(X))
    with pytest.raises(ValueError, match="features"):
        m.predict(X[:, :2].tolist())


# 8. is_cached: only an index for this Python, written by this numba for the current source
def test_is_cached_needs_a_fresh_index_for_this_python(tmp_path, monkeypatch):
    import numba
    monkeypatch.delenv("NUMBA_CACHE_DIR", raising=False)
    module = tmp_path / "mod.py"
    module.write_text("x = 1\n")
    cache = tmp_path / "__pycache__"
    cache.mkdir()
    st = os.stat(module)
    tag = f"py{sys.version_info.major}{sys.version_info.minor}"

    def index(name, version, stamp):
        (cache / name).write_bytes(pickle.dumps(version) + pickle.dumps((stamp, {})))

    other = "py27" if tag != "py27" else "py39"
    index(f"mod.f-3.{other}.nbi", numba.__version__, (st.st_mtime, st.st_size))
    assert not numba_compile.is_cached(str(module))  # another Python version
    index(f"mod.f-3.{tag}.nbi", numba.__version__, (st.st_mtime - 10, st.st_size))
    assert not numba_compile.is_cached(str(module))  # the source changed since
    index(f"mod.g-5.{tag}.nbi", "0.0.1", (st.st_mtime, st.st_size))
    assert not numba_compile.is_cached(str(module))  # another numba
    (cache / f"mod.h-7.{tag}.nbi").write_bytes(b"not a pickle")
    assert not numba_compile.is_cached(str(module))
    index(f"mod.f-3.{tag}.nbi", numba.__version__, (st.st_mtime, st.st_size))
    assert numba_compile.is_cached(str(module))


# 9. a missing value is not the level "missing"
def test_missing_values_differ_from_a_missing_level():
    s = np.array(["missing", "a", None, "b"] * 60, dtype=object)
    X = pd.DataFrame({"s": s})
    b = _Binarizer().fit(X)
    assert "s missing" in b.names_ and "s = missing" in b.names_
    B = b.transform(X)
    np.testing.assert_array_equal(B[:, b.names_.index("s missing")], pd.isna(s))
    np.testing.assert_array_equal(B[:, b.names_.index("s = missing")], s == "missing")
    # the string form of a missing value ("None", "nan") is never a level
    X2 = pd.DataFrame({"s": np.array(["None", "nan", None, np.nan] * 60, dtype=object)})
    b2 = _Binarizer().fit(X2)
    B2 = b2.transform(X2)
    assert set(b2.names_) == {"s = None", "s = nan", "s missing"}
    np.testing.assert_array_equal(B2[:, b2.names_.index("s missing")], pd.isna(X2["s"]).to_numpy())
    np.testing.assert_array_equal(B2[:, b2.names_.index("s = nan")], X2["s"].to_numpy() == "nan")
