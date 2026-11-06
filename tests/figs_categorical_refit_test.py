import numpy as np
import pandas as pd
import pytest

from imodels import FIGSClassifier, FIGSRegressor


def _input(kind):
    X = pd.DataFrame({"value": np.arange(12)})
    if kind == "categorical":
        X["kind"] = ["a", "b"] * 6
        return X, ["kind"]
    if kind == "other_categorical":
        X["town"] = ["east", "north", "west"] * 4
        return X, ["town"]
    return X, [] if kind == "numeric_empty" else None


def _check_predictions(model, fresh, X):
    np.testing.assert_array_equal(model.predict(X), fresh.predict(X))
    np.testing.assert_allclose(model.predict(X, by_tree=True), fresh.predict(X, by_tree=True))
    if isinstance(model, FIGSClassifier):
        np.testing.assert_allclose(model.predict_proba(X), fresh.predict_proba(X))


@pytest.mark.parametrize("estimator", [FIGSClassifier, FIGSRegressor])
@pytest.mark.parametrize("as_frame", [True, False])
def test_categorical_model_can_be_refit_on_numeric_inputs(estimator, as_frame):
    X_cat, features = _input("categorical")
    X_num, _ = _input("numeric")
    if not as_frame:
        X_num = X_num.to_numpy()
    y = np.repeat([0, 1], 6)
    model = estimator(max_rules=2, random_state=0).fit(X_cat, y, categorical_features=features)
    np.testing.assert_array_equal(model.predict(X_cat), y)

    # A second fit should have the behavior of a fresh model on the new inputs.
    if estimator is FIGSClassifier:
        y = np.repeat(["low", "high"], 6)
    fresh = estimator(max_rules=2, random_state=0).fit(X_num, y)
    assert model.fit(X_num, y) is model
    _check_predictions(model, fresh, X_num)


@pytest.mark.parametrize("as_frame", [True, False])
def test_categorical_regressor_can_be_refit_on_numeric_multioutput(as_frame):
    X_cat, features = _input("categorical")
    X_num, _ = _input("numeric")
    if not as_frame:
        X_num = X_num.to_numpy()
    y = np.repeat([0., 1.], 6)
    model = FIGSRegressor(max_rules=2, random_state=0).fit(X_cat, y, categorical_features=features)
    target = np.column_stack([y * 3, 9 - y])
    fresh = FIGSRegressor(max_rules=2, random_state=0).fit(X_num, target)
    model.fit(X_num, target)
    _check_predictions(model, fresh, X_num)


@pytest.mark.parametrize("estimator", [FIGSClassifier, FIGSRegressor])
@pytest.mark.parametrize("sequence", [
    ("numeric", "numeric"),
    ("numeric", "categorical"),
    ("categorical", "categorical"),
    ("categorical", "other_categorical"),
    ("categorical", "numeric", "other_categorical"),
    ("numeric", "categorical", "numeric"),
    ("categorical", "numeric_empty"),
    ("numeric_empty", "numeric"),
])
def test_figs_refit_transitions_match_fresh_models(estimator, sequence):
    y = np.repeat([0, 1], 6)
    model = estimator(max_rules=2, random_state=0)
    for kind in sequence:
        X, features = _input(kind)
        fresh = estimator(max_rules=2, random_state=0).fit(X, y, categorical_features=features)
        assert model.fit(X, y, categorical_features=features) is model
        _check_predictions(model, fresh, X)
