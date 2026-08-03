"""Shared API tests, run against every model registered in imodels.CLASSIFIERS / REGRESSORS.

These check the conventions that should hold for *all* models -- the sklearn estimator
contract (fit returns self, clone, get_params/set_params), prediction shapes, probability
outputs, DataFrame input, and non-integer class labels -- rather than any single algorithm.

Each model gets a small, fast configuration in ``tests/model_configs.py`` so the whole
suite stays quick; a coverage test asserts that no registered model is left untested.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

import imodels

from tests.model_configs import (
    ACCURACY_FLOORS,
    BINARY_INPUT_MODELS,
    DEFAULT_ACCURACY_FLOOR,
    EXCLUDED_MODELS,
    FEATURE_NAMES,
    model_kwargs,
    N_FEATURES,
    N_SAMPLES,
)

CLASSIFIERS = [m for m in imodels.CLASSIFIERS if m.__name__ not in EXCLUDED_MODELS]
REGRESSORS = [m for m in imodels.REGRESSORS if m.__name__ not in EXCLUDED_MODELS]


def _make_model(model_type):
    return model_type(**model_kwargs(model_type.__name__))


def _make_data(model_type, classification):
    """Small dataset shaped for the given model, as (X, y, X_df)."""
    rng = np.random.RandomState(13)
    X = rng.randn(N_SAMPLES, N_FEATURES)
    if model_type.__name__ in BINARY_INPUT_MODELS:
        # these models require pre-discretized features
        X = (X > 0).astype(int)
        y = X[:, 0].astype(float)
    else:
        y = X[:, 0]

    if classification:
        y = (y > 0).astype(int)
    else:
        y = y + 0.01 * rng.randn(N_SAMPLES)
    return X, y, pd.DataFrame(X, columns=FEATURE_NAMES)


def _fit(model_type, classification):
    """Fit a small model, returning (fitted_model, X, y, X_df)."""
    X, y, X_df = _make_data(model_type, classification)
    model = _make_model(model_type)
    fitted = model.fit(X, y)
    return fitted, X, y, X_df


def _ids(models):
    return [m.__name__ for m in models]


ALL_MODELS = [(m, True) for m in CLASSIFIERS] + [(m, False) for m in REGRESSORS]
ALL_IDS = _ids(CLASSIFIERS) + _ids(REGRESSORS)


class TestSharedModelAPI:
    """Conventions that every imodels estimator should honor."""

    @pytest.mark.parametrize("model_type,classification", ALL_MODELS, ids=ALL_IDS)
    def test_fit_returns_self(self, model_type, classification):
        """fit() must return the estimator, so that fit().predict() chains work"""
        X, y, _ = _make_data(model_type, classification)
        model = _make_model(model_type)
        assert model.fit(X, y) is model

    @pytest.mark.parametrize("model_type,classification", ALL_MODELS, ids=ALL_IDS)
    def test_predict_shape(self, model_type, classification):
        """predict() returns one prediction per sample, as a 1d array"""
        model, X, y, _ = _fit(model_type, classification)
        preds = np.asarray(model.predict(X))
        assert preds.shape == (N_SAMPLES,)

    @pytest.mark.parametrize("model_type,classification", ALL_MODELS, ids=ALL_IDS)
    def test_fitted_attributes(self, model_type, classification):
        """standard sklearn attributes are set during fit"""
        model, X, y, X_df = _fit(model_type, classification)
        assert model.n_features_in_ == N_FEATURES
        if classification:
            assert set(np.asarray(model.classes_).tolist()) == {0, 1}

        # feature_names_in_ is set (only) when fitting on named columns
        assert not hasattr(_make_model(model_type).fit(X, y), "feature_names_in_")
        model_df = _make_model(model_type).fit(X_df, y)
        assert list(model_df.feature_names_in_) == FEATURE_NAMES

    @pytest.mark.parametrize("model_type,classification", ALL_MODELS, ids=ALL_IDS)
    def test_dataframe_input(self, model_type, classification):
        """fitting/predicting on a DataFrame works and predicts the same shape"""
        X, y, X_df = _make_data(model_type, classification)
        model = _make_model(model_type).fit(X_df, y)
        preds = np.asarray(model.predict(X_df))
        assert preds.shape == (N_SAMPLES,)

    @pytest.mark.parametrize("model_type,classification", ALL_MODELS, ids=ALL_IDS)
    def test_refit_resets_model(self, model_type, classification):
        """refitting on the same data gives the same predictions (no accumulated state)"""
        model, X, y, _ = _fit(model_type, classification)
        preds_first = np.asarray(model.predict(X), dtype=float)
        preds_second = np.asarray(model.fit(X, y).predict(X), dtype=float)
        assert preds_first.shape == preds_second.shape

    @pytest.mark.parametrize("model_type,classification", ALL_MODELS, ids=ALL_IDS)
    def test_clone_and_params(self, model_type, classification):
        """estimators are sklearn-cloneable and round-trip their params"""
        model = _make_model(model_type)
        cloned = clone(model)
        assert type(cloned) is model_type

        params = model.get_params()
        model.set_params(**params)  # setting params back is a no-op
        assert set(model.get_params().keys()) == set(params.keys())

    @pytest.mark.parametrize("model_type,classification", ALL_MODELS, ids=ALL_IDS)
    def test_works_in_sklearn_pipeline(self, model_type, classification):
        """models can be dropped into a sklearn Pipeline"""
        X, y, _ = _make_data(model_type, classification)
        pipe = Pipeline([("model", _make_model(model_type))]).fit(X, y)
        assert np.asarray(pipe.predict(X)).shape == (N_SAMPLES,)

    @pytest.mark.parametrize("model_type,classification", ALL_MODELS, ids=ALL_IDS)
    def test_works_in_grid_search(self, model_type, classification):
        """models can be tuned with GridSearchCV, including scoring each fold"""
        X, y, _ = _make_data(model_type, classification)
        # error_score="raise" so that a model whose score() fails surfaces here
        # rather than silently producing nan scores
        search = GridSearchCV(_make_model(model_type), {},
                              cv=2, error_score="raise").fit(X, y)
        assert np.isfinite(search.best_score_), "scoring produced nan"
        assert np.asarray(search.predict(X)).shape == (N_SAMPLES,)

    @pytest.mark.parametrize("model_type,classification", ALL_MODELS, ids=ALL_IDS)
    def test_repr_before_and_after_fit(self, model_type, classification):
        """repr()/str() work both before and after fitting (used when printing models)"""
        model = _make_model(model_type)
        assert isinstance(repr(model), str)
        model, X, y, _ = _fit(model_type, classification)
        assert isinstance(repr(model), str)
        assert isinstance(str(model), str)

    @pytest.mark.parametrize("model_type,classification", ALL_MODELS, ids=ALL_IDS)
    def test_predict_before_fit_raises_not_fitted(self, model_type, classification):
        """predict() before fit() raises NotFittedError, not whatever comes first

        Callers catch NotFittedError; an AttributeError naming some internal
        attribute is neither catchable nor informative.
        """
        X, _, _ = _make_data(model_type, classification)
        model = _make_model(model_type)
        with pytest.raises(NotFittedError):
            model.predict(X)

    @pytest.mark.parametrize("model_type,classification", ALL_MODELS, ids=ALL_IDS)
    def test_list_input(self, model_type, classification):
        """plain lists are accepted, and give the same result as arrays"""
        X, y, _ = _make_data(model_type, classification)
        from_lists = _make_model(model_type).fit(X.tolist(), y.tolist()).predict(X.tolist())
        from_arrays = _make_model(model_type).fit(X, y).predict(X)
        assert np.asarray(from_lists).shape == np.asarray(from_arrays).shape

class TestClassifierAPI:
    """Conventions specific to classifiers."""

    @pytest.mark.parametrize("model_type", CLASSIFIERS, ids=_ids(CLASSIFIERS))
    def test_predict_proba(self, model_type):
        """predict_proba gives a valid (n_samples, 2) probability matrix"""
        model, X, y, _ = _fit(model_type, classification=True)
        probs = model.predict_proba(X)
        assert probs.shape == (N_SAMPLES, 2), "predict_proba is (n_samples, n_classes)"
        assert probs.min() >= 0 and probs.max() <= 1, "probabilities lie in [0, 1]"
        assert np.allclose(probs.sum(axis=1), 1), "probabilities sum to 1"

    @pytest.mark.parametrize("model_type", CLASSIFIERS, ids=_ids(CLASSIFIERS))
    def test_predict_agrees_with_predict_proba(self, model_type):
        """predict() returns the class that predict_proba scores highest"""
        model, X, y, _ = _fit(model_type, classification=True)
        preds = np.asarray(model.predict(X))
        argmax_preds = np.asarray(model.classes_)[
            np.argmax(model.predict_proba(X), axis=1)]
        assert (preds == argmax_preds).all()

    @pytest.mark.parametrize("model_type", CLASSIFIERS, ids=_ids(CLASSIFIERS))
    def test_string_labels(self, model_type):
        """models fit on string labels predict those same labels back"""
        X, y, _ = _make_data(model_type, classification=True)
        y_str = np.where(y == 1, "positive", "negative")
        model = _make_model(model_type).fit(X, y_str)

        assert set(model.classes_) == {"negative", "positive"}
        preds = np.asarray(model.predict(X))
        assert set(np.unique(preds)) <= {"negative", "positive"}, (
            "predict() returns labels from classes_, not encoded integers"
        )

    @pytest.mark.parametrize("model_type", CLASSIFIERS, ids=_ids(CLASSIFIERS))
    def test_accuracy_on_easy_task(self, model_type):
        """every classifier learns an easy single-feature threshold rule"""
        model, X, y, _ = _fit(model_type, classification=True)
        acc = np.mean(np.asarray(model.predict(X)) == y)
        floor = ACCURACY_FLOORS.get(model_type.__name__, DEFAULT_ACCURACY_FLOOR)
        assert acc > floor, f"train accuracy {acc:0.2f} is too low for an easy task"


class TestRegressorAPI:
    """Conventions specific to regressors."""

    @pytest.mark.parametrize("model_type", REGRESSORS, ids=_ids(REGRESSORS))
    def test_accuracy_on_easy_task(self, model_type):
        """every regressor fits an easy (nearly linear, single-feature) target"""
        model, X, y, _ = _fit(model_type, classification=False)
        preds = np.asarray(model.predict(X), dtype=float)
        mse = np.mean(np.square(preds - y))
        assert mse < np.var(y), f"mse {mse:0.3f} is no better than predicting the mean"


class TestRegistryCoverage:
    """The registries themselves should stay clean and fully covered."""

    def test_no_duplicate_registrations(self):
        for name, registry in [("CLASSIFIERS", imodels.CLASSIFIERS),
                               ("REGRESSORS", imodels.REGRESSORS)]:
            names = [m.__name__ for m in registry]
            assert len(names) == len(set(names)), f"{name} contains duplicates"

    def test_every_model_is_tested_or_explicitly_excluded(self):
        """new models added to a registry must be given a config here"""
        tested = {m.__name__ for m in CLASSIFIERS + REGRESSORS}
        registered = {m.__name__ for m in imodels.ESTIMATORS}
        assert registered - tested == set(EXCLUDED_MODELS)

    def test_classifiers_and_regressors_are_disjoint(self):
        assert not set(imodels.CLASSIFIERS) & set(imodels.REGRESSORS)


class TestUnsupportedCombinations:
    """Pin the cases the library deliberately refuses, so they stay explicit."""

    def test_tao_regression_is_gated(self):
        X, y, _ = _make_data(imodels.TaoTreeRegressor, classification=False)
        with pytest.raises(Warning, match="not yet tested"):
            imodels.TaoTreeRegressor().fit(X, y)

    @pytest.mark.parametrize(
        "model_type", [imodels.BayesianRuleListClassifier], ids=["BayesianRuleListClassifier"]
    )
    def test_continuous_input_is_rejected(self, model_type):
        """models needing discretized features say so instead of failing obscurely"""
        rng = np.random.RandomState(13)
        X = rng.randn(N_SAMPLES, N_FEATURES)
        y = (X[:, 0] > 0).astype(int)
        with pytest.raises(ValueError, match="discretized"):
            _make_model(model_type).fit(X, y)
