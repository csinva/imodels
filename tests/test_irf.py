"""Statistical and estimator contracts for iterative random forests."""

import importlib

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.utils.estimator_checks import check_estimator

from imodels import IRFClassifier


irf_module = importlib.import_module(
    "imodels.tree.iterative_random_forest.iterative_random_forest"
)


def _small_model(**kwargs):
    params = dict(n_estimators=5, n_iterations=2, n_bootstraps=3,
                  n_rit=8, rit_depth=2, random_state=13)
    params.update(kwargs)
    return IRFClassifier(**params)


def _fit_outer_forests_serially(monkeypatch):
    """Fit outer forests one at a time with ``fit`` so tests can inspect them.

    The estimator fits all outer trees in one worker pool and keeps only their
    paths; test_irf_forest checks that pool against this serial reference.
    """
    def forest_paths(forests, datasets, n_jobs=None):
        for forest, (X, y, feature_weights, sample_weight, n_classes) in zip(
                forests, datasets):
            forest.fit(X, y, feature_weights=feature_weights,
                       sample_weight=sample_weight, n_classes=n_classes)
            yield [leaf for tree in forest.estimators_
                   for leaf in tree.terminal_paths(X, sample_weight=sample_weight)]

    monkeypatch.setattr(irf_module, "_forest_paths", forest_paths)


def _record_forests(monkeypatch, importances):
    """A controlled learner exposes iteration/bootstrap scheduling mistakes."""
    fits = []

    class RecordedForest:
        def __init__(self, **params):
            self.params = params

        def fit(self, X, y, feature_weights, sample_weight, n_classes):
            self.fit_index = len(fits)
            self.X = X.copy()
            self.y = y.copy()
            self.weights = feature_weights.copy()
            self.sample_weight = sample_weight.copy()
            self.feature_importances_ = np.array(
                importances[min(self.fit_index, len(importances) - 1)], dtype=float
            )
            self.estimators_ = []
            self.n_classes = n_classes
            fits.append(self)
            return self

        def predict_proba(self, X):
            return np.tile([0.7, 0.3], (len(X), 1))

    monkeypatch.setattr(irf_module, "_WeightedForest", RecordedForest)
    _fit_outer_forests_serially(monkeypatch)
    return fits


def test_weights_are_learned_once_and_bootstraps_use_final_input_weights(monkeypatch):
    X = np.arange(40).reshape(20, 2)
    y = np.arange(20) % 2
    fits = _record_forests(monkeypatch, [[0.8, 0.2], [0.25, 0.75], [0.99, 0.01]])
    discovered = iter([{(0, 1)}, {(0, 1), (0,)}, set(), {(1,)}])
    monkeypatch.setattr(irf_module, "_random_intersection_trees",
                        lambda *args, **kwargs: next(discovered))
    model = _small_model(n_iterations=3, n_bootstraps=4, n_estimators=7,
                         n_estimators_bootstrap=2).fit(X, y)

    # Exactly K+B fits: reweighting is outside the outer bootstrap loop.
    assert len(fits) == 7
    expected_weights = [[0.5, 0.5], [0.8, 0.2], [0.25, 0.75]]
    assert_allclose(model.feature_weights_history_, expected_weights)
    for forest, weights in zip(fits[:3], expected_weights):
        assert_array_equal(forest.X, X)
        assert_allclose(forest.weights, weights)
        assert forest.params["n_estimators"] == 7
    for forest, indices in zip(fits[3:], model.bootstrap_samples_):
        assert_array_equal(forest.X, X[indices])
        assert_allclose(forest.weights, [0.25, 0.75])
        assert forest.params["n_estimators"] == 2
    assert model.forest_ is fits[2]
    assert len({forest.params["random_state"] for forest in fits}) == 7
    assert_allclose(model.feature_importances_, [0.99, 0.01])
    assert_allclose(model.feature_weights_, [0.25, 0.75])
    assert model.interaction_stability_ == {(0, 1): 0.5, (0,): 0.25, (1,): 0.25}
    assert model.interactions_ == [(0, 1), (0,), (1,)]


def test_one_iteration_means_uniform_bootstrap_weights(monkeypatch):
    fits = _record_forests(monkeypatch, [[0.95, 0.05]])
    model = _small_model(n_iterations=1, n_bootstraps=2).fit(
        np.arange(24).reshape(12, 2), np.arange(12) % 2
    )
    assert len(fits) == 3
    for forest in fits:
        assert_allclose(forest.weights, [0.5, 0.5])
    assert_allclose(model.feature_importances_, [0.95, 0.05])


def test_zero_gini_importance_preserves_previous_sampling_weights(monkeypatch):
    _record_forests(monkeypatch, [[1, 0], [0, 0], [0, 0]])
    model = _small_model(n_iterations=3, n_bootstraps=0).fit(
        np.arange(24).reshape(12, 2), np.arange(12) % 2
    )
    assert_allclose(model.feature_weights_history_, [[0.5, 0.5], [1, 0], [1, 0]])
    assert_allclose(model.feature_importances_, [0, 0])


def test_outer_bootstraps_retain_replacement_and_class_counts():
    X = np.arange(80).reshape(40, 2)
    y = np.repeat(["common", "rare"], [30, 10])
    model = _small_model(n_estimators=2, max_depth=1, n_bootstraps=4).fit(X, y)
    for indices in model.bootstrap_samples_:
        assert len(indices) == 40
        assert len(np.unique(indices)) < len(indices)
        assert_array_equal(np.unique(y[indices], return_counts=True)[1], [30, 10])


def test_fractional_outer_sample_size_rounds_up():
    X = np.arange(62).reshape(31, 2)
    y = np.arange(31) % 2
    model = _small_model(n_estimators=2, bootstrap_fraction=0.2).fit(X, y)
    assert all(len(indices) == 7 for indices in model.bootstrap_samples_)


def test_rit_receives_mass_from_all_outer_rows_not_inner_bootstraps(monkeypatch):
    forests, rit_inputs = [], []
    original_fit = irf_module._WeightedForest.fit

    def capture_fit(self, *args, **kwargs):
        result = original_fit(self, *args, **kwargs)
        forests.append(self)
        return result

    def capture_rit(paths, masses, **kwargs):
        rit_inputs.append((paths, masses))
        return set()

    monkeypatch.setattr(irf_module._WeightedForest, "fit", capture_fit)
    _fit_outer_forests_serially(monkeypatch)
    monkeypatch.setattr(irf_module, "_random_intersection_trees", capture_rit)
    X = np.arange(48).reshape(24, 2)
    y = (X[:, 0] >= 20).astype(int)
    sample_weight = np.linspace(0.5, 4, 24)
    model = _small_model(n_iterations=1, max_depth=1, max_features=None).fit(
        X, y, sample_weight=sample_weight
    )
    # A common rescaling does not alter the path-sampling distribution; the
    # estimator normalizes input weights for numerical stability.
    sample_weight = sample_weight / sample_weight.max()
    saw_difference_from_inbag = False
    for forest, indices, (paths, masses) in zip(
        forests[1:], model.bootstrap_samples_, rit_inputs
    ):
        expected_masses = []
        for tree in forest.estimators_:
            leaves = tree.apply(X[indices])
            routed = np.bincount(leaves, weights=sample_weight[indices],
                                 minlength=tree.node_count_)
            for node in np.flatnonzero(tree.children_left_ < 0):
                if np.argmax(tree.value_[node]) == 1 and routed[node] > 0:
                    expected_masses.append(routed[node])
                    saw_difference_from_inbag |= not np.isclose(
                        routed[node], tree.weighted_n_node_samples_[node]
                    )
        assert len(paths) == len(expected_masses)
        assert_allclose(masses, expected_masses)
    assert saw_difference_from_inbag


def test_known_and_interaction_and_predictive_forest():
    X = np.tile([[0, 0], [0, 1], [1, 0], [1, 1]], (40, 1))
    y = np.where(np.all(X == 1, axis=1), "positive", "negative")
    model = _small_model(max_features=None, bootstrap=False,
                         n_bootstraps=4, n_rit=20).fit(X, y)
    assert_array_equal(model.predict(X), y)
    assert model.interaction_class_ == "positive"
    assert model.interaction_stability_[(0, 1)] == 1.0
    assert model.forest_.apply(X).shape == (len(X), model.n_estimators)
    assert_allclose(model.predict_proba(X).sum(axis=1), 1)


def test_single_feature_paths_do_not_create_interactions():
    X = np.arange(40).reshape(20, 2)
    y = (X[:, 0] > 20).astype(int)
    model = _small_model(max_depth=1, max_features=None).fit(X, y)
    assert model.interactions_ == []
    assert all(replicate == set() for replicate in model.bootstrap_interactions_)


@pytest.mark.parametrize("depth", [0, 1])
def test_rit_requires_at_least_two_sampled_paths(depth):
    with pytest.raises(ValueError, match="rit_depth.*>= 2"):
        _small_model(rit_depth=depth).fit([[0], [1]], [0, 1])


def test_multiclass_and_zero_weight_class_keep_probability_columns():
    X = np.arange(32).reshape(16, 2)
    y = np.repeat(["absent", "blue", "green", "red"], 4)
    weights = np.ones(16)
    weights[:4] = 0
    model = _small_model(max_features=None, bootstrap=False).fit(X, y, weights)
    assert_array_equal(model.classes_, ["absent", "blue", "green", "red"])
    probabilities = model.predict_proba(X)
    assert probabilities.shape == (16, 4)
    assert_allclose(probabilities[:, 0], 0)
    assert_allclose(probabilities.sum(axis=1), 1)
    assert all(np.all(indices >= 4) for indices in model.bootstrap_samples_)
    assert all(len(indices) == 12 for indices in model.bootstrap_samples_)


def test_serial_and_parallel_fits_are_reproducible_without_global_rng_changes():
    rng = np.random.RandomState(9)
    X = rng.normal(size=(60, 4))
    y = (X[:, 0] * X[:, 1] > 0).astype(int)
    np.random.seed(314)
    before = np.random.get_state()
    serial = _small_model(n_jobs=1).fit(X, y)
    parallel = clone(serial).set_params(n_jobs=2).fit(X, y)
    after = np.random.get_state()
    assert before[0] == after[0]
    assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]
    assert_allclose(serial.predict_proba(X), parallel.predict_proba(X), atol=0, rtol=0)
    assert_array_equal(serial.forest_.apply(X), parallel.forest_.apply(X))
    assert_allclose(serial.feature_weights_history_, parallel.feature_weights_history_)
    assert serial.bootstrap_interactions_ == parallel.bootstrap_interactions_
    assert serial.interaction_stability_ == parallel.interaction_stability_
    for first, second in zip(serial.bootstrap_samples_, parallel.bootstrap_samples_):
        assert_array_equal(first, second)


def test_prediction_only_refit_clears_interaction_results():
    X = np.tile([[0, 0], [0, 1], [1, 0], [1, 1]], (10, 1))
    y = np.all(X == 1, axis=1).astype(int)
    model = _small_model(max_features=None, bootstrap=False).fit(X, y)
    assert model.interaction_stability_
    assert model.set_params(n_bootstraps=0).fit(X, y) is model
    assert model.interactions_ == []
    assert model.interaction_stability_ == {}
    assert model.bootstrap_samples_ == []
    assert model.bootstrap_interactions_ == []
    assert_array_equal(model.predict(X), y)


@pytest.mark.parametrize("weights", [[-1, 1, 1, 1], [0, 0, 0, 0],
                                      [np.inf, 1, 1, 1], [1, 1]])
def test_invalid_observation_weights_fail(weights):
    with pytest.raises(ValueError):
        _small_model().fit(np.arange(8).reshape(4, 2), [0, 0, 1, 1], weights)


def test_predict_requires_a_fit_and_rejects_changed_feature_count():
    model = _small_model(n_bootstraps=0)
    with pytest.raises(NotFittedError):
        model.predict([[0, 1]])
    model.fit(np.arange(16).reshape(8, 2), np.arange(8) % 2)
    with pytest.raises(ValueError, match="features"):
        model.predict([[0, 1, 2]])


@pytest.mark.parametrize("scale", [1e308, 1e-320])
def test_uniform_sample_weight_scaling_is_numerically_stable(scale):
    X = np.tile([[0, 0], [0, 1], [1, 0], [1, 1]], (8, 1))
    y = np.all(X == 1, axis=1).astype(int)
    reference = _small_model().fit(X, y)
    scaled = _small_model().fit(X, y, sample_weight=scale)
    assert_allclose(scaled.predict_proba(X), reference.predict_proba(X))
    assert_allclose(scaled.feature_weights_history_, reference.feature_weights_history_)
    assert scaled.interaction_stability_ == reference.interaction_stability_


def test_sklearn_estimator_contract_without_row_bootstrapping():
    # Like sklearn RF, row bootstrapping does not equate integer observation
    # weights to physically repeated input rows. Disable it for the generic
    # estimator suite's weighted-repeat equivalence check.
    check_estimator(IRFClassifier(n_estimators=5, n_iterations=2,
                                  n_bootstraps=0, bootstrap=False,
                                  random_state=0))
