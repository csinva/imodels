"""Estimator kinds must remain visible to scikit-learn's meta-estimators."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.base import clone, is_classifier, is_regressor
from sklearn.ensemble import StackingRegressor, VotingClassifier, VotingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, get_scorer
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

import imodels
from imodels.rule_set.slipper_util import SlipperBaseEstimator
from imodels.util.ensemble import ResidualBoostingRegressor
from tests.model_configs import model_kwargs


@pytest.mark.parametrize(
    "estimator_class,kind",
    [(cls, "classifier") for cls in imodels.CLASSIFIERS]
    + [(cls, "regressor") for cls in imodels.REGRESSORS],
    ids=[cls.__name__ for cls in imodels.ESTIMATORS],
)
def test_registered_estimator_kind(estimator_class, kind):
    # No fitting is needed: kind information is also used to choose CV splitters.
    estimator = estimator_class(**model_kwargs(estimator_class.__name__))
    assert is_classifier(estimator) == (kind == "classifier")
    assert is_regressor(estimator) == (kind == "regressor")


@pytest.mark.parametrize(
    "estimator,kind",
    [
        (SlipperBaseEstimator(), "classifier"),
        (ResidualBoostingRegressor(DecisionTreeRegressor()), "regressor"),
    ],
    ids=["slipper_base", "residual_boosting"],
)
def test_utility_estimator_kind(estimator, kind):
    assert is_classifier(estimator) == (kind == "classifier")
    assert is_regressor(estimator) == (kind == "regressor")


def test_stable_clustering_kind():
    estimator = imodels.StableClustering(k_values=[2, 3])
    # New sklearn versions use tags instead of the legacy type attribute.
    if hasattr(estimator, "__sklearn_tags__"):
        assert estimator.__sklearn_tags__().estimator_type == "clusterer"
    else:
        assert estimator._estimator_type == "clusterer"


def _classification_data():
    X = np.arange(16, dtype=float).reshape(-1, 1)
    y = (X[:, 0] >= 8).astype(int)
    return X, y


def test_greedy_rule_list_direct_predictions():
    X, y = _classification_data()
    rules = imodels.GreedyRuleListClassifier(max_depth=2)
    assert rules.fit(X, y) is rules
    assert_array_equal(rules.predict(X), y)
    assert_allclose(rules.predict_proba(X), np.column_stack((1 - y, y)))


def test_greedy_rule_list_probability_scorer():
    X, y = _classification_data()
    rules = imodels.GreedyRuleListClassifier(max_depth=2).fit(X, y)
    assert get_scorer("roc_auc")(rules, X, y) == pytest.approx(1.0)


@pytest.mark.parametrize("voting", ["hard", "soft"])
def test_greedy_rule_list_in_voting(voting):
    X, y = _classification_data()
    ensemble = VotingClassifier(
        [("rules", imodels.GreedyRuleListClassifier(max_depth=2)),
         ("tree", DecisionTreeClassifier(max_depth=2, random_state=0))],
        voting=voting,
        n_jobs=1,
    ).fit(X, y)
    assert_array_equal(ensemble.predict(X), y)


def _regression_data():
    X = np.arange(16, dtype=float).reshape(-1, 1)
    y = X[:, 0] / 2 + np.arange(16) % 3
    return X, y


def _residual_regressor():
    return ResidualBoostingRegressor(
        DecisionTreeRegressor(max_depth=2, random_state=0), n_estimators=3
    )


def test_residual_boosting_direct_predictions():
    X, y = _regression_data()
    estimator = _residual_regressor()
    assert estimator.fit(X, y) is estimator
    # Changing mixin order must not change the residual-fitting algorithm.
    expected = np.zeros_like(y)
    for _ in range(estimator.n_estimators):
        stage = clone(estimator.estimator).fit(X, y - expected)
        expected += stage.predict(X)
    assert_allclose(estimator.predict(X), expected)


@pytest.mark.parametrize(
    "estimator",
    [
        _residual_regressor(),
        imodels.DecisionTreeCCPRegressor(
            estimator_=DecisionTreeRegressor(random_state=0), desired_complexity=3
        ),
    ],
    ids=["residual_boosting", "pruned_tree"],
)
def test_regressor_in_voting(estimator):
    X, y = _regression_data()
    tree = DecisionTreeRegressor(max_depth=2, random_state=0)
    expected = (
        clone(estimator).fit(X, y).predict(X) + clone(tree).fit(X, y).predict(X)
    ) / 2
    ensemble = VotingRegressor(
        [("model", clone(estimator)), ("tree", tree)], n_jobs=1
    ).fit(X, y)
    assert_allclose(ensemble.predict(X), expected)


@pytest.mark.parametrize("residual_as_final", [False, True])
def test_residual_boosting_in_stacking(residual_as_final):
    X, y = _regression_data()
    residual = _residual_regressor()
    base = LinearRegression() if residual_as_final else residual
    final = residual if residual_as_final else LinearRegression()
    estimator = StackingRegressor(
        estimators=[("model", base)], final_estimator=final, cv=2, n_jobs=1
    ).fit(X, y)
    predictions = estimator.predict(X)
    assert predictions.shape == y.shape
    assert np.isfinite(predictions).all()


def test_automl_preserves_search_score():
    X = np.arange(12, dtype=float).reshape(-1, 1)
    y = np.zeros(12, dtype=int)
    y[[2, 7, 10, 11]] = 1
    estimator = imodels.AutoInterpretableClassifier(
        param_grid=[{"est": [DecisionTreeClassifier(max_depth=1, random_state=0)]}]
    ).fit(X, y, cv=2)
    expected = get_scorer("roc_auc")(estimator.est_.best_estimator_, X, y)
    # ClassifierMixin.score would silently substitute accuracy for ROC AUC.
    assert expected != pytest.approx(accuracy_score(y, estimator.predict(X)))
    assert estimator.score(X, y) == pytest.approx(expected)


def test_tao_regression_remains_experimental():
    X, y = _regression_data()
    tree_kwargs = {"max_depth": 2, "random_state": 0}
    estimator = imodels.TaoTreeRegressor(
        n_iters=0, randomize_tree=False, model_args=tree_kwargs
    )
    with pytest.warns(UserWarning, match="TAO regression is experimental"):
        assert estimator.fit(X, y) is estimator
    expected = DecisionTreeRegressor(**tree_kwargs).fit(X, y).predict(X)
    assert_allclose(estimator.predict(X), expected)
