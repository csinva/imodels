import pytest
from imodels import *
from sklearn.datasets import make_regression, make_classification
from sklearn.base import clone

classifiers = [
    SLIMClassifier(),
    OptimalRuleListClassifier(),
    GreedyRuleListClassifier(),
    OneRClassifier(),
    BoostedRulesClassifier(),
    BayesianRuleSetClassifier(),
    RuleFitClassifier(),
    SkopeRulesClassifier(),
    SlipperClassifier(),
    C45TreeClassifier(),
    GreedyTreeClassifier(),
    FIGSClassifier(),
    FIGSClassifierCV(),
    HSTreeClassifier(),
    HSTreeClassifierCV(),
    TaoTreeClassifier(),
]


regressors = [
    SLIMRegressor(),
    BoostedRulesRegressor(),
    RuleFitRegressor(),
    GreedyTreeRegressor(),
    FIGSRegressor(),
    FIGSRegressorCV(),
    HSTreeRegressor(),
    HSTreeRegressorCV(),
    TaoTreeRegressor(),
]


@pytest.mark.parametrize("classifier", classifiers)
def test_fit_before_print_classifier(classifier) -> None:
    X, y = make_classification(n_samples=25, n_features=5)
    print(classifier)


@pytest.mark.parametrize("regressor", regressors)
def test_fit_before_print_regressor(regressor) -> None:
    X, y = make_regression(n_samples=25, n_features=5)
    print(regressor)


classifiers_fit_pass = [
    SLIMClassifier(),
    GreedyRuleListClassifier(),
    OneRClassifier(),
    BoostedRulesClassifier(),
    RuleFitClassifier(),
    SkopeRulesClassifier(),
    C45TreeClassifier(),
    GreedyTreeClassifier(),
    FIGSClassifier(),
    FIGSClassifierCV(),
    HSTreeClassifier(),
    HSTreeClassifierCV(),
    TaoTreeClassifier(),
]


regressors_fit_pass = [
    SLIMRegressor(),
    BoostedRulesRegressor(),
    RuleFitRegressor(),
    GreedyTreeRegressor(),
    FIGSRegressor(),
    FIGSRegressorCV(),
    HSTreeRegressor(),
    HSTreeRegressorCV(),
]


@pytest.mark.parametrize("classifier", classifiers_fit_pass)
def test_fit_after_print_classifier(classifier) -> None:
    X, y = make_classification(n_samples=25, n_features=5)
    classifier.fit(X, y)
    print(classifier.fit(X, y))


@pytest.mark.parametrize("regressor", regressors_fit_pass)
def test_fit_after_print_regressor(regressor) -> None:
    X, y = make_regression(n_samples=25, n_features=5)
    regressor.fit(X, y)
    print(regressor)


classifiers_print = [
    GreedyRuleListClassifier(),
    BayesianRuleSetClassifier(),
    GreedyTreeClassifier(),
    FIGSClassifier(),
    HSTreeClassifier(),
]


regressors_print = [
    FIGSRegressor(),
    HSTreeRegressor(),
]


@pytest.mark.parametrize("classifier", classifiers_print)
def test_print_classifier(classifier) -> None:
    X, y = make_classification(n_samples=25, n_features=5)
    classifier_ = clone(classifier)
    classifier_.fit(X, y)
    assert classifier_.__str__ != classifier_.__class__.__name__


@pytest.mark.parametrize("regressor", regressors_print)
def test_print_regressor(regressor) -> None:
    X, y = make_regression(n_samples=25, n_features=5)
    regressor_ = clone(regressor)
    regressor_.fit(X, y)
    assert regressor_.__str__ != regressor_.__class__.__name__
