import pytest
from imodels import *
from sklearn.datasets import make_regression, make_classification


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
def test_fit_classifier(classifier) -> None:
    X, y = make_classification(n_samples=25, n_features=5)
    classifier.fit(X, y)


@pytest.mark.parametrize("regressor", regressors)
def test_fit_regressor(regressor) -> None:
    X, y = make_regression(n_samples=25, n_features=5)
    regressor.fit(X, y)
