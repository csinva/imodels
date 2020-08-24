from sklearn.utils.estimator_checks import check_estimator
from skrules import SkopeRules
from skrules.datasets import load_credit_data


def test_classifier():
    check_estimator(SkopeRules)


def test_load_credit_data():
    assert load_credit_data().data.shape[0] == 30000
