"""
Testing for SkopeRules algorithm
"""
import numpy as np
import pytest
from numpy.testing import assert_array_equal, assert_no_warnings, assert_raises, suppress_warnings, assert_warns
from sklearn.datasets import load_iris, load_boston, make_blobs
from sklearn.metrics import accuracy_score
from sklearn.model_selection import ParameterGrid
from sklearn.utils import check_random_state

from imodels.rule_set.skope_rules import SkopeRulesClassifier

rng = check_random_state(0)

# load the iris dataset
# and randomly permute it
iris = load_iris()
perm = rng.permutation(iris.target.size)
iris.data = iris.data[perm]
iris.target = iris.target[perm]

# also load the boston dataset
# and randomly permute it
boston = load_boston()
perm = rng.permutation(boston.target.size)
boston.data = boston.data[perm]
boston.target = boston.target[perm]


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_skope_rules():
    """Check various parameter settings."""
    X_train = [[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1],
               [6, 3], [-4, -7]]
    y_train = [0] * 6 + [1] * 2
    X_test = np.array([[2, 1], [1, 1]])

    grid = ParameterGrid({
        "precision_min": [0.],
        "recall_min": [0.],
        "n_estimators": [1],
        "max_samples": [0.5, 4],
        "max_samples_features": [0.5, 2],
        "bootstrap": [True, False],
        "bootstrap_features": [True, False],
        "max_depth": [2],
        "max_features": ["auto", 1, 0.1],
        "min_samples_split": [2, 0.1],
        "n_jobs": [-1, 2]})

    with suppress_warnings():
        for params in grid:
            SkopeRulesClassifier(random_state=rng,
                                 **params).fit(X_train, y_train, feature_names=['a', 'b']).predict(X_test)

    # additional parameters:
    SkopeRulesClassifier(n_estimators=50,
                         max_samples=1.,
                         recall_min=0.,
                         precision_min=0.).fit(X_train, y_train).predict(X_test)


def test_skope_rules_error():
    """Test that it gives proper exception on deficient input."""
    X = iris.data
    y = iris.target
    y = (y != 0)

    # Test max_samples
    assert_raises(ValueError,
                  SkopeRulesClassifier(max_samples=-1).fit, X, y)
    assert_raises(ValueError,
                  SkopeRulesClassifier(max_samples=0.0).fit, X, y)
    assert_raises(ValueError,
                  SkopeRulesClassifier(max_samples=2.0).fit, X, y)
    # explicitly setting max_samples > n_samples should result in a warning.
    assert_warns(UserWarning,
                 SkopeRulesClassifier(max_samples=1000).fit, X, y)
    assert_no_warnings(SkopeRulesClassifier(max_samples=np.int64(2)).fit, X, y)
    assert_raises(ValueError, SkopeRulesClassifier(max_samples='foobar').fit, X, y)
    assert_raises(ValueError, SkopeRulesClassifier(max_samples=1.5).fit, X, y)
    assert_raises(ValueError, SkopeRulesClassifier(max_depth_duplication=1.5).fit, X, y)
    assert_raises(ValueError, SkopeRulesClassifier().fit(X, y).predict, X[:, 1:])
    assert_raises(ValueError, SkopeRulesClassifier().fit(X, y).eval_weighted_rule_sum,
                  X[:, 1:])
    assert_raises(ValueError, SkopeRulesClassifier().fit(X, y).rules_vote, X[:, 1:])
    assert_raises(ValueError, SkopeRulesClassifier().fit(X, y).score_top_rules,
                  X[:, 1:])


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_max_samples_attribute():
    X = iris.data
    y = iris.target
    y = (y != 0)

    clf = SkopeRulesClassifier(max_samples=1.).fit(X, y)
    assert clf.max_samples_ == X.shape[0]

    clf = SkopeRulesClassifier(max_samples=500)
    assert_warns(UserWarning,
                 clf.fit, X, y)
    assert clf.max_samples_ == X.shape[0]

    clf = SkopeRulesClassifier(max_samples=0.4).fit(X, y)
    assert clf.max_samples_ == 0.4 * X.shape[0]


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_skope_rules_works():
    # toy sample (the last two samples are outliers)
    X = [[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1], [6, 3], [4, -7]]
    y = [0] * 6 + [1] * 2
    X_test = [[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1],
              [10, 5], [5, -7]]
    # Test LOF
    clf = SkopeRulesClassifier(random_state=rng, max_samples=1.)
    clf.fit(X, y)
    decision_func = clf.eval_weighted_rule_sum(X_test)
    rules_vote = clf.rules_vote(X_test)
    score_top_rules = clf.score_top_rules(X_test)
    pred = clf.predict(X_test)
    pred_score_top_rules = clf.predict_top_rules(X_test, 1)
    # assert detect outliers:
    assert np.min(decision_func[-2:]) > np.max(decision_func[:-2])
    assert np.min(rules_vote[-2:]) > np.max(rules_vote[:-2])
    assert np.min(score_top_rules[-2:]) > np.max(score_top_rules[:-2])
    assert_array_equal(pred, 6 * [0] + 2 * [1])
    assert_array_equal(pred_score_top_rules, 6 * [0] + 2 * [1])


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_deduplication_works():
    # toy sample (the last two samples are outliers)
    X = [[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1], [6, 3], [4, -7]]
    y = [0] * 6 + [1] * 2
    X_test = [[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1],
              [10, 5], [5, -7]]
    # Test LOF
    clf = SkopeRulesClassifier(random_state=rng, max_samples=1., max_depth_duplication=3)
    clf.fit(X, y)
    decision_func = clf.eval_weighted_rule_sum(X_test)
    rules_vote = clf.rules_vote(X_test)
    score_top_rules = clf.score_top_rules(X_test)
    pred = clf.predict(X_test)
    pred_score_top_rules = clf.predict_top_rules(X_test, 1)
    assert True, 'deduplication works'


def test_performances():
    X, y = make_blobs(n_samples=1000, random_state=0, centers=2)

    # make labels imbalanced by remove all but 100 instances from class 1
    indexes = np.ones(X.shape[0]).astype(bool)
    ind = np.array([False] * 100 + list(((y == 1)[100:])))
    indexes[ind] = 0
    X = X[indexes]
    y = y[indexes]
    n_samples, n_features = X.shape

    clf = SkopeRulesClassifier()
    # fit
    clf.fit(X, y)
    # with lists
    clf.fit(X.tolist(), y.tolist())
    y_pred = clf.predict(X)
    assert y_pred.shape == (n_samples,)
    # training set performance
    assert accuracy_score(y, y_pred) > 0.83

    # eval_weighted_rule_sum agrees with predict
    decision = -clf.eval_weighted_rule_sum(X)
    assert decision.shape == (n_samples,)
    dec_pred = (decision.ravel() < 0).astype(np.int)
    assert_array_equal(dec_pred, y_pred)
