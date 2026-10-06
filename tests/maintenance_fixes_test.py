"""Regression tests for shared-code and older-model fixes in 3.0.3.

Each test names the bug it pins: refits keeping stale feature names, continuous
targets accepted by classifiers, TAO regression, GreedyRuleList class_weight, numpy 2
aliases, single-class targets, deprecation warnings, CCP constructors and packaging.
"""
import os
import subprocess
import sys
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeClassifier

import imodels
from imodels.util.arguments import check_fit_arguments, set_feature_names_in
from tests.model_configs import (BINARY_INPUT_MODELS, EXCLUDED_MODELS,
                                 model_kwargs)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rng = np.random.RandomState(0)
X = rng.randn(60, 3)
Y_CLS = (X[:, 0] + 0.3 * rng.randn(60) > 0).astype(int)
Y_REG = X[:, 0] + 0.1 * rng.randn(60)
DF = pd.DataFrame(X, columns=['a', 'b', 'c'])
REFIT_MODELS = [m for m in imodels.ESTIMATORS
                if m.__name__ not in BINARY_INPUT_MODELS | set(EXCLUDED_MODELS)]


def _quiet(f, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return f(*args, **kwargs)


# 1. a refit on data without column names drops feature_names_in_ ##############
@pytest.mark.parametrize('model_type', REFIT_MODELS, ids=lambda m: m.__name__)
def test_numpy_refit_drops_stale_feature_names(model_type):
    y = Y_CLS if model_type in imodels.CLASSIFIERS else Y_REG
    kwargs = model_kwargs(model_type.__name__)
    if model_type.__name__ == 'FastSmallTreeClassifier':
        kwargs['regularization'] = 0.3  # keep the exact search short
    m = model_type(**kwargs)
    _quiet(m.fit, DF, y)
    assert list(m.feature_names_in_) == ['a', 'b', 'c']
    _quiet(m.fit, X[:, :2], y)
    assert not hasattr(m, 'feature_names_in_')
    assert m.n_features_in_ == 2
    _quiet(m.predict, X[:, :2])


def test_set_feature_names_in_deletes_stale_names():
    class M:
        pass
    m = M()
    assert set_feature_names_in(m, DF)
    assert not set_feature_names_in(m, X)
    assert not hasattr(m, 'feature_names_in_')


@pytest.mark.parametrize('model', [
    imodels.FIGSClassifier(), imodels.HSTreeClassifier(),
    imodels.BoostedRulesClassifier(n_estimators=3), imodels.TaoTreeClassifier(n_iters=2),
    imodels.SlipperClassifier(n_estimators=2), imodels.TreeGAMClassifier(n_boosting_rounds=2),
    imodels.GreedyTreeRegressor(), imodels.HSTreeRegressor(), imodels.HSTreeRegressorCV(cv=2),
], ids=lambda m: type(m).__name__)
def test_get_rules_after_numpy_refit(model):
    y = Y_REG if 'Regressor' in type(model).__name__ else Y_CLS
    _quiet(model.fit, DF, y)
    _quiet(model.fit, X[:, :2], y)
    rules = ' '.join(imodels.get_rules(model)['rule'].astype(str))
    assert not any(f'{c} ' in rules or rules.startswith(c) for c in 'abc'), rules
    assert 'X0' in rules or 'X1' in rules or rules.strip() == '' or 'X_' in rules


def test_ccp_refit_drops_stale_feature_names():
    m = imodels.DecisionTreeCCPClassifier(desired_complexity=2).fit(DF, Y_CLS)
    assert hasattr(m, 'feature_names_in_')
    m.fit(X, Y_CLS)
    assert not hasattr(m, 'feature_names_in_')


# 2. classification targets are checked before they are encoded ###############
@pytest.mark.parametrize('model_type', [
    imodels.GreedyTreeClassifier, imodels.BoostedRulesClassifier, imodels.OneRClassifier,
    imodels.GreedyRuleListClassifier, imodels.C45TreeClassifier],
    ids=lambda m: m.__name__)
def test_continuous_target_is_rejected(model_type):
    with pytest.raises(ValueError, match='Unknown label type'):
        model_type().fit(X, rng.randn(60))


def test_check_fit_arguments_accepts_non_ndarray_y():
    """sklearn's check_classifier_data_not_an_array forbids np functions on y"""
    from sklearn.utils.estimator_checks import _NotAnArray
    m = imodels.GreedyTreeClassifier()
    _, y, _ = check_fit_arguments(m, X, _NotAnArray(Y_CLS), None)
    assert list(m.classes_) == [0, 1] and set(y) == {0.0, 1.0}
    imodels.GreedyRuleListClassifier().fit(_NotAnArray(X), _NotAnArray(Y_CLS))


# 3. TaoTreeRegressor fits ########################################################
def test_tao_regressor_fits_and_improves_random_tree():
    with pytest.warns(UserWarning, match='experimental'):
        m = imodels.TaoTreeRegressor(n_iters=5).fit(X, Y_REG)
    assert m.score(X, Y_REG) > 0.5
    np.random.seed(0)
    before = _quiet(imodels.TaoTreeRegressor(n_iters=0, randomize_tree=True).fit, X, Y_REG)
    np.random.seed(0)
    after = _quiet(imodels.TaoTreeRegressor(n_iters=5, randomize_tree=True).fit, X, Y_REG)
    assert after.score(X, Y_REG) > before.score(X, Y_REG)
    assert imodels.TaoTreeRegressor in imodels.REGRESSORS


# 4. GreedyRuleListClassifier honours class_weight ################################
def test_greedy_rule_list_class_weight():
    r = np.random.RandomState(0)
    x = r.rand(400, 1)
    y = (r.rand(400) < x[:, 0] ** 3).astype(int)  # the best stump moves with the weights

    def cutoff(**kw):
        return imodels.GreedyRuleListClassifier(max_depth=1, **kw).fit(x, y).rules_[0]['cutoff']
    assert cutoff() != cutoff(class_weight={0: 1, 1: 50})
    assert cutoff() != cutoff(class_weight='balanced')
    p1 = imodels.GreedyRuleListClassifier(max_depth=1, class_weight={0: 1, 1: 50}).fit(
        x, y).predict_proba(x)
    with pytest.raises(ValueError):
        imodels.GreedyRuleListClassifier(class_weight='nonsense').fit(x, y)
    # keyed by the original labels, here strings
    ys = np.where(y == 1, 'pos', 'neg')
    ps = imodels.GreedyRuleListClassifier(max_depth=1, class_weight={'neg': 1, 'pos': 50}).fit(
        x, ys).predict_proba(x)
    np.testing.assert_allclose(ps, p1)
    with pytest.raises(ValueError, match='not in y'):
        imodels.GreedyRuleListClassifier(class_weight={'other': 2}).fit(x, ys)
    assert not hasattr(imodels.GreedyRuleListClassifier, '_neg_corr_criterion')


# 5. numpy 2 removed np.NaN / np.Inf ##############################################
def test_rf_plus_gradient_boosting_under_numpy2():
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.linear_model import RidgeCV
    from imodels.tree.rf_plus.rf_plus.rf_plus_models import RandomForestPlusRegressor
    m = RandomForestPlusRegressor(
        prediction_model=RidgeCV(),
        rf_model=GradientBoostingRegressor(n_estimators=3, init='zero'))
    m.fit(X, Y_REG)
    assert m.predict(X).shape == (60,)


def test_no_removed_numpy_aliases():
    import re
    pattern = re.compile(r'np\.(NaN|Inf|NINF|PINF|infty|int|float|bool)\b(?!\w)')
    hits = []
    for root, _, files in os.walk(os.path.join(REPO, 'imodels')):
        for f in files:
            if f.endswith('.py'):
                path = os.path.join(root, f)
                for i, line in enumerate(open(path, encoding='utf8'), 1):
                    if pattern.search(line.split('#')[0]):
                        hits.append(f'{path}:{i}')
    assert not hits, hits


# 6. single-class y gives a clear error #########################################
@pytest.mark.parametrize('model_type', [
    imodels.GreedyRuleListClassifier, imodels.FastFrugalTreeClassifier,
    imodels.OneRClassifier, imodels.TreeGAMClassifier, imodels.SlipperClassifier,
    imodels.C45TreeClassifier], ids=lambda m: m.__name__)
def test_single_class_target_raises(model_type):
    with pytest.raises(ValueError, match='at least 2 classes'):
        model_type().fit(X, np.zeros(60, int))


# 7. deprecated names warn once and stay listed ###################################
def test_deprecated_names_warn_once():
    code = '''
import warnings
warnings.simplefilter("always")
with warnings.catch_warnings(record=True) as w:
    from imodels import *
    from imodels import SLIMClassifier
    import imodels
    imodels.SLIMClassifier
    imodels.MarginalShrinkageLinearModelRegressor
    imodels.MarginalShrinkageLinearModelRegressor
msgs = [str(x.message) for x in w if issubclass(x.category, FutureWarning)]
assert sum("SLIMClassifier is deprecated" in m for m in msgs) == 1, msgs
assert sum("MarginalShrinkageLinearModelRegressor" in m for m in msgs) == 1, msgs
assert "SLIMClassifier" in dir(imodels) and "SLIMClassifier" in imodels.__all__
assert imodels.MarginalShrinkageLinearModelRegressor is imodels.MarginalShrinkageLinearRegressor
'''
    subprocess.run([sys.executable, '-c', code], check=True, cwd=REPO)


# 8. every registered estimator constructs with no arguments ####################
def test_all_estimators_construct_without_arguments():
    for estimator in imodels.ESTIMATORS:
        estimator()


def test_ccp_default_estimator():
    from sklearn.tree import DecisionTreeRegressor
    c = imodels.DecisionTreeCCPClassifier(desired_complexity=2).fit(X, Y_CLS)
    assert isinstance(c.estimator_, DecisionTreeClassifier)
    r = imodels.DecisionTreeCCPRegressor(desired_complexity=2).fit(X, Y_REG)
    assert isinstance(r.estimator_, DecisionTreeRegressor)
    given = DecisionTreeClassifier(max_depth=1)
    m = imodels.DecisionTreeCCPClassifier(given, desired_complexity=2).fit(X, Y_CLS)
    assert m.estimator_.max_depth == 1 and not hasattr(given, 'tree_')


# 9. the license file ships ######################################################
def test_license_file_is_declared():
    text = open(os.path.join(REPO, 'pyproject.toml'), encoding='utf8').read()
    assert 'license-files = ["license.md"]' in text
    assert os.path.exists(os.path.join(REPO, 'license.md'))
