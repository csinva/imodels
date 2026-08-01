"""RuleFit with an XGBoost tree generator (issue #227).

xgboost is not a dependency of imodels, so these tests skip when it isn't installed.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, r2_score

from imodels import RuleFitClassifier, RuleFitRegressor
from imodels.util.convert import is_xgboost_model, xgboost_to_rules

xgb = pytest.importorskip('xgboost')

N_SAMPLES = 500
FEATURE_NAMES = ['a', 'b', 'c']


def _data(seed=0):
    rng = np.random.RandomState(seed)
    X = pd.DataFrame(rng.randn(N_SAMPLES, 3), columns=FEATURE_NAMES)
    y = (X['a'] + X['b'] + 0.5 * rng.randn(N_SAMPLES) > 0).astype(int)
    return X, y


def test_detection_does_not_import_xgboost():
    """Detection is duck-typed, so imodels never imports xgboost itself"""
    from sklearn.ensemble import GradientBoostingClassifier

    X, y = _data()
    assert is_xgboost_model(xgb.XGBClassifier())
    assert not is_xgboost_model(GradientBoostingClassifier())

    import imodels.util.convert as convert_module
    source = open(convert_module.__file__).read()
    assert 'import xgboost' not in source


@pytest.mark.parametrize('depth,n_estimators', [(3, 1), (4, 3), (2, 5)])
def test_rules_reproduce_xgboost_routing(depth, n_estimators):
    """Every rule must select exactly the rows XGBoost sends to that leaf"""
    X, y = _data()
    model = xgb.XGBClassifier(
        n_estimators=n_estimators, max_depth=depth).fit(X, y)

    frame = model.get_booster().trees_to_dataframe()
    leaf_nodes = model.get_booster().predict(
        xgb.DMatrix(X), pred_leaf=True).reshape(len(X), -1)
    rules = xgboost_to_rules(model, FEATURE_NAMES, prediction_values=True)

    # xgboost compares in float32, so evaluate the rules the same way
    X32 = X.astype(np.float32)
    n_leaves = [int((frame.Tree == t).mul(frame.Feature == 'Leaf').sum())
                for t in range(n_estimators)]

    start = 0
    for tree in range(n_estimators):
        tree_rules = rules[start:start + n_leaves[tree]]
        start += n_leaves[tree]

        sub = frame[frame.Tree == tree]
        value_by_node = {int(r.Node): float(r.Gain)
                         for r in sub[sub.Feature == 'Leaf'].itertuples()}
        expected = np.array([value_by_node[n] for n in leaf_nodes[:, tree]])

        predicted = np.full(len(X), np.nan)
        matches = np.zeros(len(X), dtype=int)
        for condition, value in tree_rules:
            mask = X32.eval(condition.replace(' and ', ' & ')).values
            predicted[mask] = value[0]
            matches += mask

        assert (matches == 1).all(), 'rules should partition the rows'
        assert np.allclose(predicted, expected), 'rule values must match the leaves'


def test_rulefit_classifier_with_xgboost():
    X, y = _data()
    model = RuleFitClassifier(
        tree_generator=xgb.XGBClassifier(n_estimators=10, max_depth=3),
        random_state=0).fit(X, y)

    assert len(model.rules_) > 0
    assert accuracy_score(y, model.predict(X)) > 0.9
    # rules are written with the DataFrame's column names
    assert any(name in ' '.join(model.get_rules()['rule'])
               for name in FEATURE_NAMES)


def test_rulefit_regressor_with_xgboost():
    X, _ = _data()
    y = X['a'] + 0.3 * np.random.RandomState(1).randn(N_SAMPLES)

    model = RuleFitRegressor(
        tree_generator=xgb.XGBRegressor(n_estimators=10, max_depth=3),
        random_state=0).fit(X, y)

    assert r2_score(y, model.predict(X)) > 0.8


def test_unsupported_generator_still_rejected():
    from sklearn.linear_model import LinearRegression

    X, y = _data()
    with pytest.raises(ValueError, match='RuleFit only works with'):
        RuleFitClassifier(tree_generator=LinearRegression()).fit(X, y)
