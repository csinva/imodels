"""Tests for the shared get_rules API (imodels.get_rules / model.get_rules())."""

import numpy as np
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeClassifier

import imodels
from imodels.util.get_rules import CORE_COLUMNS

N_SAMPLES = 120
FEATURE_NAMES = ['age', 'bmi', 'bp']

# one model per way of storing rules, so every extraction path is covered
RULE_MODELS = {
    'RuleFitClassifier': dict(max_rules=4, random_state=0),
    'SkopeRulesClassifier': dict(random_state=0, max_depth_duplication=1),
    'SlipperClassifier': dict(n_estimators=2),
    'GreedyRuleListClassifier': {},
    'OneRClassifier': {},
    'FIGSClassifier': dict(max_rules=4),
    'FIGSClassifierCV': dict(n_rules_list=[3], n_trees_list=[2], cv=2),
    'BoostedRulesClassifier': dict(n_estimators=3),
    'GreedyTreeClassifier': dict(max_leaf_nodes=4),
    'HSTreeClassifier': {},
    'HSTreeClassifierCV': {},
    'C45TreeClassifier': dict(max_rules=3),
    'TaoTreeClassifier': {},
    'TreeGAMClassifier': dict(n_boosting_rounds=3, random_state=0),
    'DecisionTreeCCPClassifier': dict(
        estimator_=DecisionTreeClassifier(random_state=0), desired_complexity=3),
}

# models whose rules are defined over pre-discretized features
BINARY_INPUT_MODELS = {'BayesianRuleListClassifier', 'FPSkopeClassifier'}
RULE_MODELS.update({
    'BayesianRuleListClassifier': dict(max_iter=2000, n_chains=1),
    'FPSkopeClassifier': dict(random_state=0, recall_min=0.5,
                              max_depth_duplication=1),
})

IDS = sorted(RULE_MODELS)


def _data(model_name):
    rng = np.random.RandomState(0)
    X = pd.DataFrame(rng.randn(N_SAMPLES, 3), columns=FEATURE_NAMES)
    y = (X['age'] + 0.8 * rng.randn(N_SAMPLES) > 0).astype(int)
    if model_name in BINARY_INPUT_MODELS:
        X = (X > 0).astype(int)
    return X, y


def _fit(model_name):
    X, y = _data(model_name)
    model = getattr(imodels, model_name)(**RULE_MODELS[model_name])
    model.fit(X, y)
    return model, X, y


@pytest.mark.parametrize('model_name', IDS)
def test_get_rules_contract(model_name):
    """Every rule model returns a non-empty frame with the documented columns"""
    model, X, y = _fit(model_name)

    rules = model.get_rules()
    assert isinstance(rules, pd.DataFrame)
    assert len(rules) > 0, 'a fitted model should expose at least one rule'

    # the guaranteed columns come first, in order
    assert list(rules.columns[:len(CORE_COLUMNS)]) == CORE_COLUMNS
    assert rules['rule'].map(lambda r: isinstance(r, str)).all()
    assert (rules['rule'].str.len() > 0).all()
    assert list(rules.index) == list(range(len(rules)))

    # the module-level function gives the same thing
    pd.testing.assert_frame_equal(rules, imodels.get_rules(model))


@pytest.mark.parametrize('model_name', IDS)
def test_rules_use_feature_names(model_name):
    """Rules are written in terms of the columns the model was fitted on"""
    model, X, y = _fit(model_name)
    rules_text = ' '.join(model.get_rules()['rule'])

    # BayesianRuleList renames features internally, so only check the others
    if model_name != 'BayesianRuleListClassifier':
        assert any(name in rules_text for name in FEATURE_NAMES), rules_text


def test_feature_names_can_be_overridden():
    """Passing feature_names renames the features in the returned rules"""
    model, X, y = _fit('GreedyTreeClassifier')
    renamed = model.get_rules(feature_names=['A', 'B', 'C'])
    assert any('A' in rule for rule in renamed['rule'])
    assert not any('age' in rule for rule in renamed['rule'])


def test_numpy_input_falls_back_to_positional_names():
    """Without column names, rules refer to X0, X1, ..."""
    rng = np.random.RandomState(0)
    X = rng.randn(N_SAMPLES, 3)
    y = (X[:, 0] > 0).astype(int)

    model = imodels.GreedyTreeClassifier(max_leaf_nodes=4).fit(X, y)
    assert any('X0' in rule for rule in model.get_rules()['rule'])


def test_ensembles_label_each_tree():
    """Models made of several trees say which tree each rule came from"""
    model, X, y = _fit('BoostedRulesClassifier')
    rules = model.get_rules()
    assert 'tree' in rules.columns
    assert rules['tree'].nunique() == len(model.estimators_)

    # a single tree needs no such column
    single, _, _ = _fit('GreedyTreeClassifier')
    assert 'tree' not in single.get_rules().columns


def test_rulefit_keeps_its_extra_columns():
    """Model-specific columns are preserved alongside the core ones"""
    model, X, y = _fit('RuleFitClassifier')
    rules = model.get_rules()
    for col in ['coef', 'support', 'importance']:
        assert col in rules.columns


def test_shrinkage_rules_reflect_shrinkage():
    """HSTree reports the shrunk predictions, not the unshrunk tree's"""
    X, y = _data('HSTreeClassifier')
    tree = DecisionTreeClassifier(max_leaf_nodes=4, random_state=0)

    unshrunk = imodels.GreedyTreeClassifier(
        max_leaf_nodes=4, random_state=0).fit(X, y).get_rules()
    shrunk = imodels.HSTreeClassifier(tree, reg_param=100).fit(X, y).get_rules()

    assert list(shrunk['rule']) == list(unshrunk['rule'])  # same tree structure
    assert not np.allclose(shrunk['prediction'], unshrunk['prediction'])


def test_unsupported_model_raises_a_clear_error():
    """Models that aren't rule-based say so, naming themselves"""
    with pytest.raises(ValueError, match='SLIMClassifier'):
        imodels.get_rules(imodels.SLIMClassifier())


def test_unfitted_model_raises_a_clear_error():
    with pytest.raises(ValueError, match='fit it first'):
        imodels.get_rules(imodels.FIGSClassifier())


def _fires(X, rule):
    """Boolean mask of the rows a rule string matches."""
    if rule == 'else':
        return np.ones(len(X), dtype=bool)
    return X.eval(rule.replace(' and ', ' & ')).values


def test_figs_rule_predictions_reproduce_the_model():
    """For FIGS, summing the fired rules' predictions gives the model's output

    FIGS is a sum of trees, so each rule's `prediction` is the contribution of
    its tree. This pins that documented meaning.
    """
    X, y = _data('FIGSClassifier')
    model = imodels.FIGSClassifier(max_rules=5).fit(X, y)
    rules = model.get_rules()

    expected = np.sum([np.asarray(model._predict_tree(tree, X.values))
                       .reshape(len(X), -1)[:, -1] for tree in model.trees_], axis=0)

    total = np.zeros(len(X))
    for _, tree_rules in rules.groupby('tree'):
        contribution = np.zeros(len(X))
        for _, rule in tree_rules.iterrows():
            contribution[_fires(X, rule['rule'])] = rule['prediction']
        total += contribution

    assert np.allclose(total, expected)


def test_tree_rule_predictions_match_predict_proba():
    """For a single tree, each rule's prediction is that leaf's probability"""
    X, y = _data('GreedyTreeClassifier')
    model = imodels.GreedyTreeClassifier(max_leaf_nodes=4, random_state=0).fit(X, y)
    rules = model.get_rules()

    predicted = np.zeros(len(X))
    for _, rule in rules.iterrows():
        predicted[_fires(X, rule['rule'])] = rule['prediction']

    assert np.allclose(predicted, model.predict_proba(X)[:, 1])
