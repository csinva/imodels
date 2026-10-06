import random
import unittest

import numpy as np
import pandas as pd

from imodels import BayesianRuleSetClassifier


class TestBRSClassifier(unittest.TestCase):
    def test_brs_recovers_a_planted_rule_set(self):
        '''BRS fits DataFrame and numpy input and recovers a planted rule set

        The rule generation fits min(n_columns ** length, 4000) trees for each
        rule length up to maxlen, so five binary features (ten columns with
        their negations) keep the fit to about a second, where the 27-feature
        tic-tac-toe data this test used before took over ten.
        '''
        rng = np.random.RandomState(0)
        X = pd.DataFrame((rng.rand(400, 5) > 0.5).astype(int),
                         columns=[f'x{i}' for i in range(5)])
        Y_clean = ((X.x0 & X.x1) | X.x2).to_numpy().astype(float)  # (x0 and x1) or x2
        Y = Y_clean.copy()
        flip = rng.rand(400) < 0.05
        Y[flip] = 1 - Y[flip]
        train, test = slice(0, 200), slice(200, 400)
        y_test = Y[test]

        np.random.seed(13)
        random.seed(13)
        model = BayesianRuleSetClassifier(n_rules=100,
                                          supp=5,
                                          maxlen=3,
                                          num_iterations=100,
                                          num_chains=2,
                                          alpha_pos=500, beta_pos=1,
                                          alpha_neg=500, beta_neg=1,
                                          alpha_l=None, beta_l=None,
                                          random_state=13)

        # fit and check accuracy
        model.fit(X[train], Y[train])
        y_pred = model.predict(X[test])
        acc1 = np.mean(y_pred == y_test)
        assert acc1 > 0.85

        # try fitting np version, on noise-free labels so that the search also
        # reaches a rule set with no training errors and takes its 'clean' move
        np.random.seed(13)
        random.seed(13)
        model.fit(X[train].values, Y_clean[train])
        y_pred = model.predict(X[test].values)
        acc2 = np.mean(y_pred == y_test)
        assert acc2 > 0.85


def test_extract_rules_from_a_tree_that_never_split():
    """A forest tree with no split is a single leaf: it gives no rule instead of crashing."""
    from sklearn.tree import DecisionTreeClassifier
    from imodels.rule_set.brs import _extract_rules
    stump = DecisionTreeClassifier().fit(np.zeros((6, 2)), [0, 1, 0, 1, 0, 1])
    assert stump.tree_.node_count == 1
    assert _extract_rules(stump, ['a', 'b']) == []
    split = DecisionTreeClassifier(max_depth=1).fit(np.array([[0, 0], [1, 0], [0, 1], [1, 1]]), [0, 1, 0, 1])
    assert _extract_rules(split, ['a', 'b']) == [['a_neg'], ['a']]


def test_brs_rejects_a_single_class_target():
    """Regression: a one-class y (e.g. the first 479 rows of the class-sorted tic-tac-toe data)
    crashed deep inside rule mining; it now raises a clear error."""
    import pytest
    X = np.random.RandomState(0).randint(0, 2, (40, 5))
    with pytest.raises(ValueError, match="at least 2 classes"):
        BayesianRuleSetClassifier().fit(X, np.ones(40))
