"""FPLasso should pass its constructor arguments to RuleFit intact."""

import inspect

import numpy as np
import pytest

from imodels import FPLassoClassifier, FPLassoRegressor
from imodels.rule_set.rule_fit import RuleFit


@pytest.mark.parametrize('cls', [FPLassoRegressor, FPLassoClassifier])
def test_constructor_arguments_reach_rulefit(cls):
    """random_state used to land in cv, because RuleFit takes cv first

    FPLasso forwarded 12 positional arguments to RuleFit, whose signature has
    cv between alpha and random_state, so the seed silently became the cv
    flag: FPLassoRegressor(random_state=7) fitted with random_state=None and
    cv=7.
    """
    m = cls(random_state=7)
    assert m.random_state == 7
    assert m.cv is True  # RuleFit's default, not the seed

    # every shared parameter keeps the value it was given
    shared = set(inspect.signature(RuleFit.__init__).parameters) & set(
        inspect.signature(cls.__init__).parameters)
    passed = dict(n_estimators=13, tree_size=5, max_rules=7, memory_par=0.5,
                  alpha=0.1, cv=False, random_state=3)
    assert set(passed) <= shared
    m = cls(**passed)
    for name, value in passed.items():
        assert getattr(m, name) == value, name


def test_seed_is_actually_used():
    """A fixed seed gives a reproducible fit."""
    rng = np.random.RandomState(0)
    X = (rng.randn(60, 4) > 0).astype(int)
    y = X[:, 0]
    coefs = []
    for _ in range(2):
        m = FPLassoRegressor(random_state=7, max_rules=5)
        m.fit(X, y)
        coefs.append([r.coef for r in m.rules_])
    assert coefs[0] == coefs[1]
