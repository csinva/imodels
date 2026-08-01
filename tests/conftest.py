import random

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def seed_global_rngs():
    """Seed the global RNGs before every test.

    Some models draw from the global numpy/random state, and some (e.g.
    BayesianRuleListClassifier) reseed it themselves during fit. Without this,
    a test's result can depend on which tests ran before it.
    """
    np.random.seed(13)
    random.seed(13)
