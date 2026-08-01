"""SHAP values for hierarchically-shrunk trees (issue #202).

shap is not a dependency of imodels, so these tests skip when it isn't installed.
"""

import numpy as np
import pytest
from sklearn.tree import DecisionTreeClassifier

from imodels import HSTreeClassifier

shap = pytest.importorskip('shap')


def _data():
    rng = np.random.RandomState(0)
    X = rng.randn(300, 4)
    y = ((X[:, 0] > 0) | (X[:, 1] > 1)).astype(int)
    return X, y


def test_tree_explainer_on_the_shrunk_estimator():
    """shap.TreeExplainer works on the wrapped estimator and sees the shrinkage"""
    X, y = _data()

    plain = DecisionTreeClassifier(max_leaf_nodes=8, random_state=0).fit(X, y)
    shrunk = HSTreeClassifier(
        DecisionTreeClassifier(max_leaf_nodes=8, random_state=0),
        reg_param=100).fit(X, y)

    # the wrapper itself is not a model type TreeExplainer knows
    with pytest.raises(Exception):
        shap.TreeExplainer(shrunk)

    explainer = shap.TreeExplainer(shrunk.estimator_)
    shap_values = np.array(explainer.shap_values(X))
    assert shap_values.shape[0] == len(X)

    # shrinkage is reflected: values differ from the unshrunk tree's
    plain_values = np.array(shap.TreeExplainer(plain).shap_values(X))
    assert not np.allclose(plain_values, shap_values)

    # and they explain the shrunk model: additivity against its predict_proba
    reconstructed = shap_values[:, :, 1].sum(axis=1) + explainer.expected_value[1]
    assert np.allclose(reconstructed, shrunk.predict_proba(X)[:, 1], atol=1e-6)
