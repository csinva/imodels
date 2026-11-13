"""Refitting must not reuse post-hoc coefficients from the previous trees."""
import numpy as np
import pytest
from sklearn.base import clone
from sklearn.datasets import make_classification, make_regression

from imodels import TreeGAMClassifier, TreeGAMRegressor


@pytest.mark.parametrize('classifier', [False, True])
@pytest.mark.parametrize('posthoc', [None, 'ridge'])
def test_refit_after_posthoc_matches_fresh_estimator(classifier, posthoc):
    if classifier:
        X, y = make_classification(n_samples=80, n_features=2,
                                   n_informative=2, n_redundant=0, random_state=1)
        estimator_class = TreeGAMClassifier
    else:
        X, y = make_regression(n_samples=80, n_features=2, random_state=1)
        estimator_class = TreeGAMRegressor
    model = estimator_class(n_boosting_rounds=2,
                            fit_posthoc_tree_coefs='ridge', random_state=0)
    model.fit(X, y)
    assert hasattr(model, 'cyclic_coef_')
    model.set_params(n_boosting_rounds=8, fit_posthoc_tree_coefs=posthoc)
    fresh = clone(model).fit(X, y)
    model.fit(X, y)
    assert len(model.estimators_) == len(fresh.estimators_)
    np.testing.assert_allclose(model.predict_proba(X), fresh.predict_proba(X))
    if posthoc is None:
        assert not hasattr(model, 'cyclic_coef_')
    else:
        np.testing.assert_allclose(model.cyclic_coef_, fresh.cyclic_coef_)
