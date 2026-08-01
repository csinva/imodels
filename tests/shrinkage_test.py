import random
from functools import partial

import numpy as np
from sklearn.ensemble import VotingRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from imodels import HSTreeClassifier, HSTreeClassifierCV, \
    HSTreeRegressor, HSTreeRegressorCV, C45TreeClassifier
from imodels.tree.c45_tree.c45_tree import HSC45TreeClassifierCV
import random
from functools import partial

import numpy as np
from sklearn.ensemble import VotingRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from imodels import HSTreeClassifier, HSTreeClassifierCV, \
    HSTreeRegressor, HSTreeRegressorCV, C45TreeClassifier
from imodels.tree.c45_tree.c45_tree import HSC45TreeClassifierCV


class TestShrinkage:
    '''Tests simple classification for different models. Note: still doesn't test all the models!
    '''

    def setup_method(self):
        np.random.seed(13)
        random.seed(13)
        self.n = 20
        self.p = 2
        self.X_classification_binary = (
            np.random.randn(self.n, self.p) > 0).astype(int)

        # y = x0 > 0
        self.y_classification_binary = (
            self.X_classification_binary[:, 0] > 0).astype(int)

        # flip labels for last few
        self.y_classification_binary[-2:] = 1 - \
            self.y_classification_binary[-2:]
        self.X_regression = np.random.randn(self.n, self.p)
        self.y_regression = self.X_regression[:,
                                              0] + np.random.randn(self.n) * 0.01

    def test_classification_shrinkage(self):
        '''Test imodels on basic binary classification task
        '''

        for model_type in [
            partial(HSTreeClassifier, estimator_=DecisionTreeClassifier()),
            partial(HSTreeClassifier, estimator_=GradientBoostingClassifier()),
            partial(HSTreeClassifier, estimator_=DecisionTreeClassifier()),
            partial(HSTreeClassifierCV, estimator_=DecisionTreeClassifier()),
            partial(HSTreeClassifierCV, estimator_=RandomForestClassifier()),
            partial(HSC45TreeClassifierCV, estimator_=C45TreeClassifier()),
            HSTreeClassifierCV,  # default estimator is Decision tree with 25 max_leaf_nodes
        ]:
            init_kwargs = {}
            m = model_type(**init_kwargs)

            X = self.X_classification_binary
            m.fit(X, self.y_classification_binary)

            # test predict()
            preds = m.predict(X)  # > 0.5).astype(int)
            assert preds.size == self.n, 'predict() yields right size'

            # test preds_proba()
            preds_proba = m.predict_proba(X)
            assert len(preds_proba.shape) == 2, 'preds_proba has 2 columns'
            assert preds_proba.shape[1] == 2, 'preds_proba has 2 columns'
            assert np.max(
                preds_proba) < 1.1, 'preds_proba has no values over 1'
            assert (np.argmax(preds_proba, axis=1) == preds).all(
            ), ("predict_proba and ""predict correspond")

            # test acc
            acc_train = np.mean(preds == self.y_classification_binary)
            # print(type(m), m, 'final acc', acc_train)
            assert acc_train > 0.8, 'acc greater than 0.8'

            # complexity
            assert m.complexity_ > 0, 'complexity is greater than 0'

    def test_recognized_by_sklearn(self):
        base_models = [('hs', HSTreeRegressor(DecisionTreeRegressor())),
                       ('dt', DecisionTreeRegressor())]
        comb_model = VotingRegressor(estimators=base_models,
                                     n_jobs=10,
                                     verbose=2)
        comb_model.fit(self.X_classification_binary, self.y_regression)

    def test_regression_shrinkage(self):
        '''Test imodels on basic binary classification task
        '''
        for model_type in [partial(HSTreeRegressor, estimator_=DecisionTreeRegressor()),
                           partial(HSTreeRegressorCV,
                                   estimator_=DecisionTreeRegressor()),
                           ]:
            m = model_type()
            m.fit(self.X_regression, self.y_regression)

            preds = m.predict(self.X_regression)
            assert preds.size == self.n, 'predictions are right size'

            mse = np.mean(np.square(preds - self.y_regression))
            assert mse < 1, 'mse less than 1'

            # complexity
            assert m.complexity_ > 0, 'complexity is greater than 0'


if __name__ == '__main__':
    t = TestShrinkage()
    t.setup_method()
    t.test_classification_shrinkage()


def test_str_with_ensemble_estimator():
    """Printing a shrunk ensemble should summarize it rather than raise

    Regression test for https://github.com/csinva/imodels/issues/212:
    export_text only renders a single tree, so str() raised
    InvalidParameterError for RandomForest/GradientBoosting estimators.
    """
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.tree import DecisionTreeRegressor

    X = np.random.RandomState(0).randn(60, 3)
    y = X[:, 0]

    for estimator in [RandomForestRegressor(n_estimators=3, random_state=0),
                      GradientBoostingRegressor(n_estimators=3, random_state=0)]:
        model = HSTreeRegressor(estimator_=estimator).fit(X, y)
        printed = str(model)
        assert type(estimator).__name__ in printed
        assert '3 trees' in printed

    # single trees still print the tree itself
    single = HSTreeRegressor(DecisionTreeRegressor(max_leaf_nodes=3)).fit(X, y)
    assert '|---' in str(single)

    # unfitted models still print their parameters
    assert 'reg_param' in str(HSTreeRegressor(RandomForestRegressor()))
def test_unsupported_estimator_raises():
    """Shrinkage should reject models it cannot actually shrink

    Regression test for https://github.com/csinva/imodels/issues/199: wrapping
    an unsupported model used to fit and predict without error while leaving it
    completely unchanged.
    """
    import pytest
    from sklearn.linear_model import LinearRegression, LogisticRegression

    X = np.random.RandomState(0).randn(50, 3)

    with pytest.raises(ValueError, match='not supported by hierarchical shrinkage'):
        HSTreeClassifier(LogisticRegression()).fit(X, (X[:, 0] > 0).astype(int))

    with pytest.raises(ValueError, match='not supported by hierarchical shrinkage'):
        HSTreeRegressor(LinearRegression()).fit(X, X[:, 0])


def test_supported_estimators_still_shrink():
    """Trees and tree ensembles remain supported and are actually modified"""
    from copy import deepcopy

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.tree import DecisionTreeRegressor

    X = np.random.RandomState(0).randn(80, 3)
    y = X[:, 0]

    for estimator in [DecisionTreeRegressor(max_leaf_nodes=8),
                      RandomForestRegressor(n_estimators=3, random_state=0)]:
        unshrunk = deepcopy(estimator).fit(X, y)
        shrunk = HSTreeRegressor(deepcopy(estimator), reg_param=50).fit(X, y)
        assert not np.allclose(unshrunk.predict(X), shrunk.predict(X)), \
            f'shrinkage had no effect on {type(estimator).__name__}'


def test_missing_values_are_passed_to_the_estimator():
    """Shrinkage should not reject NaN that the wrapped estimator can handle

    Regression test for https://github.com/csinva/imodels/issues/213: imodels
    validated X itself and raised "Input contains NaN" before the estimator --
    which for sklearn trees and forests supports missing values -- saw the data.
    """
    import pytest
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

    rng = np.random.RandomState(0)
    X = rng.randn(80, 3)
    y = (X[:, 0] > 0).astype(int)
    X_nan = X.copy()
    X_nan[::7, 1] = np.nan

    for estimator in [DecisionTreeClassifier(max_leaf_nodes=8),
                      RandomForestClassifier(n_estimators=3, random_state=0)]:
        model = HSTreeClassifier(estimator, reg_param=10).fit(X_nan, y)
        assert model.predict(X_nan).shape == (80,)

    model = HSTreeRegressor(
        DecisionTreeRegressor(max_leaf_nodes=8), reg_param=10).fit(X_nan, X[:, 0])
    assert model.predict(X_nan).shape == (80,)

    # the CV variants score internally, so exercise those too
    assert HSTreeClassifierCV().fit(X_nan, y).predict(X_nan).shape == (80,)

    # estimators that can't handle missing values still say so themselves
    with pytest.raises(ValueError, match='NaN'):
        HSTreeClassifier(
            GradientBoostingClassifier(n_estimators=3)).fit(X_nan, y)


def test_models_do_not_share_a_default_estimator():
    """Two HSTree models must not wrap the same estimator object

    The default estimator used to be built once in the signature, so every
    HSTreeClassifier() shared it and fitting one model refit another's tree.
    """
    assert HSTreeClassifier().estimator_ is not HSTreeClassifier().estimator_
    assert HSTreeRegressor().estimator_ is not HSTreeRegressor().estimator_

    rng = np.random.RandomState(0)
    X = rng.randn(300, 4)
    y = (X[:, 0] + 1.5 * rng.randn(300) > 0).astype(int)

    first = HSTreeClassifier().fit(X, y)
    before = first.predict_proba(X).copy()

    HSTreeClassifier().fit(X, np.roll(y, 7))  # a different model, different labels

    assert np.allclose(before, first.predict_proba(X)), \
        'fitting one model changed another already-fitted model'

    # the defaults themselves still behave
    assert HSTreeRegressor().fit(X, X[:, 0]).predict(X).shape == (300,)
    assert isinstance(HSTreeRegressor().estimator_, DecisionTreeRegressor)
    assert isinstance(HSTreeClassifier().estimator_, DecisionTreeClassifier)
