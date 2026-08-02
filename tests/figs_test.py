import os
import random
from functools import partial

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

from imodels import FIGSClassifier, FIGSRegressor, FIGSClassifierCV, FIGSRegressorCV
from imodels.experimental.figs_ensembles import FIGSExtRegressor, FIGSExtClassifier
from sklearn.ensemble import StackingRegressor, VotingRegressor, BaggingClassifier

path_to_tests = os.path.dirname(os.path.realpath(__file__))


class TestFIGS:

    def setup_method(self):
        '''Test on synthetic dataset
        '''
        np.random.seed(13)
        random.seed(13)
        self.n = 100
        self.p = 2
        self.X = (np.random.randn(self.n, self.p) > 0).astype(int)

        # y = x0 > 0 * x1 > 0
        self.y_classification_binary = (self.X[:, 0] > 0).astype(int) * (
            self.X[:, 1] > 0).astype(int)
        self.y_reg = self.X[:, 0] + self.X[:, 1]

    def test_recognized_by_sklearn(self):
        base_models = [('figs', FIGSRegressor()),
                       ('random_forest', DecisionTreeRegressor())]
        comb_model = VotingRegressor(estimators=base_models,
                                     n_jobs=10,
                                     verbose=2)
        comb_model.fit(self.X, self.y_reg)

    def test_categorical(self):
        """Test FIGS with categorical data"""
        categories = ['cat', 'dog', 'bird', 'fish']
        categories_2 = ['bear', 'chicken', 'cow']

        self.X_cat = pd.DataFrame(self.X)
        # 1d: a (n, 1) array becomes a column of arrays under pandas >= 3
        self.X_cat['pet1'] = np.random.choice(categories, size=self.n)
        self.X_cat['pet2'] = np.random.choice(categories_2, size=self.n)

        figs_reg = FIGSRegressor()
        figs_cls = FIGSClassifier()

        figs_reg.fit(self.X_cat, self.y_reg,
                     categorical_features=["pet1", 'pet2'])
        figs_reg.predict(self.X_cat, categorical_features=["pet1", 'pet2'])

        figs_cls.fit(self.X_cat, self.y_reg,
                     categorical_features=["pet1", 'pet2'])
        figs_cls.predict_proba(
            self.X_cat, categorical_features=["pet1", 'pet2'])

    def test_fitting(self):
        '''Test on a real (small) dataset
        '''
        for model_type in [
            FIGSClassifier, FIGSRegressor,
            FIGSExtClassifier, FIGSExtRegressor,
            FIGSClassifierCV, FIGSRegressorCV,
            partial(BaggingClassifier,
                    estimator=FIGSExtClassifier(max_rules=3),
                    n_estimators=2),
        ]:

            init_kwargs = {}
            m = model_type(**init_kwargs)

            X = self.X
            m.fit(X, self.y_classification_binary)

            # test predict()
            preds = m.predict(X)  # > 0.5).astype(int)
            assert preds.size == self.n, 'predict() yields right size'

            # test preds_proba()
            if model_type in [FIGSClassifier, FIGSClassifierCV, BaggingClassifier]:
                preds_proba = m.predict_proba(X)
                assert len(preds_proba.shape) == 2, 'preds_proba has 2 columns'
                assert preds_proba.shape[1] == 2, 'preds_proba has 2 columns'
                assert np.max(
                    preds_proba) < 1.1, 'preds_proba has no values over 1'
                assert (np.argmax(preds_proba, axis=1) == preds).all(), ("predict_proba and "
                                                                         "predict correspond")

            # test acc
            acc_train = np.mean(preds == self.y_classification_binary)
            assert acc_train > 0.9, 'acc greater than 0.9'
            # print(m)

            if not type(m) in [FIGSClassifierCV, FIGSRegressorCV, BaggingClassifier]:
                trees = m.trees_
                assert len(trees) == 1, 'only one tree'
                assert trees[0].feature == 1, 'split on feat 1'
                #assert np.abs(trees[0].left.value[0]) < 0.01, 'left value 0'
                assert trees[0].left.left is None and trees[0].left.right is None, 'left is leaf'
                #assert np.abs(
                #    trees[0].right.left.value[0]) < 0.01, 'right-left value 0'
                #assert np.abs(trees[0].right.right.value[0] -
                #              1) < 0.01, 'right-right value 1'


if __name__ == '__main__':
    t = TestFIGS()
    t.setup_method()
    t.test_recognized_by_sklearn()
    # t.test_fitting()
    # t.test_categorical()


def test_class_weight():
    """FIGSClassifier should accept class_weight, like sklearn classifiers

    Feature request https://github.com/csinva/imodels/issues/195: passing
    class_weight used to raise TypeError.
    """
    import pytest
    from sklearn.metrics import balanced_accuracy_score

    from imodels import FIGSClassifier, FIGSRegressor

    rng = np.random.RandomState(0)
    X = rng.randn(400, 3)
    y = (X[:, 0] + rng.randn(400) > 1.3).astype(int)  # ~15% positives

    unweighted = FIGSClassifier(max_rules=5).fit(X, y)
    balanced = FIGSClassifier(max_rules=5, class_weight='balanced').fit(X, y)

    # upweighting the rare class should help it be predicted
    assert (balanced_accuracy_score(y, balanced.predict(X)) >
            balanced_accuracy_score(y, unweighted.predict(X)))

    # an explicit dict works too, and gets combined with sample_weight
    explicit = FIGSClassifier(max_rules=5, class_weight={0: 1, 1: 5}).fit(X, y)
    assert explicit.predict(X).shape == (400,)
    combined = FIGSClassifier(max_rules=5, class_weight='balanced').fit(
        X, y, sample_weight=np.ones(len(y)) * 3)
    assert combined.predict(X).shape == (400,)

    # it round-trips through get_params/set_params like any other parameter
    assert FIGSClassifier(class_weight='balanced').get_params()[
        'class_weight'] == 'balanced'

    # and is rejected where it has no meaning
    with pytest.raises(ValueError, match='only meaningful for classification'):
        FIGSRegressor(class_weight='balanced').fit(X, X[:, 0])


def test_feature_importances_use_sample_weight():
    """FIGS feature_importances_ should account for sample_weight

    https://github.com/csinva/imodels/issues/157: the node statistics behind
    importances (and behind the sklearn tree conversion used by plot) counted
    rows, ignoring their weights.
    """
    from imodels import FIGSClassifier
    from imodels.tree.viz_utils import extract_sklearn_tree_from_figs

    rng = np.random.RandomState(0)
    X = rng.randn(200, 3)
    y = ((X[:, 0] > 0) | (X[:, 1] > 1)).astype(int)
    weights = np.where(np.arange(200) % 3 == 0, 3.0, 1.0)

    # weighting rows must match repeating them, which is what weights mean
    weighted = FIGSClassifier(max_rules=5).fit(X, y, sample_weight=weights)
    duplicated = FIGSClassifier(max_rules=5).fit(
        np.repeat(X, weights.astype(int), axis=0),
        np.repeat(y, weights.astype(int)))
    assert np.allclose(weighted.feature_importances_,
                       duplicated.feature_importances_)

    # uniform weights change nothing
    assert np.allclose(
        FIGSClassifier(max_rules=5).fit(
            X, y, sample_weight=np.ones(len(y))).feature_importances_,
        FIGSClassifier(max_rules=5).fit(X, y).feature_importances_)

    # and weighting actually moves them
    assert not np.allclose(weighted.feature_importances_,
                           FIGSClassifier(max_rules=5).fit(
                               X, y).feature_importances_)

    # the converted sklearn tree keeps raw counts and weighted totals apart
    tree = extract_sklearn_tree_from_figs(weighted, 0, 2).tree_
    assert tree.n_node_samples[0] == len(y)
    assert np.isclose(tree.weighted_n_node_samples[0], weights.sum())


def test_verbose_progress_reporting():
    """FIGS should be able to report progress while fitting

    Feature request https://github.com/csinva/imodels/issues/92: a long fit
    gave no indication of progress.
    """
    import contextlib
    import io

    from imodels import FIGSRegressor

    rng = np.random.RandomState(0)
    X = rng.randn(200, 4)
    y = X[:, 0] + (X[:, 1] > 0)

    def fit_and_capture(model, **fit_kwargs):
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            model.fit(X, y, **fit_kwargs)
        return buffer.getvalue()

    # silent by default
    assert fit_and_capture(FIGSRegressor(max_rules=4)) == ''

    # verbose=1 reports each rule, with a running count against the budget
    output = fit_and_capture(FIGSRegressor(max_rules=4, verbose=1))
    lines = [line for line in output.splitlines() if line.startswith('rule ')]
    assert len(lines) == 4
    assert 'rule 1/4' in output and 'rule 4/4' in output
    assert 'tree(s)' in output

    # verbose=2 additionally prints the model
    assert 'FIGS' in fit_and_capture(FIGSRegressor(max_rules=2, verbose=2))

    # fit(verbose=...) still overrides, as it did before
    assert fit_and_capture(FIGSRegressor(max_rules=2), verbose=1) != ''
    assert fit_and_capture(FIGSRegressor(max_rules=2, verbose=1), verbose=0) == ''

    # and it round-trips as a normal parameter
    assert FIGSRegressor(verbose=1).get_params()['verbose'] == 1


def test_categorical_features_remembered_from_fit():
    """predict shouldn't require re-passing categorical_features

    https://github.com/csinva/imodels/issues/77: fitting with
    categorical_features worked, but predict(X) then failed with
    "TypeError: 'NoneType' object is not iterable" unless they were passed again.
    """
    import pandas as pd

    from imodels import FIGSClassifier, FIGSRegressor

    rng = np.random.RandomState(0)
    X = pd.DataFrame({
        'age': rng.randn(300),
        'pet': rng.choice(['cat', 'dog', 'bird'], 300),
        'city': rng.choice(['NY', 'LA'], 300),
    })
    y = ((X['age'] > 0) | (X['pet'] == 'dog')).astype(int)
    categorical = ['pet', 'city']

    model = FIGSClassifier(max_rules=5).fit(X, y, categorical_features=categorical)

    preds = model.predict(X)                       # used to raise
    assert preds.shape == (300,)
    assert model.categorical_features_ == categorical
    assert model.predict_proba(X).shape == (300, 2)

    # passing them explicitly gives the same answer
    assert np.array_equal(preds, model.predict(X, categorical_features=categorical))

    # regressors too
    regressor = FIGSRegressor(max_rules=4).fit(
        X, X['age'], categorical_features=categorical)
    assert regressor.predict(X).shape == (300,)

    # and a purely numeric model is unaffected
    X_num = pd.DataFrame(rng.randn(200, 3), columns=list('abc'))
    numeric = FIGSClassifier(max_rules=4).fit(X_num, (X_num['a'] > 0).astype(int))
    assert numeric.categorical_features_ is None
    assert numeric.predict(X_num).shape == (200,)
def test_single_feature_input():
    """FIGS should fit data with one feature

    Annotating the tree indexed X by the node's split feature even for leaves,
    whose feature is the -2 placeholder. With several features that silently
    picked the wrong column; with one it raised IndexError.
    """
    from imodels import FIGSClassifier, FIGSClassifierCV, FIGSRegressor

    rng = np.random.RandomState(0)
    X = rng.randn(120, 1)
    y = (X[:, 0] > 0).astype(int)

    classifier = FIGSClassifier(max_rules=5).fit(X, y)
    assert classifier.predict(X).shape == (120,)
    assert np.mean(classifier.predict(X) == y) > 0.9
    assert classifier.feature_importances_.shape == (1,)

    regressor = FIGSRegressor(max_rules=5).fit(X, X[:, 0])
    assert regressor.predict(X).shape == (120,)

    cv = FIGSClassifierCV(n_rules_list=[3], n_trees_list=[2], cv=2).fit(X, y)
    assert cv.predict(X).shape == (120,)

    # leaf membership and rules work too
    assert classifier.apply(X).shape[0] == 120
    assert len(classifier.get_rules()) > 0


def test_n_jobs_gives_identical_models():
    """Fitting candidate splits in parallel must not change the result

    https://github.com/csinva/imodels/issues/94: FIGS used one core. Candidate
    splits are independent, so they can be evaluated in threads.
    """
    from imodels import FIGSClassifier, FIGSRegressor

    rng = np.random.RandomState(0)
    X = rng.randn(2000, 10)
    y = X[:, 0] + X[:, 1] + rng.randn(2000) * 0.3

    serial = FIGSRegressor(max_rules=12).fit(X, y)
    parallel = FIGSRegressor(max_rules=12, n_jobs=4).fit(X, y)
    assert np.allclose(serial.predict(X), parallel.predict(X))
    assert len(serial.trees_) == len(parallel.trees_)

    y_binary = (y > 0).astype(int)
    serial_clf = FIGSClassifier(max_rules=10).fit(X, y_binary)
    parallel_clf = FIGSClassifier(max_rules=10, n_jobs=4).fit(X, y_binary)
    assert np.allclose(serial_clf.predict_proba(X), parallel_clf.predict_proba(X))

    # it round-trips as an ordinary parameter
    assert FIGSRegressor(n_jobs=4).get_params()['n_jobs'] == 4
    assert FIGSRegressor().get_params()['n_jobs'] is None


def test_plot_lays_trees_out_over_cols():
    """plot(cols=n) arranges the trees in n columns

    cols was accepted and then ignored: every tree went into a single column
    no matter what was passed.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    rng = np.random.RandomState(0)
    X = rng.randn(200, 6)
    # additive signal, so FIGS grows a sum of several trees
    y = X[:, 0] * 2 + np.sin(X[:, 1] * 3) + X[:, 2] ** 2 + X[:, 3]
    model = FIGSRegressor(max_rules=12).fit(X, y)
    assert len(model.trees_) > 1, 'need several trees for the layout to matter'

    for cols in (1, 2, 3):
        model.plot(cols=cols, filename=os.path.join(path_to_tests, 'figs_plot.png'))
        geometry = plt.gcf().axes[0].get_subplotspec().get_gridspec().get_geometry()
        plt.close('all')
        expected_cols = min(cols, len(model.trees_))
        assert geometry[1] == expected_cols, f'cols={cols} gave grid {geometry}'
    os.remove(os.path.join(path_to_tests, 'figs_plot.png'))


def test_classifier_has_decision_function():
    """FIGSClassifier exposes decision_function, as sklearn wrappers expect

    _init_decision_function() claimed to set one up but only bound a local that
    was immediately discarded, so the attribute never existed.
    """
    rng = np.random.RandomState(0)
    X = rng.randn(100, 4)
    y = (X[:, 0] > 0).astype(int)
    model = FIGSClassifier(max_rules=5).fit(X, y)

    scores = model.decision_function(X)
    assert scores.shape == (100,)
    # it ranks the same way predict_proba does
    assert np.allclose(scores, model.predict_proba(X)[:, 1])


def test_fit_accepts_plain_lists():
    """lists are valid input to fit, as they are for any sklearn estimator"""
    rng = np.random.RandomState(0)
    X = rng.randn(60, 3)
    y = (X[:, 0] > 0).astype(int)

    from_lists = FIGSClassifier(max_rules=4).fit(X.tolist(), y.tolist())
    from_arrays = FIGSClassifier(max_rules=4).fit(X, y)
    assert np.array_equal(from_lists.predict(X), from_arrays.predict(X))
