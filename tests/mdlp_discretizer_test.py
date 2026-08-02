"""Checks the two discretizers imodels exports from the MDLP module.

Both are advertised in imodels.DISCRETIZERS alongside sklearn-style
transformers, but neither could actually be used that way: MDLPDiscretizer had
no fit/transform at all, and BRLDiscretizer's fit crashed on a DataFrame and
returned None instead of self.
"""

import numpy as np
import pandas as pd
import pytest

import imodels
from imodels.discretization.mdlp import BRLDiscretizer, MDLPDiscretizer

FEATURES = ['a', 'b', 'c']


@pytest.fixture
def data():
    rng = np.random.RandomState(0)
    X = pd.DataFrame(rng.randn(200, 3), columns=FEATURES)
    y = (X['a'] > 0).astype(int)
    return X, y


def test_every_exported_discretizer_has_the_transformer_api():
    """Anything in DISCRETIZERS should be usable as a transformer"""
    for discretizer in imodels.DISCRETIZERS:
        for method in ('fit', 'transform', 'fit_transform'):
            assert hasattr(discretizer, method), \
                f'{discretizer.__name__} has no {method}'


class TestMDLPDiscretizer:

    def test_fit_transform(self, data):
        X, y = data
        out = MDLPDiscretizer().fit(X, y).transform(X)
        assert out.shape == X.shape
        assert list(out.columns) == FEATURES
        # the informative feature gets cut, so it is no longer numeric
        assert out['a'].nunique() > 1
        assert not np.issubdtype(np.asarray(out['a']).dtype, np.number)

    def test_transform_reuses_the_fitted_bins(self, data):
        """New data must be binned with the cut points found during fit"""
        X, y = data
        discretizer = MDLPDiscretizer().fit(X, y)
        train_bins = set(map(str, discretizer.transform(X)['a']))
        held_out = set(map(str, discretizer.transform(X.iloc[:20])['a']))
        assert held_out <= train_bins

    def test_fit_transform_matches_fit_then_transform(self, data):
        X, y = data
        assert MDLPDiscretizer().fit_transform(X, y).equals(
            MDLPDiscretizer().fit(X, y).transform(X))

    def test_accepts_numpy(self, data):
        X, y = data
        assert MDLPDiscretizer().fit(X.values, y).transform(X.values).shape \
            == X.shape

    def test_transform_before_fit_says_so(self, data):
        X, _ = data
        with pytest.raises(ValueError, match='not fitted'):
            MDLPDiscretizer().transform(X)

    def test_eager_construction_still_works(self, data):
        """BRLDiscretizer builds one by passing the dataset to __init__"""
        X, y = data
        dataset = X.copy()
        dataset['y'] = y.values
        discretizer = MDLPDiscretizer(dataset=dataset, class_label='y',
                                      features=FEATURES)
        assert list(discretizer._data.columns) == FEATURES + ['y']
        assert sorted(discretizer.bin_labels_) == FEATURES


class TestBRLDiscretizer:

    @pytest.mark.parametrize('as_numpy', [False, True], ids=['df', 'numpy'])
    def test_fit_transform(self, data, as_numpy):
        X, y = data
        X_in = X.values if as_numpy else X
        discretizer = BRLDiscretizer(feature_labels=FEATURES).fit(X_in, y)
        assert isinstance(discretizer, BRLDiscretizer), 'fit must return self'
        out = discretizer.transform(X_in)
        assert out.shape[0] == len(X)

    def test_dataframe_and_numpy_agree(self, data):
        X, y = data
        from_df = BRLDiscretizer(feature_labels=FEATURES).fit(X, y).transform(X)
        from_np = BRLDiscretizer(
            feature_labels=FEATURES).fit(X.values, y).transform(X.values)
        assert list(from_df.columns) == list(from_np.columns)

    def test_binary_data_passes_through(self, data):
        """Nothing to discretize -- the columns should survive untouched"""
        X, y = data
        X_binary = (X > 0).astype(int)
        out = BRLDiscretizer(feature_labels=FEATURES).fit(
            X_binary, y).transform(X_binary)
        assert list(out.columns) == FEATURES

    def test_mixed_string_and_numeric(self, data):
        X, y = data
        X_mixed = X.copy()
        X_mixed['c'] = np.where(X_mixed['c'] > 0, 'hi', 'lo')
        out = BRLDiscretizer(feature_labels=FEATURES).fit(
            X_mixed, y).transform(X_mixed)
        assert out.shape[0] == len(X)
        assert any('hi' in str(col) for col in out.columns)
