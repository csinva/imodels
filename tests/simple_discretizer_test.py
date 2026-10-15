"""The advertised simple discretizer fits numeric bins before using them."""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris
from sklearn.preprocessing import KBinsDiscretizer

from imodels.discretization import SimpleDiscretizer


@pytest.mark.parametrize('strategy', ['uniform', 'quantile'])
def test_numeric_features_fit_and_transform(strategy):
    iris = load_iris()
    model = SimpleDiscretizer(n_bins=3, strategy=strategy)
    result = model.fit_transform(iris.data, iris.feature_names)
    expected = KBinsDiscretizer(n_bins=3, encode='onehot', strategy=strategy)
    np.testing.assert_array_equal(result.values, expected.fit_transform(iris.data).toarray())
    assert result.shape == (150, 12)
    assert all(any(name in column for name in iris.feature_names) for column in result.columns)
    pd.testing.assert_frame_equal(model.transform(iris.data[:5]), result.iloc[:5])


def test_mixed_features_keep_binary_columns_after_numeric_bins():
    X = np.array([[0., 0.], [1., 1.], [0., 2.], [1., 3.]])
    model = SimpleDiscretizer(n_bins=2)
    result = model.fit_transform(X, ['binary', 'numeric'])
    assert list(result.columns) == ['numeric_0.0_to_1.5', 'numeric_1.5_to_3.0', 'binary']
    np.testing.assert_array_equal(result.values, [[1, 0, 0], [1, 0, 1], [0, 1, 0], [0, 1, 1]])
    transformed = model.transform(np.array([[1., 0.5], [0., 2.5]]))
    np.testing.assert_array_equal(transformed.values, [[1, 0, 1], [0, 1, 0]])


def test_binary_features_still_pass_through():
    X = np.array([[0, 1], [1, 0]])
    result = SimpleDiscretizer().fit_transform(X, ['a', 'b'])
    pd.testing.assert_frame_equal(result, pd.DataFrame(X, columns=['a', 'b']))
