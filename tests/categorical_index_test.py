import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import OneHotEncoder

from imodels import FIGSClassifier, FIGSRegressor
from imodels.util.data_util import encode_categories


INDEXES = [
    pytest.param(pd.RangeIndex(4, name='row'), id='default'),
    pytest.param(pd.Index([10, 11, 12, 13], name='row'), id='offset'),
    pytest.param(pd.Index([3, 2, 1, 0], name='row'), id='permuted'),
    pytest.param(pd.Index(['w', 'x', 'y', 'z'], name='row'), id='string'),
    pytest.param(pd.Index([7, 7, 8, 8], name='row'), id='duplicate'),
    pytest.param(pd.date_range('2025-01-01', periods=4, name='row'), id='datetime'),
    pytest.param(pd.MultiIndex.from_tuples(
        [('a', 1), ('a', 2), ('b', 1), ('b', 2)], names=['group', 'row']),
        id='multi'),
]


@pytest.mark.parametrize('index', INDEXES)
@pytest.mark.parametrize('reuse_encoder', [False, True])
@pytest.mark.parametrize('numeric_column', [False, True])
def test_category_encoding_keeps_sample_alignment(index, reuse_encoder, numeric_column):
    X = pd.DataFrame({'kind': ['a', 'b', 'a', 'b']}, index=index)
    if numeric_column:
        X.insert(0, 'value', np.arange(4))
    original = X.copy(deep=True)
    encoder = OneHotEncoder(sparse_output=False).fit(X[['kind']]) if reuse_encoder else None

    result = encode_categories(X, ['kind'], encoder)
    encoded = result if reuse_encoder else result[0]

    expected = pd.DataFrame({'kind_a': [1., 0., 1., 0.],
                             'kind_b': [0., 1., 0., 1.]}, index=index)
    if numeric_column:
        expected.insert(0, 'value', np.arange(4))
    pd.testing.assert_frame_equal(encoded, expected)
    pd.testing.assert_frame_equal(X, original)


@pytest.mark.parametrize('model_type', [FIGSClassifier, FIGSRegressor])
@pytest.mark.parametrize('custom_index_at', ['neither', 'fit', 'predict'])
def test_figs_categorical_fit_and_predict_with_row_labels(model_type, custom_index_at):
    X = pd.DataFrame({'value': [0., 0., 0., 0.], 'kind': ['a', 'b', 'a', 'b']})
    y = np.array([0, 1, 0, 1])
    reference = model_type(max_rules=2, random_state=0).fit(
        X, y, categorical_features=['kind'])
    train, query = X.copy(), X.copy()
    if custom_index_at == 'fit':
        train.index = pd.Index([10, 11, 12, 13], name='sample')
    elif custom_index_at == 'predict':
        query.index = pd.Index(['w', 'x', 'y', 'z'], name='sample')
    original_train, original_query = train.copy(deep=True), query.copy(deep=True)

    model = model_type(max_rules=2, random_state=0).fit(
        train, y, categorical_features=['kind'])
    np.testing.assert_allclose(model.predict(query), reference.predict(X))
    if model_type is FIGSClassifier:
        np.testing.assert_allclose(model.predict_proba(query), reference.predict_proba(X))
    pd.testing.assert_frame_equal(train, original_train)
    pd.testing.assert_frame_equal(query, original_query)
