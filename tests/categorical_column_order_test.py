import itertools
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import OneHotEncoder

from imodels.util.data_util import encode_categories


@pytest.mark.parametrize('numeric_columns', list(itertools.permutations(['age', 'income', 'score'])))
@pytest.mark.parametrize('reuse_encoder', [False, True])
@pytest.mark.parametrize('feature_container', [list, tuple, pd.Index], ids=['list', 'tuple', 'index'])
def test_encode_categories_keeps_numeric_column_order(numeric_columns, reuse_encoder, feature_container):
    X = pd.DataFrame({'age': [0, 1], 'income': [10, 20], 'score': [2, 3],
                      'kind': ['a', 'b'], 'city': ['x', 'y']}, index=['first', 'second'])
    X = X[[numeric_columns[0], 'kind', numeric_columns[1], 'city', numeric_columns[2]]]
    original = X.copy(deep=True)
    features = feature_container(['city', 'kind'])
    encoder = OneHotEncoder(sparse_output=False).fit(X[['city', 'kind']]) if reuse_encoder else None
    result = encode_categories(X, features, encoder)
    encoded = result if reuse_encoder else result[0]

    assert list(encoded.columns) == [*numeric_columns, 'city_x', 'city_y', 'kind_a', 'kind_b']
    pd.testing.assert_frame_equal(encoded.loc[:, list(numeric_columns)],
                                  X.loc[:, list(numeric_columns)])
    np.testing.assert_array_equal(encoded[['city_x', 'city_y', 'kind_a', 'kind_b']],
                                  [[1., 0., 1., 0.], [0., 1., 0., 1.]])
    pd.testing.assert_frame_equal(X, original)


def test_duplicate_numeric_columns_are_not_repeated():
    X = pd.DataFrame([[10, 20, 'a'], [30, 40, 'b']], columns=['score', 'score', 'kind'])
    encoded, _ = encode_categories(X, ['kind'])
    assert list(encoded.columns) == ['score', 'score', 'kind_a', 'kind_b']
    pd.testing.assert_frame_equal(encoded.iloc[:, :2], X.iloc[:, :2])
    np.testing.assert_array_equal(encoded.iloc[:, 2:], [[1., 0.], [0., 1.]])


ROUND_TRIP_SCRIPT = '''
import json
import sys
import joblib
import pandas as pd
import imodels

X = pd.DataFrame({"age": [0., 0., 1., 1.], "income": [10., 20., 10., 20.],
                  "score": [2., 3., 2., 3.], "kind": ["a", "b", "a", "b"]})
if sys.argv[1] == "fit":
    model_type = getattr(imodels, sys.argv[2])
    features = ("kind",) if sys.argv[3] == "tuple" else ["kind"]
    model = model_type(max_rules=2, random_state=0).fit(
        X, [0, 0, 1, 1], categorical_features=features)
    joblib.dump(model, "model.joblib")
else:
    model = joblib.load("model.joblib")
print(json.dumps(model.predict(X).tolist()))
'''


@pytest.mark.parametrize('model_name', ['FIGSClassifier', 'FIGSRegressor'])
@pytest.mark.parametrize('feature_container', ['list', 'tuple'])
def test_figs_joblib_round_trip_between_hash_seeds(tmp_path, model_name, feature_container):
    script = tmp_path / 'round_trip.py'
    script.write_text(ROUND_TRIP_SCRIPT)

    def run(action, seed):
        env = dict(os.environ, PYTHONHASHSEED=str(seed))
        result = subprocess.run([sys.executable, str(script), action, model_name, feature_container],
                                cwd=tmp_path, env=env, capture_output=True, text=True,
                                check=True)
        return json.loads(result.stdout)

    fitted = run('fit', 1)
    assert run('predict', 1) == fitted
    assert run('predict', 2) == fitted
