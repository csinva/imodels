import numpy as np

import imodels


def test_get_data():
    X, y, feature_names = imodels.get_clean_dataset('friedman1', data_source='synthetic')
    assert X.shape[0] == 200
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert isinstance(feature_names, list)
