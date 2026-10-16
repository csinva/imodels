import numpy as np
import pytest
from numpy.testing import assert_array_equal

from imodels import StableClustering


@pytest.mark.parametrize('query', [
    [[-10.0], [-9.0]],
    [[9.0], [10.0]],
    [[-10.0], [9.0], [10.0]],
    [[-10.0], [-9.0], [10.0]],
    [[-10.0], [-9.0], [9.0], [10.0]],
])
@pytest.mark.parametrize('n_closest', [0, 1, 2, 10])
def test_closest_points_stay_in_their_clusters(query, n_closest):
    model = StableClustering(k_values=[2], n_repetitions=2).fit(
        np.array([[-10.0], [-9.0], [9.0], [10.0]]))
    X = np.array(query)
    assignments = model.predict(X)

    selected = model.predict_closest_points_to_centroids(X, n_closest=n_closest)

    assert_array_equal(selected[selected >= 0], assignments[selected >= 0])
    for cluster in range(model.best_k_):
        assert np.sum(selected == cluster) == min(
            n_closest, np.sum(assignments == cluster))


def test_closest_points_selects_nearest_within_each_cluster():
    model = StableClustering(k_values=[2], n_repetitions=2).fit(
        np.array([[-10.0], [-9.0], [9.0], [10.0]]))
    X = np.array([[-15.0], [-9.5], [9.5], [15.0]])
    expected = np.array([-1, model.predict(X[1:2])[0],
                         model.predict(X[2:3])[0], -1])

    assert_array_equal(model.predict_closest_points_to_centroids(X), expected)
