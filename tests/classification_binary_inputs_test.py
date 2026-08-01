"""Classification on pre-discretized (binary) inputs, for every registered classifier.

Continuous inputs and the general estimator contract are covered by model_api_test.py;
this file pins the binary-input path, which several rule-based models are built around.
"""

import numpy as np
import pytest

import imodels
from tests.model_configs import EXCLUDED_MODELS, MODEL_KWARGS

CLASSIFIERS = [m for m in imodels.CLASSIFIERS if m.__name__ not in EXCLUDED_MODELS]
IDS = [m.__name__ for m in CLASSIFIERS]

N_SAMPLES = 60
N_FEATURES = 4


@pytest.fixture(scope="module")
def binary_data():
    """Binary features with y = x0, plus a couple of flipped labels as noise."""
    rng = np.random.RandomState(13)
    X = (rng.randn(N_SAMPLES, N_FEATURES) > 0).astype(int)
    y = X[:, 0].copy()
    y[-2:] = 1 - y[-2:]
    return X, y


@pytest.mark.parametrize("model_type", CLASSIFIERS, ids=IDS)
def test_classification_binary_inputs(model_type, binary_data):
    X, y = binary_data
    model = model_type(**MODEL_KWARGS.get(model_type.__name__, {}))
    model.fit(X, y)

    preds = np.asarray(model.predict(X))
    assert preds.size == N_SAMPLES, "predict() yields one prediction per sample"

    probs = model.predict_proba(X)
    assert probs.shape == (N_SAMPLES, 2), "predict_proba is (n_samples, n_classes)"
    assert np.allclose(probs.sum(axis=1), 1), "probabilities sum to 1"

    acc = np.mean(preds == y)
    assert acc > 0.9, f"train accuracy {acc:0.2f} is too low for an easy task"


def test_incompatible_gosdt_falls_back_with_clear_message():
    """An incompatible gosdt install should not fail deep inside fit

    Regression test for https://github.com/csinva/imodels/issues/219: the
    unrelated `gosdt` package on PyPI imports under the same name but has a
    different API, so `import gosdt` succeeding was not enough and fitting died
    with `AttributeError: module 'gosdt' has no attribute 'configure'`.
    """
    import warnings

    import numpy as np
    from imodels import OptimalTreeClassifier
    from imodels.tree.gosdt import pygosdt

    X = np.random.RandomState(0).randn(60, 3)
    y = (X[:, 0] > 0).astype(int)

    installed, supported = pygosdt.gosdt_installed, pygosdt.gosdt_supported
    try:
        pygosdt.gosdt_installed, pygosdt.gosdt_supported = True, False
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            model = OptimalTreeClassifier().fit(X, y)
            messages = [str(w.message) for w in caught]

        # falls back to a working tree instead of raising
        assert model.predict(X).shape == (60,)
        assert any('gosdt-deprecated' in m for m in messages), messages
    finally:
        pygosdt.gosdt_installed, pygosdt.gosdt_supported = installed, supported
