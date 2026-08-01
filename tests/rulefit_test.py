import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._testing import ignore_warnings

from imodels.rule_set.rule_fit import RuleFitRegressor
from imodels.util.transforms import FriedScale


## Testing FriedScale():
def test_fried_scale():
    x_scale_test = np.zeros([100, 2])
    x_scale_test[0:5, 0] = -100
    x_scale_test[5:10, 0] = 100
    x_scale_test[10:55, 0] = 1
    x_scale_test[5:55,
    1] = 1  # winsorised version of first column at trim=0.1: note, will not be scaled because it is already an indicator function, as per FP004
    fs = FriedScale()  # trim_quantile=0.1)
    fs.train(x_scale_test)
    '''
    np.testing.assert_array_equal(fs.scale(x_scale_test),
                                  np.hstack([x_scale_test[:, 1].reshape([-1, 1]) * 0.4 / np.std(x_scale_test[:, 1]),
                                             x_scale_test[:, 1].reshape([-1, 1])]))
                                             
'''


# @ignore_warnings(category=ConvergenceWarning)
def test_integration():
    X = np.array([[1, 99, 43, 34],
                  [1, 76, 22, 10],
                  [0, 83, 11, 0],
                  [0, 99, 74, 33],
                  [0, 53, 40, 34]])
    y = np.array([1, 0, 1, 1, 0])

    rfr = RuleFitRegressor(exp_rand_tree_size=False, n_estimators=500, random_state=1, include_linear=False,
                           max_rules=None, alpha=0.1)
    rfr.fit(X, y)
    print(len(rfr._get_rules()))
    expected = np.array([0.83333333, 0.25, 0.83333333, 0.83333333, 0.25])
    assert np.allclose(rfr.predict(X), expected, atol=1.0e-04)

    rfr = RuleFitRegressor(exp_rand_tree_size=False, n_estimators=5, random_state=0, max_rules=None, alpha=0.01)
    rfr.fit(X, y)
    expected = np.array([0.89630491, 0.15375469, 0.89624531, 1.05000033, 0.00369476])
    assert np.allclose(rfr.predict(X), expected)


    rfr = RuleFitRegressor(exp_rand_tree_size=False, n_estimators=5, random_state=0,
                           max_rules=None, alpha=0.01, tree_generator=RandomForestClassifier())
    rfr.fit(X, y)
    # expected = np.array([0.89630491, 0.15375469, 0.89624531, 1.05000033, 0.00369476])
    # assert np.allclose(rfr.predict(X), expected)

def test_sample_fract_is_used():
    """sample_fract should control the subsample used to generate the trees

    Regression test for https://github.com/csinva/imodels/issues/200: the
    parameter was stored but never passed on, so it had no effect.
    """
    X = np.random.RandomState(0).randn(200, 4)
    y = X[:, 0] + 0.1 * np.random.RandomState(1).randn(200)

    def fit_rules(sample_fract):
        m = RuleFitRegressor(sample_fract=sample_fract, random_state=0,
                             exp_rand_tree_size=False, n_estimators=10)
        m.fit(X, y)
        return [str(rule) for rule in m.rules_]

    # different subsampling fractions should give different rule sets
    assert fit_rules(0.2) != fit_rules(0.9)

    # the default is unchanged (Friedman & Popescu's heuristic), and is not
    # simply whatever an explicit fraction gives
    assert fit_rules('default') == fit_rules('default')
    assert fit_rules('default') != fit_rules(0.9)
def test_set_params_lin_trim_quantile():
    """lin_trim_quantile set after construction should be used when fitting

    Regression test for https://github.com/csinva/imodels/issues/222: the
    winsorizer/scaler used to be built in __init__, so set_params had no effect.
    """
    X = np.random.RandomState(0).randn(50, 3)
    y = (X[:, 0] > 0).astype(int)

    via_init = RuleFitRegressor(lin_trim_quantile=0.04, random_state=0)
    via_set_params = RuleFitRegressor(random_state=0).set_params(
        lin_trim_quantile=0.04)

    via_init.fit(X, y)
    via_set_params.fit(X, y)

    assert via_set_params.friedscale.winsorizer.trim_quantile == 0.04
    assert (via_init.friedscale.winsorizer.trim_quantile ==
            via_set_params.friedscale.winsorizer.trim_quantile)
    assert np.allclose(via_init.predict(X), via_set_params.predict(X))
