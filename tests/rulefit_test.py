import numpy as np

from imodels.util.rules import RuleCondition, Rule
from imodels.util.transforms import FriedScale
from imodels.rule_set.rule_fit import RuleFitRegressor

rule_condition_smaller = RuleCondition(1, 5, "<=", 0.4)
rule_condition_greater = RuleCondition(0, 1, ">", 0.1)

X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])


## Testing RuleCondition
def test_rule_condition_hashing_equal1():
    assert (RuleCondition(1, 5, "<=", 0.4) == RuleCondition(1, 5, "<=", 0.4))


def test_rule_condition_hashing_equal2():
    assert (RuleCondition(1, 5, "<=", 0.5) == RuleCondition(1, 5, "<=", 0.4))


def test_rule_condition_hashing_different1():
    assert (RuleCondition(1, 4, "<=", 0.4) != RuleCondition(1, 5, "<=", 0.4))


def test_rule_condition_hashing_different2():
    assert (RuleCondition(1, 5, ">", 0.4) != RuleCondition(1, 5, "<=", 0.4))


def test_rule_condition_hashing_different3():
    assert (RuleCondition(2, 5, ">", 0.4) != RuleCondition(1, 5, ">", 0.4))


def test_rule_condition_smaller():
    np.testing.assert_array_equal(rule_condition_smaller.transform(X),
                                  np.array([1, 1, 0]))


'''

'''


def test_rule_condition_greater():
    np.testing.assert_array_equal(rule_condition_greater.transform(X),
                                  np.array([0, 1, 1]))


## Testing rule
rule = Rule([rule_condition_smaller, rule_condition_greater], 0)


def test_rule_transform():
    np.testing.assert_array_equal(rule.transform(X),
                                  np.array([0, 1, 0]))


def test_rule_equality():
    rule2 = Rule([rule_condition_greater, rule_condition_smaller], 0)
    assert rule == rule2


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


def test_integration():

    X = np.array([[1, 99, 43, 34],
                  [1, 76, 22, 10],
                  [0, 83, 11, 0],
                  [0, 99, 74, 33],
                  [0, 53, 40, 34]])
    y = np.array([1, 0, 1, 1, 0])

    rfr = RuleFitRegressor(exp_rand_tree_size=False, random_state=1, include_linear=False)
    rfr.fit(X, y)
    print(len(rfr.get_rules()))
    expected = np.array([0.95068613, 0.0739708 , 0.95068613, 0.95068613, 0.0739708])
    assert np.allclose(rfr.predict(X), expected, atol=1.0e-04)

    rfr = RuleFitRegressor(exp_rand_tree_size=False, max_rules=20, random_state=0)
    rfr.fit(X, y)
    expected = np.array([0.998019929, 0.00296281305, 0.997986350, 1.00094900, 0.0000819108129])
    assert np.allclose(rfr.predict(X), expected)
