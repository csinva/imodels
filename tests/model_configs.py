"""Shared per-model test configuration, used by the test modules in this directory.

Keeping the registry of small/fast model settings in one place means a model added to
imodels.CLASSIFIERS or imodels.REGRESSORS only needs configuring once (and
model_api_test.TestRegistryCoverage fails if it isn't).
"""

from copy import deepcopy

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

N_SAMPLES = 50
N_FEATURES = 3
FEATURE_NAMES = ["feat_a", "feat_b", "feat_c"]

# keyword args that keep each model small (and deterministic where it supports a seed)
MODEL_KWARGS = {
    # the marginal-likelihood fit is O(P^3) per step in the total bin count, so
    # the test config keeps the grids coarse and the step count small
    "GPGamRegressor": dict(schedule=False, n_bins=8, n_pairs=1, pair_bins=4,
                           n_steps=15),
    # max_depth_duplication dedups subsumed rules (e.g. both "x0" and "x0 and x1");
    # without it their weighted votes average out and dilute the good rule
    "SkopeRulesClassifier": dict(
        random_state=0, max_samples_features=1.0, max_depth_duplication=1),
    "FPSkopeClassifier": dict(
        random_state=0, recall_min=0.5, max_depth_duplication=1),
    "SlipperClassifier": dict(n_estimators=1),
    # the defaults run 50k MCMC iterations over 3 chains, which dominates the suite
    "BayesianRuleListClassifier": dict(max_iter=2000, n_chains=1, random_state=0),
    "BoostedRulesClassifier": dict(n_estimators=5, random_state=0),
    "BoostedRulesRegressor": dict(n_estimators=5, random_state=0),
    "RuleFitClassifier": dict(max_rules=5, n_estimators=5, random_state=0),
    "RuleFitRegressor": dict(max_rules=5, n_estimators=5, random_state=0),
    "TreeGAMClassifier": dict(n_boosting_rounds=10, random_state=0),
    "TreeGAMRegressor": dict(n_boosting_rounds=50, random_state=0),
    "FIGSClassifierCV": dict(n_rules_list=[3], n_trees_list=[2], cv=2),
    "FIGSRegressorCV": dict(n_rules_list=[3], n_trees_list=[2], cv=2),
    "BART": dict(n_samples=5, n_burn=5, n_trees=3, n_chains=1),
    "DecisionTreeCCPClassifier": dict(
        estimator_=DecisionTreeClassifier(random_state=0), desired_complexity=3
    ),
    "DecisionTreeCCPRegressor": dict(
        estimator_=DecisionTreeRegressor(random_state=0), desired_complexity=3
    ),
    "AutoInterpretableClassifier": dict(
        param_grid=[{"est": [DecisionTreeClassifier(random_state=0)],
                     "est__max_leaf_nodes": [2, 4]}],
    ),
    "AutoInterpretableRegressor": dict(
        param_grid=[{"est": [DecisionTreeRegressor(random_state=0)],
                     "est__max_leaf_nodes": [2, 4]}],
    ),
}

def model_kwargs(model_name):
    """A fresh copy of the kwargs for `model_name`.

    Some entries hold estimator instances (the CCP models take an ``estimator_``).
    Handing the same instance to every test lets one test's fit leak into the
    next, which hides state bugs -- a model built from the registry would
    already be fitted. Always build from a copy.
    """
    return deepcopy(MODEL_KWARGS.get(model_name, {}))


# models that require pre-discretized (binary) features
BINARY_INPUT_MODELS = {
    "BayesianRuleListClassifier",
    "BayesianRuleSetClassifier",
    "FPLassoClassifier",
    "FPLassoRegressor",
    "FPSkopeClassifier",
}

# per-model overrides of the accuracy bar on the small shared task
ACCURACY_FLOORS = {}
DEFAULT_ACCURACY_FLOOR = 0.8

# models excluded from the shared suite, each with the test that covers them instead
EXCLUDED_MODELS = {
    # needs enough binary features to mine rules from; covered by brs_test.py
    "BayesianRuleSetClassifier": "requires a larger binary dataset",
    # TAO regression is deliberately gated off in the library
    # (pinned by TestUnsupportedCombinations::test_tao_regression_is_gated)
    "TaoTreeRegressor": "TAO regression is not supported yet",
}

