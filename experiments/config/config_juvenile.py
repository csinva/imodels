from functools import partial

import numpy as np
from imodels import (
    SkopeRulesClassifier as skope, RuleFitClassifier as rfit, FPLassoClassifier as fpl, 
    FPSkopeClassifier as fps, BayesianRuleListClassifier as brl, BoostedRulesClassifier as brs
)
from sklearn.ensemble import RandomForestClassifier as rf, GradientBoostingClassifier as gb
from sklearn.tree import DecisionTreeClassifier

from experiments.util import Model
from experiments.config.config_general import RULEFIT_KWARGS, BRL_KWARGS, FPL_KWARGS

JUVENILE_TEST_ESTIMATORS = []

JUVENILE_ESTIMATORS = []
JUVENILE_ESTIMATORS.append(
    [Model('random_forest - mid_0', rf, 'n_estimators', n, 'min_impurity_decrease', 0., {'max_depth': 2}) for n in np.arange(1, 8)]
    + [Model('random_forest - mid_1e-4', rf, 'n_estimators', n, 'min_impurity_decrease', 1e-4, {'max_depth': 2}) for n in np.arange(1, 8)]
    + [Model('random_forest - mid_5e-4', rf, 'n_estimators', n, 'min_impurity_decrease', 5e-4, {'max_depth': 2}) for n in np.arange(1, 8)]
)
JUVENILE_ESTIMATORS.append(
    [Model('gradient_boosting - mid_0', gb, 'n_estimators', n, 'min_impurity_decrease', 0., {'max_depth': 2}) for n in np.arange(1, 8)]
    + [Model('gradient_boosting - mid_10', gb, 'n_estimators', n, 'min_impurity_decrease', 10, {'max_depth': 2}) for n in np.arange(1, 10)]
    + [Model('gradient_boosting - mid_20', gb, 'n_estimators', n, 'min_impurity_decrease', 15, {'max_depth': 2}) for n in np.arange(1, 40, 4)]
)
JUVENILE_ESTIMATORS.append(
    [Model('skope_rules - prec_0.5', skope, 'n_estimators', n, 'precision_min', 0.5, {'max_depth': 2}) for n in [1, 2, 3] + np.arange(5, 40, 4).tolist()]
    + [Model('skope_rules - prec_.45', skope, 'n_estimators', n, 'precision_min', 0.45, {'max_depth': 2}) for n in [1, 2] + np.arange(3, 20, 2).tolist()]
    + [Model('skope_rules - prec_0.4', skope, 'n_estimators', n, 'precision_min', 0.4, {'max_depth': 2}) for n in np.arange(1, 10)]
)
JUVENILE_ESTIMATORS.append(
    [Model('rulefit - alpha_30', rfit, 'n_estimators', n, 'alpha', 30, RULEFIT_KWARGS) for n in np.arange(1, 7).tolist() + [9, 13, 17, 21, 25]]
    + [Model('rulefit - alpha_13', rfit, 'n_estimators', n, 'alpha', 13, RULEFIT_KWARGS) for n in np.arange(1, 11)]
    + [Model('rulefit - alpha_5', rfit, 'n_estimators', n, 'alpha', 5, RULEFIT_KWARGS) for n in np.arange(1, 11)]
    + [Model('rulefit - alpha_2', rfit, 'n_estimators', n, 'alpha', 2, RULEFIT_KWARGS) for n in np.arange(1, 11)]
    + [Model('rulefit - alpha_1', rfit, 'n_estimators', n, 'alpha', 1, RULEFIT_KWARGS) for n in np.arange(1, 11)]
)
JUVENILE_ESTIMATORS.append(
    [Model('fplasso - minsup_0.05', fpl, 'alpha', a, 'minsupport', 0.05, FPL_KWARGS) for a in np.logspace(2, 3, 10)]
    + [Model('fplasso - minsup_0.1', fpl, 'alpha', a, 'minsupport', 0.1, FPL_KWARGS) for a in np.logspace(2, 3, 10)]
    + [Model('fplasso - minsup_0.15', fpl, 'alpha', a, 'minsupport', 0.15, FPL_KWARGS) for a in np.logspace(2, 3, 10)]
)
JUVENILE_ESTIMATORS.append(
    [Model('brl - minsup', brl, 'listlengthprior', n, 'minsupport', 0.1, BRL_KWARGS) for n in np.arange(1, 20, 2)]
    + [Model('brl - minsup', brl, 'listlengthprior', n, 'minsupport', 0.1, BRL_KWARGS) for n in np.arange(1, 20, 2)]
)
JUVENILE_ESTIMATORS.append(
    [Model('brs - mid_0', brs, 'n_estimators', n, 'estimator', partial(DecisionTreeClassifier, min_impurity_decrease=0, max_depth=2)) for n in np.arange(1, 9)]
    + [Model('brs - mid_1e-3', brs, 'n_estimators', n, 'estimator', partial(DecisionTreeClassifier, min_impurity_decrease=1e-3, max_depth=2)) for n in np.arange(1, 9)]
    + [Model('brs - mid_2e-3', brs, 'n_estimators', n, 'estimator', partial(DecisionTreeClassifier, min_impurity_decrease=2e-3, max_depth=2)) for n in np.arange(1, 9)]
    + [Model('brs - mid_5e-3', brs, 'n_estimators', n, 'estimator', partial(DecisionTreeClassifier, min_impurity_decrease=5e-3, max_depth=2)) for n in np.arange(1, 9)]
    + [Model('brs - mid_1e-2', brs, 'n_estimators', n, 'estimator', partial(DecisionTreeClassifier, min_impurity_decrease=1e-2, max_depth=2)) for n in np.arange(1, 9)]
)
