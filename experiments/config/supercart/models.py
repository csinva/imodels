import numpy as np

from experiments.models.supercart import SuperCARTClassifier, SuperCARTRegressor
from experiments.util import Model
from imodels import GreedyTreeClassifier, GreedyTreeRegressor
from imodels import RuleFitClassifier, RuleFitRegressor

RULEFIT_DEFAULT_KWARGS = {'random_state': 0, 'max_rules': None, 'include_linear': False, 'alpha': 1}

ESTIMATORS_CLASSIFICATION = (
    [Model('rulefit', RuleFitClassifier, 'n_estimators', n, other_params=RULEFIT_DEFAULT_KWARGS)
     for n in np.arange(1, 11, 1)],  # can also vary n_estimators and get a good spread
    [Model('tree', GreedyTreeClassifier, 'max_depth', n)
     for n in np.arange(1, 6, 1)],
    [Model('supercart', SuperCARTClassifier, 'max_rules', n)
     for n in np.concatenate((np.arange(1, 19, 3), [25, 30]))]
)

ESTIMATORS_REGRESSION = (
    [Model('rulefit - alpha_1', RuleFitRegressor, 'n_estimators', n, other_params=RULEFIT_DEFAULT_KWARGS)
     for n in np.arange(1, 11, 1)],  # can also vary n_estimators and get a good spread
    [Model('tree', GreedyTreeRegressor, 'max_depth', n)
     for n in np.arange(1, 6, 1)],
    [Model('supercart', SuperCARTRegressor, 'max_rules', n)
     for n in np.concatenate((np.arange(1, 19, 3), [25, 30]))]
)
