from experiments.util import Model
from imodels import RuleFitClassifier as rfit
from experiments.models.supercart import SuperCARTClassifier as scart
from imodels import GreedyTreeClassifier as gt
import numpy as np

RULEFIT_DEFAULT_KWARGS = {'random_state': 0, 'max_rules': None, 'include_linear': False}

ESTIMATORS = (
    [Model('rulefit - alpha_1', rfit, 'n_estimators', n, 'alpha', 1, RULEFIT_DEFAULT_KWARGS)
     for n in np.arange(1, 11, 1)], # can also vary n_estimators and get a good spread
    [Model('tree', gt, 'max_depth', n)
     for n in np.arange(1, 6, 1)],
    [Model('supercart', scart, 'max_rules', n)
     for n in np.arange(1, 11, 3)]
)
