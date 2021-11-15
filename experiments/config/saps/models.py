import numpy as np
from numpy import concatenate as cat
from experiments.util import Model
from imodels import GreedyTreeClassifier, GreedyTreeRegressor, C45TreeClassifier, SaplingSumClassifier
from imodels import RuleFitClassifier, RuleFitRegressor, SaplingSumRegressor

RULEFIT_DEFAULT_KWARGS_CLASSIFICATION = {'random_state': 0, 'max_rules': None, 'include_linear': False, 'alpha': 1}
ESTIMATORS_CLASSIFICATION = (
    [Model('C4.5', C45TreeClassifier, 'max_rules', n)
     for n in np.concatenate((np.arange(1, 19, 3), [25, 30]))],
    [Model('CART', GreedyTreeClassifier, 'max_depth', n)
     for n in np.arange(1, 6, 1)],
    [Model('SAPS', SaplingSumClassifier, 'max_rules', n)
     for n in cat((np.arange(1, 19, 3), [25, 30]))],
    [Model('Rulefit', RuleFitClassifier, 'n_estimators', n, other_params=RULEFIT_DEFAULT_KWARGS_CLASSIFICATION)
     for n in np.arange(1, 11, 1)],  # can also vary n_estimators and get a good spread
)

RULEFIT_DEFAULT_KWARGS_REGRESSION = {'random_state': 0, 'max_rules': None, 'include_linear': False, 'alpha': 0.15}
ESTIMATORS_REGRESSION = (
    [Model('Rulefit', RuleFitRegressor, 'n_estimators', n, other_params=RULEFIT_DEFAULT_KWARGS_REGRESSION)
     for n in cat((np.arange(1, 11, 1), [15]))],  # can also vary n_estimators and get a good spread
    [Model('CART (MSE)', GreedyTreeRegressor, 'max_depth', n)
     for n in np.arange(1, 6, 1)],
    [Model('CART (MAE)', GreedyTreeRegressor, 'max_depth', n, other_params={'criterion': 'absolute_error'})
     for n in np.arange(1, 6, 1)],
    [Model('SAPS', SaplingSumRegressor, 'max_rules', n)
     for n in cat((np.arange(1, 19, 3), [25, 30]))],
)
