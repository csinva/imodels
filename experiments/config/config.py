from copy import deepcopy

import numpy as np
from imodels import (
    SkopeRulesClassifier as skope, RuleFitClassifier as rfit, FPLassoClassifier as fpl, 
    FPSkopeClassifier as fps, BayesianRuleListClassifier as brl, GreedyRuleListClassifier as grl,
    OneRClassifier as oner, BoostedRulesClassifier as brs
)
from sklearn.ensemble import RandomForestClassifier as rf, GradientBoostingClassifier as gb

from experiments.util import Model, get_best_models_under_complexity, DATASET_PATH
from experiments.models.stablelinear import StableLinearClassifier as stbl
from experiments.models.stableskope import StableSkopeClassifier as stbs


EASY_DATASETS = [
    ("breast-cancer", DATASET_PATH + "breast_cancer.csv"), 
    ("credit-g", DATASET_PATH + "credit_g.csv"), 
    ("haberman", DATASET_PATH + "haberman.csv"), 
    ("heart", DATASET_PATH + "heart.csv"), 
]
HARD_DATASETS = [
    ("recidivism", DATASET_PATH + "compas-analysis/compas_two_year_clean.csv"),
    ("credit", DATASET_PATH + "credit_card/UCI_Credit_Card.csv"),
    ("juvenile", DATASET_PATH + "ICPSR_03986/DS0001/data_clean.csv")
]

BEST_ESTIMATORS = []


ALL_ESTIMATORS = []
ALL_ESTIMATORS.append(
    [Model('random_forest - depth_1', rf, 'n_estimators', n, 'max_depth', 1) for n in np.arange(1, 38, 4)]
    + [Model('random_forest - depth_2', rf, 'n_estimators', n, 'max_depth', 2) for n in np.arange(1, 15)]
    + [Model('random_forest - depth_3', rf, 'n_estimators', n, 'max_depth', 3) for n in np.arange(1, 8)]
)
ALL_ESTIMATORS.append(
    [Model('gradient_boosting - depth_1', gb, 'n_estimators', n, 'max_depth', 1) for n in np.arange(1, 38, 4)]
    + [Model('gradient_boosting - depth_2', gb, 'n_estimators', n, 'max_depth', 2) for n in np.arange(1, 15)]
    + [Model('gradient_boosting - depth_3', gb, 'n_estimators', n, 'max_depth', 3) for n in np.arange(1, 8)]
)
ALL_ESTIMATORS.append(
    [Model('skope_rules - max_depth_1_prec_0.3', skope, 'n_estimators', n, 'max_depth', 1, {'precision_min': 0.3}) for n in np.arange(1, 137, 14)]
    + [Model('skope_rules - max_depth_1_prec_0.4', skope, 'n_estimators', n, 'max_depth', 1, {'precision_min': 0.4}) for n in np.arange(1, 137, 14)]
    + [Model('skope_rules - max_depth_1_prec_0.5', skope, 'n_estimators', n, 'max_depth', 1, {'precision_min': 0.5}) for n in np.arange(1, 137, 14)]
    + [Model('skope_rules - max_depth_2_prec_0.3', skope, 'n_estimators', n, 'max_depth', 2, {'precision_min': 0.3}) for n in np.arange(1, 20, 2)]
    + [Model('skope_rules - max_depth_2_prec_0.4', skope, 'n_estimators', n, 'max_depth', 2, {'precision_min': 0.4}) for n in np.arange(1, 20, 2)]
    + [Model('skope_rules - max_depth_2_prec_0.5', skope, 'n_estimators', n, 'max_depth', 2, {'precision_min': 0.5}) for n in np.arange(1, 20, 2)]
    + [Model('skope_rules - max_depth_3_prec_0.3', skope, 'n_estimators', n, 'max_depth', 3, {'precision_min': 0.3}) for n in np.arange(1, 6)]
    + [Model('skope_rules - max_depth_3_prec_0.4', skope, 'n_estimators', n, 'max_depth', 3, {'precision_min': 0.4}) for n in np.arange(1, 6)]
    + [Model('skope_rules - max_depth_3_prec_0.5', skope, 'n_estimators', n, 'max_depth', 3, {'precision_min': 0.5}) for n in np.arange(1, 6)]
)
rulefit_kwargs = {'random_state': 0, 'max_rules': None, 'include_linear': False}
ALL_ESTIMATORS.append(
    [Model('rulefit - alpha_30', rfit, 'n_estimators', n, 'alpha', 30, rulefit_kwargs) for n in [1, 3] + list(np.arange(7, 50, 6))]
    + [Model('rulefit - alpha_13', rfit, 'n_estimators', n, 'alpha', 13, rulefit_kwargs) for n in np.arange(1, 24, 2)]
    + [Model('rulefit - alpha_5', rfit, 'n_estimators', n, 'alpha', 5, rulefit_kwargs) for n in np.arange(1, 11)]
    + [Model('rulefit - alpha_2', rfit, 'n_estimators', n, 'alpha', 2, rulefit_kwargs) for n in np.arange(1, 11)]
    + [Model('rulefit - alpha_1', rfit, 'n_estimators', n, 'alpha', 1, rulefit_kwargs) for n in np.arange(1, 11)]
)
fpl_kwargs = {'disc_strategy': 'simple', 'max_rules': None}
ALL_ESTIMATORS.append(
    [Model('fplasso - max_card_1', fpl, 'alpha', a, 'maxcardinality', 1, fpl_kwargs) for a in np.logspace(1, 2.8, 10)]
    + [Model('fplasso - max_card_2', fpl, 'alpha', a, 'maxcardinality', 2, fpl_kwargs) for a in np.logspace(2, 3, 10)]
)
ALL_ESTIMATORS.append(
    [Model('fpskope - max_card_1_prec_0.3', fps, 'minsupport', n, 'maxcardinality', 1, {'disc_strategy': 'simple', 'precision_min': 0.3}) for n in np.linspace(0.03, 0.5, 10)]
    + [Model('fpskope - max_card_1_prec_0.4', fps, 'minsupport', n, 'maxcardinality', 1, {'disc_strategy': 'simple', 'precision_min': 0.4}) for n in np.linspace(0.01, 0.5, 10)]
    + [Model('fpskope - max_card_1_prec_0.5', fps, 'minsupport', n, 'maxcardinality', 1, {'disc_strategy': 'simple', 'precision_min': 0.5}) for n in np.linspace(0.01, 0.5, 10)]
    + [Model('fpskope - max_card_2_prec_0.3', fps, 'minsupport', n, 'maxcardinality', 2, {'disc_strategy': 'simple', 'precision_min': 0.3}) for n in np.linspace(0.2, 0.6, 10)]
    + [Model('fpskope - max_card_2_prec_0.4', fps, 'minsupport', n, 'maxcardinality', 2, {'disc_strategy': 'simple', 'precision_min': 0.4}) for n in np.linspace(0.2, 0.6, 10)]
    + [Model('fpskope - max_card_2_prec_0.5', fps, 'minsupport', n, 'maxcardinality', 2, {'disc_strategy': 'simple', 'precision_min': 0.5}) for n in np.linspace(0.2, 0.6, 10)]
)
brl_kwargs = {'disc_strategy': 'simple', 'max_iter': 2000}
ALL_ESTIMATORS.append(
    [Model('brl - list_width_1', brl, 'listlengthprior', n, 'listwidthprior', 1, brl_kwargs) for n in np.arange(1, 20, 2)]
    + [Model('brl - list_width_2', brl, 'listlengthprior', n, 'listwidthprior', 2, brl_kwargs) for n in np.arange(1, 20, 2)]
    + [Model('brl - list_width_3', brl, 'listlengthprior', n, 'listwidthprior', 3, brl_kwargs) for n in np.arange(1, 16, 2)]
)
ALL_ESTIMATORS.append([Model('brs - ', brs, 'n_estimators', n) for n in np.arange(1, 35, 3)])


BEST_EASY_ESTIMATORS = []

EASY_ESTIMATORS = deepcopy(ALL_ESTIMATORS)

EASY_ESTIMATORS[3] = (
    [Model('rulefit - alpha_30', rfit, 'n_estimators', n, 'alpha', 30, rulefit_kwargs) for n in np.arange(1, 92, 10)]
    + [Model('rulefit - alpha_13', rfit, 'n_estimators', n, 'alpha', 13, rulefit_kwargs) for n in np.arange(1, 38, 4)]
    + [Model('rulefit - alpha_5', rfit, 'n_estimators', n, 'alpha', 5, rulefit_kwargs) for n in np.arange(1, 38, 4)]
    + [Model('rulefit - alpha_2', rfit, 'n_estimators', n, 'alpha', 2, rulefit_kwargs) for n in np.arange(1, 20, 2)]
    + [Model('rulefit - alpha_1', rfit, 'n_estimators', n, 'alpha', 1, rulefit_kwargs) for n in np.arange(1, 11)]
)
EASY_ESTIMATORS[4] = (
    [Model('fplasso - max_card_1', fpl, 'alpha', a, 'maxcardinality', 1, fpl_kwargs) for a in np.logspace(-0.5, 1, 10)]
    + [Model('fplasso - max_card_2', fpl, 'alpha', a, 'maxcardinality', 2, fpl_kwargs) for a in np.logspace(0.5, 1.4, 10)]
)
EASY_ESTIMATORS[5] = (
    [Model('fpskope - max_card_1_prec_0.3', fps, 'minsupport', n, 'maxcardinality', 1, {'disc_strategy': 'simple', 'precision_min': 0.3}) for n in np.linspace(0.1, 0.5, 10)]
    + [Model('fpskope - max_card_1_prec_0.4', fps, 'minsupport', n, 'maxcardinality', 1, {'disc_strategy': 'simple', 'precision_min': 0.4}) for n in np.linspace(0.08, 0.5, 10)]
    + [Model('fpskope - max_card_1_prec_0.5', fps, 'minsupport', n, 'maxcardinality', 1, {'disc_strategy': 'simple', 'precision_min': 0.5}) for n in np.linspace(0.08, 0.5, 10)]
    + [Model('fpskope - max_card_2_prec_0.3', fps, 'minsupport', n, 'maxcardinality', 2, {'disc_strategy': 'simple', 'precision_min': 0.3}) for n in np.linspace(0.3, 0.6, 10)]
    + [Model('fpskope - max_card_2_prec_0.4', fps, 'minsupport', n, 'maxcardinality', 2, {'disc_strategy': 'simple', 'precision_min': 0.4}) for n in np.linspace(0.3, 0.6, 10)]
    + [Model('fpskope - max_card_2_prec_0.5', fps, 'minsupport', n, 'maxcardinality', 2, {'disc_strategy': 'simple', 'precision_min': 0.5}) for n in np.linspace(0.3, 0.6, 10)]
)
