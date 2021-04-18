import numpy as np
from imodels import (
    SkopeRulesClassifier as skope, RuleFitClassifier as rfit, FPLassoClassifier as fpl, 
    FPSkopeClassifier as fps, BayesianRuleListClassifier as brl, GreedyRuleListClassifier as grl,
    OneRClassifier as oner, BoostedRulesClassifier as brs
)
from sklearn.ensemble import RandomForestClassifier as rf, GradientBoostingClassifier as gb

from experiments.util import Model, get_best_models_under_complexity
from experiments.models.stablelinear import StableLinearClassifier as stbl
from experiments.models.stableskope import StableSkopeClassifier as stbs


COMPARISON_DATASETS = [
        ("breast-cancer", 13),
        ("breast-w", 15),
        ("credit-g", 31),
        ("haberman", 43),
        ("heart", 1574),
        ("labor", 4),
        ("vote", 56),
    ]

EASY_DATASETS = ["breast-w", "labor", "vote"]
MEDIUM_DATASETS = ["breast-cancer", "credit-g", "haberman", "heart"]

BEST_ESTIMATORS = [
    [Model('random_forest', rf, 'n_estimators', n, 'max_depth', 1) for n in np.arange(1, 16, 2)],
    [Model('gradient_boosting', gb, 'n_estimators', n, 'max_depth', 1) for n in np.linspace(1, 20, 10, dtype=int)],
    [Model('skope_rules', skope, 'n_estimators', n, 'max_depth', 1) for n in np.linspace(2, 200, 10, dtype=int)],
    [Model('rulefit', rfit, 'max_rules', n, 'tree_size', 2) for n in np.linspace(2, 100, 10, dtype=int)],
    [Model('fplasso', fpl, 'max_rules', n, 'maxcardinality', 1) for n in np.linspace(2, 100, 10, dtype=int)],
    [Model('fpskope', fps, 'maxcardinality', n, 'max_depth_duplication', 3) for n in np.arange(1, 5)],
    [Model('brl', brl, 'listlengthprior', n, 'maxcardinality', 1) for n in np.linspace(1, 20, 10, dtype=int)],
    [Model('grl', grl, 'max_depth', n) for n in np.arange(1, 6)],
    [Model('oner', oner, 'max_depth', n) for n in np.arange(1, 6)],
    [Model('brs', brs, 'n_estimators', n) for n in np.linspace(1, 32, 10, dtype=int)]
]

weak_learners = [('skope_rules', skope), ('rulefit', rfit), ('fplasso', fpl), ('fpskope', fps), ('brs', brs)]
weak_learners_inst = [get_best_models_under_complexity(c, weak_learners) for c in np.arange(5, 30, 2)]
stbl_kw = [{'weak_learners': wl_lst} for wl_lst in weak_learners_inst]
stbl_cs = lambda: enumerate(np.arange(5, 30, 2))

BEST_ESTIMATORS += [
    [Model('stbl_l2_mm0', stbl, 'max_complexity', c, 'min_mult', 0, {**stbl_kw[i], 'penalty': 'l2'}) for i, c in stbl_cs()],
    [Model('stbl_l2_mm1', stbl, 'max_complexity', c, 'min_mult', 1, {**stbl_kw[i], 'penalty': 'l2'}) for i, c in stbl_cs()],
    [Model('stbl_l2_mm2', stbl, 'max_complexity', c, 'min_mult', 2, {**stbl_kw[i], 'penalty': 'l2'}) for i, c in stbl_cs()],
    [Model('stbl_l1_mm0', stbl, 'max_complexity', c, 'min_mult', 0, stbl_kw[i]) for i, c in stbl_cs()],
    [Model('stbl_l1_mm1', stbl, 'max_complexity', c, 'min_mult', 1, stbl_kw[i]) for i, c in stbl_cs()],
    [Model('stbl_l1_mm2', stbl, 'max_complexity', c, 'min_mult', 2, stbl_kw[i]) for i, c in stbl_cs()],
    # [Model('stbl_skp_mm0', stbs, 'max_complexity', c, 'min_mult', 0, stbl_kw[i]) for i, c in stbl_cs()],
    # [Model('stbl_skp_mm1', stbs, 'max_complexity', c, 'min_mult', 1, stbl_kw[i]) for i, c in stbl_cs()],
    # [Model('stbl_skp_mm2', stbs, 'max_complexity', c, 'min_mult', 2, stbl_kw[i]) for i, c in stbl_cs()]
]

ALL_ESTIMATORS = []
ALL_ESTIMATORS.append(
    [Model('random_forest - depth_1', rf, 'n_estimators', n, 'max_depth', 1) for n in np.linspace(1, 40, 10, dtype=int)]
    + [Model('random_forest - depth_2', rf, 'n_estimators', n, 'max_depth', 2) for n in np.linspace(1, 15, 10, dtype=int)]
    + [Model('random_forest - depth_3', rf, 'n_estimators', n, 'max_depth', 3) for n in np.arange(1, 8)]
)
ALL_ESTIMATORS.append(
    [Model('gradient_boosting - depth_1', gb, 'n_estimators', n, 'max_depth', 1) for n in np.linspace(1, 40, 10, dtype=int)]
    + [Model('gradient_boosting - depth_2', gb, 'n_estimators', n, 'max_depth', 2) for n in np.linspace(1, 15, 10, dtype=int)]
    + [Model('gradient_boosting - depth_3', gb, 'n_estimators', n, 'max_depth', 3) for n in np.arange(1, 8)]
)
ALL_ESTIMATORS.append(
    [Model('skope_rules - depth_1', skope, 'n_estimators', n, 'max_depth', 1) for n in np.linspace(2, 200, 10, dtype=int)]
    + [Model('skope_rules - depth_2', skope, 'n_estimators', n, 'max_depth', 2) for n in np.linspace(2, 200, 10, dtype=int)]
    + [Model('skope_rules - depth_3', skope, 'n_estimators', n, 'max_depth', 3) for n in np.linspace(2, 80, 10, dtype=int)]
)
ALL_ESTIMATORS.append(
    [Model('rulefit - depth_1', rfit, 'max_rules', n, 'tree_size', 2) for n in np.linspace(2, 100, 10, dtype=int)]
    + [Model('rulefit - depth_2', rfit, 'max_rules', n, 'tree_size', 4) for n in np.linspace(2, 50, 10, dtype=int)]
    + [Model('rulefit - depth_3', rfit, 'max_rules', n, 'tree_size', 8) for n in np.linspace(2, 50, 10, dtype=int)]
)
ALL_ESTIMATORS.append(
    [Model('fplasso - max_card_1', fpl, 'max_rules', n, 'maxcardinality', 1) for n in np.linspace(2, 100, 10, dtype=int)]
    + [Model('fplasso - max_card_2', fpl, 'max_rules', n, 'maxcardinality', 2) for n in np.linspace(2, 60, 10, dtype=int)]
    + [Model('fplasso - max_card_3', fpl, 'max_rules', n, 'maxcardinality', 3) for n in np.linspace(2, 50, 10, dtype=int)]
)
ALL_ESTIMATORS.append(
    [Model('fpskope - No dedup', fps, 'maxcardinality', n, 'max_depth_duplication', None) for n in [1, 2]]
    + [Model('fpskope - max_dedup_1', fps, 'maxcardinality', n, 'max_depth_duplication', 1) for n in [1, 2, 3, 4]]
    + [Model('fpskope - max_dedup_2', fps, 'maxcardinality', n, 'max_depth_duplication', 2) for n in [1, 2, 3, 4]]
    + [Model('fpskope - max_dedup_3', fps, 'maxcardinality', n, 'max_depth_duplication', 3) for n in [1, 2, 3, 4]]
)
ALL_ESTIMATORS.append(
    [Model('brl - max_card_1', brl, 'listlengthprior', n, 'maxcardinality', 1) for n in np.linspace(1, 20, 10, dtype=int)]
    + [Model('brl - max_card_2', brl, 'listlengthprior', n, 'maxcardinality', 2) for n in np.linspace(1, 16, 8, dtype=int)]
    + [Model('brl - max_card_3', brl, 'listlengthprior', n, 'maxcardinality', 3) for n in np.linspace(1, 16, 8, dtype=int)]
)
ALL_ESTIMATORS.append([Model('grl - ', grl, 'max_depth', n) for n in np.arange(1, 6)])
ALL_ESTIMATORS.append([Model('oner - ', oner, 'max_depth', n) for n in np.arange(1, 6)])
ALL_ESTIMATORS.append([Model('brs - ', brs, 'n_estimators', n) for n in np.linspace(1, 32, 10, dtype=int)])
