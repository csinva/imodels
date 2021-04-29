import numpy as np
from imodels import (
    SkopeRulesClassifier as skope, RuleFitClassifier as rfit, 
    BayesianRuleListClassifier as brl, BoostedRulesClassifier as brs
)

from experiments.util import Model, get_best_models_under_complexity

weak_learners = [('skope_rules', skope), ('rulefit', rfit), ('brl', brl), ('brs', brs)]
weak_learners_inst = [get_best_models_under_complexity(c, weak_learners, metric='hard_mean_ROCAUC') for c in np.arange(5, 30, 2)]
stbl_kw = [{'weak_learners': wl_lst} for wl_lst in weak_learners_inst]
stbl_cs = lambda: enumerate(np.arange(5, 30, 2))

BEST_ENSEMBLES = []

ALL_ENSEMBLES = []

kwargs_1 = {**stbl_kw[i], 'penalty': 'l2', 'alpha': 30, 'max_rules': None}
kwargs_2 = {**stbl_kw[i], 'penalty': 'l2', 'alpha': 13, 'max_rules': None}
kwargs_3 = {**stbl_kw[i], 'penalty': 'l2', 'alpha': 5, 'max_rules': None}

ALL_ENSEMBLES.append([
    [Model('stbl_l2_mm0_alpha_30', stbl, 'max_complexity', c, 'min_mult', 0, kwargs_1) for i, c in stbl_cs()]
    + [Model('stbl_l2_mm0_alpha_13', stbl, 'max_complexity', c, 'min_mult', 0, kwargs_2) for i, c in stbl_cs()]
    + [Model('stbl_l2_mm0_alpha_5', stbl, 'max_complexity', c, 'min_mult', 0, kwargs_3) for i, c in stbl_cs()]
    + [Model('stbl_l2_mm1_alpha_30', stbl, 'max_complexity', c, 'min_mult', 1, kwargs_1) for i, c in stbl_cs()]
    + [Model('stbl_l2_mm1_alpha_13', stbl, 'max_complexity', c, 'min_mult', 1, kwargs_2) for i, c in stbl_cs()]
    + [Model('stbl_l2_mm1_alpha_5', stbl, 'max_complexity', c, 'min_mult', 1, kwargs_3) for i, c in stbl_cs()]
    + [Model('stbl_l2_mm2_alpha_30', stbl, 'max_complexity', c, 'min_mult', 2, kwargs_1) for i, c in stbl_cs()]
    + [Model('stbl_l2_mm2_alpha_13', stbl, 'max_complexity', c, 'min_mult', 2, kwargs_2) for i, c in stbl_cs()]
    + [Model('stbl_l2_mm2_alpha_5', stbl, 'max_complexity', c, 'min_mult', 2, kwargs_3) for i, c in stbl_cs()]
])

kwargs_1 = {**stbl_kw[i], 'alpha': 30, 'max_rules': None}
kwargs_2 = {**stbl_kw[i], 'alpha': 13, 'max_rules': None}
kwargs_3 = {**stbl_kw[i], 'alpha': 5, 'max_rules': None}

ALL_ENSEMBLES.append([
    [Model('stbl_l1_mm0_alpha_30', stbl, 'max_complexity', c, 'min_mult', 0, kwargs_1) for i, c in stbl_cs()]
    + [Model('stbl_l1_mm0_alpha_13', stbl, 'max_complexity', c, 'min_mult', 0, kwargs_2) for i, c in stbl_cs()]
    + [Model('stbl_l1_mm0_alpha_5', stbl, 'max_complexity', c, 'min_mult', 0, kwargs_3) for i, c in stbl_cs()]
    + [Model('stbl_l1_mm1_alpha_30', stbl, 'max_complexity', c, 'min_mult', 1, kwargs_1) for i, c in stbl_cs()]
    + [Model('stbl_l1_mm1_alpha_13', stbl, 'max_complexity', c, 'min_mult', 1, kwargs_2) for i, c in stbl_cs()]
    + [Model('stbl_l1_mm1_alpha_5', stbl, 'max_complexity', c, 'min_mult', 1, kwargs_3) for i, c in stbl_cs()]
    + [Model('stbl_l1_mm2_alpha_30', stbl, 'max_complexity', c, 'min_mult', 2, kwargs_1) for i, c in stbl_cs()]
    + [Model('stbl_l1_mm2_alpha_13', stbl, 'max_complexity', c, 'min_mult', 2, kwargs_2) for i, c in stbl_cs()]
    + [Model('stbl_l1_mm2_alpha_5', stbl, 'max_complexity', c, 'min_mult', 2, kwargs_3) for i, c in stbl_cs()]
])
