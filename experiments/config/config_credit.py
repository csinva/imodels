from functools import partial

import numpy as np
from imodels import (
    SkopeRulesClassifier as skope, RuleFitClassifier as rfit, FPLassoClassifier as fpl, 
    FPSkopeClassifier as fps, BayesianRuleListClassifier as brl, BoostedRulesClassifier as brs
)
from sklearn.ensemble import RandomForestClassifier as rf, GradientBoostingClassifier as gb
from sklearn.tree import DecisionTreeClassifier

from experiments.config.config_general import RULEFIT_KWARGS, BRL_KWARGS, FPL_KWARGS
from experiments.models.stablelinear import StableLinearClassifier as stbl
from experiments.util import Model
from experiments.util import get_best_model_under_complexity


CREDIT_ESTIMATORS = []
CREDIT_ESTIMATORS.append(
    [Model('random_forest - mid_0', rf, 'n_estimators', n, 'min_impurity_decrease', 0., {'max_depth': 2}) for n in np.arange(1, 8)]
    + [Model('random_forest - mid_1e-4', rf, 'n_estimators', n, 'min_impurity_decrease', 1e-4, {'max_depth': 2}) for n in np.arange(1, 8)]
    + [Model('random_forest - mid_5e-4', rf, 'n_estimators', n, 'min_impurity_decrease', 5e-4, {'max_depth': 2}) for n in np.arange(1, 8)]
)
CREDIT_ESTIMATORS.append(
    [Model('gradient_boosting - mid_0', gb, 'n_estimators', n, 'min_impurity_decrease', 0., {'max_depth': 2}) for n in np.arange(1, 8)]
    + [Model('gradient_boosting - mid_50', gb, 'n_estimators', n, 'min_impurity_decrease', 50, {'max_depth': 2}) for n in np.arange(1, 30, 3)]
    + [Model('gradient_boosting - mid_80', gb, 'n_estimators', n, 'min_impurity_decrease', 80, {'max_depth': 2}) for n in np.arange(1, 40, 4)]
)
CREDIT_ESTIMATORS.append(
    [Model('skope_rules - prec_0.5', skope, 'n_estimators', n, 'precision_min', 0.5, {'max_depth': 2}) for n in np.arange(1, 80, 8)]
    + [Model('skope_rules - prec_.45', skope, 'n_estimators', n, 'precision_min', 0.45, {'max_depth': 2}) for n in [1, 2, 3] + np.arange(5, 40, 4).tolist()]
    + [Model('skope_rules - prec_0.4', skope, 'n_estimators', n, 'precision_min', 0.4, {'max_depth': 2}) for n in np.arange(1, 11).tolist() + [13, 15, 17]]
)
CREDIT_ESTIMATORS.append(
    [Model('rulefit - alpha_30', rfit, 'n_estimators', n, 'alpha', 30, RULEFIT_KWARGS) for n in np.arange(1, 13)]
    + [Model('rulefit - alpha_13', rfit, 'n_estimators', n, 'alpha', 13, RULEFIT_KWARGS) for n in np.arange(1, 11)]
    + [Model('rulefit - alpha_5', rfit, 'n_estimators', n, 'alpha', 5, RULEFIT_KWARGS) for n in np.arange(1, 11)]
    + [Model('rulefit - alpha_neg', rfit, 'n_estimators', n, 'alpha', 2, RULEFIT_KWARGS) for n in np.arange(1, 11)]
    + [Model('rulefit - alpha_01', rfit, 'n_estimators', n, 'alpha', 1, RULEFIT_KWARGS) for n in np.arange(1, 11)]
)
CREDIT_ESTIMATORS.append(
    [Model('fplasso - minsup_0.05', fpl, 'alpha', a, 'minsupport', 0.05, FPL_KWARGS) for a in np.logspace(2.5, 3.4, 10)]
    + [Model('fplasso - minsup_0.10', fpl, 'alpha', a, 'minsupport', 0.1, FPL_KWARGS) for a in np.logspace(2.5, 3.1, 10)]
    + [Model('fplasso - minsup_0.15', fpl, 'alpha', a, 'minsupport', 0.15, FPL_KWARGS) for a in np.logspace(2.5, 3.1, 10)]
)
CREDIT_ESTIMATORS.append(
    [Model('brl - minsup_0.5', brl, 'listlengthprior', n, 'minsupport', 0.5, BRL_KWARGS) for n in np.arange(1, 20, 2)]
    + [Model('brl - minsup_0.3', brl, 'listlengthprior', n, 'minsupport', 0.3, BRL_KWARGS) for n in np.arange(1, 20, 2)]
    + [Model('brl - minsup_0.1', brl, 'listlengthprior', n, 'minsupport', 0.1, BRL_KWARGS) for n in np.arange(1, 20, 2)]
)
CREDIT_ESTIMATORS.append(
    [Model('brs - mid_0', brs, 'n_estimators', n, 'estimator', partial(DecisionTreeClassifier, min_impurity_decrease=0, max_depth=2)) for n in np.arange(1, 9)]
    + [Model('brs - mid_1e-3', brs, 'n_estimators', n, 'estimator', partial(DecisionTreeClassifier, min_impurity_decrease=1e-3, max_depth=2)) for n in np.arange(1, 9)]
    + [Model('brs - mid_2e-3', brs, 'n_estimators', n, 'estimator', partial(DecisionTreeClassifier, min_impurity_decrease=2e-3, max_depth=2)) for n in np.arange(1, 9)]
    + [Model('brs - mid_5e-3', brs, 'n_estimators', n, 'estimator', partial(DecisionTreeClassifier, min_impurity_decrease=5e-3, max_depth=2)) for n in np.arange(1, 9)]
    + [Model('brs - mid_1e-2', brs, 'n_estimators', n, 'estimator', partial(DecisionTreeClassifier, min_impurity_decrease=1e-2, max_depth=2)) for n in np.arange(1, 9)]
)

CREDIT_TEST_ESTIMATORS = [
    [Model('random_forest', rf, 'n_estimators', n, 'min_impurity_decrease', 5e-4, {'max_depth': 2}) for n in np.arange(1, 8)],
    [Model('gradient_boosting', gb, 'n_estimators', n, 'min_impurity_decrease', 80, {'max_depth': 2}) for n in np.arange(1, 40, 4)],
    [Model('skope_rules', skope, 'n_estimators', n, 'precision_min', 0.4, {'max_depth': 2}) for n in np.arange(1, 11).tolist() + [13, 15, 17]],
    [Model('rulefit', rfit, 'n_estimators', n, 'alpha', 30, RULEFIT_KWARGS) for n in np.arange(1, 13)],
    [Model('fplasso', fpl, 'alpha', a, 'minsupport', 0.05, FPL_KWARGS) for a in np.logspace(2.5, 3.4, 10)],
    [Model('brl', brl, 'listlengthprior', n, 'minsupport', 0.3, BRL_KWARGS) for n in np.arange(1, 20, 2)],
    [Model('brs', brs, 'n_estimators', n, 'estimator', partial(DecisionTreeClassifier, min_impurity_decrease=2e-3, max_depth=2)) for n in np.arange(1, 9)]
]


def get_weak_learner_inst_list_credit(complexity_limits):
    weak_learners = []
    for complexity_limit in complexity_limits:
        common_kwargs = {'c': complexity_limit, 'dataset': 'credit'}
        weak_learners_c = [
            get_best_model_under_complexity(**common_kwargs,
                                            model_name='skope_rules',
                                            model_cls=skope,
                                            metric='mean_avg_precision',
                                            curve_param=0.4,
                                            kwargs={'max_depth': 2}),
            get_best_model_under_complexity(**common_kwargs,
                                            model_name='rulefit',
                                            model_cls=rfit,
                                            metric='mean_avg_precision',
                                            curve_param=30,
                                            kwargs=RULEFIT_KWARGS),
            get_best_model_under_complexity(**common_kwargs,
                                            model_name='fplasso',
                                            model_cls=fpl,
                                            metric='mean_avg_precision',
                                            curve_param=0.05,
                                            kwargs=FPL_KWARGS),
            get_best_model_under_complexity(**common_kwargs,
                                            model_name='brl',
                                            model_cls=brl,
                                            metric='mean_avg_precision',
                                            curve_param=0.3,
                                            kwargs=BRL_KWARGS),
            get_best_model_under_complexity(**common_kwargs,
                                            model_name='brs',
                                            model_cls=brs,
                                            curve_param=2e-3,
                                            metric='mean_avg_precision')
        ]
        weak_learners_c_clean = [wl for wl in weak_learners_c if wl is not None]
        weak_learners.append(weak_learners_c_clean)
    return weak_learners


def get_ensembles_for_credit(test: bool = False):
    c_limits = np.concatenate((np.arange(2, 10), np.arange(10, 29, 2)))
    stbl_cs = lambda: enumerate(c_limits)
    stbl_kw = [{'weak_learners': wl_lst} for wl_lst in get_weak_learner_inst_list_credit(c_limits)]

    kwargs_2 = [{**kw, 'penalty': 'l2', 'alpha': 100, 'max_rules': None} for kw in stbl_kw]
    kwargs_3 = [{**kw, 'penalty': 'l2', 'alpha': 10, 'max_rules': None} for kw in stbl_kw]
    kwargs_4 = [{**kw, 'penalty': 'l2', 'alpha': 1, 'max_rules': None} for kw in stbl_kw]
    kwargs_5 = [{**kw, 'penalty': 'l2', 'alpha': 0.1, 'max_rules': None} for kw in stbl_kw]

    ALL_ENSEMBLES = []
    ALL_ENSEMBLES.append(
        [Model('stbl_l2 - mm0_alpha_10.', stbl, 'max_complexity', c, 'min_mult', 0, kwargs_3[i]) for i, c in stbl_cs() if i < 8]
        + [Model('stbl_l2 - mm0_alpha_1.', stbl, 'max_complexity', c, 'min_mult', 0, kwargs_4[i]) for i, c in stbl_cs() if i < 8]
        + [Model('stbl_l2 - mm0_alpha_0.1', stbl, 'max_complexity', c, 'min_mult', 0, kwargs_5[i]) for i, c in stbl_cs() if i < 8]
        + [Model('stbl_l2 - mm1_alpha_10.', stbl, 'max_complexity', c, 'min_mult', 1, kwargs_3[i]) for i, c in stbl_cs()]
        + [Model('stbl_l2 - mm1_alpha_1.', stbl, 'max_complexity', c, 'min_mult', 1, kwargs_4[i]) for i, c in stbl_cs()]
        + [Model('stbl_l2 - mm1_alpha_0.1', stbl, 'max_complexity', c, 'min_mult', 1, kwargs_5[i]) for i, c in stbl_cs()]
        + [Model('stbl_l2 - mm2_alpha_10.', stbl, 'max_complexity', c, 'min_mult', 2, kwargs_3[i]) for i, c in stbl_cs()]
        + [Model('stbl_l2 - mm2_alpha_1.', stbl, 'max_complexity', c, 'min_mult', 2, kwargs_4[i]) for i, c in stbl_cs()]
        + [Model('stbl_l2 - mm2_alpha_0.1', stbl, 'max_complexity', c, 'min_mult', 2, kwargs_5[i]) for i, c in stbl_cs()]
    )

    BEST_ENSEMBLES = []
    BEST_ENSEMBLES += [
        [Model('stbl_l2_mm0', stbl, 'max_complexity', c, 'min_mult', 0, kwargs_3[i]) for i, c in stbl_cs() if i < 8],
        [Model('stbl_l2_mm1', stbl, 'max_complexity', c, 'min_mult', 1, kwargs_3[i]) for i, c in stbl_cs()]
    ]

    kwargs_2 = [{**kw, 'penalty': 'l1', 'alpha': 100, 'max_rules': None} for kw in stbl_kw]
    kwargs_3 = [{**kw, 'penalty': 'l1', 'alpha': 10, 'max_rules': None} for kw in stbl_kw]
    kwargs_4 = [{**kw, 'penalty': 'l1', 'alpha': 1, 'max_rules': None} for kw in stbl_kw]
    kwargs_5 = [{**kw, 'penalty': 'l1', 'alpha': 0.1, 'max_rules': None} for kw in stbl_kw]

    ALL_ENSEMBLES.append(
        [Model('stbl_l1 - mm0_alpha_100.', stbl, 'max_complexity', c, 'min_mult', 0, kwargs_2[i]) for i, c in stbl_cs()]
        + [Model('stbl_l1 - mm0_alpha_10.', stbl, 'max_complexity', c, 'min_mult', 0, kwargs_3[i]) for i, c in stbl_cs()]
        + [Model('stbl_l1 - mm0_alpha_1.', stbl, 'max_complexity', c, 'min_mult', 0, kwargs_4[i]) for i, c in stbl_cs()]
        + [Model('stbl_l1 - mm0_alpha_0.1', stbl, 'max_complexity', c, 'min_mult', 0, kwargs_5[i]) for i, c in stbl_cs()]
        + [Model('stbl_l1 - mm1_alpha_100.', stbl, 'max_complexity', c, 'min_mult', 1, kwargs_2[i]) for i, c in stbl_cs()]
        + [Model('stbl_l1 - mm1_alpha_10.', stbl, 'max_complexity', c, 'min_mult', 1, kwargs_3[i]) for i, c in stbl_cs()]
        + [Model('stbl_l1 - mm1_alpha_1.', stbl, 'max_complexity', c, 'min_mult', 1, kwargs_4[i]) for i, c in stbl_cs()]
        + [Model('stbl_l1 - mm1_alpha_0.1', stbl, 'max_complexity', c, 'min_mult', 1, kwargs_5[i]) for i, c in stbl_cs()]
        + [Model('stbl_l1 - mm2_alpha_100.', stbl, 'max_complexity', c, 'min_mult', 2, kwargs_2[i]) for i, c in stbl_cs()]
        + [Model('stbl_l1 - mm2_alpha_10.', stbl, 'max_complexity', c, 'min_mult', 2, kwargs_3[i]) for i, c in stbl_cs()]
        + [Model('stbl_l1 - mm2_alpha_1.', stbl, 'max_complexity', c, 'min_mult', 2, kwargs_4[i]) for i, c in stbl_cs()]
        + [Model('stbl_l1 - mm2_alpha_0.1', stbl, 'max_complexity', c, 'min_mult', 2, kwargs_5[i]) for i, c in stbl_cs()]
    )

    BEST_ENSEMBLES += [
        [Model('stbl_l1_mm0', stbl, 'max_complexity', c, 'min_mult', 0, kwargs_2[i]) for i, c in stbl_cs()],
        [Model('stbl_l1_mm1', stbl, 'max_complexity', c, 'min_mult', 1, kwargs_2[i]) for i, c in stbl_cs()]
    ]

    return ALL_ENSEMBLES if not test else BEST_ENSEMBLES
