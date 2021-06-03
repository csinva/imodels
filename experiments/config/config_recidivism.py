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


RECIDIVISM_ESTIMATORS = []
RECIDIVISM_ESTIMATORS.append(
    [Model('random_forest - mss_2', rf, 'n_estimators', n, 'min_samples_split', 2, {'max_depth': 3}) for n in np.arange(1, 8)]
    + [Model('random_forest - mss_100', rf, 'n_estimators', n, 'min_samples_split', 100, {'max_depth': 3}) for n in np.arange(1, 8)]
    + [Model('random_forest - mss_500', rf, 'n_estimators', n, 'min_samples_split', 500, {'max_depth': 3}) for n in np.arange(1, 8)]
    + [Model('random_forest - mss_1000', rf, 'n_estimators', n, 'min_samples_split', 1000, {'max_depth': 3}) for n in np.arange(1, 8)]
    + [Model('random_forest - mss_1500', rf, 'n_estimators', n, 'min_samples_split', 1500, {'max_depth': 3}) for n in np.arange(1, 8)]
    + [Model('random_forest - mss_2000', rf, 'n_estimators', n, 'min_samples_split', 2000, {'max_depth': 3}) for n in np.arange(1, 18)]
    + [Model('random_forest - mss_2500', rf, 'n_estimators', n, 'min_samples_split', 2500, {'max_depth': 3}) for n in np.arange(1, 50)]
)
RECIDIVISM_ESTIMATORS.append(
    [Model('gradient_boosting - mss_2', gb, 'n_estimators', n, 'min_samples_split', 2, {'max_depth': 3}) for n in np.arange(1, 8)]
    + [Model('gradient_boosting - mss_100', gb, 'n_estimators', n, 'min_samples_split', 100, {'max_depth': 3}) for n in np.arange(1, 8)]
    + [Model('gradient_boosting - mss_500', gb, 'n_estimators', n, 'min_samples_split', 500, {'max_depth': 3}) for n in np.arange(1, 8)]
    + [Model('gradient_boosting - mss_1000', gb, 'n_estimators', n, 'min_samples_split', 1000, {'max_depth': 3}) for n in np.arange(1, 8)]
    + [Model('gradient_boosting - mss_1500', gb, 'n_estimators', n, 'min_samples_split', 1500, {'max_depth': 3}) for n in np.arange(1, 8)]
    + [Model('gradient_boosting - mss_2000', gb, 'n_estimators', n, 'min_samples_split', 2000, {'max_depth': 3}) for n in np.arange(1, 8)]
    + [Model('gradient_boosting - mss_3000', gb, 'n_estimators', n, 'min_samples_split', 3000, {'max_depth': 3}) for n in np.arange(1, 18)]
)
RECIDIVISM_ESTIMATORS.append(
    [Model('skope_rules - mss_100_prec_0.5', skope, 'n_estimators', n, 'min_samples_split', 100, {'max_depth': 3}) for n in np.arange(1, 20)]
    + [Model('skope_rules - mss_1000_prec_0.5', skope, 'n_estimators', n, 'min_samples_split', 1000, {'max_depth': 3}) for n in np.arange(1, 20)]
    + [Model('skope_rules - mss_2000_prec_0.5', skope, 'n_estimators', n, 'min_samples_split', 2000, {'max_depth': 3}) for n in np.arange(1, 30).tolist() + np.arange(30, 200, 5).tolist()]
    + [Model('skope_rules - mss_100_prec_0.4', skope, 'n_estimators', n, 'min_samples_split', 100, {'precision_min': 0.4, 'max_depth': 3}) for n in np.arange(1, 20)]
    + [Model('skope_rules - mss_1000_prec_0.4', skope, 'n_estimators', n, 'min_samples_split', 1000, {'precision_min': 0.4, 'max_depth': 3}) for n in np.arange(1, 20)]
    + [Model('skope_rules - mss_2000_prec_0.4', skope, 'n_estimators', n, 'min_samples_split', 2000, {'precision_min': 0.4, 'max_depth': 3}) for n in np.arange(1, 30).tolist() + np.arange(30, 200, 5).tolist()]
)
RECIDIVISM_ESTIMATORS.append(
    [Model('rulefit - alpha_300', rfit, 'n_estimators', n, 'alpha', 300, RULEFIT_KWARGS) for n in np.arange(1, 30).tolist() + np.arange(30, 200, 5).tolist()]
    + [Model('rulefit - alpha_200', rfit, 'n_estimators', n, 'alpha', 200, RULEFIT_KWARGS) for n in np.arange(1, 30).tolist() + np.arange(30, 200, 5).tolist()]
    + [Model('rulefit - alpha_100', rfit, 'n_estimators', n, 'alpha', 100, RULEFIT_KWARGS) for n in np.arange(1, 30).tolist() + np.arange(30, 200, 5).tolist()]
    + [Model('rulefit - alpha_50', rfit, 'n_estimators', n, 'alpha', 50, RULEFIT_KWARGS) for n in np.arange(1, 30).tolist() + np.arange(30, 100, 5).tolist()]
    + [Model('rulefit - alpha_10', rfit, 'n_estimators', n, 'alpha', 10, RULEFIT_KWARGS) for n in np.arange(1, 15)]
    + [Model('rulefit - alpha_1', rfit, 'n_estimators', n, 'alpha', 1, RULEFIT_KWARGS) for n in np.arange(1, 13)]
)
RECIDIVISM_ESTIMATORS.append(
    [Model('fplasso - minsup_0.05', fpl, 'alpha', a, 'minsupport', 0.05, FPL_KWARGS) for a in np.logspace(1.5, 3, 40)]
    + [Model('fplasso - minsup_0.1', fpl, 'alpha', a, 'minsupport', 0.1, FPL_KWARGS) for a in np.logspace(1.5, 3, 40)]
    + [Model('fplasso - minsup_0.15', fpl, 'alpha', a, 'minsupport', 0.15, FPL_KWARGS) for a in np.logspace(1.5, 3, 40)]
)
RECIDIVISM_ESTIMATORS.append(
    [Model('brl - minsup_0.7', brl, 'listlengthprior', n, 'minsupport', 0.7, BRL_KWARGS) for n in np.arange(1, 20)]
    + [Model('brl - minsup_0.5', brl, 'listlengthprior', n, 'minsupport', 0.5, BRL_KWARGS) for n in np.arange(1, 20)]
    + [Model('brl - minsup_0.3', brl, 'listlengthprior', n, 'minsupport', 0.3, BRL_KWARGS) for n in np.arange(1, 20)]
    + [Model('brl - minsup_0.1', brl, 'listlengthprior', n, 'minsupport', 0.1, BRL_KWARGS) for n in np.arange(1, 20)]
    + [Model('brl - minsup_0.05', brl, 'listlengthprior', n, 'minsupport', 0.05, BRL_KWARGS) for n in np.arange(1, 20)]
)
RECIDIVISM_ESTIMATORS.append(
    [Model('brs - mss_2', brs, 'n_estimators', n, 'estimator', partial(DecisionTreeClassifier, min_samples_split=2, max_depth=3)) for n in np.arange(1, 9)]
    + [Model('brs - mss_100', brs, 'n_estimators', n, 'estimator', partial(DecisionTreeClassifier, min_samples_split=100, max_depth=3)) for n in np.arange(1, 9)]
    + [Model('brs - mss_500', brs, 'n_estimators', n, 'estimator', partial(DecisionTreeClassifier, min_samples_split=500, max_depth=3)) for n in np.arange(1, 9)]
    + [Model('brs - mss_1000', brs, 'n_estimators', n, 'estimator', partial(DecisionTreeClassifier, min_samples_split=1000, max_depth=3)) for n in np.arange(1, 9)]
    + [Model('brs - mss_1500', brs, 'n_estimators', n, 'estimator', partial(DecisionTreeClassifier, min_samples_split=1500, max_depth=3)) for n in np.arange(1, 9)]
    + [Model('brs - mss_2000', brs, 'n_estimators', n, 'estimator', partial(DecisionTreeClassifier, min_samples_split=2000, max_depth=3)) for n in np.arange(1, 9)]
    + [Model('brs - mss_2500', brs, 'n_estimators', n, 'estimator', partial(DecisionTreeClassifier, min_samples_split=2500, max_depth=3)) for n in np.arange(1, 9)]
    + [Model('brs - mss_3000', brs, 'n_estimators', n, 'estimator', partial(DecisionTreeClassifier, min_samples_split=3000, max_depth=3)) for n in np.arange(1, 11)]
)

RECIDIVISM_TEST_ESTIMATORS = [
    [Model('random_forest - mss_2500', rf, 'n_estimators', n, 'min_samples_split', 2500, {'max_depth': 3}) for n in np.arange(1, 50)],
    [Model('gradient_boosting - mss_3000', gb, 'n_estimators', n, 'min_samples_split', 3000, {'max_depth': 3}) for n in np.arange(1, 18)],
    (
        [Model('skope_rules - mss_2000_prec_0.5', skope, 'n_estimators', n, 'min_samples_split', 2000, {'max_depth': 3}) for n in np.arange(1, 30).tolist() + np.arange(30, 200, 5).tolist()]
        + [Model('skope_rules - mss_1000_prec_0.5', skope, 'n_estimators', n, 'min_samples_split', 1000, {'max_depth': 3}) for n in np.arange(1, 20)]
    ),
    (
        [Model('rulefit - alpha_300', rfit, 'n_estimators', n, 'alpha', 300, RULEFIT_KWARGS) for n in np.arange(1, 30).tolist() + np.arange(30, 200, 5).tolist()]
        + [Model('rulefit - alpha_50', rfit, 'n_estimators', n, 'alpha', 50, RULEFIT_KWARGS) for n in np.arange(1, 30).tolist() + np.arange(30, 100, 5).tolist()]
    ),
    [Model('fplasso - minsup_0.05', fpl, 'alpha', a, 'minsupport', 0.05, FPL_KWARGS) for a in np.logspace(1.5, 3, 40)],
    (
        [Model('brl - minsup_0.7', brl, 'listlengthprior', n, 'minsupport', 0.7, BRL_KWARGS) for n in np.arange(1, 20)]
        + [Model('brl - minsup_0.3', brl, 'listlengthprior', n, 'minsupport', 0.3, BRL_KWARGS) for n in np.arange(1, 20)]
        + [Model('brl - minsup_0.1', brl, 'listlengthprior', n, 'minsupport', 0.1, BRL_KWARGS) for n in np.arange(1, 20)]
    ),
    [Model('brs - mss_3000', brs, 'n_estimators', n, 'estimator', partial(DecisionTreeClassifier, min_samples_split=3000, max_depth=3)) for n in np.arange(1, 11)]
]


def get_weak_learner_inst_list_recidivism(complexity_limits):
    weak_learners = []
    for complexity_limit in complexity_limits:
        common_kwargs = {'c': complexity_limit, 'dataset': 'recidivism'}
        weak_learners_c = [
            get_best_model_under_complexity(**common_kwargs,
                                            model_name='skope_rules',
                                            model_cls=skope,
                                            curve_params=[2000, 1000],
                                            kwargs={'max_depth': 3}),
            get_best_model_under_complexity(**common_kwargs,
                                            model_name='rulefit',
                                            model_cls=rfit,
                                            curve_params=[300, 50],
                                            kwargs=RULEFIT_KWARGS),
            get_best_model_under_complexity(**common_kwargs,
                                            model_name='fplasso',
                                            model_cls=fpl,
                                            curve_params=[0.05],
                                            kwargs=FPL_KWARGS),
            get_best_model_under_complexity(**common_kwargs,
                                            model_name='brl',
                                            model_cls=brl,
                                            kwargs=BRL_KWARGS,
                                            curve_params=[0.7, 0.3, 0.1]),
            get_best_model_under_complexity(**common_kwargs,
                                            model_name='brs',
                                            model_cls=brs,
                                            curve_params=[3000])
        ]
        weak_learners_c_clean = [wl for wl in weak_learners_c if wl is not None]
        weak_learners.append(weak_learners_c_clean)
    return weak_learners



def get_ensembles_for_recidivism(test: bool = False):
    c_limits = np.arange(2, 40)
    stbl_cs = lambda: enumerate(c_limits)
    stbl_kw = [{'weak_learners': wl_lst} for wl_lst in get_weak_learner_inst_list_recidivism(c_limits)]

    kwargs_0 = [{**kw, 'penalty': 'l2', 'alpha': 200, 'max_rules': None} for kw in stbl_kw]
    kwargs_1 = [{**kw, 'penalty': 'l2', 'alpha': 100, 'max_rules': None} for kw in stbl_kw]
    kwargs_2 = [{**kw, 'penalty': 'l2', 'alpha': 50, 'max_rules': None} for kw in stbl_kw]
    kwargs_3 = [{**kw, 'penalty': 'l2', 'alpha': 10, 'max_rules': None} for kw in stbl_kw]
    kwargs_4 = [{**kw, 'penalty': 'l2', 'alpha': 1, 'max_rules': None} for kw in stbl_kw]

    ALL_ENSEMBLES = []
    ALL_ENSEMBLES.append(
        [Model('stbl_l2 - mm1_alpha_200', stbl, 'max_complexity', c, 'min_mult', 1, kwargs_0[i]) for i, c in stbl_cs()]
        + [Model('stbl_l2 - mm1_alpha_100', stbl, 'max_complexity', c, 'min_mult', 1, kwargs_1[i]) for i, c in stbl_cs()]
        + [Model('stbl_l2 - mm1_alpha_50', stbl, 'max_complexity', c, 'min_mult', 1, kwargs_2[i]) for i, c in stbl_cs()]
        + [Model('stbl_l2 - mm1_alpha_10', stbl, 'max_complexity', c, 'min_mult', 1, kwargs_3[i]) for i, c in stbl_cs()]
        + [Model('stbl_l2 - mm1_alpha_1', stbl, 'max_complexity', c, 'min_mult', 1, kwargs_4[i]) for i, c in stbl_cs()]
    )

    BEST_ENSEMBLES = []
    BEST_ENSEMBLES += [
        [Model('stbl_l2_mm1', stbl, 'max_complexity', c, 'min_mult', 1, kwargs_2[i]) for i, c in stbl_cs()]
    ]

    kwargs_0 = [{**kw, 'penalty': 'l1', 'alpha': 200, 'max_rules': None} for kw in stbl_kw]
    kwargs_1 = [{**kw, 'penalty': 'l1', 'alpha': 100, 'max_rules': None} for kw in stbl_kw]
    kwargs_2 = [{**kw, 'penalty': 'l1', 'alpha': 50, 'max_rules': None} for kw in stbl_kw]
    kwargs_3 = [{**kw, 'penalty': 'l1', 'alpha': 10, 'max_rules': None} for kw in stbl_kw]
    kwargs_4 = [{**kw, 'penalty': 'l1', 'alpha': 1, 'max_rules': None} for kw in stbl_kw]

    ALL_ENSEMBLES.append(
        [Model('stbl_l1 - mm0_alpha_200', stbl, 'max_complexity', c, 'min_mult', 0, kwargs_0[i]) for i, c in stbl_cs()]
        + [Model('stbl_l1 - mm0_alpha_100', stbl, 'max_complexity', c, 'min_mult', 0, kwargs_1[i]) for i, c in stbl_cs()]
        + [Model('stbl_l1 - mm0_alpha_50', stbl, 'max_complexity', c, 'min_mult', 0, kwargs_2[i]) for i, c in stbl_cs()]
        + [Model('stbl_l1 - mm0_alpha_10', stbl, 'max_complexity', c, 'min_mult', 0, kwargs_3[i]) for i, c in stbl_cs()]
        + [Model('stbl_l1 - mm0_alpha_1', stbl, 'max_complexity', c, 'min_mult', 0, kwargs_4[i]) for i, c in stbl_cs()]
        + [Model('stbl_l1 - mm1_alpha_200', stbl, 'max_complexity', c, 'min_mult', 1, kwargs_0[i]) for i, c in stbl_cs()]
        + [Model('stbl_l1 - mm1_alpha_100', stbl, 'max_complexity', c, 'min_mult', 1, kwargs_1[i]) for i, c in stbl_cs()]
        + [Model('stbl_l1 - mm1_alpha_50', stbl, 'max_complexity', c, 'min_mult', 1, kwargs_2[i]) for i, c in stbl_cs()]
        + [Model('stbl_l1 - mm1_alpha_10', stbl, 'max_complexity', c, 'min_mult', 1, kwargs_3[i]) for i, c in stbl_cs()]
        + [Model('stbl_l1 - mm1_alpha_1', stbl, 'max_complexity', c, 'min_mult', 1, kwargs_4[i]) for i, c in stbl_cs()]
    )

    BEST_ENSEMBLES += [
        [Model('stbl_l1_mm0', stbl, 'max_complexity', c, 'min_mult', 0, kwargs_2[i]) for i, c in stbl_cs()],
        [Model('stbl_l1_mm1', stbl, 'max_complexity', c, 'min_mult', 1, kwargs_2[i]) for i, c in stbl_cs()]
    ]

    return ALL_ENSEMBLES if not test else BEST_ENSEMBLES
