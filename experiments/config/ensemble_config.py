import numpy as np
import imodels

from experiments.config.config import RULEFIT_KWARGS, FPL_KWARGS, BRL_KWARGS
from experiments.models.stablelinear import StableLinearClassifier as stbl
from experiments.util import Model, get_best_model_under_complexity


def get_weak_learner_inst_list(complexity_limits, easy: bool = False):
    weak_learners = []
    for complexity_limit in complexity_limits:
        weak_learners_c = [
            get_best_model_under_complexity(c=complexity_limit,
                                            model_name='skope_rules',
                                            model_cls=imodels.SkopeRulesClassifier,
                                            curve_param=1,
                                            kwargs={'precision_min': 0.3 if not easy else 0.5},
                                            easy=easy),
            get_best_model_under_complexity(c=complexity_limit,
                                            model_name='rulefit',
                                            model_cls=imodels.RuleFitClassifier,
                                            curve_param=30 if not easy else 13,
                                            kwargs=RULEFIT_KWARGS,
                                            easy=easy),
            get_best_model_under_complexity(c=complexity_limit,
                                            model_name='fplasso',
                                            model_cls=imodels.FPLassoClassifier,
                                            curve_param=1,
                                            kwargs=FPL_KWARGS,
                                            easy=easy),
            # get_best_model_under_complexity(c=complexity_limit,
            #                                 model_name='brl',
            #                                 model_cls=imodels.BayesianRuleListClassifier,
            #                                 curve_param=3 if not easy else 1,
            #                                 kwargs=BRL_KWARGS,
            #                                 easy=easy),
            get_best_model_under_complexity(c=complexity_limit,
                                            model_name='brs',
                                            model_cls=imodels.BoostedRulesClassifier,
                                            easy=easy)
        ]
        weak_learners_c_clean = [wl for wl in weak_learners_c if wl is not None]
        weak_learners.append(weak_learners_c_clean)
    return weak_learners

c_limits = np.concatenate((np.arange(2, 10), np.arange(10, 29, 2)))
stbl_cs = lambda: enumerate(c_limits)

def get_ensembles_hard():
    stbl_kw_hard = [{'weak_learners': wl_lst} for wl_lst in get_weak_learner_inst_list(c_limits)]

    kwargs_1 = [{**kw, 'penalty': 'l2', 'alpha': 30, 'max_rules': None} for kw in stbl_kw_hard]
    kwargs_2 = [{**kw, 'penalty': 'l2', 'alpha': 13, 'max_rules': None} for kw in stbl_kw_hard]
    kwargs_3 = [{**kw, 'penalty': 'l2', 'alpha': 5, 'max_rules': None} for kw in stbl_kw_hard]
    kwargs_4 = [{**kw, 'penalty': 'l2', 'alpha': 2, 'max_rules': None} for kw in stbl_kw_hard]
    kwargs_5 = [{**kw, 'penalty': 'l2', 'alpha': 1, 'max_rules': None} for kw in stbl_kw_hard]

    BEST_ENSEMBLES = []

    ALL_ENSEMBLES = []
    ALL_ENSEMBLES.append(
        # [Model('stbl_l2 - mm0_alpha_30', stbl, 'max_complexity', c, 'min_mult', 0, kwargs_1[i]) for i, c in stbl_cs()]
        # + [Model('stbl_l2 - mm0_alpha_13', stbl, 'max_complexity', c, 'min_mult', 0, kwargs_2[i]) for i, c in stbl_cs()]
        # + [Model('stbl_l2 - mm0_alpha_5', stbl, 'max_complexity', c, 'min_mult', 0, kwargs_3[i]) for i, c in stbl_cs()]
        # + [Model('stbl_l2 - mm0_alpha_2', stbl, 'max_complexity', c, 'min_mult', 0, kwargs_3[i]) for i, c in stbl_cs()]
        # + [Model('stbl_l2 - mm0_alpha_1', stbl, 'max_complexity', c, 'min_mult', 0, kwargs_3[i]) for i, c in stbl_cs()]
        [Model('stbl_l2 - mm1_alpha_30', stbl, 'max_complexity', c, 'min_mult', 1, kwargs_1[i]) for i, c in stbl_cs()]
        + [Model('stbl_l2 - mm1_alpha_13', stbl, 'max_complexity', c, 'min_mult', 1, kwargs_2[i]) for i, c in stbl_cs()]
        + [Model('stbl_l2 - mm1_alpha_5', stbl, 'max_complexity', c, 'min_mult', 1, kwargs_3[i]) for i, c in stbl_cs()]
        + [Model('stbl_l2 - mm1_alpha_2', stbl, 'max_complexity', c, 'min_mult', 1, kwargs_4[i]) for i, c in stbl_cs()]
        + [Model('stbl_l2 - mm1_alpha_1', stbl, 'max_complexity', c, 'min_mult', 1, kwargs_5[i]) for i, c in stbl_cs()]
        + [Model('stbl_l2 - mm2_alpha_30', stbl, 'max_complexity', c, 'min_mult', 2, kwargs_1[i]) for i, c in stbl_cs()]
        + [Model('stbl_l2 - mm2_alpha_13', stbl, 'max_complexity', c, 'min_mult', 2, kwargs_2[i]) for i, c in stbl_cs()]
        + [Model('stbl_l2 - mm2_alpha_5', stbl, 'max_complexity', c, 'min_mult', 2, kwargs_3[i]) for i, c in stbl_cs()]
        + [Model('stbl_l2 - mm2_alpha_2', stbl, 'max_complexity', c, 'min_mult', 2, kwargs_4[i]) for i, c in stbl_cs()]
        + [Model('stbl_l2 - mm2_alpha_1', stbl, 'max_complexity', c, 'min_mult', 2, kwargs_5[i]) for i, c in stbl_cs()]
    )

    kwargs_1 = [{**kw, 'penalty': 'l1', 'alpha': 30, 'max_rules': None} for kw in stbl_kw_hard]
    kwargs_2 = [{**kw, 'penalty': 'l1', 'alpha': 13, 'max_rules': None} for kw in stbl_kw_hard]
    kwargs_3 = [{**kw, 'penalty': 'l1', 'alpha': 5, 'max_rules': None} for kw in stbl_kw_hard]

    ALL_ENSEMBLES.append(
        # [Model('stbl_l1 - mm0_alpha_30', stbl, 'max_complexity', c, 'min_mult', 0, kwargs_1[i]) for i, c in stbl_cs()]
        # + [Model('stbl_l1 - mm0_alpha_13', stbl, 'max_complexity', c, 'min_mult', 0, kwargs_2[i]) for i, c in stbl_cs()]
        # + [Model('stbl_l1 - mm0_alpha_5', stbl, 'max_complexity', c, 'min_mult', 0, kwargs_3[i]) for i, c in stbl_cs()]
        # + [Model('stbl_l1 - mm0_alpha_2', stbl, 'max_complexity', c, 'min_mult', 0, kwargs_3[i]) for i, c in stbl_cs()]
        # + [Model('stbl_l1 - mm0_alpha_1', stbl, 'max_complexity', c, 'min_mult', 0, kwargs_3[i]) for i, c in stbl_cs()]
        [Model('stbl_l1 - mm1_alpha_30', stbl, 'max_complexity', c, 'min_mult', 1, kwargs_1[i]) for i, c in stbl_cs()]
        + [Model('stbl_l1 - mm1_alpha_13', stbl, 'max_complexity', c, 'min_mult', 1, kwargs_2[i]) for i, c in stbl_cs()]
        + [Model('stbl_l1 - mm1_alpha_5', stbl, 'max_complexity', c, 'min_mult', 1, kwargs_3[i]) for i, c in stbl_cs()]
        + [Model('stbl_l1 - mm1_alpha_2', stbl, 'max_complexity', c, 'min_mult', 1, kwargs_4[i]) for i, c in stbl_cs()]
        + [Model('stbl_l1 - mm1_alpha_1', stbl, 'max_complexity', c, 'min_mult', 1, kwargs_5[i]) for i, c in stbl_cs()]
        + [Model('stbl_l1 - mm2_alpha_30', stbl, 'max_complexity', c, 'min_mult', 2, kwargs_1[i]) for i, c in stbl_cs()]
        + [Model('stbl_l1 - mm2_alpha_13', stbl, 'max_complexity', c, 'min_mult', 2, kwargs_2[i]) for i, c in stbl_cs()]
        + [Model('stbl_l1 - mm2_alpha_5', stbl, 'max_complexity', c, 'min_mult', 2, kwargs_3[i]) for i, c in stbl_cs()]
        + [Model('stbl_l1 - mm2_alpha_2', stbl, 'max_complexity', c, 'min_mult', 2, kwargs_4[i]) for i, c in stbl_cs()]
        + [Model('stbl_l1 - mm2_alpha_1', stbl, 'max_complexity', c, 'min_mult', 2, kwargs_5[i]) for i, c in stbl_cs()]
    )
    return ALL_ENSEMBLES

def get_ensembles_easy():
    stbl_kw_easy = [{'weak_learners': wl_lst} for wl_lst in get_weak_learner_inst_list(c_limits, easy=True)]

    BEST_EASY_ENSEMBLES = []


    kwargs_1 = [{**kw, 'penalty': 'l2', 'alpha': 30, 'max_rules': None} for kw in stbl_kw_easy]
    kwargs_2 = [{**kw, 'penalty': 'l2', 'alpha': 13, 'max_rules': None} for kw in stbl_kw_easy]
    kwargs_3 = [{**kw, 'penalty': 'l2', 'alpha': 5, 'max_rules': None} for kw in stbl_kw_easy]
    kwargs_4 = [{**kw, 'penalty': 'l2', 'alpha': 2, 'max_rules': None} for kw in stbl_kw_easy]
    kwargs_5 = [{**kw, 'penalty': 'l2', 'alpha': 1, 'max_rules': None} for kw in stbl_kw_easy]

    EASY_ENSEMBLES = []
    EASY_ENSEMBLES.append(
        # [Model('stbl_l2 - mm0_alpha_30', stbl, 'max_complexity', c, 'min_mult', 0, kwargs_1[i]) for i, c in stbl_cs()]
        # + [Model('stbl_l2 - mm0_alpha_13', stbl, 'max_complexity', c, 'min_mult', 0, kwargs_2[i]) for i, c in stbl_cs()]
        # + [Model('stbl_l2 - mm0_alpha_5', stbl, 'max_complexity', c, 'min_mult', 0, kwargs_3[i]) for i, c in stbl_cs()]
        # + [Model('stbl_l2 - mm0_alpha_2', stbl, 'max_complexity', c, 'min_mult', 0, kwargs_3[i]) for i, c in stbl_cs()]
        # + [Model('stbl_l2 - mm0_alpha_1', stbl, 'max_complexity', c, 'min_mult', 0, kwargs_3[i]) for i, c in stbl_cs()]
        [Model('stbl_l2 - mm1_alpha_30', stbl, 'max_complexity', c, 'min_mult', 1, kwargs_1[i]) for i, c in stbl_cs()]
        + [Model('stbl_l2 - mm1_alpha_13', stbl, 'max_complexity', c, 'min_mult', 1, kwargs_2[i]) for i, c in stbl_cs()]
        + [Model('stbl_l2 - mm1_alpha_5', stbl, 'max_complexity', c, 'min_mult', 1, kwargs_3[i]) for i, c in stbl_cs()]
        + [Model('stbl_l2 - mm1_alpha_2', stbl, 'max_complexity', c, 'min_mult', 1, kwargs_4[i]) for i, c in stbl_cs()]
        + [Model('stbl_l2 - mm1_alpha_1', stbl, 'max_complexity', c, 'min_mult', 1, kwargs_5[i]) for i, c in stbl_cs()]
        + [Model('stbl_l2 - mm2_alpha_30', stbl, 'max_complexity', c, 'min_mult', 2, kwargs_1[i]) for i, c in stbl_cs()]
        + [Model('stbl_l2 - mm2_alpha_13', stbl, 'max_complexity', c, 'min_mult', 2, kwargs_2[i]) for i, c in stbl_cs()]
        + [Model('stbl_l2 - mm2_alpha_5', stbl, 'max_complexity', c, 'min_mult', 2, kwargs_3[i]) for i, c in stbl_cs()]
        + [Model('stbl_l2 - mm2_alpha_2', stbl, 'max_complexity', c, 'min_mult', 2, kwargs_4[i]) for i, c in stbl_cs()]
        + [Model('stbl_l2 - mm2_alpha_1', stbl, 'max_complexity', c, 'min_mult', 2, kwargs_5[i]) for i, c in stbl_cs()]
    )

    kwargs_1 = [{**kw, 'penalty': 'l1', 'alpha': 30, 'max_rules': None} for kw in stbl_kw_easy]
    kwargs_2 = [{**kw, 'penalty': 'l1', 'alpha': 13, 'max_rules': None} for kw in stbl_kw_easy]
    kwargs_3 = [{**kw, 'penalty': 'l1', 'alpha': 5, 'max_rules': None} for kw in stbl_kw_easy]
    kwargs_4 = [{**kw, 'penalty': 'l1', 'alpha': 2, 'max_rules': None} for kw in stbl_kw_easy]
    kwargs_5 = [{**kw, 'penalty': 'l1', 'alpha': 1, 'max_rules': None} for kw in stbl_kw_easy]

    EASY_ENSEMBLES.append(
        # [Model('stbl_l1 - mm0_alpha_30', stbl, 'max_complexity', c, 'min_mult', 0, kwargs_1[i]) for i, c in stbl_cs()]
        # + [Model('stbl_l1 - mm0_alpha_13', stbl, 'max_complexity', c, 'min_mult', 0, kwargs_2[i]) for i, c in stbl_cs()]
        # + [Model('stbl_l1 - mm0_alpha_5', stbl, 'max_complexity', c, 'min_mult', 0, kwargs_3[i]) for i, c in stbl_cs()]
        # + [Model('stbl_l1 - mm0_alpha_2', stbl, 'max_complexity', c, 'min_mult', 0, kwargs_3[i]) for i, c in stbl_cs()]
        # + [Model('stbl_l1 - mm0_alpha_1', stbl, 'max_complexity', c, 'min_mult', 0, kwargs_3[i]) for i, c in stbl_cs()]
        [Model('stbl_l1 - mm1_alpha_30', stbl, 'max_complexity', c, 'min_mult', 1, kwargs_1[i]) for i, c in stbl_cs()]
        + [Model('stbl_l1 - mm1_alpha_13', stbl, 'max_complexity', c, 'min_mult', 1, kwargs_2[i]) for i, c in stbl_cs()]
        + [Model('stbl_l1 - mm1_alpha_5', stbl, 'max_complexity', c, 'min_mult', 1, kwargs_3[i]) for i, c in stbl_cs()]
        + [Model('stbl_l1 - mm1_alpha_2', stbl, 'max_complexity', c, 'min_mult', 1, kwargs_4[i]) for i, c in stbl_cs()]
        + [Model('stbl_l1 - mm1_alpha_1', stbl, 'max_complexity', c, 'min_mult', 1, kwargs_5[i]) for i, c in stbl_cs()]
        + [Model('stbl_l1 - mm2_alpha_30', stbl, 'max_complexity', c, 'min_mult', 2, kwargs_1[i]) for i, c in stbl_cs()]
        + [Model('stbl_l1 - mm2_alpha_13', stbl, 'max_complexity', c, 'min_mult', 2, kwargs_2[i]) for i, c in stbl_cs()]
        + [Model('stbl_l1 - mm2_alpha_5', stbl, 'max_complexity', c, 'min_mult', 2, kwargs_3[i]) for i, c in stbl_cs()]
        + [Model('stbl_l1 - mm2_alpha_2', stbl, 'max_complexity', c, 'min_mult', 2, kwargs_4[i]) for i, c in stbl_cs()]
        + [Model('stbl_l1 - mm2_alpha_1', stbl, 'max_complexity', c, 'min_mult', 2, kwargs_5[i]) for i, c in stbl_cs()]
    )

    return EASY_ENSEMBLES
