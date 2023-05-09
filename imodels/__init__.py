"""
.. include:: ../readme.md
"""
# Python `imodels` package for interpretable models compatible with scikit-learn.
# Github repo available [here](https://github.com/csinva/imodels)

from .algebraic.slim import SLIMRegressor, SLIMClassifier
from .discretization.discretizer import RFDiscretizer, BasicDiscretizer
from .discretization.mdlp import MDLPDiscretizer, BRLDiscretizer
from .experimental.bartpy import BART
from .rule_list.bayesian_rule_list.bayesian_rule_list import BayesianRuleListClassifier
from .rule_list.corels_wrapper import OptimalRuleListClassifier
from .rule_list.greedy_rule_list import GreedyRuleListClassifier
from .rule_list.one_r import OneRClassifier
from .rule_set import boosted_rules
from .rule_set.boosted_rules import *
from .rule_set.boosted_rules import BoostedRulesClassifier
from .rule_set.brs import BayesianRuleSetClassifier
from .rule_set.fplasso import FPLassoRegressor, FPLassoClassifier
from .rule_set.fpskope import FPSkopeClassifier
from .rule_set.rule_fit import RuleFitRegressor, RuleFitClassifier
from .rule_set.skope_rules import SkopeRulesClassifier
from .rule_set.slipper import SlipperClassifier
from .tree.c45_tree.c45_tree import C45TreeClassifier
from .tree.cart_ccp import DecisionTreeCCPClassifier, DecisionTreeCCPRegressor, HSDecisionTreeCCPClassifierCV, \
    HSDecisionTreeCCPRegressorCV
# from .tree.iterative_random_forest.iterative_random_forest import IRFClassifier
# from .tree.optimal_classification_tree import OptimalTreeModel
from .tree.cart_wrapper import GreedyTreeClassifier, GreedyTreeRegressor
from .tree.figs import FIGSRegressor, FIGSClassifier, FIGSRegressorCV, FIGSClassifierCV
from .tree.gosdt.pygosdt import OptimalTreeClassifier
from .tree.gosdt.pygosdt_shrinkage import HSOptimalTreeClassifier, HSOptimalTreeClassifierCV
from .tree.hierarchical_shrinkage import HSTreeRegressor, HSTreeClassifier, HSTreeRegressorCV, HSTreeClassifierCV
from .tree.tao import TaoTreeClassifier, TaoTreeRegressor
from .util.data_util import get_clean_dataset
from .util.distillation import DistilledRegressor
from .util.explain_errors import explain_classification_errors

CLASSIFIERS = [BayesianRuleListClassifier, GreedyRuleListClassifier, SkopeRulesClassifier,
               BoostedRulesClassifier, SLIMClassifier, SlipperClassifier, BayesianRuleSetClassifier,
               C45TreeClassifier, OptimalTreeClassifier, OptimalRuleListClassifier, OneRClassifier,
               SlipperClassifier, RuleFitClassifier, TaoTreeClassifier,
               FIGSClassifier, HSTreeClassifier, HSTreeClassifierCV]  # , IRFClassifier
REGRESSORS = [RuleFitRegressor, SLIMRegressor, GreedyTreeClassifier, FIGSRegressor,
              TaoTreeRegressor, HSTreeRegressor, HSTreeRegressorCV, BART]
DISCRETIZERS = [RFDiscretizer, BasicDiscretizer, MDLPDiscretizer, BRLDiscretizer]
