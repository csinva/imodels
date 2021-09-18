"""
.. include:: ../readme.md
"""
# Python `imodels` package for interpretable models compatible with scikit-learn.
# Github repo available [here](https://github.com/csinva/interpretability-implementations-demos).

# from .tree.iterative_random_forest.iterative_random_forest import IRFClassifier
# from .tree.optimal_classification_tree import OptimalTreeModel
from .algebraic.slim import SLIMRegressor, SLIMClassifier
from .rule_list.bayesian_rule_list.bayesian_rule_list import BayesianRuleListClassifier
from .rule_list.greedy_rule_list import GreedyRuleListClassifier
from .rule_list.one_r import OneRClassifier
from .rule_list.corels_wrapper import CorelsRuleListClassifier
from .rule_set.boosted_rules import BoostedRulesClassifier
from .rule_set.fplasso import FPLassoRegressor, FPLassoClassifier
from .rule_set.fpskope import FPSkopeClassifier
from .rule_set.rule_fit import RuleFitRegressor, RuleFitClassifier
from .rule_set.skope_rules import SkopeRulesClassifier
from .rule_set.slipper import SlipperClassifier
from .rule_set.boa import BOAClassifier
from .rule_set import boosted_rules
from .rule_set.boosted_rules import *
from .discretization.discretizer import RFDiscretizer, BasicDiscretizer
from .discretization.mdlp import MDLPDiscretizer, BRLDiscretizer

CLASSIFIERS = [BayesianRuleListClassifier, GreedyRuleListClassifier, SkopeRulesClassifier,
               BoostedRulesClassifier, SLIMClassifier, SlipperClassifier, BOAClassifier]  # , IRFClassifier
REGRESSORS = [RuleFitRegressor, SLIMRegressor]
DISCRETIZERS = [RFDiscretizer, BasicDiscretizer, MDLPDiscretizer, BRLDiscretizer]

