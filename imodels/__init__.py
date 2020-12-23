"""
.. include:: ../readme.md
"""
# Python `imodels` package for interpretable models compatible with scikit-learn.
# Github repo available [here](https://github.com/csinva/interpretability-implementations-demos).

from .rule_list.bayesian_rule_list.bayesian_rule_list import BayesianRuleListClassifier
from .rule_list.greedy_rule_list import GreedyRuleListClassifier
from .rule_list.one_r import OneRClassifier
from .rule_set.rule_fit import RuleFitRegressor
from .rule_set.skope_rules import SkopeRulesClassifier
from .rule_set.boosted_rules import BoostedRulesClassifier
# from .tree.iterative_random_forest.iterative_random_forest import IRFClassifier
# from .tree.optimal_classification_tree import OptimalTreeModel
from .algebraic.slim import SLIMRegressor

CLASSIFIERS = BayesianRuleListClassifier, GreedyRuleListClassifier, SkopeRulesClassifier, BoostedRulesClassifier #, IRFClassifier
REGRESSORS = RuleFitRegressor, SLIMRegressor
