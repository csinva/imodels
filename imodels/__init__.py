"""
.. include:: ../readme.md
"""
# Python `imodels` package for interpretable models compatible with scikit-learn.
# Github repo available [here](https://github.com/csinva/interpretability-implementations-demos).

from .bayesian_rule_list.bayesian_rule_list import BayesianRuleListClassifier
from .greedy_rule_list import GreedyRuleListClassifier
from .iterative_random_forest.iterative_random_forest import IRFClassifier
from .skulefit.rule_fit import RuleFitRegressor
from .skulefit.skope_rules import SkopeRulesClassifier
from .slim import SLIMRegressor

CLASSIFIERS = BayesianRuleListClassifier, GreedyRuleListClassifier, IRFClassifier
REGRESSORS = RuleFitRegressor, SkopeRulesClassifier, SLIMRegressor
# from .optimal_classification_tree import OptimalTreeModel
