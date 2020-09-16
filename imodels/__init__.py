"""
Python `imodels` package for interpretable models compatible with scikit-learn.
Github repo available [here](https://github.com/csinva/interpretability-implementations-demos).
.. include:: ../readme.md
"""

from .bayesian_rule_list.RuleListClassifier import RuleListClassifier
from .greedy_rule_list import GreedyRuleListClassifier
from .iterative_random_forest.iterative_random_forest import IRFClassifier
from .rule_fit import RuleFitRegressor
from .skope_rules import SkopeRulesClassifier
from .slim import SLIMRegressor

CLASSIFIERS = RuleListClassifier, GreedyRuleListClassifier, IRFClassifier
REGRESSORS = RuleFitRegressor, SkopeRulesClassifier, SLIMRegressor
# from .optimal_classification_tree import OptimalTreeModel
