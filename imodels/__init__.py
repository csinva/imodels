"""
Python `imodels` package for interpretable models compatible with scikit-learn.
Github repo available [here](https://github.com/csinva/interpretability-implementations-demos).
.. include:: ./documentation.md
"""

from .bayesian_rule_list.RuleListClassifier import RuleListClassifier
from .rule_fit import RuleFit
from .slim import SLIM
from .greedy_rule_list import GreedyRuleList
from .skrules.skope_rules import SkopeRules
from .iterative_random_forest.iterative_random_forest import IRFClassifier
# from .optimal_classification_tree import OptimalTreeModel