"""
Python `imodels` package for interpretable models compatible with scikit-learn.
Github repo available [here](https://github.com/csinva/interpretability-implementations-demos).
.. include:: ./documentation.md
"""

from .bayesian_rule_list.RuleListClassifier import RuleListClassifier
from .rulefit.rulefit import RuleFit
from .slim.SLIM import SLIM
from .greedy_rule_list.GreedyRuleList import GreedyRuleList
from .skrules.skope_rules import SkopeRules
from .irf.irf import IRFClassifier
# from .optimal_classification_tree import OptimalTreeModel