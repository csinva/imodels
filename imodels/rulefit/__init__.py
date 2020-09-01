'''[rulefit](http://statweb.stanford.edu/~jhf/ftp/RuleFit.pdf) (based on [this implementation](https://github.com/christophM/rulefit)) - find rules from a decision tree and build a linear model with them
'''
from .rulefit import RuleCondition, Rule, RuleEnsemble, RuleFit, FriedScale

__all__ = ["rulefit"]