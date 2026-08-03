"""
.. include:: ../readme.md
"""
# Python `imodels` package for interpretable models compatible with scikit-learn.
# Github repo available [here](https://github.com/csinva/imodels)

from .algebraic.slim import SLIMRegressor, SLIMClassifier
from .algebraic.tree_gam import TreeGAMClassifier, TreeGAMRegressor
from .algebraic.marginal_shrinkage_linear_model import (
    MarginalShrinkageLinearModelRegressor,
)
from .discretization.discretizer import RFDiscretizer, BasicDiscretizer
from .discretization.mdlp import MDLPDiscretizer, BRLDiscretizer
from .experimental.bartpy import BART
from .rule_list.bayesian_rule_list.bayesian_rule_list import BayesianRuleListClassifier
from .rule_list.fast_frugal_tree import FastFrugalTreeClassifier
from .rule_list.greedy_rule_list import GreedyRuleListClassifier
from .rule_list.one_r import OneRClassifier
from .rule_set.boosted_rules import BoostedRulesClassifier, BoostedRulesRegressor
from .rule_set.brs import BayesianRuleSetClassifier
from .rule_set.fplasso import FPLassoRegressor, FPLassoClassifier
from .rule_set.fpskope import FPSkopeClassifier
from .rule_set.rule_fit import RuleFitRegressor, RuleFitClassifier
from .rule_set.skope_rules import SkopeRulesClassifier
from .rule_set.slipper import SlipperClassifier
from .tree.c45_tree.c45_tree import C45TreeClassifier
from .tree.cart_ccp import (
    DecisionTreeCCPClassifier,
    DecisionTreeCCPRegressor,
    HSDecisionTreeCCPClassifierCV,
    HSDecisionTreeCCPRegressorCV,
)

from .tree.cart_wrapper import GreedyTreeClassifier, GreedyTreeRegressor
from .tree.figs import FIGSRegressor, FIGSClassifier, FIGSRegressorCV, FIGSClassifierCV
from .tree.hierarchical_shrinkage import (
    HSTreeRegressor,
    HSTreeClassifier,
    HSTreeRegressorCV,
    HSTreeClassifierCV,
)
from .tree.tao import TaoTreeClassifier, TaoTreeRegressor
from .util.automl import AutoInterpretableClassifier, AutoInterpretableRegressor
from .util.data_util import get_clean_dataset
from .util.get_rules import get_rules
from .util.tree_viz import shadow_tree
from .util.distillation import DistilledRegressor
from .util.explain_errors import explain_classification_errors
from .clustering.stableclustering import StableClustering

CLASSIFIERS = [
    BayesianRuleListClassifier,
    GreedyRuleListClassifier,
    FastFrugalTreeClassifier,
    SkopeRulesClassifier,
    BoostedRulesClassifier,
    SLIMClassifier,
    SlipperClassifier,
    BayesianRuleSetClassifier,
    C45TreeClassifier,
    OneRClassifier,
    RuleFitClassifier,
    FPLassoClassifier,
    FPSkopeClassifier,
    TaoTreeClassifier,
    TreeGAMClassifier,
    FIGSClassifier,
    FIGSClassifierCV,
    HSTreeClassifier,
    HSTreeClassifierCV,
    GreedyTreeClassifier,
    DecisionTreeCCPClassifier,
    AutoInterpretableClassifier,
]
REGRESSORS = [
    RuleFitRegressor,
    FPLassoRegressor,
    SLIMRegressor,
    GreedyTreeRegressor,
    FIGSRegressor,
    FIGSRegressorCV,
    TaoTreeRegressor,
    TreeGAMRegressor,
    BoostedRulesRegressor,
    MarginalShrinkageLinearModelRegressor,
    HSTreeRegressor,
    HSTreeRegressorCV,
    DecisionTreeCCPRegressor,
    BART,
    AutoInterpretableRegressor,
]
ESTIMATORS = CLASSIFIERS + REGRESSORS
DISCRETIZERS = [RFDiscretizer, BasicDiscretizer,
                MDLPDiscretizer, BRLDiscretizer]

# The public API. Kept explicit so that `from imodels import *` brings in the
# models and helpers rather than whatever each submodule happened to import.
__all__ = [
    "AutoInterpretableClassifier", "AutoInterpretableRegressor", "BART",
    "BRLDiscretizer", "BasicDiscretizer", "BayesianRuleListClassifier",
    "BayesianRuleSetClassifier", "BoostedRulesClassifier",
    "BoostedRulesRegressor", "C45TreeClassifier", "CLASSIFIERS",
    "DISCRETIZERS", "DecisionTreeCCPClassifier", "DecisionTreeCCPRegressor",
    "DistilledRegressor", "ESTIMATORS", "FIGSClassifier",
    "FIGSClassifierCV", "FIGSRegressor", "FIGSRegressorCV",
    "FPLassoClassifier", "FPLassoRegressor", "FPSkopeClassifier",
    "FastFrugalTreeClassifier", "GreedyRuleListClassifier",
    "GreedyTreeClassifier", "GreedyTreeRegressor",
    "HSDecisionTreeCCPClassifierCV", "HSDecisionTreeCCPRegressorCV",
    "HSTreeClassifier", "HSTreeClassifierCV", "HSTreeRegressor",
    "HSTreeRegressorCV", "MDLPDiscretizer",
    "MarginalShrinkageLinearModelRegressor", "OneRClassifier", "REGRESSORS",
    "RFDiscretizer", "RuleFitClassifier", "RuleFitRegressor",
    "SLIMClassifier", "SLIMRegressor", "SkopeRulesClassifier",
    "SlipperClassifier", "StableClustering", "TaoTreeClassifier",
    "TaoTreeRegressor", "TreeGAMClassifier", "TreeGAMRegressor",
    "explain_classification_errors", "get_clean_dataset", "get_rules",
    "shadow_tree",
]
