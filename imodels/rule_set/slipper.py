from imodels.rule_set.boosted_rules import BoostedRulesClassifier
from imodels.rule_set.slipper_util import SlipperBaseEstimator


class SlipperClassifier(BoostedRulesClassifier):
    def __init__(self, n_estimators=10):
        '''
        An estimator that supports building rules as described in
        A Simple, Fast, and Effective Rule Learner (1999).
        Parameters
        ----------
        n_estimators
        '''
        super().__init__(n_estimators, SlipperBaseEstimator)
