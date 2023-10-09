from imodels.rule_set.boosted_rules import BoostedRulesClassifier
from imodels.rule_set.slipper_util import SlipperBaseEstimator


class SlipperClassifier(BoostedRulesClassifier):
    def __init__(self, n_estimators=10, **kwargs):
        '''
        An estimator that supports building rules as described in
        A Simple, Fast, and Effective Rule Learner (1999).
        Parameters
        ----------
        n_estimators
        '''
        super().__init__(estimator=SlipperBaseEstimator(), n_estimators=n_estimators, **kwargs)
        # super().__init__(n_estimators, SlipperBaseEstimator)
