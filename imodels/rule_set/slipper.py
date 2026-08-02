from imodels.rule_set.boosted_rules import BoostedRulesClassifier
from imodels.rule_set.slipper_util import SlipperBaseEstimator
from imodels.util.arguments import check_binary_target


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

    def fit(self, X, y, feature_names=None, **kwargs):
        # its base estimator learns a single binary rule, so a multiclass target
        # otherwise fails deep inside boosting with a shape mismatch
        check_binary_target(self, y)
        return super().fit(X, y, feature_names=feature_names, **kwargs)
