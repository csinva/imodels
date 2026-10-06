from imodels.rule_set.boosted_rules import BoostedRulesClassifier
from imodels.rule_set.slipper_util import SlipperBaseEstimator
from imodels.util.arguments import check_binary_target, check_two_classes


class SlipperClassifier(BoostedRulesClassifier):
    """SLIPPER: boosted rules, as described in A Simple, Fast, and Effective Rule Learner
    (Cohen and Singer, 1999).

    Parameters
    ----------
    n_estimators : int, default=10
        Number of boosting rounds (rules).
    **kwargs
        Passed on to BoostedRulesClassifier.
    """

    def __init__(self, n_estimators=10, **kwargs):
        super().__init__(estimator=SlipperBaseEstimator(), n_estimators=n_estimators, **kwargs)

    def fit(self, X, y, feature_names=None, **kwargs):
        # its base estimator learns a single binary rule, so a multiclass target
        # otherwise fails deep inside boosting with a shape mismatch
        check_binary_target(self, y)
        check_two_classes(self, y)
        return super().fit(X, y, feature_names=feature_names, **kwargs)
