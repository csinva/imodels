import numpy as np

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
        # each boosted estimator is a single rule, which is inherently binary;
        # without this the ensemble fails later on a shape mismatch
        n_classes = len(np.unique(y))
        if n_classes > 2:
            raise ValueError(
                f"{type(self).__name__} does not yet support multiclass "
                f"classification (found {n_classes} classes: "
                f"{sorted(np.unique(y).tolist())}); it is a binary classifier."
            )
        return super().fit(X, y, feature_names=feature_names, **kwargs)
        # super().__init__(n_estimators, SlipperBaseEstimator)

    def fit(self, X, y, feature_names=None, **kwargs):
        # its base estimator learns a single binary rule, so a multiclass target
        # otherwise fails deep inside boosting with a shape mismatch
        check_binary_target(self, y)
        return super().fit(X, y, feature_names=feature_names, **kwargs)
