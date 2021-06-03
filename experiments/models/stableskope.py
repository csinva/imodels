import numpy as np
from typing import List

from imodels.rule_set.skope_rules import SkopeRulesClassifier
from imodels.util.rule import Rule
from imodels.util.score import score_precision_recall
from sklearn.base import BaseEstimator

from .util import extract_ensemble


class StableSkopeClassifier(SkopeRulesClassifier):

    def __init__(self,
                 weak_learners: List[BaseEstimator],
                 max_complexity: int,
                 min_mult: int = 1,
                 precision_min=0.5,
                 recall_min=0.4,
                 n_estimators=10,
                 max_samples=.8,
                 max_samples_features=.8,
                 bootstrap=False,
                 bootstrap_features=False,
                 max_depth=3,
                 max_depth_duplication=None,
                 max_features=1.,
                 min_samples_split=2,
                 n_jobs=1,
                 random_state=None):
        super().__init__(precision_min,
                         recall_min,
                         n_estimators,
                         max_samples,
                         max_samples_features,
                         bootstrap,
                         bootstrap_features,
                         max_depth,
                         max_depth_duplication,
                         max_features,
                         min_samples_split,
                         n_jobs,
                         random_state)
        self.weak_learners = weak_learners
        self.max_complexity = max_complexity
        self.min_mult = min_mult

    def fit(self, X, y=None, feature_names=None, sample_weight=None):
        super().fit(X, y, feature_names=feature_names, sample_weight=sample_weight)
        return self

    def _extract_rules(self, X, y) -> List[str]:
        return [extract_ensemble(self.weak_learners, X, y, self.min_mult)], [np.arange(X.shape[0])], [np.arange(len(self.feature_names))]

    def _score_rules(self, X, y, rules) -> List[Rule]:
        return score_precision_recall(X, y,
                                      rules,
                                      self.estimators_samples_,
                                      self.estimators_features_,
                                      self.feature_placeholders,
                                      oob=False)
