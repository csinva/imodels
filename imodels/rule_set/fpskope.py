from typing import List

import numpy as np
import pandas as pd

from imodels.rule_set.skope_rules import SkopeRulesClassifier
from imodels.util.convert import itemsets_to_rules
from imodels.util.extract import extract_fpgrowth
from imodels.util.rule import Rule
from imodels.util.score import score_precision_recall


class FPSkopeClassifier(SkopeRulesClassifier):

    def __init__(self,
                 minsupport=0.1,
                 maxcardinality=2,
                 verbose=False,
                 precision_min=0.5,
                 recall_min=0.01,
                 n_estimators=10,
                 max_samples=.8,
                 max_samples_features=1.,
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
                         random_state,
                         verbose)
        self.minsupport = minsupport
        self.maxcardinality = maxcardinality
        self.verbose = verbose

    def fit(self, X, y=None, feature_names=None, undiscretized_features=[], sample_weight=None):
        self.undiscretized_features = undiscretized_features
        super().fit(X, y, feature_names=feature_names, sample_weight=sample_weight)
        return self

    def _extract_rules(self, X, y) -> List[str]:
        X = pd.DataFrame(X, columns=self.feature_placeholders)
        itemsets = extract_fpgrowth(X, minsupport=self.minsupport,
                                    maxcardinality=self.maxcardinality,
                                    verbose=self.verbose)
        return [itemsets_to_rules(itemsets)], [np.arange(X.shape[0])], [np.arange(len(self.feature_names))]

    def _score_rules(self, X, y, rules) -> List[Rule]:
        return score_precision_recall(X, y,
                                      rules,
                                      self.estimators_samples_,
                                      self.estimators_features_,
                                      self.feature_placeholders,
                                      oob=False)
