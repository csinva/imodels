'''The oneR algorithm returns a rule list that splits on only one (usually continuous) feature
It works by building a greedy rule list using only one feature at a time, and then returning
the rule list with the highest accuracy
'''

import math
import numpy as np
from copy import deepcopy
from sklearn.base import BaseEstimator
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.multiclass import check_classification_targets, unique_labels
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from imodels import GreedyRuleListClassifier
from imodels.rule_list.rule_list import RuleList
from imodels.util.arguments import check_fit_arguments


class OneRClassifier(GreedyRuleListClassifier):
    def __init__(self, max_depth=5, class_weight=None, criterion='gini'):
        self.max_depth = max_depth
        self.feature_names_ = None
        self.class_weight = class_weight
        self.criterion = criterion
        self._estimator_type = 'classifier'

    def fit(self, X, y, feature_names=None):
        """Fit oneR
        """
        X, y, feature_names = check_fit_arguments(self, X, y, feature_names)

        ms = []
        accs = np.zeros(X.shape[1])
        for col_idx in range(X.shape[1]):
            x = X[:, col_idx].reshape(-1, 1)
            m = GreedyRuleListClassifier(max_depth=self.max_depth, class_weight=self.class_weight,
                                         criterion=self.criterion)
            feat_names_single = [self.feature_names_[col_idx]]
            m.fit(x, y, feature_names=feat_names_single)
            accs[col_idx] = np.mean(m.predict(x) == y)
            ms.append(m)
            # print('acc', feat_names_single[0], f'{accs[col_idx]:0.2f}')
        col_idx_best = np.argmax(accs)
        self.rules_ = ms[col_idx_best].rules_
        self.complexity_ = len(self.rules_)

        # need to adjust index_col since was fitted with only 1 col
        for rule in self.rules_:
            if 'index_col' in rule:
                rule['index_col'] += col_idx_best
        self.depth = len(self.rules_)
        return self
