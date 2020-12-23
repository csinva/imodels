'''The oneR algorithm returns a rule list that splits on only one (usually continuous) feature
It works by building a greedy rule list using only one feature at a time, and then returning
the rule list with the highest accuracy
'''

import math
import numpy as np
from copy import deepcopy
from sklearn.base import BaseEstimator

from imodels import GreedyRuleListClassifier
from imodels.rule_list.rule_list import RuleList


class OneRClassifier(GreedyRuleListClassifier):
    def __init__(self, max_depth=5, class_weight=None, criterion='gini'):
        self.max_depth = max_depth
        self.feature_names = None
        self.class_weight = class_weight
        self.criterion = criterion

    def fit(self, X, y, depth=0, feature_names=None, verbose=False):
        """Fit oneR
        """

        # set self.feature_names and make sure x, y are not pandas type
        if 'pandas' in str(type(X)):
            self.feature_names = X.columns
            X = X.values
        else:
            if self.feature_names is None:
                self.feature_names = ['feat ' + str(i) for i in range(X.shape[1])]
        if feature_names is not None:
            self.feature_names = feature_names
        if 'pandas' in str(type(y)):
            y = y.values

        ms = []
        accs = np.zeros(X.shape[1])
        for col_idx in range(X.shape[1]):
            x = X[:, col_idx].reshape(-1, 1)
            m = GreedyRuleListClassifier(max_depth=self.max_depth, class_weight=self.class_weight,
                                         criterion=self.criterion)
            feat_names_single = [self.feature_names[col_idx]]
            m.fit(x, y, feature_names=feat_names_single)
            accs[col_idx] = np.mean(m.predict(x) == y)
            ms.append(m)
            # print('acc', feat_names_single[0], f'{accs[col_idx]:0.2f}')
        col_idx_best = np.argmax(accs)
        self.rules_ = ms[col_idx_best].rules_

        # need to adjust index_col since was fitted with only 1 col
        for rule in self.rules_:
            if 'index_col' in rule:
                rule['index_col'] += col_idx_best
        self.depth = len(self.rules_)
