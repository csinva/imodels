'''Greedy rule list.
Greedily splits on one feature at a time along a single path.
Tries to find rules which maximize the probability of class 1.
Currently only supports binary classification.
'''

import math
from copy import deepcopy

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.tree import DecisionTreeClassifier
from imodels.rule_list.rule_list import RuleList
from imodels.util.arguments import check_fit_arguments


class GreedyRuleListClassifier(BaseEstimator, RuleList, ClassifierMixin):
    def __init__(self, max_depth: int = 5, class_weight=None,
                 criterion: str = 'gini'):
        '''
        Params
        ------
        max_depth
            Maximum depth the list can achieve
        criterion: str
            Criterion used to split
            'gini', 'entropy', or 'log_loss'
        '''

        self.max_depth = max_depth
        self.class_weight = class_weight
        self.criterion = criterion
        self.depth = 0  # tracks the fitted depth

    def fit(self, X, y, depth: int = 0, feature_names=None, verbose=False):
        """
        Params
        ------
        X: array_like
            Feature set
        y: array_like
            target variable
        depth
            the depth of the current layer (used to recurse)
        """
        X, y, feature_names = check_fit_arguments(self, X, y, feature_names)
        return self.fit_node_recursive(X, y, depth=0, verbose=verbose)

    def fit_node_recursive(self, X, y, depth: int, verbose):

        # base case 1: no data in this group
        if y.size == 0:
            return []

        # base case 2: all y is the same in this group
        elif np.all(y == y[0]):
            return [{'val': y[0], 'num_pts': y.size}]

         # base case 3: max depth reached
        elif depth == self.max_depth:
            return [{'val': np.mean(y), 'num_pts': y.size}]

        # recursively generate rule list
        else:

            # find a split with the best value for the criterion
            m = DecisionTreeClassifier(max_depth=1, criterion=self.criterion)
            m.fit(X, y)
            col = m.tree_.feature[0]
            cutoff = m.tree_.threshold[0]
            # col, cutoff, criterion_val = self._find_best_split(X, y)
            if col == -2:
                return []
                
            y_left = y[X[:, col] < cutoff]  # left-hand side data
            y_right = y[X[:, col] >= cutoff]  # right-hand side data


            # put higher probability of class 1 on the right-hand side
            if len(y_left) > 0 and np.mean(y_left) > np.mean(y_right):
                flip = True
                tmp = deepcopy(y_left)
                y_left = deepcopy(y_right)
                y_right = tmp
                x_left = X[X[:, col] >= cutoff]
            else:
                flip = False
                x_left = X[X[:, col] < cutoff]

            # print
            if verbose:
                print(
                    f'{np.mean(100 * y):.2f} -> {self.feature_names_[col]} -> {np.mean(100 * y_left):.2f} ({y_left.size}) {np.mean(100 * y_right):.2f} ({y_right.size})')

            # save info
            par_node = [{
                'col': self.feature_names_[col],
                'index_col': col,
                'cutoff': cutoff,
                'val': np.mean(y_left),  # will be the values before splitting in the next lower level
                'flip': flip,
                'val_right': np.mean(y_right),
                'num_pts': y.size,
                'num_pts_right': y_right.size
            }]

            # generate tree for the non-leaf data
            par_node = par_node + \
                self.fit_node_recursive(x_left, y_left, depth + 1, verbose=verbose)

            self.depth += 1  # increase the depth since we call fit once
            self.rules_ = par_node
            self.complexity_ = len(self.rules_)
            self.classes_ = unique_labels(y)
            return par_node

    def predict_proba(self, X):
        check_is_fitted(self)
        X = check_array(X)
        n = X.shape[0]
        probs = np.zeros(n)
        for i in range(n):
            x = X[i]
            for j, rule in enumerate(self.rules_):
                if j == len(self.rules_) - 1:
                    probs[i] = rule['val']
                    continue
                regular_condition = x[rule["index_col"]] >= rule["cutoff"]
                flipped_condition = x[rule["index_col"]] < rule["cutoff"]
                condition = flipped_condition if rule["flip"] else regular_condition
                if condition:
                    probs[i] = rule['val_right']
                    break
        return np.vstack((1 - probs, probs)).transpose()  # probs (n, 2)

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        return np.argmax(self.predict_proba(X), axis=1)

    """
    def __str__(self):
        # s = ''
        # for rule in self.rules_:
        #     s += f"mean {rule['val'].round(3)} ({rule['num_pts']} pts)\n"
        #     if 'col' in rule:
        #         s += f"if {rule['col']} >= {rule['cutoff']} then {rule['val_right'].round(3)} ({rule['num_pts_right']} pts)\n"
        # return s
    """

    def __str__(self):
        '''Print out the list in a nice way
        '''
        s = '> ------------------------------\n> Greedy Rule List\n> ------------------------------\n'

        def red(s):
            # return f"\033[91m{s}\033[00m"
            return s

        def cyan(s):
            # return f"\033[96m{s}\033[00m"
            return s

        def rule_name(rule):
            if rule['flip']:
                return '~' + rule['col']
            return rule['col']

        # rule = self.rules_[0]
        #     s += f"{red((100 * rule['val']).round(3))}% IwI ({rule['num_pts']} pts)\n"
        for rule in self.rules_:
            s += u'\u2193\n' + f"{cyan((100 * rule['val']).round(2))}% risk ({rule['num_pts']} pts)\n"
            #         s += f"\t{'Else':>45} => {cyan((100 * rule['val']).round(2)):>6}% IwI ({rule['val'] * rule['num_pts']:.0f}/{rule['num_pts']} pts)\n"
            if 'col' in rule:
                #             prefix = f"if {rule['col']} >= {rule['cutoff']}"
                prefix = f"if {rule_name(rule)}"
                val = f"{100 * rule['val_right'].round(3)}"
                s += f"\t{prefix} ==> {red(val)}% risk ({rule['num_pts_right']} pts)\n"
        # rule = self.rules_[-1]
        #     s += f"{red((100 * rule['val']).round(3))}% IwI ({rule['num_pts']} pts)\n"
        return s

    ######## HERE ONWARDS CUSTOM SPLITTING (DEPRECATED IN FAVOR OF SKLEARN STUMP) ########
    ######################################################################################
    def _find_best_split(self, x, y):
        """
        Find the best split from all features
        returns: the column to split on, the cutoff value, and the actual criterion_value
        """
        col = None
        min_criterion_val = 1e10
        cutoff = None

        # iterating through each feature
        for i, c in enumerate(x.T):

            # find the best split of that feature
            criterion_val, cur_cutoff = self._split_on_feature(c, y)

            # found perfect cutoff
            if criterion_val == 0:
                return i, cur_cutoff, criterion_val

            # check if it's best so far
            elif criterion_val <= min_criterion_val:
                min_criterion_val = criterion_val
                col = i
                cutoff = cur_cutoff
        return col, cutoff, min_criterion_val

    def _split_on_feature(self, col, y):
        """
        col: the column we split on
        y: target var
        """
        min_criterion_val = 1e10
        cutoff = 0.5

        # iterate through each value in the column
        for value in np.unique(col):
            # separate y into 2 groups
            y_predict = col < value

            # get criterion val of this split
            criterion_val = self._weighted_criterion(y_predict, y)

            # check if it's the smallest one so far
            if criterion_val <= min_criterion_val:
                min_criterion_val = criterion_val
                cutoff = value
        return min_criterion_val, cutoff

    def _weighted_criterion(self, split_decision, y_real):
        """Returns criterion calculated over a split
        split decision, True/False, and y_true can be multi class
        """
        if split_decision.shape[0] != y_real.shape[0]:
            print('They have to be the same length')
            return None

        # choose the splitting criterion
        if self.criterion == 'entropy':
            criterion_func = self._entropy_criterion
        elif self.criterion == 'gini':
            criterion_func = self._gini_criterion
        elif self.criterion == 'neg_corr':
            return self._neg_corr_criterion(split_decision, y_real)

        # left-hand side criterion
        s_left = criterion_func(y_real[split_decision])

        # right-hand side criterion
        s_right = criterion_func(y_real[~split_decision])

        # overall criterion, again weighted average
        n = y_real.shape[0]
        if self.class_weight is not None:
            sample_weights = np.ones(n)
            for c in self.class_weight.keys():
                idxs_c = y_real == c
                sample_weights[idxs_c] = self.class_weight[c]
            total_weight = np.sum(sample_weights)
            weight_left = np.sum(sample_weights[split_decision]) / total_weight
            # weight_right = np.sum(sample_weights[~split_decision]) / total_weight
        else:
            tot_left_samples = np.sum(split_decision == 1)
            weight_left = tot_left_samples / n

        s = weight_left * s_left + (1 - weight_left) * s_right
        return s

    def _gini_criterion(self, y):
        '''Returns gini index for one node
        = sum(pc * (1 – pc))
        '''
        s = 0
        n = y.shape[0]
        classes = np.unique(y)

        # for each class, get entropy
        for c in classes:
            # weights for each class
            n_c = np.sum(y == c)
            p_c = n_c / n

            # weighted avg
            s += p_c * (1 - p_c)

        return s

    def _entropy_criterion(self, y):
        """Returns entropy of a divided group of data
        Data may have multiple classes
        """
        s = 0
        n = len(y)
        classes = set(y)

        # for each class, get entropy
        for c in classes:
            # weights for each class
            weight = sum(y == c) / n

            def _entropy_from_counts(c1, c2):
                """Returns entropy of a group of data
                c1: count of one class
                c2: count of another class
                """
                if c1 == 0 or c2 == 0:  # when there is only one class in the group, entropy is 0
                    return 0

                def _entropy_func(p): return -p * math.log(p, 2)

                p1 = c1 * 1.0 / (c1 + c2)
                p2 = c2 * 1.0 / (c1 + c2)
                return _entropy_func(p1) + _entropy_func(p2)

            # weighted avg
            s += weight * _entropy_from_counts(sum(y == c), sum(y != c))
        return s

    def _neg_corr_criterion(self, split_decision, y):
        '''Returns negative correlation between y
        and the binary splitting variable split_decision
        y must be binary
        '''
        if np.unique(y).size < 2:
            return 0
        elif np.unique(y).size != 2:
            print('y must be binary output for corr criterion')

        # y should be 1 more often on the "right side" of the split
        if y.sum() < y.size / 2:
            y = 1 - y

        return -1 * np.corrcoef(split_decision.astype(np.int), y)[0, 1]
