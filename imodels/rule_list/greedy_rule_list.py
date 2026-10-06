'''Greedy rule list.
Greedily splits on one feature at a time along a single path.
Tries to find rules which maximize the probability of class 1.
Currently only supports binary classification.
'''

from copy import deepcopy

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.tree import DecisionTreeClassifier
from imodels.rule_list.rule_list import RuleList
from imodels.util.arguments import (check_binary_target, check_fit_arguments,
                                    check_predict_X, check_two_classes, decode_labels)


class GreedyRuleListClassifier(BaseEstimator, RuleList, ClassifierMixin):
    def __init__(self, max_depth: int = 5, class_weight=None,
                 criterion: str = 'gini'):
        '''
        Params
        ------
        max_depth
            Maximum depth the list can achieve
        class_weight: dict, 'balanced' or None
            Weights of the classes when choosing each split, keyed by the original
            labels; passed on to the sklearn stump that finds the split
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
        check_binary_target(self, y)
        check_two_classes(self, y)
        X, y, feature_names = check_fit_arguments(self, X, y, feature_names)
        self._class_weight = self._encode_class_weight()
        self.depth = 0  # reset so that refitting doesn't accumulate depth
        self.rules_ = self.fit_node_recursive(X, y, depth=0, verbose=verbose)
        self.complexity_ = len(self.rules_)
        return self

    def _encode_class_weight(self):
        """Key a class_weight dict by the integer codes the stumps are fit on."""
        if not isinstance(self.class_weight, dict):
            return self.class_weight  # None, 'balanced', or invalid (sklearn raises)
        missing = [c for c in self.class_weight if c not in list(self.classes_)]
        if missing:
            raise ValueError(f"class_weight has labels {missing} that are not in y "
                             f"(classes: {list(self.classes_)})")
        return {float(code): self.class_weight[c] for code, c in enumerate(self.classes_)
                if c in self.class_weight}

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
            m = DecisionTreeClassifier(max_depth=1, criterion=self.criterion,
                                       class_weight=getattr(self, '_class_weight',
                                                            self.class_weight))
            m.fit(X, y)
            col = m.tree_.feature[0]
            cutoff = m.tree_.threshold[0]
            # base case 4: no split found, so emit a leaf holding this group's mean.
            # (returning [] here would leave a split rule as the list's final entry,
            # which predict_proba treats as the default rule and applies to everything)
            if col == -2:
                return [{'val': np.mean(y), 'num_pts': y.size}]

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
                'depth': depth,
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
            return par_node

    def predict_proba(self, X):
        check_is_fitted(self)
        X = check_array(check_predict_X(self, X))
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
        X = check_array(check_predict_X(self, X))
        return decode_labels(self, np.argmax(self.predict_proba(X), axis=1))


    def __str__(self):
        '''Print out the list in a nice way
        '''

        s = '> ------------------------------\n> Greedy Rule List\n> ------------------------------\n'
        precision = 2
        for rule in self.rules_:
            if 'col' in rule:
                prefix = "if" if rule['depth'] == 0 else "else if"
                sign = '<=' if rule['flip'] else '>'
                threshold = rule['cutoff'].round(precision)
                condition = f"{prefix} {rule['col']} {sign} {threshold}"
                pred_prob = (100 * rule['val_right']).round(precision)
                num_pts = rule['num_pts_right']
                s += f"> {condition} | {pred_prob}% pred prob ({num_pts} obs)\n"
            else:
                s += f"> else | {(100 * rule['val']).round(precision)}% pred prob ({rule['num_pts']} obs)\n"
        s += '> ------------------------------\n'
        return s
