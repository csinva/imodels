from copy import deepcopy
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.datasets
from sklearn import datasets
from sklearn import tree
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import plot_tree, DecisionTreeClassifier
from sklearn.utils import check_X_y, check_array

from imodels.tree.viz_utils import DecisionTreeViz

plt.rcParams['figure.dpi'] = 300


class Node:
    def __init__(self, feature: int = None, threshold: int = None,
                 value=None, idxs=None, is_root: bool = False, left=None,
                 impurity_reduction: float = None, tree_num: int = None,
                 right=None):
        """Node class for splitting
        """

        # split or linear
        self.is_root = is_root
        self.idxs = idxs
        self.tree_num = tree_num
        self.feature = feature
        self.impurity_reduction = impurity_reduction

        # different meanings
        self.value = value  # for split this is mean, for linear this is weight

        # split-specific
        self.threshold = threshold
        self.left = left
        self.right = right
        self.left_temp = None
        self.right_temp = None

    def setattrs(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __str__(self):
        if self.is_root:
            return f'X_{self.feature} <= {self.threshold:0.3f} (Tree #{self.tree_num} root)'
        elif self.left is None and self.right is None:
            return f'Val: {self.value[0][0]:0.3f} (leaf)'
        else:
            return f'X_{self.feature} <= {self.threshold:0.3f} (split)'
    
    def print_root(self, y):
        try:
            one_count = pd.Series(y).value_counts()[1.0]
        except KeyError:
            one_count = 0
        one_proportion = f' {one_count}/{y.shape[0]} ({round(100*one_count/y.shape[0], 2)}%)'

        if self.is_root:
            return f'X_{self.feature} <= {self.threshold:0.3f}' + one_proportion
        elif self.left is None and self.right is None:
            return f'ΔRisk = {self.value[0][0]:0.2f}' + one_proportion
        else:
            return f'X_{self.feature} <= {self.threshold:0.3f}' + one_proportion

    def __repr__(self):
        return self.__str__()


class FIGS(BaseEstimator):
    """FIGS (sum of trees) classifier.
    Fast Interpretable Greedy-Tree Sums (FIGS) is an algorithm for fitting concise rule-based models.
    Specifically, FIGS generalizes CART to simultaneously grow a flexible number of trees in a summation.
    The total number of splits across all the trees can be restricted by a pre-specified threshold, keeping the model interpretable.
    Experiments across real-world datasets show that FIGS achieves state-of-the-art prediction performance when restricted to just a few splits (e.g. less than 20).
    https://arxiv.org/abs/2201.11931
    """

    def __init__(self, max_rules: int = 12, min_impurity_decrease: float = 0.0):
        super().__init__()
        self.max_rules = max_rules
        self.min_impurity_decrease = min_impurity_decrease
        self._init_estimator_type()  # decides between regressor and classifier
        self._init_decision_function()

    def _init_estimator_type(self):
        """
        FIGSRegressor and FIGSClassifier override this method
        to alter the prediction task. When using this class directly,
        it is equivalent to FIGSRegressor
        """
        self._estimator_type = 'regressor'

    def _init_decision_function(self):
        """Sets decision function based on _estimator_type
        """
        # used by sklearn GrriidSearchCV, BaggingClassifier
        if self._estimator_type == 'classifier':
            decision_function = lambda x: self.predict_proba(x)[:, 1]
        elif self._estimator_type == 'regressor':
            decision_function = self.predict

    def _construct_node_with_stump(self, X, y, idxs, tree_num, sample_weight=None):
        # array indices
        SPLIT = 0
        LEFT = 1
        RIGHT = 2

        # fit stump
        stump = tree.DecisionTreeRegressor(max_depth=1)
        if sample_weight is not None:
            sample_weight = sample_weight[idxs]
        stump.fit(X[idxs], y[idxs], sample_weight=sample_weight)

        # these are all arrays, arr[0] is split node
        # note: -2 is dummy
        feature = stump.tree_.feature
        threshold = stump.tree_.threshold

        impurity = stump.tree_.impurity
        n_node_samples = stump.tree_.n_node_samples
        value = stump.tree_.value

        # no split
        if len(feature) == 1:
            # print('no split found!', idxs.sum(), impurity, feature)
            return Node(idxs=idxs, value=value[SPLIT], tree_num=tree_num,
                        feature=feature[SPLIT], threshold=threshold[SPLIT],
                        impurity_reduction=None)

        # split node
        impurity_reduction = (
                                     impurity[SPLIT] -
                                     impurity[LEFT] * n_node_samples[LEFT] / n_node_samples[SPLIT] -
                                     impurity[RIGHT] * n_node_samples[RIGHT] / n_node_samples[SPLIT]
                             ) * idxs.sum()

        node_split = Node(idxs=idxs, value=value[SPLIT], tree_num=tree_num,
                          feature=feature[SPLIT], threshold=threshold[SPLIT],
                          impurity_reduction=impurity_reduction)
        # print('\t>>>', node_split, 'impurity', impurity, 'num_pts', idxs.sum(), 'imp_reduc', impurity_reduction)

        # manage children
        idxs_split = X[:, feature[SPLIT]] <= threshold[SPLIT]
        idxs_left = idxs_split & idxs
        idxs_right = ~idxs_split & idxs
        node_left = Node(idxs=idxs_left, value=value[LEFT], tree_num=tree_num)
        node_right = Node(idxs=idxs_right, value=value[RIGHT], tree_num=tree_num)
        node_split.setattrs(left_temp=node_left, right_temp=node_right, )
        return node_split

    def fit(self, X, y=None, feature_names=None, verbose=False, sample_weight=None):
        """
        Params
        ------
        _sample_weight: array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted.
            Splits that would create child nodes with net zero or negative weight
            are ignored while searching for a split in each node.
        """
        X, y = check_X_y(X, y)
        y = y.astype(float)
        if feature_names is not None:
            self.feature_names_ = feature_names

        self.trees_ = []  # list of the root nodes of added trees
        self.complexity_ = 0  # tracks the number of rules in the model
        y_predictions_per_tree = {}  # predictions for each tree
        y_residuals_per_tree = {}  # based on predictions above

        # set up initial potential_splits
        # everything in potential_splits either is_root (so it can be added directly to self.trees_)
        # or it is a child of a root node that has already been added
        idxs = np.ones(X.shape[0], dtype=bool)
        node_init = self._construct_node_with_stump(X=X, y=y, idxs=idxs, tree_num=-1, sample_weight=sample_weight)
        potential_splits = [node_init]
        for node in potential_splits:
            node.setattrs(is_root=True)
        potential_splits = sorted(potential_splits, key=lambda x: x.impurity_reduction)

        # start the greedy fitting algorithm
        finished = False
        while len(potential_splits) > 0 and not finished:
            # print('potential_splits', [str(s) for s in potential_splits])
            split_node = potential_splits.pop()  # get node with max impurity_reduction (since it's sorted)

            # don't split on node
            if split_node.impurity_reduction < self.min_impurity_decrease:
                finished = True
                break

            # split on node
            if verbose:
                print('\nadding ' + str(split_node))
            self.complexity_ += 1

            # if added a tree root
            if split_node.is_root:

                # start a new tree
                self.trees_.append(split_node)

                # update tree_num
                for node_ in [split_node, split_node.left_temp, split_node.right_temp]:
                    if node_ is not None:
                        node_.tree_num = len(self.trees_) - 1

                # add new root potential node
                node_new_root = Node(is_root=True, idxs=np.ones(X.shape[0], dtype=bool),
                                     tree_num=-1)
                potential_splits.append(node_new_root)

            # add children to potential splits
            # assign left_temp, right_temp to be proper children
            # (basically adds them to tree in predict method)
            split_node.setattrs(left=split_node.left_temp, right=split_node.right_temp)

            # add children to potential_splits
            potential_splits.append(split_node.left)
            potential_splits.append(split_node.right)

            # update predictions for altered tree
            for tree_num_ in range(len(self.trees_)):
                y_predictions_per_tree[tree_num_] = self._predict_tree(self.trees_[tree_num_], X)
            y_predictions_per_tree[-1] = np.zeros(X.shape[0])  # dummy 0 preds for possible new trees

            # update residuals for each tree
            # -1 is key for potential new tree
            for tree_num_ in list(range(len(self.trees_))) + [-1]:
                y_residuals_per_tree[tree_num_] = deepcopy(y)

                # subtract predictions of all other trees
                for tree_num_other_ in range(len(self.trees_)):
                    if not tree_num_other_ == tree_num_:
                        y_residuals_per_tree[tree_num_] -= y_predictions_per_tree[tree_num_other_]

            # recompute all impurities + update potential_split children
            potential_splits_new = []
            for potential_split in potential_splits:
                y_target = y_residuals_per_tree[potential_split.tree_num]

                # re-calculate the best split
                potential_split_updated = self._construct_node_with_stump(X=X,
                                                                          y=y_target,
                                                                          idxs=potential_split.idxs,
                                                                          tree_num=potential_split.tree_num,
                                                                          sample_weight=sample_weight, )

                # need to preserve certain attributes from before (value at this split + is_root)
                # value may change because residuals may have changed, but we want it to store the value from before
                potential_split.setattrs(
                    feature=potential_split_updated.feature,
                    threshold=potential_split_updated.threshold,
                    impurity_reduction=potential_split_updated.impurity_reduction,
                    left_temp=potential_split_updated.left_temp,
                    right_temp=potential_split_updated.right_temp,
                )

                # this is a valid split
                if potential_split.impurity_reduction is not None:
                    potential_splits_new.append(potential_split)

            # sort so largest impurity reduction comes last (should probs make this a heap later)
            potential_splits = sorted(potential_splits_new, key=lambda x: x.impurity_reduction)
            if verbose:
                print(self)
            if self.max_rules is not None and self.complexity_ >= self.max_rules:
                finished = True
                break
        return self

    def _tree_to_str(self, root: Node, prefix=''):
        if root is None:
            return ''
        elif root.threshold is None:
            return ''
        pprefix = prefix + '\t'
        return prefix + str(root) + '\n' + self._tree_to_str(root.left, pprefix) + self._tree_to_str(root.right,
                                                                                                     pprefix)

    def _tree_to_str_with_data(self, X, y, root: Node, prefix=''):
        if root is None:
            return ''
        elif root.threshold is None:
            return ''
        pprefix = prefix + '\t'
        left = X[:, root.feature] <= root.threshold
        return (
            prefix + root.print_root(y) + '\n' + 
            self._tree_to_str_with_data(X[left], y[left], root.left, pprefix) + 
            self._tree_to_str_with_data(X[~left], y[~left], root.right, pprefix))

    def __str__(self):
        s = '------------\n' + '\n\t+\n'.join([self._tree_to_str(t) for t in self.trees_])
        if hasattr(self, 'feature_names_') and self.feature_names_ is not None:
            for i in range(len(self.feature_names_))[::-1]:
                s = s.replace(f'X_{i}', self.feature_names_[i])
        return s
    
    def print_tree(self, X, y):
        s = '------------\n' + '\n\t+\n'.join([self._tree_to_str_with_data(X, y, t) for t in self.trees_])
        if hasattr(self, 'feature_names_') and self.feature_names_ is not None:
            for i in range(len(self.feature_names_))[::-1]:
                s = s.replace(f'X_{i}', self.feature_names_[i])
        return s

    def predict(self, X):
        X = check_array(X)
        preds = np.zeros(X.shape[0])
        for tree in self.trees_:
            preds += self._predict_tree(tree, X)
        if self._estimator_type == 'regressor':
            return preds
        elif self._estimator_type == 'classifier':
            return (preds > 0.5).astype(int)

    def predict_proba(self, X):
        X = check_array(X)
        if self._estimator_type == 'regressor':
            return NotImplemented
        preds = np.zeros(X.shape[0])
        for tree in self.trees_:
            preds += self._predict_tree(tree, X)
        preds = np.clip(preds, a_min=0., a_max=1.)  # constrain to range of probabilities
        return np.vstack((1 - preds, preds)).transpose()

    def _predict_tree(self, root: Node, X):
        """Predict for a single tree
        """

        def _predict_tree_single_point(root: Node, x):
            if root.left is None and root.right is None:
                return root.value
            left = x[root.feature] <= root.threshold
            if left:
                if root.left is None:  # we don't actually have to worry about this case
                    return root.value
                else:
                    return _predict_tree_single_point(root.left, x)
            else:
                if root.right is None:  # we don't actually have to worry about this case
                    return root.value
                else:
                    return _predict_tree_single_point(root.right, x)

        preds = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            preds[i] = _predict_tree_single_point(root, X[i])
        return preds

    def plot(self, cols=2, feature_names=None, filename=None, label="all", impurity=False, tree_number=None):
        is_single_tree =  len(self.trees_) < 2 or tree_number is not None
        n_cols = int(cols)
        n_rows = int(np.ceil(len(self.trees_) / n_cols))
        # if is_single_tree:
        #     fig, ax = plt.subplots(1)
        # else:
        #     fig, axs = plt.subplots(n_rows, n_cols)
        n_plots = int(len(self.trees_)) if tree_number is None else 1
        fig, axs = plt.subplots(n_plots)
        criterion = "squared_error" if self._estimator_type == "regressor" else "gini"
        n_classes = 1 if self._estimator_type == 'regressor' else 2
        ax_size = int(len(self.trees_))#n_cols * n_rows
        for i in range(n_plots):
            r = i // n_cols
            c = i % n_cols
            if not is_single_tree:
                # ax = axs[r, c]
                ax = axs[i]
            else:
                ax = axs
            try:
                tree = self.trees_[i] if tree_number is None else self.trees_[tree_number]
                plot_tree(DecisionTreeViz(tree, criterion, n_classes), ax=ax, feature_names=feature_names, label=label,
                          impurity=impurity)
            except IndexError:
                ax.axis('off')
                continue

            ax.set_title(f"Tree {i}")
        if filename is not None:
            plt.savefig(filename)
            return
        plt.show()


class FIGSRegressor(FIGS, RegressorMixin):
    def _init_estimator_type(self):
        self._estimator_type = 'regressor'


class FIGSClassifier(FIGS, ClassifierMixin):
    def _init_estimator_type(self):
        self._estimator_type = 'classifier'


class FIGSCV:
    def __init__(self, figs,
                 n_rules_list: List[float] = [6, 12, 24, 30, 50],
                 cv: int = 3, scoring=None, *args, **kwargs):

        self._figs_class = figs
        self.n_rules_list = np.array(n_rules_list)
        self.cv = cv
        self.scoring = scoring

    def fit(self, X, y):
        self.scores_ = []
        for n_rules in self.n_rules_list:
            est = self._figs_class(max_rules=n_rules)
            cv_scores = cross_val_score(est, X, y, cv=self.cv, scoring=self.scoring)
            mean_score = np.mean(cv_scores)
            if len(self.scores_) == 0:
                self.figs = est
            elif mean_score > np.max(self.scores_):
                self.figs = est

            self.scores_.append(mean_score)
        self.figs.fit(X=X, y=y)

    def predict_proba(self, X):
        return self.figs.predict_proba(X)

    def predict(self, X):
        return self.figs.predict(X)

    @property
    def max_rules(self):
        return self.figs.max_rules


class FIGSRegressorCV(FIGSCV):
    def __init__(self,
                 n_rules_list: List[int] = [6, 12, 24, 30, 50],
                 cv: int = 3, scoring='r2', *args, **kwargs):
        super(FIGSRegressorCV, self).__init__(figs=FIGSRegressor, n_rules_list=n_rules_list,
                                              cv=cv, scoring=scoring, *args, **kwargs)


class FIGSClassifierCV(FIGSCV):
    def __init__(self,
                 n_rules_list: List[int] = [6, 12, 24, 30, 50],
                 cv: int = 3, scoring="accuracy", *args, **kwargs):
        super(FIGSClassifierCV, self).__init__(figs=FIGSClassifier, n_rules_list=n_rules_list,
                                               cv=cv, scoring=scoring, *args, **kwargs)


if __name__ == '__main__':
    from sklearn import datasets

    X_cls, Y_cls = datasets.load_breast_cancer(return_X_y=True)
    X_reg, Y_reg = datasets.make_friedman1(100)

    est = FIGSRegressorCV()
    est.fit(X_reg, Y_reg)
    est.predict(X_reg)
    print(est.max_rules)
    est.figs.plot(tree_number=0)

    est = FIGSClassifierCV()
    est.fit(X_cls, Y_cls)
    est.predict(X_cls)
    print(est.max_rules)
    est.figs.plot(tree_number=0)
