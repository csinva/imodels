from copy import deepcopy

import numpy as np
from sklearn import datasets
from sklearn import tree
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split


class Node:
    def __init__(self, feature: int = None, threshold: int = None,
                 value=None, idxs=None, is_root: bool = False, left=None,
                 impurity_reduction: float = None, tree_num: int = None,
                 right=None):
        self.feature = feature
        self.threshold = threshold
        self.is_root = is_root
        self.idxs = idxs
        self.left = left
        self.right = right
        self.value = value
        self.impurity_reduction = impurity_reduction
        self.tree_num = tree_num
        self.left_temp = None
        self.right_temp = None

    def setattrs(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __str__(self):
        node_type = 'split'
        if self.is_root:
            node_type = 'root'
            return f'X_{self.feature} <= {self.threshold:0.3f} (Tree #{self.tree_num} {node_type})'
        elif self.left is None and self.right is None:
            node_type = 'leaf'
            return f'Val: {self.value[0][0]:0.3f} ({node_type})'
        else:
            return f'X_{self.feature} <= {self.threshold:0.3f} ({node_type})'

    def __repr__(self):
        return self.__str__()


class SuperCART(BaseEstimator):

    def __init__(self):
        super().__init__()
        self._init_prediction_task()  # decides between regressor and classifier

    def _init_prediction_task(self):
        """
        SuperCARTRegressor and SuperCARTClassifier override this method
        to alter the prediction task. When using this class directly,
        it is equivalent to SuperCARTRegressor
        """
        self.prediction_task = 'regression'

    def construct_node_from_stump(self, stump, idxs, X, tree_num):
        # array indices
        SPLIT = 0
        LEFT = 1
        RIGHT = 2

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

    def fit_stump(self, X, y, idxs):
        """

        Parameters
        ----------
        X_
            probably the same as X
        y_
            might change if we are predicting residuals
        idxs
            indexes of subset to fit to
        """
        stump = tree.DecisionTreeRegressor(max_depth=1)
        return stump.fit(X[idxs], y[idxs])

    def fit(self, X, y=None, feature_names=None, min_impurity_decrease=0.0, verbose=True, max_rules=5):

        y = y.astype(float)

        idxs = np.ones(X.shape[0], dtype=bool)
        stump = self.fit_stump(X, y, idxs)
        node_init = self.construct_node_from_stump(stump, idxs=idxs, X=X, tree_num=0)
        node_init.setattrs(is_root=True)

        # should eventually make this a heap, for now just sort so largest impurity reduction comes last
        potential_splits = [node_init]

        self.trees_ = []
        y_predictions_per_tree = {}
        y_residuals_per_tree = {}  # based on predictions above
        self.complexity_ = 0 # tracks the number of rules in the model
        while len(potential_splits) > 0:
            # print('potential_splits', [str(s) for s in potential_splits])
            split_node = potential_splits.pop()  # get node with max impurity_reduction (since it's sorted)

            # don't split on node
            if split_node.impurity_reduction < min_impurity_decrease:
                return self

            # split on node
            if verbose:
                print('\nadding ' + str(split_node))
            self.complexity_ += 1

            # assign left_temp, right_temp to be proper children
            # (basically adds them to tree in predict method)
            split_node.setattrs(left=split_node.left_temp, right=split_node.right_temp)

            # add children to potential_splits
            potential_splits.append(split_node.left)
            potential_splits.append(split_node.right)

            # if added a tree root
            if split_node.is_root:
                self.trees_.append(split_node)  # start a new tree

                # add new root potential node
                node_new_root = Node(is_root=True, idxs=np.ones(X.shape[0], dtype=bool), tree_num=len(self.trees_))
                potential_splits.append(node_new_root)

            # update predictions for altered tree
            for tree_num_ in range(len(self.trees_)):
                y_predictions_per_tree[tree_num_] = self.predict_tree(self.trees_[tree_num_], X)
            y_predictions_per_tree[len(self.trees_)] = np.zeros(X.shape[0])  # dummy 0 preds for possible new tree

            # update residuals for each tree
            for tree_num_ in range(len(self.trees_) + 1):
                y_residuals_per_tree[tree_num_] = deepcopy(y)

                # subtract predictions of all other trees
                for tree_num_2_ in range(len(self.trees_) + 1):
                    if not tree_num_2_ == tree_num_:
                        y_residuals_per_tree[tree_num_] -= y_predictions_per_tree[tree_num_2_]

            # debugging
            if self.complexity_ == 1:
                assert np.array_equal(y_predictions_per_tree[0], stump.predict(X)), \
                    'For one rule, prediction should match stump'
                assert np.array_equal(y_residuals_per_tree[0], y), \
                    'For one rule, residual should match y since there are no other trees'

            # recompute all impurities + update potential_split children
            potential_splits_new = []
            for potential_split in potential_splits:
                y_target = y_residuals_per_tree[potential_split.tree_num]
                stump = self.fit_stump(X, y_target, idxs=potential_split.idxs)
                potential_split_updated = self.construct_node_from_stump(stump,
                                                                         idxs=potential_split.idxs,
                                                                         X=X,
                                                                         tree_num=potential_split.tree_num)

                # need to preserve certain attributes from before (value + is_root)
                # value may change because we are predicting something different (e.g. residuals)
                potential_split.setattrs(
                    feature=potential_split_updated.feature,
                    threshold=potential_split_updated.threshold,
                    left_temp=potential_split_updated.left_temp,
                    right_temp=potential_split_updated.right_temp,
                    impurity_reduction=potential_split_updated.impurity_reduction,
                )

                if potential_split.impurity_reduction is not None:  # there was a split found
                    potential_splits_new.append(potential_split)

            # sort so largest impurity reduction comes last
            potential_splits = sorted(potential_splits_new, key=lambda x: x.impurity_reduction)
            if verbose:
                print(self)

            if self.complexity_ >= max_rules:
                return self
        return self

    def tree_to_str(self, root: Node, prefix=''):
        if root is None or root.threshold is None:
            return ''
        pprefix = prefix + '\t'
        return prefix + str(root) + '\n' + self.tree_to_str(root.left, pprefix) + self.tree_to_str(root.right, pprefix)

    def __str__(self):
        return '------------\n' + '\n\t+\n'.join([self.tree_to_str(t) for t in self.trees_])

    def predict(self, X):
        preds = np.zeros(X.shape[0])
        for tree in self.trees_:
            preds += self.predict_tree(tree, X)
        if self.prediction_task == 'regression':
            return preds
        elif self.prediction_task == 'classification':
            return (preds > 0.5).astype(int)

    def predict_proba(self, X):
        if self.prediction_task == 'regression':
            return NotImplemented
        else:
            preds = np.zeros(X.shape[0])
            for tree in self.trees_:
                preds += self.predict_tree(tree, X)
            return np.vstack((1 - preds, preds)).transpose()

    def predict_tree(self, root: Node, X):
        """This can be made way faster
        """
        preds = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            preds[i] = self.predict_tree_single_point(root, X[i])
        return preds

    def predict_tree_single_point(self, root: Node, x):
        if root.left is None and root.right is None:
            return root.value
        left = x[root.feature] <= root.threshold
        if left:
            if root.left is None:  # we don't actually have to worry about this case
                return root.value
            else:
                return self.predict_tree_single_point(root.left, x)
        else:
            if root.right is None:  # we don't actually have to worry about this case
                return root.value
            else:
                return self.predict_tree_single_point(root.right, x)


class SuperCARTRegressor(SuperCART):
    def _init_prediction_task(self):
        self.prediction_task = 'regression'


class SuperCARTClassifier(SuperCART):
    def _init_prediction_task(self):
        self.prediction_task = 'classification'


if __name__ == '__main__':
    np.random.seed(13)
    X, y = datasets.load_breast_cancer(return_X_y=True)  # binary classification
    # X, y = datasets.load_diabetes(return_X_y=True)  # regression
    # X = np.random.randn(500, 10)
    # y = (X[:, 0] > 0).astype(float) + (X[:, 1] > 1).astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )
    print('X.shape', X.shape)
    print('ys', np.unique(y_train), '\n\n')

    m = SuperCARTClassifier()
    m.fit(X_train, y_train)
    print(m.predict_proba(X_train))
