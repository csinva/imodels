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

    def setattrs(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __str__(self):
        node_type = ''
        if self.is_root:
            node_type = 'root'
        elif self.left is None and self.right is None:
            node_type = 'leaf'
        return f'{self.feature} <= {self.threshold:0.3f} (Tree #{self.tree_num} {node_type})'


class SuperCART(BaseEstimator):

    def __init__(self):
        self.prediction_task = 'regression'
        super().__init__()

    def fit(self, X, y=None, feature_names=None, impurity_dec_thresh=10, verbose=True):

        def fit_stump(X_, y_, idxs):
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
            if self.prediction_task == 'regression':
                stump = tree.DecisionTreeRegressor(max_depth=1)
            else:
                stump = tree.DecisionTreeClassifier(max_depth=1)
            return stump.fit(X_[idxs], y_[idxs])

        def construct_node_from_stump(stump, idxs, tree_num):
            # array indices
            SPLIT = 0
            LEFT = 1
            RIGHT = 2

            # these are all arrays, arr[0] is split node
            feature = stump.tree_.feature
            threshold = stump.tree_.threshold
            impurity = stump.tree_.impurity
            n_node_samples = stump.tree_.n_node_samples
            value = stump.tree_.value
            # print('impurity', impurity, 'n_node_samples', n_node_samples, 'value', value)
            impurity_reduction = impurity[SPLIT] - (
                    impurity[LEFT] * n_node_samples[LEFT] + impurity[RIGHT] * n_node_samples[RIGHT]) / n_node_samples[
                                     SPLIT]
            # manage children
            idxs_split = X[:, feature[SPLIT]] <= threshold[SPLIT]
            idxs_left = idxs_split & idxs
            idxs_right = ~idxs_split & idxs
            node_left = Node(idxs=idxs_left, value=value[LEFT], tree_num=tree_num)
            node_right = Node(idxs=idxs_right, value=value[RIGHT], tree_num=tree_num)
            node_split = Node(idxs=idxs, value=value[SPLIT], tree_num=tree_num,
                              feature=feature[SPLIT], threshold=threshold[SPLIT],
                              impurity_reduction=impurity_reduction)
            node_split.setattrs(left_temp=node_left, right_temp=node_right,)
            return node_split

        idxs = np.ones(X.shape[0], dtype=bool)
        stump = fit_stump(X, y, idxs)
        node_init = construct_node_from_stump(stump, idxs=idxs, tree_num=0)
        node_init.setattrs(is_root=True)

        # should eventually make this a heap, for now just sort so largest impurity reduction comes last
        potential_splits = [node_init]

        trees = []
        y_predictions_per_tree = {}
        y_residuals_per_tree = {}  # based on predictions above
        total_num_rules = 0
        while len(potential_splits) > 0:
            split_node = potential_splits.pop()  # get node with max impurity_reduction (since it's sorted)
            print([str(s) for s in potential_splits])

            # don't split on node
            if split_node.impurity_reduction < impurity_dec_thresh:
                break

            # split on node
            if verbose:
                print('adding ' + str(split_node))
            total_num_rules += 1

            # assign left_temp, right_temp to be proper children
            # (basically adds them to tree in predict method)
            split_node.setattrs(left=split_node.left_temp, right=split_node.right_temp)
            split_node.setattrs(left_temp=None, right_temp=None)  # clean up some memory

            # add children to potential_splits
            potential_splits.append(split_node.left)
            potential_splits.append(split_node.right)

            # if added a tree root
            if split_node.is_root:
                trees.append(split_node)  # add to trees

                # add new root potential node
                node_new_root = Node(is_root=True, idxs=np.ones(X.shape[0], dtype=bool), tree_num=len(trees))
                potential_splits.append(node_new_root)

            # update predictions for altered tree
            for tree_num_ in range(len(trees)):
                """
                predictor, n_rules = self.root_to_sklearn_tree(trees[tree_num])
                y_predictions_per_tree[tree_num_] = predictor.predict(X)
                """
                y_predictions_per_tree[tree_num_] = self.predict_tree(trees[tree_num_], X)
            y_predictions_per_tree[len(trees)] = np.zeros(X.shape[0])  # dummy 0 preds for possible new tree

            # update residuals for each tree
            for tree_num_ in range(len(trees) + 1):
                y_residual = y
                for tree_num_2_ in range(len(trees) + 1):
                    if not tree_num_2_ == tree_num_:
                        y_residual -= y_predictions_per_tree[tree_num_2_]
                y_residuals_per_tree[tree_num_] = y_residual

            # debugging
            if total_num_rules == 1:
                assert np.array_equal(y_predictions_per_tree[0],
                                      stump.predict(X)), 'For one rule, prediction should match stump'
                assert np.array_equal(y_residuals_per_tree[0],
                                      y), 'For one rule, residual should match y since there are no other trees'
                print('passed basic rule1 checks!')

            # recompute all impurities + update potential_split children
            potential_splits_new = []
            for potential_split in potential_splits:
                y_target = y_residuals_per_tree[potential_split.tree_num]
                stump = fit_stump(X, y_=y_target, idxs=potential_split.idxs)
                potential_split_updated = construct_node_from_stump(stump, idxs=idxs, tree_num=0)

                # need to preserve certain attributes from before
                # value may change because we are predicting something different (e.g. residuals)
                potential_split_updated.setattrs(
                    value=potential_split.value,
                    is_root=potential_split.is_root,
                )
                potential_splits_new.append(potential_split_updated)

            # sort so largest impurity reduction comes last
            potential_splits = sorted(potential_splits_new, key=lambda x: x.impurity_reduction)

        self.trees_ = trees
        return self

    def predict(self, X):
        return self.clf.predict(X)

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


'''
def root_to_sklearn_tree(self, root):
    """Try converting to sklearn format
    """
    # start with a deep tree
    if self.prediction_task == 'regression':
        stump = tree.DecisionTreeRegressor(max_depth=100).fit(X, y)
    else:
        stump = tree.DecisionTreeClassifier().fit(X[:1], y[:1])

    # traverse tree and convert to sklearn format
    feature = []
    threshold = []
    impurity = []
    # n_node_samples = []
    value = []
    q = [root]
    node_index = 0
    while len(q) > 0:
        node = q.pop(0)  # pop off front
        feature.append(node.feature)
        threshold.append(node.threshold)
        # t.impurity.append(node.impurity)
        value.append(node.value)
        if node.left is not None:
            q.append(node.left)
        if node.right is not None:
            q.append(node.right)

    t = stump.tree_
    assert len(t.feature) >= len(feature), 'dummy tree must be at least as big as new tree'
    for i in range(len(feature)):
        t.feature[i] = feature[i]
        t.threshold[i] = threshold[i]
        t.value[i] = value[i]
    for i in range(len(feature), len(t.feature)):
        t.children_left[i] = -1
        t.children_right[i] = -1
    t.node_count = len(feature)

    return stump, t.node_count
'''

if __name__ == '__main__':
    # X, y = datasets.load_breast_cancer(return_X_y=True) # binary classification
    X, y = datasets.load_diabetes(return_X_y=True)  # regression

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )
    print('X.shape', X.shape)
    print('ys', np.unique(y_train))

    m = SuperCART()
    m.fit(X_train, y_train)
