from collections import namedtuple

import numpy as np
from sklearn import datasets
from sklearn import tree
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split


class Node:
    def __init__(self, feature: int = None, threshold: int = None, value=None, is_root: bool = False, left=None,
                 right=None):
        self.feature = feature
        self.threshold = threshold
        self.is_root = is_root
        self.left = left
        self.right = right
        self.value = value


class SuperCART(BaseEstimator):

    def __init__(self):
        self.prediction_task = 'regression'
        super().__init__()

    def fit(self, X, y=None, feature_names=None, impurity_dec_thresh=10):

        def construct_node_from_stump(stump):
            # these are all arrays, arr[0] is split node
            SPLIT = 0
            LEFT = 1
            RIGHT = 2
            n_nodes = stump.tree_.node_count
            # children_left = clf.tree_.children_left
            # children_right = clf.tree_.children_right
            feature = stump.tree_.feature
            threshold = stump.tree_.threshold
            impurity = stump.tree_.impurity
            n_node_samples = stump.tree_.n_node_samples
            value = stump.tree_.value
            print('impurity', impurity, 'n_node_samples', n_node_samples, 'value', value)
            impurity_reduction = impurity[SPLIT] - (
                    impurity[LEFT] * n_node_samples[LEFT] + impurity[RIGHT] * n_node_samples[RIGHT]) / n_nodes
            # print(children_left)
            return [Node(feature=feature[x], threshold=threshold[x], value=value[x]) for x in [SPLIT, LEFT, RIGHT]] + [
                impurity_reduction]

        Split = namedtuple('Split', 'impurity_reduction node idxs tree_num')
        if self.prediction_task == 'regression':
            stump = tree.DecisionTreeRegressor(max_depth=1)
        else:
            stump = tree.DecisionTreeClassifier(max_depth=1)
        stump.fit(X, y)
        node_split, node_left, node_right, impurity_reduction = construct_node_from_stump(stump)
        node_split.is_root = True

        potential_splits = [
            Split(impurity_reduction, node_split, idxs=np.ones(X.shape[0], dtype=bool), tree_num=0)
        ]  # should eventually make this a heap, for now just sort so largest impurity reduction comes last

        trees = []
        y_predictions_per_tree = {}
        y_residuals_per_tree = {}  # based on predictions above
        total_num_rules = 0
        while len(potential_splits) > 0:
            impurity_reduction, node, idxs, tree_num = potential_splits.pop()
            if impurity_reduction <= impurity_dec_thresh:
                total_num_rules += 1

                # add children to potential_splits w/ appropriate idxs
                idxs_split = X[:, node.feature] <= node.threshold
                idxs_left = idxs_split & idxs
                idxs_right = ~idxs_split & idxs
                # node_left = Node()
                # node_right = Node()
                node.left = node_left
                node.right = node_right
                potential_splits.append(Split(None, node_left, idxs_left, tree_num))
                potential_splits.append(Split(None, node_right, idxs_right, tree_num))

                # if added root
                if node.is_root:
                    trees.append(node)  # add to trees

                    # add new root potential node
                    node = Node()
                    node.is_root = True
                    potential_splits.append(
                        Split(None, node, idxs=np.ones(X.shape[0], dtype=bool), tree_num=len(trees))
                    )

                # update predictions for altered tree
                for tree_num_ in range(len(trees)):
                    """
                    predictor, n_rules = self.root_to_sklearn_tree(trees[0])
                    import matplotlib.pyplot as plt
                    fig = plt.figure(figsize=(25, 20))
                    _ = tree.plot_tree(predictor, filled=True)
                    """

                    y_predictions_per_tree[tree_num_] = self.predict_tree(trees[tree_num], X)

                # debugging
                if total_num_rules == 1:
                    assert np.array_equal(y_predictions_per_tree[0],
                                          stump.predict(X)), 'For one rule, prediction should match stump'
                    print(stump)

                # recompute all impurities
                potential_splits.sort(
                    key=lambda x: x.impurity_reduction
                )  # sort so largest impurity reduction comes last



            else:
                break

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
            if root.left is None:
                return root.value
            else:
                return self.predict_tree_single_point(root.left, x)
        else:
            if root.right is None:
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
