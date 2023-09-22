from sklearn.tree._tree import Tree
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn import __version__
from collections import namedtuple
import pandas as pd
import numpy as np

from collections import namedtuple

from sklearn import __version__
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.tree._tree import Tree
import imodels.util.tree


class Node:
    def __init__(self, impurity, num_samples, num_samples_per_class, predicted_probs):
        self.impurity = impurity
        self.num_samples = num_samples
        self.num_samples_per_class = num_samples_per_class
        self.predicted_probs = predicted_probs
        self.feature_index = 0
        self.threshold = 0
        self.left = None
        self.right = None


class CustomDecisionTreeClassifier(ClassifierMixin):
    def __init__(self, max_leaf_nodes=None, impurity_func='gini'):
        self.root = None
        self.max_leaf_nodes = max_leaf_nodes
        self.impurity_func = impurity_func

    def fit(self, X, y, feature_costs=None):
        self.n_classes_ = len(set(y))
        self.n_features = X.shape[1]
        self.feature_costs_ = imodels.util.tree._validate_feature_costs(
            feature_costs, self.n_features)
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y):
        stack = []
        num_samples_per_class = np.array([np.sum(y == i)
                                          for i in range(self.n_classes_)])
        root = Node(
            impurity=self._calc_impurity(y),
            num_samples=y.size,
            num_samples_per_class=num_samples_per_class,
            predicted_probs=num_samples_per_class / y.size,
        )
        root.impurity_reduction = self._best_split(X, y)[-1]

        stack.append((root, X, y))
        self.n_splits = 0
        while stack:
            node, X_node, y_node = stack.pop()
            idx, thr, _ = self._best_split(X_node, y_node)
            if idx is not None:
                self.n_splits += 1
                indices_left = X_node[:, idx] < thr
                X_left, y_left = X_node[indices_left], y_node[indices_left]
                X_right, y_right = X_node[~indices_left], y_node[~indices_left]
                node.feature_index = idx
                node.threshold = thr
                num_samples_per_class_left = np.array([
                    np.sum(y_left == i) for i in range(self.n_classes_)])
                node.left = Node(
                    impurity=self._calc_impurity(y_left),
                    num_samples=y_left.size,
                    num_samples_per_class=num_samples_per_class_left,
                    predicted_probs=num_samples_per_class_left / y_left.size,
                )
                # some redundant calculation going on here, but it's okay....
                node.left.impurity_reduction = self._best_split(
                    X_left, y_left)[-1]

                num_samples_per_class_right = np.array([
                    np.sum(y_right == i) for i in range(self.n_classes_)])
                node.right = Node(
                    impurity=self._calc_impurity(y_right),
                    num_samples=y_right.size,
                    num_samples_per_class=num_samples_per_class_right,
                    predicted_probs=num_samples_per_class_right / y_right.size,
                )
                node.right.impurity_reduction = self._best_split(
                    X_right, y_right)[-1]
                stack.append((node.right, X_right, y_right))
                stack.append((node.left, X_left, y_left))

            # early stop
            if self.max_leaf_nodes and self.n_splits >= self.max_leaf_nodes - 1:
                return root

            # sort stack by impurity_reduction
            stack = sorted(
                stack, key=lambda x: x[0].impurity_reduction, reverse=True)

        return root

    def _best_split(self, X, y):
        n = y.size
        if n <= 1:
            return None, None, 0

        orig_impurity = self._gini(y)
        impurity_reduction = 0
        best_impurity_reduction = 0
        best_idx, best_thr = None, None

        # loop over features
        for idx in range(self.n_features):
            thresholds, y_classes = zip(*sorted(zip(X[:, idx], y)))
            y_classes = np.array(y_classes)

            # consider every point where threshold value changes
            idx_thresholds = (1 + np.where(np.diff(thresholds))[0]).tolist()
            for i in idx_thresholds:

                # calculate impurity for left and right
                y_left = y_classes[:i]
                y_right = y_classes[i:]
                impurity_reduction = orig_impurity - (y_left.size * self._gini(y_left) +
                                                      y_right.size * self._gini(y_right)) / n

                if self.impurity_func == 'information_gain_ratio':
                    split_info = - (y_left.size / n * np.log2(y_left.size / n)) - (
                        y_right.size / n * np.log2(y_right.size / n))
                    if ~np.isnan(split_info):
                        impurity_reduction = impurity_reduction / split_info
                if self.impurity_func == 'cost_information_gain_ratio':
                    impurity_reduction /= self.feature_costs_[idx]

                if impurity_reduction > best_impurity_reduction:
                    best_impurity_reduction = impurity_reduction
                    best_idx = idx
                    best_thr = (thresholds[i] + thresholds[i - 1]) / 2

        return best_idx, best_thr, impurity_reduction

    def _calc_impurity(self, y):
        if self.impurity_func == 'gini':
            return self._gini(y)
        elif self.impurity_func in ['entropy', 'information_gain_ratio', 'cost_information_gain_ratio']:
            return self._entropy(y)

    def _gini(self, y):
        n = y.size
        return 1.0 - sum((np.sum(y == c) / n) ** 2 for c in range(self.n_classes_))

    def _entropy(self, y):
        n = y.size
        return -sum((np.sum(y == c) / n) * np.log2(np.sum(y == c) / n) for c in range(self.n_classes_))

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def predict_proba(self, X):
        return np.array([self._predict_single_proba(x) for x in X])

    def _predict_single_proba(self, x):
        node = self.root
        while node.left or node.right:
            if x[node.feature_index] < node.threshold and node.left:
                node = node.left
            elif node.right:
                node = node.right
        return node.predicted_probs


if __name__ == '__main__':
    from sklearn.datasets import load_breast_cancer, load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    # data = load_breast_cancer()
    data = load_iris()
    X = data.data
    y = data.target
    # print(np.unique(y))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=42, test_size=0.5)
    m = CustomDecisionTreeClassifier(
        # max_leaf_nodes=20,
        impurity_func='cost_information_gain_ratio')
    m.fit(X_train, y_train)
    y_pred = m.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print('n_nodes', m.n_splits + 1)
    print('shapes', m.predict_proba(X_test).shape, m.predict(X_test).shape)

    cost = imodels.util.tree.calculate_mean_depth_of_points_in_custom_tree(m)
    print('Cost', cost)
