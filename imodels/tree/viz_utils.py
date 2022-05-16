import numpy as np

from collections import namedtuple

from sklearn.tree import BaseDecisionTree

TreeData = namedtuple('TreeData', "children_left children_right "
                                  "feature threshold n_node_samples impurity value n_classes n_outputs")


def extract_figs_tree(node, n_classes):
    tree_data = TreeData(
        children_left=[],
        children_right=[],
        feature=[],
        threshold=[],
        n_node_samples=[],
        impurity=[],
        value=[],
        n_classes=np.array([n_classes]),
        n_outputs=np.array([1]))

    node_counter = iter(range(1, int(1e06)))

    def _update_node(nd):
        if nd is None:
            return
        has_children = nd.right is not None
        left = right = -1
        feature = threshold = -2
        value = np.expand_dims(np.array([0]), axis=-1) if nd.value is None else nd.value
        impurity_reduction = 0 if nd.impurity_reduction is None else nd.impurity_reduction
        if has_children:
            right = next(node_counter)
            left = next(node_counter)
            feature = nd.feature
            threshold = nd.threshold

        tree_data.children_left.append(left)
        tree_data.children_right.append(right)
        tree_data.feature.append(feature)
        tree_data.threshold.append(threshold)
        tree_data.n_node_samples.append(np.sum(nd.idxs))
        tree_data.impurity.append(impurity_reduction)
        tree_data.value.append(np.array(value))

        _update_node(nd.right)
        _update_node(nd.left)

    _update_node(node)
    return tree_data


class LightTreeViz:
    def __init__(self, figs_tree, n_classes):
        tree = extract_figs_tree(figs_tree, n_classes)
        self.children_left = tree.children_left
        self.children_right = tree.children_right
        self.feature = tree.feature
        self.threshold = tree.threshold
        self.n_node_samples = tree.n_node_samples
        self.impurity = tree.impurity
        self.value = tree.value
        self.n_classes = tree.n_classes
        self.n_outputs = tree.n_outputs


class DecisionTreeViz(BaseDecisionTree):
    def __init__(self, dt, criterion, n_classes):

        tree = LightTreeViz(dt, n_classes)
        self.tree_ = tree
        self.criterion = criterion