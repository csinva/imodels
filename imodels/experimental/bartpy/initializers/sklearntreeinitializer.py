from typing import Tuple, List
from operator import gt, le

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from imodels.tree.figs import FIGSRegressor, Node, FIGSRegressorCV
from sklearn.exceptions import NotFittedError
from sklearn.tree import DecisionTreeRegressor

from ..data import make_bartpy_data
from ..initializers.initializer import Initializer
from ..mutation import GrowMutation
from ..node import split_node, LeafNode
from ..split import Split
from ..splitcondition import SplitCondition
from ..tree import Tree, mutate


class SklearnTreeInitializer(Initializer):
    """
    Initialize tree structure and leaf node values by fitting a single Sklearn GBR tree

    Both tree structure and leaf node parameters are copied across
    """

    def __init__(self,
                 max_depth: int = 4,
                 min_samples_split: int = 2,
                 loss: str = 'squared_error',
                 tree_=None):
        super().__init__()
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.loss = loss
        self._tree = tree_

    def initialize_tree(self,
                        tree: Tree, tree_number: int) -> None:
        # # trees = list(trees)
        # if not self._tree:
        #     params = {
        #         'n_estimators': 1,
        #         'max_depth': self.max_depth,
        #         'min_samples_split': self.min_samples_split,
        #         'learning_rate': 0.8,
        #         'loss': self.loss
        #     }
        #     # for tree in trees:
        #
        #     self._tree = GradientBoostingRegressor(**params)
        #     # clf = FIGSRegressor()
        #
        #     self._tree.fit(tree.nodes[0].data.X.values, tree.nodes[0].data.y.values)

        # for i, tree in enumerate(trees):
        # sklearn_tree = self._tree.estimators_[0][0].tree_
        sklearn_tree = self._get_sklearn_tree(tree_number)
        map_sklearn_tree_into_bartpy(tree, sklearn_tree)
        pass

    def _get_sklearn_tree(self, tree_number):
        if isinstance(self._tree, GradientBoostingRegressor):
            return self._tree.estimators_[0][tree_number].tree_
        elif isinstance(self._tree, DecisionTreeRegressor):
            return self._tree.tree_
        elif isinstance(self._tree, FIGSRegressor):
            return SkTree(self._tree.trees_[tree_number])
        elif isinstance(self._tree, FIGSRegressorCV):
            return SkTree(self._tree.figs.trees_[tree_number])
        elif isinstance(self._tree, RandomForestRegressor):
            return self._tree.estimators_[tree_number]


def get_child(node, direction):
    if hasattr(node, direction):
        return getattr(node, direction)
    elif hasattr(node, f"{direction}_child"):
        return getattr(node, f"{direction}_child")
    return


def enumarate_tree(tree: Node, num_iter=iter(range(int(1e+06)))):
    if tree is None:
        return
    tree.number = next(num_iter)
    # if hasattr(tree, 'left'):
    enumarate_tree(get_child(tree, 'left'), num_iter)
    # if hasattr(tree, 'right'):
    enumarate_tree(get_child(tree, 'right'), num_iter)


def fill_nodes_dict(tree: Node, node_dict: dict):
    if tree is None:
        return
    node_dict[tree.number] = tree
    fill_nodes_dict(get_child(tree, 'left'), node_dict)
    fill_nodes_dict(get_child(tree, 'right'), node_dict)

    # return node_dict


class SkTree:
    def __init__(self, tree: Node):
        nodes_dict = {}
        enumarate_tree(tree, num_iter=iter(range(int(1e+06))))
        fill_nodes_dict(tree, nodes_dict)
        self.children_left = []
        self.children_right = []
        self.feature = []
        self.threshold = []
        self.n_node_samples = []
        self.impurity = []
        self.value = []
        for node in nodes_dict.values():
            r_c = get_child(node, "right")
            l_c = get_child(node, "left")
            right_number = r_c.number if r_c else -1
            left_number = l_c.number if l_c else -1
            f_node = node.feature if l_c else -2
            t_node = node.threshold if l_c else -2

            self.children_right.append(right_number)
            self.children_left.append(left_number)
            self.feature.append(f_node)
            self.threshold.append(t_node)
            self.n_node_samples.append(np.sum(node.idxs))
            self.impurity.append(node.impurity_reduction)
            self.value.append(node.value)
    #
    # def predict(self, *args, **kwargs):
    #     """ Predict target for X. """
    #     self._figs.predict(*args, **kwargs)


class SkFigs:
    def __init__(self, figs: FIGSRegressor):
        self.trees_ = [SkTree(t) for t in figs.trees_]


def get_figs_sklearn(figs: FIGSRegressor):
    return SkFigs(figs)


def map_sklearn_split_into_bartpy_split_conditions(sklearn_tree, index: int) -> Tuple[SplitCondition, SplitCondition]:
    """
    Convert how a split is stored in sklearn's gradient boosted trees library to the bartpy representation

    Parameters
    ----------
    sklearn_tree: The full tree object
    index: The index of the node in the tree object

    Returns
    -------

    """
    return (
        SplitCondition(sklearn_tree.feature[index], sklearn_tree.threshold[index], le),
        SplitCondition(sklearn_tree.feature[index], sklearn_tree.threshold[index], gt)

    )


def get_bartpy_tree_from_sklearn(sklearn_tree, X, y):
    data = make_bartpy_data(X, y, normalize=False)
    bartpy_tree = Tree([LeafNode(Split(data))])
    map_sklearn_tree_into_bartpy(bartpy_tree, sklearn_tree)
    return bartpy_tree.nodes[0]


def map_sklearn_tree_into_bartpy(bartpy_tree, sklearn_tree):
    nodes = [None for x in sklearn_tree.children_left]
    nodes[0] = bartpy_tree.nodes[0]

    def search(index: int = 0):
        left_child_index, right_child_index = sklearn_tree.children_left[index], sklearn_tree.children_right[index]

        if left_child_index == -1:  # Trees are binary splits, so only need to check left tree
            return

        searched_node: LeafNode = nodes[index]

        split_conditions = map_sklearn_split_into_bartpy_split_conditions(sklearn_tree, index)
        decision_node = split_node(searched_node, split_conditions)

        left_child: LeafNode = decision_node.left_child
        right_child: LeafNode = decision_node.right_child
        left_child.set_value(sklearn_tree.value[left_child_index][0][0])
        right_child.set_value(sklearn_tree.value[right_child_index][0][0])

        mutation = GrowMutation(searched_node, decision_node)
        mutate(bartpy_tree, mutation)

        nodes[index] = decision_node
        nodes[left_child_index] = decision_node.left_child
        nodes[right_child_index] = decision_node.right_child

        search(left_child_index)
        search(right_child_index)

    search()
