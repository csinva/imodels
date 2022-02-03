from typing import List

import numpy as np

from bartpy.mutation import TreeMutation
from bartpy.node import TreeNode, LeafNode, DecisionNode, deep_copy_node


class Tree:
    """
    An encapsulation of the structure of a single decision tree
    Contains no logic, but keeps track of 4 different kinds of nodes within the tree:
      - leaf nodes
      - decision nodes
      - splittable leaf nodes
      - prunable decision nodes

    Parameters
    ----------
    nodes: List[Node]
        All nodes contained in the tree, i.e. decision and leaf nodes
    """

    def __init__(self, nodes: List[TreeNode]):
        self._nodes = nodes
        self.cache_up_to_date = False
        self._prediction = None

    @property
    def nodes(self) -> List[TreeNode]:
        """
        List of all nodes contained in the tree
        """
        return self._nodes

    @property
    def leaf_nodes(self) -> List[LeafNode]:
        """
        List of all of the leaf nodes in the tree
        """
        return [x for x in self._nodes if type(x) == LeafNode]

    @property
    def splittable_leaf_nodes(self) -> List[LeafNode]:
        """
        List of all leaf nodes in the tree which can be split in a non-degenerate way
        i.e. not all rows of the covariate matrix are duplicates
        """
        return [x for x in self.leaf_nodes if x.is_splittable()]

    @property
    def decision_nodes(self) -> List[DecisionNode]:
        """
        List of decision nodes in the tree.
        Decision nodes are internal split nodes, i.e. not leaf nodes
        """
        return [x for x in self._nodes if type(x) == DecisionNode]

    @property
    def prunable_decision_nodes(self) -> List[DecisionNode]:
        """
        List of decision nodes in the tree that are suitable for pruning
        In particular, decision nodes that have two leaf node children
        """
        return [x for x in self.decision_nodes if x.is_prunable()]

    def update_y(self, y: np.ndarray) -> None:
        """
        Update the cached value of the target array in all nodes
        Used to pass in the residuals from the sum of all of the other trees
        """
        self.cache_up_to_date = False
        for node in self.nodes:
            node.update_y(y)

    def predict(self, X: np.ndarray=None) -> np.ndarray:
        """
        Generate a set of predictions with the same dimensionality as the target array
        Note that the prediction is from one tree, so represents only (1 / number_of_trees) of the target
        """
        if X is not None:
            return self._out_of_sample_predict(X)

        if self.cache_up_to_date:
            return self._prediction
        for leaf in self.leaf_nodes:
            if self._prediction is None:
                self._prediction = np.zeros(self.nodes[0].data.X.n_obsv)
            self._prediction[leaf.split.condition()] = leaf.predict()
        self.cache_up_to_date = True
        return self._prediction

    def _out_of_sample_predict(self, X) -> np.ndarray:
        """
        Prediction for a covariate matrix not used for training

        Note that this is quite slow

        Parameters
        ----------
        X: pd.DataFrame
            Covariates to predict for
        Returns
        -------
        np.ndarray
        """
        prediction = np.array([0.] * len(X))
        for leaf in self.leaf_nodes:
            prediction[leaf.split.condition(X)] = leaf.predict()
        return prediction

    def remove_node(self, node: TreeNode) -> None:
        """
        Remove a single node from the tree
        Note that this is non-recursive, only drops the node and not any children
        """
        self._nodes.remove(node)

    def add_node(self, node: TreeNode) -> None:
        """
        Add a node to the tree
        Note that this is non-recursive, only adds the node and not any children
        """
        self._nodes.append(node)


def mutate(tree: Tree, mutation: TreeMutation) -> None:
    """
    Apply a change to the structure of the tree
    Modifies not only the tree, but also the links between the TreeNodes

    Parameters
    ----------
    tree: Tree
        The tree to mutate
    mutation: TreeMutation
        The mutation to apply to the tree
    """
    tree.cache_up_to_date = False

    if mutation.kind == "prune":
        tree.remove_node(mutation.existing_node)
        tree.remove_node(mutation.existing_node.left_child)
        tree.remove_node(mutation.existing_node.right_child)
        tree.add_node(mutation.updated_node)

    if mutation.kind == "grow":
        tree.remove_node(mutation.existing_node)
        tree.add_node(mutation.updated_node.left_child)
        tree.add_node(mutation.updated_node.right_child)
        tree.add_node(mutation.updated_node)

    for node in tree.nodes:
        if node.right_child == mutation.existing_node:
            node._right_child = mutation.updated_node
        if node.left_child == mutation.existing_node:
            node._left_child = mutation.updated_node


def deep_copy_tree(tree: Tree):
    """
    Efficiently create a copy of the tree for storage
    Creates a memory-light version of the tree with access to important information
    Parameters
    ----------
    tree: Tree
        Tree to copy

    Returns
    -------
    Tree
        Version of the tree optimized to be low memory
    """
    return Tree([deep_copy_node(x) for x in tree.nodes])
