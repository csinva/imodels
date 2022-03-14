from typing import Generator

from ..tree import Tree


class Initializer:
    """
    The abstract interface for the tree initializers.

    Initializers are responsible for setting the starting values of the model, in particular:
      - structure of decision and leaf nodes
      - variables and values used in splits
      - values of leaf nodes

    Good initialization of trees helps speed up convergence of sampling

    Default behaviour is to leave trees uninitialized
    """

    def __init__(self):
        self.n_trees = 1

    def initialize_tree(self, tree: Tree, tree_number: int) -> None:
        pass

    def initialize_trees(self, trees: Generator[Tree, None, None]) -> None:
        n_trees = 0
        for tree_number, tree in enumerate(trees):
            self.initialize_tree(tree, tree_number)
            n_trees += 1
        self.n_trees = n_trees
