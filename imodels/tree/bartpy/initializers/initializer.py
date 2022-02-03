from typing import Generator

from bartpy.tree import Tree


class Initializer(object):
    """
    The abstract interface for the tree initializers.

    Initializers are responsible for setting the starting values of the model, in particular:
      - structure of decision and leaf nodes
      - variables and values used in splits
      - values of leaf nodes

    Good initialization of trees helps speed up convergence of sampling

    Default behaviour is to leave trees uninitialized
    """

    def initialize_tree(self, tree: Tree) -> None:
        pass

    def initialize_trees(self, trees: Generator[Tree, None, None]) -> None:
        for tree in trees:
            self.initialize_tree(tree)
