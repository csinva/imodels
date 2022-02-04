from operator import le, gt
from typing import Callable, List, Mapping, Optional, Tuple

import numpy as np

from bartpy.errors import NoSplittableVariableException, NoPrunableNodeException
from bartpy.mutation import TreeMutation, GrowMutation, PruneMutation
from bartpy.node import LeafNode, DecisionNode, split_node
from bartpy.samplers.scalar import DiscreteSampler
from bartpy.samplers.treemutation import TreeMutationProposer
from bartpy.split import SplitCondition
from bartpy.tree import Tree


def uniformly_sample_grow_mutation(tree: Tree) -> TreeMutation:
    node = random_splittable_leaf_node(tree)
    updated_node = sample_split_node(node)
    return GrowMutation(node, updated_node)


def uniformly_sample_prune_mutation(tree: Tree) -> TreeMutation:
    node = random_prunable_decision_node(tree)
    updated_node = LeafNode(node.split, depth=node.depth)
    return PruneMutation(node, updated_node)


class UniformMutationProposer(TreeMutationProposer):

    def __init__(self,
                 prob_method: List[float]=None,
                 prob_method_lookup: Mapping[Callable[[Tree], TreeMutation], float]=None):
        if prob_method_lookup is not None:
            self.prob_method_lookup = prob_method_lookup
        else:
            if prob_method is None:
                prob_method = [0.5, 0.5]
            self.prob_method_lookup = {x[0]: x[1] for x in zip([uniformly_sample_grow_mutation, uniformly_sample_prune_mutation], prob_method)}
        self.methods = list(self.prob_method_lookup.keys())
        self.method_sampler = DiscreteSampler(list(self.prob_method_lookup.keys()),
                                              list(self.prob_method_lookup.values()),
                                              cache_size=1000)

    def propose(self, tree: Tree) -> TreeMutation:
        method = self.method_sampler.sample()
        try:
            return method(tree)
        except NoSplittableVariableException:
            return self.propose(tree)
        except NoPrunableNodeException:
            return self.propose(tree)


def random_splittable_leaf_node(tree: Tree) -> LeafNode:
    """
    Returns a random leaf node that can be split in a non-degenerate way
    i.e. a random draw from the set of leaf nodes that have at least two distinct values in their covariate matrix
    """
    splittable_nodes = tree.splittable_leaf_nodes
    if len(splittable_nodes) == 0:
        raise NoSplittableVariableException()
    else:
        return np.random.choice(splittable_nodes)


def random_prunable_decision_node(tree: Tree) -> DecisionNode:
    """
    Returns a random decision node that can be pruned
    i.e. a random draw from the set of decision nodes that have two leaf node children
    """
    leaf_parents = tree.prunable_decision_nodes
    if len(leaf_parents) == 0:
        raise NoPrunableNodeException()
    return np.random.choice(leaf_parents)


def sample_split_condition(node: LeafNode) -> Optional[Tuple[SplitCondition, SplitCondition]]:
    """
    Randomly sample a splitting rule for a particular leaf node
    Works based on two random draws

      - draw a node to split on based on multinomial distribution
      - draw an observation within that variable to split on

    Returns None if there isn't a possible non-degenerate split
    """
    split_variable = np.random.choice(list(node.split.data.X.splittable_variables()))
    split_value = node.data.X.random_splittable_value(split_variable)
    if split_value is None:
        return None
    return SplitCondition(split_variable, split_value, le), SplitCondition(split_variable, split_value, gt)


def sample_split_node(node: LeafNode) -> DecisionNode:
    """
    Split a leaf node into a decision node with two leaf children
    The variable and value to split on is determined by sampling from their respective distributions
    """
    conditions = sample_split_condition(node)
    return split_node(node, conditions)
