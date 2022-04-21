from typing import Optional, Tuple

import numpy as np

from ...model import Model
from ...mutation import TreeMutation
from ...samplers.sampler import Sampler
from ...samplers.scalar import UniformScalarSampler
from ...samplers.treemutation import TreeMutationLikihoodRatio
from ...samplers.treemutation import TreeMutationProposer
from ...samplers.unconstrainedtree.likihoodratio import UniformTreeMutationLikihoodRatio
from ...samplers.unconstrainedtree.proposer import UniformMutationProposer
from ...tree import Tree, mutate


class UnconstrainedTreeMutationSampler(Sampler):
    """
    A sampler for tree mutation space.
    Responsible for producing samples of ways to mutate a tree within a model

    Works by combining a proposer and likihood evaluator into:
     - propose a mutation
     - assess likihood
     - accept if likihood higher than a uniform(0, 1) draw

    Parameters
    ----------
    proposer: TreeMutationProposer
    likihood_ratio: TreeMutationLikihoodRatio
    """

    def __init__(self,
                 proposer: TreeMutationProposer,
                 likihood_ratio: TreeMutationLikihoodRatio,
                 scalar_sampler=UniformScalarSampler()):
        self.proposer = proposer
        self.likihood_ratio = likihood_ratio
        self._scalar_sampler = scalar_sampler

    def sample(self, model: Model, tree: Tree) -> (Optional[TreeMutation], float):
        proposal = self.proposer.propose(tree)
        ratio, (l_new, l_old), (prob_new, prob_old) = self.likihood_ratio.log_probability_ratio(model, tree, proposal)
        n = proposal.existing_node.n_obs
        move = proposal.kind
        d = proposal.existing_node.depth
        mutation = proposal.existing_node if move == "prune" else proposal.updated_node
        variable = mutation.splitting_variable
        sample_data = {"ratio":ratio, "d":d, "n":n, "move":move, "variable": variable}

        if self._scalar_sampler.sample() < ratio:
            sample_data['accepted'] = True
            return proposal, sample_data
        else:
            sample_data['accepted'] = False
            return None, sample_data

    def step(self, model: Model, tree: Tree, tree_num: int) -> Tuple[Optional[TreeMutation], float]:
        mutation, data = self.sample(model, tree)
        data['tree_num'] = tree_num
        if mutation is not None:
            mutate(tree, mutation)
        return mutation, data


def get_tree_sampler(p_grow: float,
                     p_prune: float) -> Sampler:
    proposer = UniformMutationProposer([p_grow, p_prune])
    likihood = UniformTreeMutationLikihoodRatio([p_grow, p_prune])
    return UnconstrainedTreeMutationSampler(proposer, likihood)