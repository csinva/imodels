from typing import Optional, Tuple

import numpy as np

from ...model import Model
from ...mutation import TreeMutation
from ...samplers.sampler import Sampler
from ...samplers.scalar import UniformScalarSampler
from ...samplers.treemutation import TreeMutationLikelihoodRatio
from ...samplers.treemutation import TreeMutationProposer
from ...samplers.unconstrainedtree.likelihoodratio import UniformTreeMutationLikelihoodRatio
from ...samplers.unconstrainedtree.proposer import UniformMutationProposer
from ...tree import Tree, mutate


class UnconstrainedTreeMutationSampler(Sampler):
    """
    A sampler for tree mutation space.
    Responsible for producing samples of ways to mutate a tree within a model

    Works by combining a proposer and likelihood evaluator into:
     - propose a mutation
     - assess likelihood
     - accept if likelihood higher than a uniform(0, 1) draw

    Parameters
    ----------
    proposer: TreeMutationProposer
    likelihood_ratio: TreeMutationLikelihoodRatio
    """

    def __init__(self,
                 proposer: TreeMutationProposer,
                 likelihood_ratio: TreeMutationLikelihoodRatio,
                 scalar_sampler=UniformScalarSampler()):
        self.proposer = proposer
        self.likelihood_ratio = likelihood_ratio
        self._scalar_sampler = scalar_sampler

    def sample(self, model: Model, tree: Tree) -> (Optional[TreeMutation], float):
        proposal = self.proposer.propose(tree)
        ratio, (l_new, l_old), (prob_new, prob_old) = self.likelihood_ratio.log_probability_ratio(model, tree, proposal)
        if self._scalar_sampler.sample() < ratio:
            return proposal, np.exp(l_new) - np.exp(l_old), np.exp(prob_new) - np.exp(prob_old)
        else:
            return None, 0, 0

    def step(self, model: Model, tree: Tree) -> Tuple[Optional[TreeMutation], float]:
        mutation, likelihood, prob = self.sample(model, tree)
        if mutation is not None:
            mutate(tree, mutation)
        return mutation, likelihood, prob


def get_tree_sampler(p_grow: float,
                     p_prune: float) -> Sampler:
    proposer = UniformMutationProposer([p_grow, p_prune])
    likelihood = UniformTreeMutationLikelihoodRatio([p_grow, p_prune])
    return UnconstrainedTreeMutationSampler(proposer, likelihood)