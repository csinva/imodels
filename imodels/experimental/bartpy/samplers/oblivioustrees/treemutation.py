from typing import Optional, List

import numpy as np

from ...model import Model
from ...mutation import TreeMutation
from ...samplers.sampler import Sampler
from ...samplers.scalar import UniformScalarSampler
from ...samplers.treemutation import TreeMutationLikelihoodRatio
from ...samplers.treemutation import TreeMutationProposer
from ...samplers.oblivioustrees.likelihoodratio import UniformTreeMutationLikelihoodRatio
from ...samplers.oblivioustrees.proposer import UniformMutationProposer
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

    def sample(self, model: Model, tree: Tree) -> Optional[List[TreeMutation]]:
        proposals: List[TreeMutation] = self.proposer.propose(tree)
        ratio = np.sum([self.likelihood_ratio.log_probability_ratio(model, tree, x) for x in proposals])
        if self._scalar_sampler.sample() < ratio:
            return proposals
        else:
            return None

    def step(self, model: Model, tree: Tree) -> Optional[List[TreeMutation]]:
        mutations = self.sample(model, tree)
        if mutations is not None:
            for mutation in mutations:
                mutate(tree, mutation)
        return mutations


def get_tree_sampler(p_grow: float,
                     p_prune: float):
    proposer = UniformMutationProposer(p_grow, p_prune)
    likelihood = UniformTreeMutationLikelihoodRatio([p_grow, p_prune])
    return UnconstrainedTreeMutationSampler(proposer, likelihood)
