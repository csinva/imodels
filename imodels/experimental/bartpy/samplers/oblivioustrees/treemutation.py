from typing import Optional, List

import numpy as np

from imodels.experimental.bartpy.model import Model
from imodels.experimental.bartpy.mutation import TreeMutation
from imodels.experimental.bartpy.samplers.sampler import Sampler
from imodels.experimental.bartpy.samplers.scalar import UniformScalarSampler
from imodels.experimental.bartpy.samplers.treemutation import TreeMutationLikihoodRatio
from imodels.experimental.bartpy.samplers.treemutation import TreeMutationProposer
from imodels.experimental.bartpy.samplers.oblivioustrees.likihoodratio import UniformTreeMutationLikihoodRatio
from imodels.experimental.bartpy.samplers.oblivioustrees.proposer import UniformMutationProposer
from imodels.experimental.bartpy.tree import Tree, mutate


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

    def sample(self, model: Model, tree: Tree) -> Optional[List[TreeMutation]]:
        proposals: List[TreeMutation] = self.proposer.propose(tree)
        ratio = np.sum([self.likihood_ratio.log_probability_ratio(model, tree, x) for x in proposals])
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
    likihood = UniformTreeMutationLikihoodRatio([p_grow, p_prune])
    return UnconstrainedTreeMutationSampler(proposer, likihood)
