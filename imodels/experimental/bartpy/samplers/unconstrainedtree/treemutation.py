from typing import Optional

from bartpy.model import Model
from bartpy.mutation import TreeMutation
from bartpy.samplers.sampler import Sampler
from bartpy.samplers.scalar import UniformScalarSampler
from bartpy.samplers.treemutation import TreeMutationLikihoodRatio
from bartpy.samplers.treemutation import TreeMutationProposer
from bartpy.samplers.unconstrainedtree.likihoodratio import UniformTreeMutationLikihoodRatio
from bartpy.samplers.unconstrainedtree.proposer import UniformMutationProposer
from bartpy.tree import Tree, mutate


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

    def sample(self, model: Model, tree: Tree) -> Optional[TreeMutation]:
        proposal = self.proposer.propose(tree)
        ratio = self.likihood_ratio.log_probability_ratio(model, tree, proposal)
        if self._scalar_sampler.sample() < ratio:
            return proposal
        else:
            return None

    def step(self, model: Model, tree: Tree) -> Optional[TreeMutation]:
        mutation = self.sample(model, tree)
        if mutation is not None:
            mutate(tree, mutation)
        return mutation


def get_tree_sampler(p_grow: float,
                     p_prune: float) -> Sampler:
    proposer = UniformMutationProposer([p_grow, p_prune])
    likihood = UniformTreeMutationLikihoodRatio([p_grow, p_prune])
    return UnconstrainedTreeMutationSampler(proposer, likihood)