import numpy as np

from ..model import Model
from ..node import LeafNode
from ..samplers.sampler import Sampler
from ..samplers.scalar import NormalScalarSampler


class LeafNodeSampler(Sampler):
    """
    Responsible for generating samples of the leaf node predictions
    Essentially just draws from a normal distribution with prior specified by model parameters

    Uses a cache of draws from a normal(0, 1) distribution to improve sampling performance
    """

    def __init__(self,
                 scalar_sampler=NormalScalarSampler(60000)):
        self._scalar_sampler = scalar_sampler

    def step(self, model: Model, node: LeafNode) -> float:
        sampled_value = self.sample(model, node)
        node.set_value(sampled_value)
        return sampled_value

    def sample(self, model: Model, node: LeafNode) -> float:
        prior_var = model.sigma_m ** 2
        n = node.data.X.n_obsv
        likelihood_var = (model.sigma.current_value() ** 2) / n
        likelihood_mean = node.data.y.summed_y() / n
        node.set_mean_response(likelihood_mean)
        posterior_variance = 1. / (1. / prior_var + 1. / likelihood_var)
        posterior_mean = likelihood_mean * 1#(prior_var / (likelihood_var + prior_var))
        val = posterior_mean# + (self._scalar_sampler.sample() * np.power(posterior_variance / model.n_trees, 0.5))
        return val

# class VectorizedLeafNodeSampler(Sampler):

#     def step(self, model: Model, nodes: List[LeafNode]) -> float:
#         sampled_values = self.sample(model, nodes)
#         for (node, sample) in zip(nodes, sampled_values):
#             node.set_value(sample)
#         return sampled_values[0]

#     def sample(self, model: Model, nodes: List[LeafNode]) -> List[float]:
#         prior_var = model.sigma_m ** 2
#         n_s = []
#         sum_s = []
