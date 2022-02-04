import numpy as np

from imodels.experimental.bartpy.model import Model
from imodels.experimental.bartpy.samplers.sampler import Sampler
from imodels.experimental.bartpy.sigma import Sigma


class SigmaSampler(Sampler):

    def step(self, model: Model, sigma: Sigma) -> float:
        sample_value = self.sample(model, sigma)
        sigma.set_value(sample_value)
        return sample_value

    @staticmethod
    def sample(model: Model, sigma: Sigma) -> float:
        posterior_alpha = sigma.alpha + (model.data.X.n_obsv / 2.)
        posterior_beta = sigma.beta + (0.5 * (np.sum(np.square(model.residuals()))))
        draw = np.power(np.random.gamma(posterior_alpha, 1./posterior_beta), -0.5)
        return draw
