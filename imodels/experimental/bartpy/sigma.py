
class Sigma:
    """
    A representation of the sigma term in the model.
    Specifically, this is the sigma of y itself, i.e. the sigma in
        y ~ Normal(sum_of_trees, sigma)

    The default prior is an inverse gamma distribution on the variance
    The parametrization is slightly different to the numpy gamma version, with the scale parameter inverted

    Parameters
    ----------
    alpha - the shape of the prior
    beta - the scale of the prior
    scaling_factor - the range of the original distribution
                     needed to rescale the variance into the original scale rather than on (-0.5, 0.5)

    """

    def __init__(self, alpha: float, beta: float, scaling_factor: float, classification :bool = False):
        self.alpha = alpha
        self.beta = beta
        self._current_value = 1.0
        self.scaling_factor = scaling_factor
        self._classification = classification

    def set_value(self, value: float) -> None:
        self._current_value = value

    def current_value(self) -> float:
        if self._classification:
            return 1
        return self._current_value

    def current_unnormalized_value(self) -> float:
        return self.current_value() * self.scaling_factor
