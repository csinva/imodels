from matplotlib import pyplot as plt

from bartpy.sklearnmodel import SklearnModel


def plot_sigma_convergence(model: SklearnModel, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    sigma_samples = [x.sigma.current_value() for x in model.model_samples]
    ax.plot(sigma_samples)
    ax.set_title("Sigma Convergence")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Sigma")
    return ax
