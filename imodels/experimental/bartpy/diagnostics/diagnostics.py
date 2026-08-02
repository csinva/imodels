from matplotlib import pyplot as plt

from ..diagnostics.residuals import plot_qq, plot_homoscedasticity_diagnostics
from ..diagnostics.sampling import plot_tree_mutation_acceptance_rate
from ..diagnostics.sigma import plot_sigma_convergence
from ..diagnostics.trees import plot_tree_depth
from ..sklearnmodel import SklearnModel


def plot_diagnostics(model: SklearnModel):
    fig, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, _)) = plt.subplots(2, 4, figsize=(10, 10))
    fig.suptitle("Diagnostics")
    plot_qq(model, ax1)
    plot_tree_depth(model, ax2)
    plot_sigma_convergence(model, ax3)
    plot_homoscedasticity_diagnostics(model, ax4)
    plot_tree_mutation_acceptance_rate(model, ax5)
    # plot_tree_likelihood(model, ax6)
    # plot_tree_probs(model, ax7)

    plt.show()
