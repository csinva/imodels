from matplotlib import pyplot as plt

from imodels.experimental.bartpy.diagnostics.residuals import plot_qq, plot_homoskedasity_diagnostics
from imodels.experimental.bartpy.diagnostics.sampling import plot_tree_mutation_acceptance_rate
from imodels.experimental.bartpy.diagnostics.sigma import plot_sigma_convergence
from imodels.experimental.bartpy.diagnostics.trees import plot_tree_depth
from imodels.experimental.bartpy.sklearnmodel import SklearnModel


def plot_diagnostics(model: SklearnModel):
    fig, ((ax1, ax2, ax3), (ax4, ax5, _)) = plt.subplots(2, 3, figsize=(10, 10))
    fig.suptitle("Diagnostics")
    plot_qq(model, ax1)
    plot_tree_depth(model, ax2)
    plot_sigma_convergence(model, ax3)
    plot_homoskedasity_diagnostics(model, ax4)
    plot_tree_mutation_acceptance_rate(model, ax5)
    plt.show()
