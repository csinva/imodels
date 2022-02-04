from matplotlib import pyplot as plt

from bartpy.sklearnmodel import SklearnModel


def plot_tree_mutation_acceptance_rate(model: SklearnModel, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1)

    ax.plot([x["Tree"] for x in model.acceptance_trace])
    ax.set_title("Tree Mutation Acceptance Rate")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Acceptance Rate")
    ax.set_ylim((0, 1.1))
    return ax
