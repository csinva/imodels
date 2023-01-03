import numpy as np
from matplotlib import pyplot as plt

from ..sklearnmodel import SklearnModel


def plot_tree_mutation_acceptance_rate(model: SklearnModel, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1)

    ax.scatter(np.arange(len(model.acceptance_trace)),[x["Tree"] for x in model.acceptance_trace])
    ax.set_title("Tree Mutation Acceptance Rate")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Acceptance Rate")
    ax.set_ylim((0, 1.1))
    return ax

def plot_tree_likelihood(model: SklearnModel, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1)


    ax.scatter(np.arange(len(model.likelihood)), model.likelihood)
    ax.set_title("Likelihood")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Likelihood")
    # ax.set_ylim((0, 1.1))
    return ax


def plot_tree_probs(model: SklearnModel, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1)

    ax.scatter(np.arange(len(model.probs)), model.probs)
    ax.set_title("Probs")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Probs")
    # ax.set_ylim((0, 1.1))
    return ax
