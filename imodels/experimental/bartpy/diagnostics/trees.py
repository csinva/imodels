import numpy as np
from matplotlib import pyplot as plt

from ..sklearnmodel import SklearnModel


def plot_tree_depth(model: SklearnModel, ax=None, title="", x_label=False):
    if ax is None:
        _, ax = plt.subplots(1, 1)
    # min_depth, mean_depth, max_depth = [], [], []
    complexity = {i:[] for i in range(model.n_trees)}
    for sample in model.model_samples:
        # model_depths = []
        for i, tree in enumerate(sample.trees):
            complexity[i].append(len(tree.leaf_nodes))
        #     model_depths += [x.depth for x in tree.nodes]
        # min_depth.append(np.min(model_depths))
        # mean_depth.append(np.mean(model_depths))
        # max_depth.append(np.max(model_depths))

    # ax.plot(min_depth, label="Min Depth")
    # ax.plot(mean_depth, label="Mean Depth")
    # ax.plot(max_depth, label="Max Depth")
    for tree_number, comp in complexity.items():
        ax.plot(np.arange(len(model.model_samples)), comp, label=f"tree {tree_number}", alpha=0.5)

    ax.set_ylabel("# Leaves")
    if x_label:
        ax.set_xlabel("Iteration")
    ax.legend()
    ax.set_title(title)
    return ax