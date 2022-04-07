import argparse
import itertools
import os
from functools import partial

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from tqdm import tqdm
from sklearn import model_selection, datasets
from sklearn.metrics import mean_squared_error

from imodels import get_clean_dataset
from imodels.experimental.bartpy.model import Model
from imodels.experimental.bartpy.tree import Tree
from ..sklearnmodel import BART, SklearnModel

ART_PATH = "/accounts/campus/omer_ronen/projects/tree_shrink/imodels/art"
DATASETS_REGRESSION = [
    # leo-breiman paper random forest uses some UCI datasets as well
    # pg 23: https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf
    ('friedman1', 'friedman1', 'synthetic'),
    ('friedman2', 'friedman2', 'synthetic'),
    ('friedman3', 'friedman3', 'synthetic'),
    ('abalone', '183', 'openml'),
    ("diabetes-regr", "diabetes", 'sklearn'),
    ("california-housing", "california_housing", 'sklearn'),  # this replaced boston-housing due to ethical issues
    ("satellite-image", "294_satellite_image", 'pmlb'),
    ("echo-months", "1199_BNG_echoMonths", 'pmlb'),
    ("breast-tumor", "1201_BNG_breastTumor", 'pmlb'),  # this one is v big (100k examples)

]


def parse_args():
    parser = argparse.ArgumentParser(description='BART Research motivation')
    parser.add_argument('datasets', metavar='datasets', type=str, nargs='+',
                        help='datasets to run sim over')

    args = parser.parse_args()
    return args


def mse_functional(model: SklearnModel, sample: Model, X, y):
    predictions_transformed = sample.predict(X)
    predictions = model.data.y.unnormalize_y(predictions_transformed)
    return mean_squared_error(predictions, y)


def n_leaves_functional(model: SklearnModel, sample: Model, X, y):
    n_leaves = 0
    for tree in sample.trees:
        n_leaves += len(tree.leaf_nodes)
    return n_leaves / len(sample.trees)


def analyze_functional(model: SklearnModel, functional: callable, ax=None, name=None, X=None, y=None):
    if ax is None:
        _, ax = plt.subplots(1, 1)
    n_chains = model.n_chains
    chain_len = int(len(model.model_samples) / n_chains)
    color = iter(cm.rainbow(np.linspace(0, 1, n_chains)))

    functional_specific = partial(functional, X=X, y=y, model=model)

    for c in range(n_chains):
        clr = next(color)
        chain_sample = model.model_samples[c * chain_len:(c + 1) * chain_len]
        chain_functional = [functional_specific(sample=s) for s in chain_sample]
        ax.plot(np.arange(chain_len), chain_functional, color=clr, label=f"Chain {c}")

    ax.set_ylabel(name)
    ax.set_xlabel("Iteration")
    ax.legend()
    # ax.set_title(title)
    return ax


def plot_chains_leaves(model: SklearnModel, ax=None, title="Tree Structure/Prediction Variation", x_label=False, X=None,
                       y=None):
    if ax is None:
        _, ax = plt.subplots(1, 1)
    complexity = {i: [] for i in range(model.n_trees)}
    n_chains = model.n_chains
    for sample in model.model_samples:
        for i, tree in enumerate(sample.trees):
            complexity[i].append(len(tree.leaf_nodes))

    chain_len = int(len(model.model_samples) / n_chains)
    color = iter(cm.rainbow(np.linspace(0, 1, n_chains)))

    for c in range(n_chains):
        clr = next(color)
        chain_preds = model.predict_chain(X, c)
        chain_std = np.round(model.chain_mse_std(X, y, c), 2)
        mse_chain = np.round(mean_squared_error(chain_preds, y), 2)

        trees_chain = np.stack([complexity[t][c * chain_len:(c + 1) * chain_len] for t in range(model.n_trees)], axis=1)
        y_plt = np.mean(trees_chain, axis=1)
        ax.plot(np.arange(chain_len), y_plt, color=clr, label=f"Chain {c} (mse: {mse_chain} std: {chain_std})")

    ax.set_ylabel("# Leaves")
    if x_label:
        ax.set_xlabel("Iteration")
    ax.legend()
    ax.set_title(title)
    return ax


def plot_within_chain(model: SklearnModel, ax=None, title="Within Chain Variation", x_label=False, X=None, y=None):
    if ax is None:
        _, ax = plt.subplots(1, 1)
    n_chains = model.n_chains

    chain_len = int(len(model.model_samples) / n_chains)
    color = iter(cm.rainbow(np.linspace(0, 1, n_chains)))

    for c in range(n_chains):
        clr = next(color)
        chain_preds = model.chain_precitions(X, c)
        mean_pred = np.array(chain_preds).mean(axis=0)

        y_plt = [mean_squared_error(mean_pred, p) for p in chain_preds]
        ax.plot(np.arange(chain_len), y_plt, color=clr, label=f"Chain {c} (Average {np.round(np.mean(y_plt), 2)})")

    ax.set_ylabel("mean squared distance to average iteration")
    if x_label:
        ax.set_xlabel("Iteration")
    ax.legend()
    ax.set_title(title)
    return ax


def plot_across_chains(model: SklearnModel, ax=None, title="Across Chain Variation", x_label=False, X=None, y=None):
    if ax is None:
        _, ax = plt.subplots(1, 1)
    n_chains = model.n_chains

    preds = []
    mat = np.zeros(shape=(n_chains, n_chains))

    for c in range(n_chains):
        preds.append(model.predict_chain(X, c))
    for c_i, c_j in itertools.combinations(range(n_chains), 2):
        mat[c_i, c_j] = mean_squared_error(preds[c_i], preds[c_j])
        mat[c_j, c_i] = mean_squared_error(preds[c_i], preds[c_j])
    ax.matshow(mat)
    for c_i, c_j in itertools.combinations(range(n_chains), 2):
        c = np.round(mat[c_i, c_j], 2)
        ax.text(c_i, c_j, c, va='center', ha='center')
        ax.text(c_j, c_i, c, va='center', ha='center')
    ax.set_xlabel("mean squared distance between predictions")

    ax.legend()
    ax.set_title(f"{title} (Between Chains Var {np.round(model.between_chains_var(X), 2)})")
    return ax


def main():
    n_trees = 50
    n_samples = 5000
    n_burn = 10000
    n_chains = 5
    with tqdm(DATASETS_REGRESSION) as t:
        for d in t:
            t.set_description(f'{d[0]}')
            X, y, feat_names = get_clean_dataset(d[1], data_source=d[2])
            n = len(y)

            X_train, X_test, y_train, y_test = model_selection.train_test_split(
                X, y, test_size=0.3, random_state=4)

            bart_zero = BART(classification=False, store_acceptance_trace=True, n_trees=n_trees, n_samples=n_samples,
                             n_burn=n_burn, n_chains=n_chains, thin=1)
            bart_zero.fit(X_train, y_train)

            fig, axs = plt.subplots(4, 1, figsize=(10, 22))
            # fig.tight_layout()
            fig.subplots_adjust(hspace=.2)

            # plot_chains_leaves(bart_zero, axs[0], X=X_test, y=y_test)
            analyze_functional(bart_zero, functional=mse_functional, ax=axs[0], X=X_test, y=y_test, name="Test MSE")
            analyze_functional(bart_zero, functional=n_leaves_functional, ax=axs[1], X=X_test, y=y_test,
                               name="# Leaves")
            plot_within_chain(bart_zero, axs[2], X=X_test, y=y_test)
            plot_across_chains(bart_zero, axs[3], X=X_test, y=y_test)

            title = f"Dataset: {d[0].capitalize()}, (n = {n}, burn = {n_burn})"
            plt.suptitle(title)

            plt.savefig(os.path.join(ART_PATH, "functional", f"{d[0]}_samples_{n_samples}.png"))
            plt.close()


if __name__ == '__main__':
    main()
