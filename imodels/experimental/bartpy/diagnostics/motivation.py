import argparse
import itertools
import os
from functools import partial
from typing import List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.inspection import permutation_importance
from tqdm import tqdm
from sklearn import model_selection, datasets
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

from imodels import get_clean_dataset
from imodels.experimental.bartpy.initializers.sklearntreeinitializer import SklearnTreeInitializer
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
DATASETS_SYNTHETIC = [
    ('friedman1', 'friedman1', 'synthetic'),
    ('radchenko_james', 'radchenko_james', 'synthetic'),
    ('vo_pati', 'vo_pati', 'synthetic'),
    ('bart', 'bart', 'synthetic'),

]

def parse_args():
    parser = argparse.ArgumentParser(description='BART Research motivation')
    parser.add_argument('datasets', metavar='datasets', type=str,nargs='+',
                        help='datasets to run sim over')
    parser.add_argument('n_trees', metavar='n_trees', type=int,
                        help='number of trees')
    parser.add_argument('--display', action='store_true',
                        help='display figure')

    args = parser.parse_args()
    return args


def get_important_features(dataset_name):
    if dataset_name in ["friedman1", "radchenko_james", "vo_pati"]:
        return [0, 1, 2, 3, 4]
    elif dataset_name in ["friedman2", "friedman3"]:
        return [0, 1, 2, 3]
    elif dataset_name == "bart":
        return [0, 1]



def log_rmse(x, y):
    return np.log(1 + np.sqrt(mean_squared_error(x, y)))


def mse_functional(model: SklearnModel, sample: Model, X, y, ds_name):
    predictions_transformed = sample.predict(X)
    predictions = model.data.y.unnormalize_y(predictions_transformed)
    return log_rmse(predictions, y)


def n_leaves_functional(model: SklearnModel, sample: Model, X, y, ds_name):
    n_leaves = 0
    for tree in sample.trees:
        n_leaves += len(tree.leaf_nodes)
    return n_leaves / len(sample.trees)


def importance_functional(model: SklearnModel, sample: Model, X, y, ds_name):
    important_features = get_important_features(ds_name)
    if important_features is None:
        return
    if sample.importances is not None:
        rel_imp = np.sum(sample.importances[important_features]) / np.sum(sample.importances)
        return rel_imp
    result = permutation_importance(
        sample, X, y, n_repeats=10, random_state=42, n_jobs=2, scoring="neg_mean_absolute_percentage_error"
    )
    # feature_names = [f"feature {i}" for i in range(X.shape[1])]

    importances = pd.Series(result.importances_mean)
    sample.set_importances(importances)
    return importances
    # rel_imp = np.sum(sample.importances[important_features]) / np.sum(sample.importances)
    #
    # return rel_imp


def analyze_functional(models: Dict[str, SklearnModel], functional: callable, axs=None, name=None, X=None, y=None,
                       ds_name=None):
    if axs is None:
        _, axs = plt.subplots(2, 1)
    colors = {0: cm.Blues, 1: cm.Greens, 2: cm.Reds}
    min_hist = np.inf
    max_hist = -1 * np.inf
    plt_data = {"plot": {m: [] for m in models.keys()}, "hist": {m: [] for m in models.keys()}}

    for i, (mdl_name, model) in enumerate(models.items()):
        n_chains = model.n_chains
        chain_len = int(len(model.model_samples) / n_chains)
        # color = iter(colors[i](np.linspace(0.3, 0.7, n_chains)))

        functional_specific = partial(functional, X=X, y=y, model=model, ds_name=ds_name)
        hist_len = int(chain_len / 3)

        for c in range(n_chains):
            chain_sample = model.model_samples[c * chain_len:(c + 1) * chain_len]
            chain_functional = [functional_specific(sample=s) for s in chain_sample]

            plt_data['plot'][mdl_name].append((np.arange(chain_len), chain_functional))
            hist_data = chain_functional[(chain_len - hist_len):chain_len]
            plt_data['hist'][mdl_name].append(hist_data)
            max_hist = np.maximum(max_hist, np.max(hist_data))
            min_hist = np.minimum(min_hist, np.min(hist_data))

    for i, (mdl_name, model) in enumerate(models.items()):
        n_chains = model.n_chains

        # chain_len = int(len(model.model_samples) / n_chains)
        pallete = colors[i] if len(models) > 1 else cm.rainbow
        l,u = (0.3, 0.7) if len(models) > 1 else (0,1)
        color = iter(pallete(np.linspace(l, u, n_chains)))
        for i, c in enumerate(range(n_chains)):
            clr = next(color)
            plt_x, plt_y = plt_data['plot'][mdl_name][i]
            axs[0].plot(plt_x, plt_y, color=clr, label=f"Chain {c} ({mdl_name})")
            hist_x = plt_data['hist'][mdl_name][i]
            axs[1].hist(hist_x, color=clr, label=f"Chain {c} ({mdl_name})",
                        alpha=0.75, bins=50, range=[min_hist, max_hist])

    axs[0].set_ylabel(name)
    axs[0].set_xlabel("Iteration")

    axs[1].set_xlabel(name)
    axs[1].set_ylabel("Count")

    axs[0].legend()
    # ax.set_title(title)
    return axs


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


def plot_within_chain(models: Dict[str, SklearnModel], ax=None, title="Within Chain Variation", x_label=False, X=None,
                      y=None):
    if ax is None:
        _, ax = plt.subplots(1, 1)
    lines = {0: "-", 1: '--'}

    for i, (mdl_name, model) in enumerate(models.items()):
        n_chains = model.n_chains

        chain_len = int(len(model.model_samples) / n_chains)
        color = iter(cm.rainbow(np.linspace(0, 1, n_chains)))

        for c in range(n_chains):
            clr = next(color)
            chain_preds = model.chain_precitions(X, c)
            mean_pred = np.array(chain_preds).mean(axis=0)

            y_plt = [np.log(np.sqrt(mean_squared_error(mean_pred, p))) for p in chain_preds]
            ax.plot(np.arange(chain_len), y_plt, color=clr,
                    label=f"Chain {c}, {mdl_name} (Average {np.round(np.mean(y_plt), 2)})", linestyle=lines[i])

    ax.set_ylabel("mean squared distance to average iteration")
    if x_label:
        ax.set_xlabel("Iteration")
    ax.legend()
    ax.set_title(title)
    return ax


def plot_across_chains(models: Dict[str, SklearnModel], ax=None, title="Across Chain Variation", x_label=False, X=None,
                       y=None, fig=None):
    if ax is None:
        _, ax = plt.subplots(1, 1)

    model_0 = list(models.values())[0]

    n_chains = model_0.n_chains * len(models)

    preds = []
    mat = np.zeros(shape=(n_chains, n_chains + 1))
    label_list = []

    for j, (mdl_name, model) in enumerate(models.items()):
        for c in range(model.n_chains):
            chain_preds = model.chain_precitions(X, c)
            chain_len = len(chain_preds)
            s = (chain_len - int(chain_len / 3))
            preds.append(model.predict_chain(X, c, s))

            posterior = chain_preds[s:chain_len]

            mean_pred = np.array(posterior).mean(axis=0)

            within_chain_mse = [log_rmse(mean_pred, p) for p in posterior]
            mat[j, 0] = np.round(np.mean(within_chain_mse), 2)
            label_list.append(f"Chain {c} ({mdl_name})")

    for c_i, c_j in itertools.combinations(range(n_chains), 2):
        mat[c_i, c_j + 1] = log_rmse(preds[c_i], preds[c_j])
        mat[c_j, c_i + 1] = log_rmse(preds[c_i], preds[c_j])
    im = ax.matshow(mat)
    # for c_i, c_j in itertools.combinations(range(n_chains), 2):
    #     c = np.round(mat[c_i, c_j], 2)
    #     ax.text(c_i, c_j, c, va='center', ha='center')
    #     ax.text(c_j, c_i, c, va='center', ha='center')
    ax.set_xlabel("log root mean squared distance between predictions")

    ax.set_xticks(np.arange(0, len(label_list) + 1))
    ax.set_xticklabels(["Within"] + label_list, rotation=90)

    ax.set_yticks(np.arange(0, len(label_list)))
    ax.set_yticklabels(label_list)

    ax.set_title(title)
    divider = make_axes_locatable(ax)
    cax = divider.new_vertical(size="5%", pad=0.7, pack_start=True)
    fig.add_axes(cax)
    fig.colorbar(im, cax=cax, orientation="horizontal")
    # ax.set_title(f"{title} (Between Chains Var {np.round(model.between_chains_var(X), 2)})")
    return ax


def fig_1(barts: Dict[str, SklearnModel], X, y, dataset, display):
    X_test = X
    y_test = y
    n, p = X.shape
    n = int(n/3 * 10)
    n_samples = list(barts.values())[0].n_samples
    n_burn = list(barts.values())[0].n_burn
    n_trees = list(barts.values())[0].n_trees

    is_synthetic = dataset in DATASETS_SYNTHETIC

    ds_name = dataset[0]

    fig, axs = plt.subplots(3, 2, figsize=(10, 22))
    # fig.tight_layout()
    fig.subplots_adjust(hspace=0.6)

    analyze_functional(barts, functional=mse_functional, axs=axs[0, 0:2], X=X_test, y=y_test,
                       name="Test log-RMSE", ds_name=ds_name)
    analyze_functional(barts, functional=n_leaves_functional, axs=axs[1, 0:2], X=X_test, y=y_test,
                       name="# Leaves", ds_name=ds_name)

    # plot_within_chain(barts, axs[2], X=X_test, y=y_test)
    plot_across_chains(barts, axs[2, 1], X=X_test, y=y_test, fig=fig)
    axs[2, 0].axis('off')
    # if is_synthetic:
    #     analyze_functional(barts, functional=importance_functional, axs=axs[3, 0:2], X=X_test, y=y_test,
    #                        name="Permutation importance", ds_name=dataset[0])

    #
    title = f"Dataset: {dataset[0].capitalize()}, (n, p) = ({n}, {p}), burn = {n_burn}"
    plt.suptitle(title)
    if display:
        plt.show()
    else:
        synthetic_name = f"{dataset[0]}_samples_{n_samples}_n_dp_{n}_trees_{n_trees}.png"
        real_name = f"{dataset[0]}_samples_{n_samples}_trees_{n_trees}.png"
        fig_name = synthetic_name if is_synthetic else real_name
        plt.savefig(os.path.join(ART_PATH,"features",  fig_name))


# def fig_2(barts: Dict[str, SklearnModel], X, y, dataset, display):
#     fig, axs = plt.subplots(3, 2, figsize=(10, 22))
#     # fig.tight_layout()
#     fig.subplots_adjust(hspace=.8)
#     X_test = X
#     y_test = y
#     n, p = X.shape
#     n_samples = list(barts.values())[0].n_samples
#     n_burn = list(barts.values())[0].n_burn
#     n_trees = list(barts.values())[0].n_trees
#     analyze_functional(barts, functional=importance_functional, axs=axs[0, 0:2], X=X_test, y=y_test,
#                        name="Permutation importance")

def _get_feature_acceptance_sample_data(mcmc_data, f_num):
    var_idx = np.array(mcmc_data.variable == f_num)
    prob = np.minimum(mcmc_data.ratio, 1)
    accpt = np.array(mcmc_data.accepted, dtype=np.int)
    # positive = np.array(np.logical_and(data_var.move == "grow", data_var.accepted), dtype=np.int)

    # negative = np.array(np.logical_and(data_var.move == "prune", data_var.accepted), dtype=np.int)
    grow = np.array(np.logical_and(var_idx,mcmc_data.move == "grow"), dtype=np.int)
    prune = np.array(np.logical_and(var_idx,mcmc_data.move == "prune"), dtype=np.int)

    net = (accpt * grow - accpt * prune) * prob

    return np.cumsum(net), np.arange(mcmc_data.shape[0])


def fig_2(bart: SklearnModel, X, y, dataset, display):
    fig, axs = plt.subplots(bart.n_chains, 1, figsize=(10, 22))
    n, p = X.shape
    n_samples = bart.n_samples
    n_burn = bart.n_burn
    n_trees = bart.n_trees
    # fig.tight_layout()
    fig.subplots_adjust(hspace=0.6)
    is_synthetic = dataset in DATASETS_SYNTHETIC

    ds_name = dataset[0]
    important_features = get_important_features(ds_name)
    mcmc_data = bart.mcmc_data

    chain_draws = n_trees * n_samples

    for c in range(bart.n_chains):
        chain_data = mcmc_data.iloc[int(c*chain_draws): int((c+1)*chain_draws), :]
        color_important = iter(cm.Blues(np.linspace(0.2, 0.8, len(important_features))))
        color_not_important = iter(cm.Reds(np.linspace(0.3, 0.7, p-len(important_features))))
        custom_lines = [Line2D([0], [0], color="blue", lw=4, label="True"),
                        Line2D([0], [0], color="red", lw=4, label="Null")]


        # chain_len = int(len(model.model_samples) / n_chains)
        for f in range(p):
            acpt, smpl = _get_feature_acceptance_sample_data(chain_data, f)
            is_important = f in important_features
            clr = next(color_important) if is_important else next(color_not_important)
            axs[c].plot(smpl, acpt, color=clr)
        axs[c].set_xlabel("Iteration")
        axs[c].set_ylabel("Cumulative net acceptance")
        if c == 0:
            axs[c].legend(custom_lines, ['True', 'Null'])


        #
        #
        # importance_f = partial(importance_functional, X=X, y=y, model=model, ds_name=ds_name)
        #
        # for c in range(1):
        #     chain_sample = model.model_samples[c * chain_len:(c + 1) * chain_len]
        #     chain_functional = np.stack([importance_f(sample=s) for s in chain_sample], axis=0)
        #     colors_i = iter(cm.rainbow(np.linspace(0, 1, len(important_features) + 1)))
        #     for f in important_features:
        #         axs[i].plot(np.arange(chain_len), chain_functional[:, f], color=next(colors_i), label=f"Feature {f}")
        #     non_imp = np.sum(chain_functional[:, non_important_idx], axis=1)
        #     axs[i].plot(np.arange(chain_len), non_imp, color=next(colors_i), label=f"Non Important")
        axs[c].set_title(f"chain {c}")
        # axs[c].legend()

    title = f"Dataset: {dataset[0].capitalize()}, (n, p) = ({n}, {p}), burn = {n_burn}"
    plt.suptitle(title)
    if display:
        plt.show()
    else:
        synthetic_name = f"{dataset[0]}_samples_{n_samples}_n_dp_{n}_trees_{n_trees}_importance.png"
        real_name = f"{dataset[0]}_samples_{n_samples}_trees_{n_trees}_importance.png"
        fig_name = synthetic_name if is_synthetic else real_name
        plt.savefig(os.path.join(ART_PATH,"features", fig_name))
        # plt.savefig(
            # os.path.join(ART_PATH, f"{dataset[0]}_samples_{n_samples}_trees_{n_trees}_importance.png"))


def main():
    # n_trees = 100
    n_samples = 10000  # 7500  # 00
    n_burn = 0  # 10000
    n_chains = 4  # 2
    args = parse_args()
    ds = args.datasets
    n_trees = args.n_trees
    display = args.display
    ds = [d for d in DATASETS_SYNTHETIC if d[0] in ds]
    with tqdm(ds) as t:
        for d in t:
            t.set_description(f'{d[0]}')

            for n_ds_samples in [2000, 20000, 200000]:
                X, y, feat_names = get_clean_dataset(d[1], data_source=d[2], n_samples=n_ds_samples, p=10)

                X_train, X_test, y_train, y_test = model_selection.train_test_split(
                    X, y, test_size=0.3, random_state=4)

                # X_rand, y_rand = np.random.random(size=X_train.shape), np.random.random(size=y_train.shape)

                bart_zero = BART(classification=False, store_acceptance_trace=True, n_trees=n_trees, n_samples=n_samples,
                                 n_burn=n_burn, n_chains=n_chains, thin=1)
                bart_zero.fit(X_train, y_train)
                # bart_zero.mcmc_data.to_csv(os.path.join(ART_PATH, f"{d[0]}_samples_{n_samples}_trees_{n_trees}_zero.csv"))
                #
                # sgb = GradientBoostingRegressor(n_estimators=n_trees)
                # sgb.fit(X_train, bart_zero.data.y.values)
                #
                # bart_sgb = BART(classification=False, store_acceptance_trace=True, n_trees=n_trees, n_samples=n_samples,
                #                 n_burn=n_burn, n_chains=n_chains, thin=1, initializer=SklearnTreeInitializer(tree_=sgb))
                # bart_sgb.fit(X_train, y_train)
                # bart_sgb.mcmc_data.to_csv(os.path.join(ART_PATH, f"{d[0]}_samples_{n_samples}_trees_{n_trees}_sgb.csv"))

                # rf = RandomForestRegressor(n_estimators=n_trees, max_leaf_nodes=10)
                # rf = rf.fit(X_train, y_rand)
                #
                # bart_rand = BART(classification=False, store_acceptance_trace=True, n_trees=n_trees, n_samples=n_samples,
                #                  n_burn=n_burn, n_chains=n_chains, thin=1, initializer=SklearnTreeInitializer(tree_=rf))
                # bart_rand.fit(X_train, y_train)
                # bart_rand.mcmc_data.to_csv(os.path.join(ART_PATH, f"{d[0]}_samples_{n_samples}_trees_{n_trees}_rand.csv"))

                # barts = {"SGB": bart_sgb, "Single Leaf": bart_zero, "Random": bart_rand}
                barts = {"Single Leaf": bart_zero}
                fig_1(barts, X_test, y_test, d, display)
                fig_2(bart_zero, X_test, y_test, d, display)
            #
            # fig, axs = plt.subplots(4, 2, figsize=(10, 22))
            # # fig.tight_layout()
            # fig.subplots_adjust(hspace=.8)
            #
            # barts = {"SGB": bart_sgb, "Single Leaf": bart_zero, "Random": bart_rand}
            #
            # # plot_chains_leaves(bart_zero, axs[0], X=X_test, y=y_test)
            # analyze_functional(barts, functional=importance_functional, axs=axs[0, 0:2], X=X_test, y=y_test,
            #                    name="Permutation importance")
            # analyze_functional(barts, functional=mse_functional, axs=axs[1, 0:2], X=X_test, y=y_test,
            #                    name="Test log-RMSE")
            # analyze_functional(barts, functional=n_leaves_functional, axs=axs[2, 0:2], X=X_test, y=y_test,
            #                    name="# Leaves")
            # # plot_within_chain(barts, axs[2], X=X_test, y=y_test)
            # plot_across_chains(barts, axs[3, 1], X=X_test, y=y_test, fig=fig)
            # axs[3, 0].axis('off')
            #
            # #
            # title = f"Dataset: {d[0].capitalize()}, (n, p) = ({n}, {p}), burn = {n_burn}"
            # plt.suptitle(title)
            # if display:
            #     plt.show()
            # else:
            #     plt.savefig(os.path.join(ART_PATH, "functional", f"{d[0]}_samples_{n_samples}_trees_{n_trees}_new.png"))
            # plt.close()


if __name__ == '__main__':
    main()
