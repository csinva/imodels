import copy
import logging
import argparse
import os.path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.lines import Line2D

from sklearn import datasets, model_selection
from sklearn.metrics import mean_squared_error

from imodels import get_clean_dataset
from imodels.experimental.bartpy.node import DecisionNode
from ..sklearnmodel import BARTChainCV, BART
from . import DATASETS_SYNTHETIC, DATASETS_REGRESSION, ART_PATH

LOGGER = logging.getLogger(__name__)
COV_DIR = os.path.join(ART_PATH, "coverage")


def parse_args():
    parser = argparse.ArgumentParser(description='BART Warm Start Study')
    parser.add_argument('dataset', metavar='dataset', type=str,
                        help='dataset to run sim over')
    parser.add_argument('n_trees', metavar='n_trees', type=int,
                        help='number of trees')
    parser.add_argument('--n_samples', metavar='n_samples', type=int,
                        help='number of mcmc samples')
    parser.add_argument('--analysis', metavar='analysis', type=str,
                        help='analysis type (s - synthetic or i - initialization)', choices={"s", "i"})

    args = parser.parse_args()
    return args


def _get_coverage(predictions, y_true):
    u, l = np.max(predictions, axis=0), np.min(predictions, axis=0)
    coverage = np.mean(np.logical_and(y_true < u, y_true > l))
    return coverage


def _get_rmse(predictions, y_true):
    mean_pred = np.mean(predictions, axis=0)
    rmse = np.sqrt(mean_squared_error(mean_pred, y_true))
    return rmse


def ds_study(d, n_samples, n_burn, n_chains, n_trees):
    X, y, feat_names = get_clean_dataset(d[1], data_source=d[2], n_samples=3000)
    # perf = {chain: {} for chain in range(n_chains)}

    bart = BART(classification=False, n_samples=n_samples, n_burn=n_burn,
                n_chains=n_chains, n_trees=n_trees)

    bart_sc = BART(classification=False, n_samples=n_samples * n_chains, n_burn=n_burn,
                   n_chains=1, n_trees=n_trees)



    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, test_size=0.3, random_state=42)
    bart.fit(X_train, y_train)
    bart_sc.fit(X_train, y_train)

    chain_num = []
    color = iter(cm.rainbow(np.linspace(0, 1, bart.n_chains)))

    for chain in range(bart.n_chains):
        # predictions = bart.chain_precitions(X_test, chain)
        # chains_rmse += [_get_rmse(predictions[0:i], y_test) for i in range(1, bart.n_samples + 1)]
        # chains_coverage += [_get_coverage(predictions[0:i], y_test) for i in range(1, bart.n_samples + 1)]
        chain_num += [next(color)] * bart.n_samples

    predictions = bart.chain_precitions(X_test, list(np.arange(bart.n_chains)))
    rmse = [_get_rmse(predictions[0:i], y_test) for i in range(1, len(predictions) + 1)]
    coverage = [_get_coverage(predictions[0:i], y_test) for i in range(1, len(predictions) + 1)]

    predictions = bart_sc.chain_precitions(X_test, list(np.arange(bart_sc.n_chains)))
    rmse_sc = [_get_rmse(predictions[0:i], y_test) for i in range(1, len(predictions) + 1)]
    coverage_sc = [_get_coverage(predictions[0:i], y_test) for i in range(1, len(predictions) + 1)]

    return {"bart":pd.DataFrame({"rmse": rmse, "coverage": coverage, "chain_number": chain_num}),
            "bart_sc":pd.DataFrame({"rmse": rmse_sc, "coverage": coverage_sc, "chain_number": chain_num})}


def main():
    if not os.path.exists(COV_DIR):
        os.mkdir(COV_DIR)

    n_burn = int(1e+05)
    args = parse_args()
    n_samples = args.n_samples  # 0000  # 7500  # 00

    ds = args.dataset
    is_synthetic = args.analysis == "s"
    datasets = DATASETS_SYNTHETIC if is_synthetic else DATASETS_REGRESSION
    d = [d for d in datasets if d[0] == ds][0]

    n_trees = args.n_trees
    n_chains = 10
    performance = ds_study(d, n_samples, n_burn, n_chains, n_trees)
    perf_bart = performance['bart']
    perf_bart_sc = performance['bart_sc']
    # df = pd.DataFrame(performance)
    fig, ax = plt.subplots(1)
    ax.scatter(perf_bart.loc[:, 'coverage'], perf_bart.loc[:, 'rmse'], c=perf_bart.loc[:, "chain_number"])
    ax.scatter(perf_bart_sc.loc[:, 'coverage'], perf_bart_sc.loc[:, 'rmse'], c="black", alpha=0.3, marker="v")

    ax.set_xlabel("coverage")
    ax.set_ylabel("rmse")
    color = list(iter(cm.rainbow(np.linspace(0, 1, n_chains))))

    custom_lines = [Line2D([0], [0], color=c, lw=4, label=f"Chain {i}") for i, c in enumerate(color)]
    ax.legend(custom_lines, [f"Chain {i}" for i in range(n_chains)])
    # plt.show()
    # return
    config = f"burn_{n_burn}_trees_{n_trees}_chains_{n_chains}"
    if not os.path.exists(os.path.join(COV_DIR, config)):
        os.mkdir(os.path.join(COV_DIR, config))
    plt.savefig(os.path.join(COV_DIR, config, f"{ds}_{args.analysis}.png"))
    # LOGGER.info()


if __name__ == '__main__':
    main()
