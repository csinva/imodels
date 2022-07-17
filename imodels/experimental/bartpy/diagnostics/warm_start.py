import copy
import logging
import argparse
import os.path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn import datasets, model_selection
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor

from imodels import get_clean_dataset
from imodels.experimental.bartpy.node import DecisionNode
from ..sklearnmodel import BARTChainCV, BART
from . import DATASETS_SYNTHETIC, DATASETS_REGRESSION, ART_PATH

LOGGER = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='BART Research motivation')
    parser.add_argument('datasets', metavar='datasets', type=str, nargs='+',
                        help='datasets to run sim over')
    parser.add_argument('n_trees', metavar='n_trees', type=int,
                        help='number of trees')
    parser.add_argument('--n_samples', metavar='n_samples', type=int,
                        help='number of mcmc samples')
    parser.add_argument('--analysis', metavar='analysis', type=str,
                        help='analysis type (s - synthetic or i - initialization)', choices={"s", "i"})
    parser.add_argument('--display', action='store_true',
                        help='display figure')

    args = parser.parse_args()
    return args


def get_survivors(bart_cv: BARTChainCV):
    sample = bart_cv.model_samples[-1]
    survivors_data = {"tree": [], "depth": [], "variable": [], "value": []}  # , "iteration":[]}
    # for iter,sample in enumerate(bart_cv.model_samples):
    for tree_num, tree in enumerate(sample.trees):
        for node in tree.nodes:
            if node.original:
                var = node.splitting_variable if isinstance(node, DecisionNode) else -1
                survivors_data['tree'].append(tree_num)
                survivors_data['depth'].append(node.depth)
                survivors_data['variable'].append(var)
                survivors_data['value'].append(node.current_value)
                # survivors_data['iteration'].append(iter)

    return pd.DataFrame(survivors_data)


def survival_sample_size_analysis(d, n_samples, n_burn, n_chains, n_trees):
    n_survivors = []
    data_sizes = [100, 500, 1000, 5000, 10000, 50000, 100000]
    X, y, feat_names = get_clean_dataset(d[1], data_source=d[2], n_samples=200)
    bart = BART(classification=False, n_samples=1, n_burn=1,
                n_chains=1, n_trees=1)
    bart.fit(X, y)

    model = DecisionTreeRegressor(max_depth=4) if n_trees == 1 else GradientBoostingRegressor(n_estimators=n_trees)
    model.fit(X, bart.data.y.values)

    for i, sample_size in enumerate(data_sizes):
        X, y, feat_names = get_clean_dataset(d[1], data_source=d[2], n_samples=sample_size)
        bart_cv = BARTChainCV(classification=False, n_samples=n_samples, n_burn=n_burn,
                              n_chains=n_chains, n_trees=n_trees)
        X_train, X_test, y_train, y_test = model_selection.train_test_split(
            X, y, test_size=0.3, random_state=i)
        bart_cv.fit(X_train, y_train, sgb_init=True, model=model)
        survivors = get_survivors(bart_cv)
        n_survivors.append(survivors.shape[0])
    fig, ax = plt.subplots(1)
    ax.scatter(data_sizes, np.log(n_survivors))
    ax.set_ylabel("# of Splits Survived")
    ax.set_xlabel("Sample Size (log scale)")


def survival_real_ds_analysis(n_samples, n_burn, n_chains, n_trees):
    n_survivors = []
    data_sizes = []

    for d in DATASETS_REGRESSION:
        X, y, feat_names = get_clean_dataset(d[1], data_source=d[2])
        data_sizes.append(len(y))
        bart = BART(classification=False, n_samples=1, n_burn=1,
                    n_chains=1, n_trees=1)
        bart.fit(X, y)

        model = DecisionTreeRegressor(max_depth=4) if n_trees == 1 else GradientBoostingRegressor(n_estimators=n_trees)
        model.fit(X, bart.data.y.values)

        bart_cv = BARTChainCV(classification=False, n_samples=n_samples, n_burn=n_burn,
                              n_chains=n_chains, n_trees=n_trees)
        X_train, X_test, y_train, y_test = model_selection.train_test_split(
            X, y, test_size=0.3, random_state=42)
        bart_cv.fit(X_train, y_train, sgb_init=True, model=model)
        survivors = get_survivors(bart_cv)
        n_survivors.append(survivors.shape[0])
    fig, ax = plt.subplots(1)
    ax.scatter(data_sizes, np.log(n_survivors))
    for i, d in enumerate(DATASETS_REGRESSION):
        ax.annotate(d[0], (data_sizes[i], np.log(n_survivors)[i]))
    ax.set_ylabel("# of Splits Survived")
    ax.set_xlabel("Sample Size (log scale)")


def survival_analysis(d, n_samples, n_burn, n_chains, n_trees):
    is_synthetic = len([ds for ds in DATASETS_SYNTHETIC if ds[0] == d]) > 0

    config_str = f"trees_{n_trees}_burn_{n_burn}_chains_{n_chains}_samples_{n_samples}"
    config_path = os.path.join(ART_PATH, "warm_start", config_str)
    if not os.path.exists(config_path):
        os.mkdir(config_path)
    fig_name = f"{d}.png" if is_synthetic else "real_ds.png"
    if is_synthetic:
        d_s = [ds for ds in DATASETS_SYNTHETIC if ds[0] == d][0]
        survival_sample_size_analysis(d_s, n_samples, n_burn, n_chains, n_trees)
    else:
        survival_real_ds_analysis(n_samples, n_burn, n_chains, n_trees)
    plt.savefig(os.path.join(config_path, fig_name))


def compare_ds(d, n_rep, n_samples, n_burn, n_chains, n_trees):
    X, y, feat_names = get_clean_dataset(d[1], data_source=d[2], n_samples=3000)
    bart_err = []
    bart_cv_err = []
    config_str = f"trees_{n_trees}_burn_{n_burn}_chains_{n_chains}_samples_{n_samples}"
    for i in range(n_rep):
        bart_cv = BARTChainCV(classification=False, n_samples=n_samples, n_burn=n_burn,
                              n_chains=n_chains, n_trees=n_trees)
        bart = BART(classification=False, n_samples=n_samples, n_burn=n_burn,
                    n_chains=n_chains, n_trees=n_trees)

        X_train, X_test, y_train, y_test = model_selection.train_test_split(
            X, y, test_size=0.3, random_state=i)
        bart_cv.fit(X_train, y_train, sgb_init=True)
        bart.fit(X_train, y_train)
        bart.prediction_intervals(X_test)

        survivors = get_survivors(bart_cv)
        if not os.path.exists(os.path.join(ART_PATH, "warm_start", config_str)):
            os.mkdir(os.path.join(ART_PATH, "warm_start", config_str))
        survivors.to_csv(os.path.join(ART_PATH, "warm_start", f"{d[0]}_{i}.csv"))

        bart_err.append(np.sqrt(mean_squared_error(bart.predict(X_test), y_test)))
        bart_cv_err.append(np.sqrt(mean_squared_error(bart_cv.predict(X_test), y_test)))
    return pd.DataFrame({"bart": bart_err, "bart_cv": bart_cv_err})


def main():
    n_burn = int(1e+5)
    args = parse_args()
    n_samples = args.n_samples  # 0000  # 7500  # 00
    n_trees = args.n_trees
    n_chains = 1

    datasets = args.datasets
    # survival_real_ds_analysis(n_samples, n_burn, n_chains, n_trees)
    # config_str = f"trees_{n_trees}_burn_{n_burn}_chains_{n_chains}_samples_{n_samples}"
    # config_path = os.path.join(ART_PATH, "warm_start", config_str)
    # if not os.path.exists(config_path):
    #     os.mkdir(config_path)
    # plt.savefig(os.path.join(config_path, f"real_datasets.png"))
    for ds in datasets:

        # d = [d for d in DATASETS_SYNTHETIC if d[0] == ds][0]

        # n_rep = args.n_rep
        survival_analysis(ds, n_samples, n_burn, n_chains, n_trees)
        # config_str = f"trees_{n_trees}_burn_{n_burn}_chains_{n_chains}_samples_{n_samples}"
        # config_path = os.path.join(ART_PATH, "warm_start", config_str)
        # if not os.path.exists(config_path):
        #     os.mkdir(config_path)
        # plt.savefig(os.path.join(config_path, f"{ds}.png"))
        # compare_ds(d, n_rep, n_samples, n_burn, n_chains, n_trees)


#

if __name__ == '__main__':
    main()
