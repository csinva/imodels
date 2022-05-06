import copy
import logging
import argparse
import os.path

import numpy as np
import pandas as pd

from sklearn import datasets, model_selection
from sklearn.metrics import mean_squared_error

from imodels import get_clean_dataset
from imodels.experimental.bartpy.node import DecisionNode
from ..sklearnmodel import BARTChainCV, BART
from . import DATASETS_SYNTHETIC, DATASETS_REGRESSION, ART_PATH

LOGGER = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='BART Warm Start Study')
    parser.add_argument('dataset', metavar='dataset', type=str,
                        help='dataset to run sim over')
    parser.add_argument('n_trees', metavar='n_trees', type=int,
                        help='number of trees')
    parser.add_argument('--n_samples', metavar='n_samples', type=int,
                        help='number of mcmc samples')
    parser.add_argument('--n_rep', metavar='n_rep', type=int,
                        help='number of splitting repetitions')

    args = parser.parse_args()
    return args


def get_survivors(bart_cv: BARTChainCV):
    # last_sample = bart_cv.model_samples[-1]
    survivors_data = {"tree": [], "depth": [], "variable": [], "value": [], "iteration":[]}
    for iter,sample in enumerate(bart_cv.model_samples):
        for tree_num, tree in enumerate(sample.trees):
            for node in tree.nodes:
                if node.original:
                    var = node.splitting_variable if isinstance(node, DecisionNode) else -1
                    survivors_data['tree'].append(tree_num)
                    survivors_data['depth'].append(node.depth)
                    survivors_data['variable'].append(var)
                    survivors_data['value'].append(node.current_value)
                    survivors_data['iteration'].append(iter)

    return pd.DataFrame(survivors_data)


def compare_ds(d, n_rep, n_samples, n_burn, n_chains, n_trees):
    X, y, feat_names = get_clean_dataset(d[1], data_source=d[2], p=3000)
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
        if not os.path.exists(os.path.join(ART_PATH, "warm_start")):
            os.mkdir(os.path.join(ART_PATH, "warm_start"))
        survivors.to_csv(os.path.join(ART_PATH, "warm_start", f"{d[0]}_{i}_{config_str}.csv"))

        bart_err.append(np.sqrt(mean_squared_error(bart.predict(X_test), y_test)))
        bart_cv_err.append(np.sqrt(mean_squared_error(bart_cv.predict(X_test), y_test)))
    return pd.DataFrame({"bart": bart_err, "bart_cv": bart_cv_err})


def main():
    n_burn = 0
    args = parse_args()
    n_samples = args.n_samples  # 0000  # 7500  # 00

    ds = args.dataset
    d = [d for d in DATASETS_SYNTHETIC if d[0] == ds][0]

    n_trees = args.n_trees
    n_chains = 1
    n_rep = args.n_rep
    compare_ds(d, n_rep, n_samples, n_burn, n_chains, n_trees)


if __name__ == '__main__':
    main()
