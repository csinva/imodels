import copy
import logging
import argparse
import os.path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

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


    args = parser.parse_args()
    return args


def ds_study(d, n_samples, n_burn, n_chains, n_trees):
    X, y, feat_names = get_clean_dataset(d[1], data_source=d[2], n_samples=3000)
    perf = {chain: {} for chain in range(n_chains)}

    bart = BART(classification=False, n_samples=n_samples, n_burn=n_burn,
                n_chains=n_chains, n_trees=n_trees)

    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, test_size=0.3, random_state=42)
    bart.fit(X_train, y_train)
    for chain in range(bart.n_chains):
        intervals = bart.prediction_intervals(X_test, list(np.arange(chain+1)))
        coverage = np.mean(np.logical_and(y_test < intervals[:, 0], y_test > intervals[:, 1]))
        point_prediction = bart.predict_chain(X_test, list(np.arange(chain+1)))
        rmse = np.sqrt(mean_squared_error(point_prediction, y_test))

        perf[chain]['coverage'] = coverage
        perf[chain]['rmse'] = rmse
    return perf


def main():
    n_burn = 2000
    args = parse_args()
    n_samples = args.n_samples  # 0000  # 7500  # 00

    ds = args.dataset
    d = [d for d in DATASETS_SYNTHETIC if d[0] == ds][0]

    n_trees = args.n_trees
    n_chains = 30
    performance = ds_study(d, n_samples, n_burn, n_chains, n_trees)
    df = pd.DataFrame(performance)
    fig, ax = plt.subplots(1)
    ax.plot(df.loc['coverage', :], df.loc[ 'rmse', :])
    ax.set_xlabel("coverage")
    ax.set_ylabel("rmse")
    plt.show()
    # LOGGER.info()


if __name__ == '__main__':
    main()
