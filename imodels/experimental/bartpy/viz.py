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
from .sklearnmodel import BARTChainCV, BART
from diagnostics import DATASETS_SYNTHETIC, DATASETS_REGRESSION, ART_PATH
#
from imodels.tree.viz_utils import extract_figs_tree


def main():
    for d in DATASETS_SYNTHETIC:
        X, y, feat_names = get_clean_dataset(d[1], data_source=d[2])
        # data_sizes.append(len(y))
        bart = BART(classification=False, n_samples=1, n_burn=1,
                    n_chains=1, n_trees=1)
        bart.fit(X, y)

        tree = bart.trees[0]

        extract_figs_tree(tree, 1)


if __name__ == '__main__':
    main()
