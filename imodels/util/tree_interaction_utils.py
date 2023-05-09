import itertools
from typing import Set, Tuple

import numpy as np
import pandas as pd


def make_rj(n=300, p=50):
    """Generates data according to the model in Radchenko & James, 2010
    X_i ~ Unif([0,1]^p)
    y = sqrt(0.5)[sum_{i=1}^5 f_i(x) + f_1(x)f_2(x) + f_1(x)f_3(x)] + N(0,1)
    f_1(x) = x1, f_2(x) = (1+x2)^{-1}, f_3(x) = sin(x3), f_4(x) = e^x4, f_5(x) = x5^2
    function withing the sum are normalized

    Params
    ------
        n (int): number of sample
        p (int): number of features

    Returns
    -------
        Tuple[np.array, np.array]: design matrix and label vector

    """

    X = np.random.uniform(0, 1, size=(n, p))
    f_1 = X[:, 0]
    f_2 = (1 + X[:, 1]) ** (-1)
    f_3 = np.sin(X[:, 2])
    f_4 = np.exp(X[:, 3])
    f_5 = X[:, 4] ** (2)

    def _normalize_vec(v):
        return (v - np.mean(v)) / np.std(v)

    effects = _normalize_vec(f_1) + _normalize_vec(f_2) + _normalize_vec(f_3) + _normalize_vec(f_4) + _normalize_vec(
        f_5)
    interactions = f_1 * f_2 + f_1 * f_3

    y = effects + interactions + np.random.normal(size=n)

    return X, y


def make_vp(n=100, p=100):
    """Generates data according to https://arxiv.org/abs/1607.02670 (Sparse additive Gaussian process with soft interactions)
    X_i ~ N(0, I)
    y = x1 + x2^2 + x3 + x4^2 + x5 + x1x2 + x2x3 + x3x4 + N(0, 0.14)

    Args:
        n (int): number of sample
        p (int): number of features

    Returns:
        Tuple[np.array, np.array]: design matrix and label vector

    """
    X = np.random.normal(size=(n, p))
    effects = X[:, 0] + X[:, 1] ** 2 + X[:, 2] + X[:, 3] ** 2 + X[:, 4]
    interactions = X[:, 0] * X[:, 1] + X[:, 1] * X[:, 2] + X[:, 2] * X[:, 3]

    y = effects + interactions + np.random.normal(scale=0.14, size=n)

    return X, y


def get_gt(dataset_name):
    important_features = []
    interactions = []
    if dataset_name == "friedman1":
        important_features = [0, 1, 2, 3, 4]
        interactions = [(0, 1)]
    elif dataset_name == "radchenko_james":
        important_features = [0, 1, 2, 3, 4]
        interactions = [(0, 1), (0, 2)]
    elif dataset_name == "vo_pati":
        important_features = [0, 1, 2, 3, 4]
        interactions = [(0, 1), (1, 2), (2, 3)]

    return set(important_features), set(interactions)


def get_important_features(importance, k):
    return set(np.argsort(importance)[0:k])


def get_interacting_features(interaction, k):
    scores_list = []
    for ind_1, ind_2 in itertools.combinations(range(interaction.shape[0]), 2):
        scores_list.append([interaction[ind_1, ind_2], ind_1, ind_2])
    df = pd.DataFrame(scores_list)
    df = df.sort_values(0, ascending=False)
    interactions = []
    for i in range(k):
        interactions.append((df.iloc[i, 1], df.iloc[i, 2]))
    return set(interactions)


def interaction_fpr(i_gt: Set[Tuple], i_hat: Set[Tuple], p: int):
    if len(i_gt) == 0:
        return
    n_pairs = 0.5 * p * (p - 1)
    n_non_interacting_pairs = (n_pairs - len(i_gt))
    return len(i_hat.difference(i_gt)) / n_non_interacting_pairs


def interaction_tpr(i_gt: Set[Tuple], i_hat: Set[Tuple], p: int):
    if len(i_gt) == 0:
        return
    n_interactions = len(i_gt)
    return len(i_hat.intersection(i_gt)) / n_interactions


def interaction_f1(i_gt: Set[Tuple], i_hat: Set[Tuple], p: int):
    if len(i_gt) == 0:
        return
    recall = len(i_gt.intersection(i_hat)) / len(i_gt)
    precision = interaction_tpr(i_hat, i_gt, p)
    if recall + precision == 0:
        return 0
    return 2 * ((precision * recall) / (precision + recall))
