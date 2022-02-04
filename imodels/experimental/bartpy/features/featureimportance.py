from copy import deepcopy
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold

from bartpy.runner import run_models
from bartpy.sklearnmodel import SklearnModel


def original_model_rmse(model: SklearnModel,
                        X: Union[pd.DataFrame, np.ndarray],
                        y: np.ndarray,
                        n_k_fold_splits: int) -> List[float]:
    """
    Calculate the RMSE of the original model
    Used as a benchmark to compare against the null

    Parameters
    ----------
    model: SklearnModel
    X: np.ndarray
    y: np.ndarray
    n_k_fold_splits: int

    Returns
    -------
    List[float]
        List of the out of sample RMSEs for each fold of the covariate matrix
    """
    kf = KFold(n_k_fold_splits, shuffle=True)

    base_line_rmses = []

    for train_index, test_index in kf.split(X):
        model = deepcopy(model)
        model.fit(X[train_index], y[train_index])
        base_line_rmses.append(model.rmse(X[test_index], y[test_index]))

    return base_line_rmses


def null_rmse_distribution(model: SklearnModel,
                           X: Union[pd.DataFrame, np.ndarray],
                           y: np.ndarray,
                           variable: int,
                           n_k_fold_splits: int,
                           n_permutations: int=10) -> List[float]:
    """
    Calculate a null distribution on the RMSEs after scrambling a variable

    Works by randomly permuting y to remove any true dependence of y on X and calculating feature importance

    RMSEs are calculated on out of sample data

    Parameters
    ----------
    model: SklearnModel
        Model specification to work with
    X: np.ndarray
        Covariate matrix
    y: np.ndarray
        Target data
    variable: int
        Which column of the covariate matrix to permute
    n_k_fold_splits: int
        How many K-fold splits to make of the data
    n_permutations: int
        How many permutations to run
        The higher the number of permutations, the more accurate the null distribution, but the longer it will take to run
    Returns
    -------
    List[float]
        A list of predict set RMSEs - one entry for each fold of each permutation
    """

    kf = KFold(n_k_fold_splits, shuffle=True)

    permuted_train_X_s = []
    permuted_test_X_s = []
    train_y_s = []
    test_y_s = []

    for train_index, test_index in kf.split(X):
        for _ in range(n_permutations):
            permuted_X = deepcopy(X)
            permuted_X[:, variable] = np.random.permutation(permuted_X[:, variable])
            permuted_train_X_s.append(permuted_X[train_index])
            permuted_test_X_s.append(permuted_X[train_index])
            train_y_s.append(y[train_index])
            test_y_s.append(y[test_index])

    fit_models = run_models(model, permuted_train_X_s, train_y_s)

    rmses = []
    for i, m in enumerate(fit_models):
        rmses.append(m.rmse(permuted_test_X_s[i], test_y_s[i]))
    return rmses


def feature_importance(model: SklearnModel,
                       X: Union[pd.DataFrame, np.ndarray],
                       y: np.ndarray,
                       variable: int,
                       n_k_fold_splits: int=2,
                       n_permutations: int=10) -> Tuple[List[float], List[float]]:
    """
    Assess the importance to the RMSE of a single column of the covariate matrix

    Parameters
    ----------
    model: SklearnModel
        An instance of the model with the parameters to train with
        The model instance itself doesn't have to be trained
    X: np.ndarray
        Covariate matrix
    y: np.ndarray
        Target array
    variable: int
        Which column of the covariate matrix to assess
    n_k_fold_splits: int
        How many folds to take of the covariate matrix
    n_permutations: int
        How many runs of the model to make when generating the null distribution
        The more permutations, the better the approximation to the true null, but the more computation will be required

    Returns
    -------
    Tuple[List[float], List[float]]
        First entry is a List of the RMSEs of the original model
        Second entry is a list of RMSEs of the null distribution
    """
    original_model = original_model_rmse(model, X, y, n_k_fold_splits)
    null_distribution = null_rmse_distribution(model, X, y, variable, n_k_fold_splits, n_permutations)

    plt.hist(null_distribution, label="Null Distribution")
    plt.hist(original_model, label="Original Model")
    plt.title("RMSE of full model against null distribution for variable {}".format(variable))
    plt.xlabel("RMSE")
    plt.ylabel("Density")

    return original_model, null_distribution
