# imports
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from imodels.tree.rf_plus.data_transformations.block_transformers import _blocked_train_test_split

def _fast_r2_score(y_true, y_pred, multiclass=False):
    """
    Evaluates the r-squared value between the observed and estimated responses.
    Equivalent to sklearn.metrics.r2_score but without the robust error
    checking, thus leading to a much faster implementation (at the cost of
    this error checking). For multi-class responses, returns the mean
    r-squared value across each column in the response matrix.

    Parameters
    ----------
    y_true: array-like of shape (n_samples, n_targets)
        Observed responses.
    y_pred: array-like of shape (n_samples, n_targets)
        Predicted responses.
    multiclass: bool
        Whether or not the responses are multi-class.

    Returns
    -------
    Scalar quantity, measuring the r-squared value.
    """
    numerator = ((y_true - y_pred) ** 2).sum(axis=0, dtype=np.float64)
    denominator = ((y_true - np.mean(y_true, axis=0)) ** 2). \
        sum(axis=0, dtype=np.float64)
    if multiclass:
        return np.mean(1 - numerator / denominator)
    else:
        return 1 - numerator / denominator


def _neg_log_loss(y_true, y_pred):
    """
    Evaluates the negative log-loss between the observed and
    predicted responses.

    Parameters
    ----------
    y_true: array-like of shape (n_samples, n_targets)
        Observed responses.
    y_pred: array-like of shape (n_samples, n_targets)
        Predicted probabilies.

    Returns
    -------
    Scalar quantity, measuring the negative log-loss value.
    """
    return -log_loss(y_true, y_pred)


def _check_Xy(X,y):
    if isinstance(X, pd.DataFrame):
        feature_names_ = list(X.columns)
        X_array = X.values
    elif isinstance(X, np.ndarray):
        X_array = X
    else:
        raise ValueError("Input X must be a pandas DataFrame or numpy array.")
    if isinstance(y, pd.DataFrame):
        y = y.values.ravel()
    elif not isinstance(y, np.ndarray):
        raise ValueError("Input y must be a pandas DataFrame or numpy array.")
    return X_array, y

def _check_X(X):
    if isinstance(X, pd.DataFrame):
        feature_names_ = list(X.columns)
        X_array = X.values
    elif isinstance(X, np.ndarray):
        X_array = X
    else:
        raise ValueError("Input X must be a pandas DataFrame or numpy array.")
    return X_array


def _get_sample_split_data(blocked_data, y, random_state):
    in_bag_blocked_data, oob_blocked_data, y_in_bag, y_oob, in_bag_indices, oob_indices = _blocked_train_test_split(blocked_data,y,random_state)
    return in_bag_blocked_data, oob_blocked_data, y_in_bag, y_oob, in_bag_indices,oob_indices


    