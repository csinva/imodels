from os.path import join as oj
from typing import Tuple

import numpy as np
import pandas as pd
import sklearn.datasets
from pmlb import fetch_data
from scipy.sparse import issparse
from sklearn.datasets import fetch_openml

from experiments.util import DATASET_PATH


def get_openml_dataset(data_id: int) -> pd.DataFrame:
    dataset = fetch_openml(data_id=data_id, as_frame=False)
    X = dataset.data
    if issparse(X):
        X = X.toarray()
    y = (dataset.target == dataset.target[0]).astype(int)
    feature_names = dataset.feature_names

    target_name = dataset.target_names
    if target_name[0].lower() == 'class':
        target_name = [dataset.target[0]]

    X_df = pd.DataFrame(X, columns=feature_names)
    y_df = pd.DataFrame(y, columns=target_name)
    return pd.concat((X_df, y_df), axis=1)


def define_openml_outcomes(y, data_id: str):
    if data_id == '59':  # ionosphere, positive is "good" class
        y = (y == 'g').astype(int)
    return y


def get_clean_dataset(dataset: str = None, data_source: str = 'local') -> Tuple[np.ndarray, np.ndarray, list]:
    """Return

    Parameters
    ----------
    dataset: str
        csv_file path or dataset id if data_source is specified
    data_source: str
        options: 'local', 'pmlb', 'sklearn'
        boolean - whether dataset is a pmlb dataset name

    Returns
    -------
    X, y, feature_names
    """
    assert data_source in ['local', 'pmlb', 'sklearn', 'openml']
    if data_source == 'local':
        df = pd.read_csv(dataset)
        X, y = df.iloc[:, :-1].values, df.iloc[:, -1].values
        feature_names = df.columns.values[:-1]
        return np.nan_to_num(X.astype('float32')), y, feature_names
    elif data_source == 'pmlb':
        feature_names = fetch_data(dataset, return_X_y=False, local_cache_dir=oj(DATASET_PATH, 'pmlb_data')).columns
        X, y = fetch_data(dataset, return_X_y=True, local_cache_dir=oj(DATASET_PATH, 'pmlb_data'))
        return X, y, feature_names
    elif data_source == 'sklearn':
        if dataset == 'diabetes':
            data = sklearn.datasets.load_diabetes()
        elif dataset == 'california_housing':
            data = sklearn.datasets.fetch_california_housing(data_home=oj(DATASET_PATH, 'sklearn_data'))
        return data['data'], data['target'], data['feature_names']
    elif data_source == 'openml':  # note this api might change in newer sklearn - should give dataset-id not name
        data = sklearn.datasets.fetch_openml(data_id=dataset, data_home=oj(DATASET_PATH, 'openml_data'))
        X, y, feature_names = data['data'], data['target'], data['feature_names']
        y = define_openml_outcomes(y, dataset)
        if issparse(X):
            X = X.toarray()
        return X, y, feature_names
