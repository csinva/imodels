import os.path
from os.path import join as oj
from typing import Tuple

import numpy as np
import pandas as pd
import requests
import sklearn.datasets
from scipy.sparse import issparse
from sklearn.datasets import fetch_openml

from ..util.tree_interaction_utils import make_rj, make_vp


def _define_openml_outcomes(y, data_id: str):
    if data_id == '59':  # ionosphere, positive is "good" class
        y = (y == 'g').astype(int)
    if data_id == '183':  # abalone, need to convert strings to floats
        y = y.astype(float)
    return y


def _clean_feat_names(feature_names):
    # shouldn't start with a digit
    return ['X_' + x if x[0].isdigit()
            else x
            for x in feature_names]


def _clean_features(X):
    if issparse(X):
        X = X.toarray()
    try:
        return X.astype(float)
    except:
        for j in range(X.shape[1]):
            try:
                X[:, j].astype(float)
            except:
                # non-numeric get replaced with numerical values
                classes, X[:, j] = np.unique(X[:, j], return_inverse=True)
    return X.astype(float)


def get_clean_dataset(dataset_name: str, data_source: str = 'imodels', data_path='data', convertna=True) -> Tuple[
    np.ndarray, np.ndarray, list]:
    """Fetch clean data (as numpy arrays) from various sources including imodels, pmlb, openml, and sklearn.
    If data is not downloaded, will download and cache. Otherwise will load locally.
    Cleans features so that they are type float and features names don't start with a digit.

    Parameters
    ----------
    dataset_name: str
        dataset_name - unique dataset identifier (see https://github.com/csinva/imodels-data for unique identifiers)
    data_source: str
        options: 'imodels', 'pmlb', 'sklearn', 'openml', 'synthetic'
    data_path: str
        path to load/save data (default: 'data')

    Returns
    -------
    X: np.ndarray
        features
    y: np.ndarray
        outcome
    feature_names: list

    Example
    -------
    ```
    # download compas dataset from imodels
    X, y, feature_names = imodels.get_clean_dataset('compas_two_year_clean', data_source='imodels')
    # download ionosphere dataset from pmlb
    X, y, feature_names = imodels.get_clean_dataset('ionosphere', data_source='pmlb')
    # download liver dataset from openml
    X, y, feature_names = imodels.get_clean_dataset('8', data_source='openml')
    # download ca housing from sklearn
    X, y, feature_names = imodels.get_clean_dataset('california_housing', data_source='sklearn')
    ```
    """
    assert data_source in ['imodels', 'pmlb', 'sklearn', 'openml', 'synthetic'], data_source + ' not correct'
    if data_source == 'imodels':
        if not dataset_name.endswith('csv'):
            dataset_name = dataset_name + '.csv'
        if not os.path.isfile(dataset_name):
            _download_imodels_dataset(dataset_name, data_path)
        df = pd.read_csv(oj(data_path, 'imodels_data', dataset_name))
        X, y = df.iloc[:, :-1].values, df.iloc[:, -1].values
        feature_names = df.columns.values[:-1]
        if convertna:
            X = np.nan_to_num(X.astype('float32'))
        return X, y, _clean_feat_names(feature_names)
    elif data_source == 'pmlb':
        from pmlb import fetch_data
        feature_names = list(
            fetch_data(dataset_name, return_X_y=False, local_cache_dir=oj(data_path, 'pmlb_data')).columns)
        feature_names.remove('target')
        X, y = fetch_data(dataset_name, return_X_y=True, local_cache_dir=oj(data_path, 'pmlb_data'))
        if np.unique(y).size == 2:  # if binary classification, ensure that the classes are 0 and 1
            y -= np.min(y)
        return _clean_features(X), y, _clean_feat_names(feature_names)
    elif data_source == 'sklearn':
        if dataset_name == 'diabetes':
            data = sklearn.datasets.load_diabetes()
        elif dataset_name == 'california_housing':
            data = sklearn.datasets.fetch_california_housing(data_home=oj(data_path, 'sklearn_data'))
        return data['data'], data['target'], _clean_feat_names(data['feature_names'])
    elif data_source == 'openml':  # note this api might change in newer sklearn - should give dataset-id not name
        data = sklearn.datasets.fetch_openml(data_id=dataset_name, data_home=oj(data_path, 'openml_data'))
        X, y, feature_names = data['data'], data['target'], _clean_feat_names(data['feature_names'])
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        y = _define_openml_outcomes(y, dataset_name)
        return _clean_features(X), y, _clean_feat_names(feature_names)
    elif data_source == 'synthetic':
        if dataset_name == 'friedman1':
            X, y = sklearn.datasets.make_friedman1(n_samples=200, n_features=10)
        elif dataset_name == 'friedman2':
            X, y = sklearn.datasets.make_friedman2(n_samples=200)
        elif dataset_name == 'friedman3':
            X, y = sklearn.datasets.make_friedman3(n_samples=200)
        elif dataset_name == "radchenko_james":
            X, y = make_rj()
        elif dataset_name == "vo_pati":
            X, y = make_vp()
        return X, y, ['X_' + str(i + 1) for i in range(X.shape[1])]


def _get_openml_dataset(data_id: int) -> pd.DataFrame:
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


def _download_imodels_dataset(dataset_fname, data_path: str):
    dataset_fname = dataset_fname.split('/')[-1]  # remove anything about the path
    download_path = f'https://raw.githubusercontent.com/csinva/imodels-data/master/data_cleaned/{dataset_fname}'
    r = requests.get(download_path)
    if r.status_code == 404:
        raise Exception(
            f'404 Error for dataset {dataset_fname} (see valid files at https://github.com/csinva/imodels-data/tree/master/data_cleaned)')

    os.makedirs(oj(data_path, 'imodels_data'), exist_ok=True)
    with open(oj(data_path, 'imodels_data', dataset_fname), 'w') as f:
        f.write(r.text)
