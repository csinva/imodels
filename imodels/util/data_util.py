import os.path
from os.path import join as oj
from typing import Tuple

import numpy as np
import pandas as pd
import requests
import sklearn.datasets
from scipy.sparse import issparse
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from imodels.util.tree_interaction_utils import make_rj, make_vp


DSET_CLASSIFICATION_KWARGS = {
    # classification
    'iris': {'dataset_name': 61, 'data_source': 'openml'},
    "pima_diabetes": {"dataset_name": 40715, "data_source": "openml"},
    "sonar": {"dataset_name": "sonar", "data_source": "pmlb"},
    "heart": {"dataset_name": "heart", "data_source": "imodels"},
    "diabetes": {"dataset_name": "diabetes", "data_source": "pmlb"},
    "breast_cancer_recurrence": {
        "dataset_name": "breast_cancer",
        "data_source": "imodels",
    },
    "breast_cancer_wisconsin": {
        "dataset_name": "breast_cancer",
        "data_source": "sklearn",
    },
    "credit_g": {"dataset_name": "credit_g", "data_source": "imodels"},
    "juvenile": {"dataset_name": "juvenile_clean", "data_source": "imodels"},
    "compas": {"dataset_name": "compas_two_year_clean", "data_source": "imodels"},
    "fico": {"dataset_name": "fico", "data_source": "imodels"},
    "readmission": {
        "dataset_name": "readmission_clean",
        "data_source": "imodels",
    },  # big, 100k points
    # big, 1e6 points
    "adult": {"dataset_name": 1182, "data_source": "openml"},
    # CDI classification
    "csi_pecarn": {"dataset_name": "csi_pecarn_pred", "data_source": "imodels"},
    "iai_pecarn": {"dataset_name": "iai_pecarn_pred", "data_source": "imodels"},
    "tbi_pecarn": {"dataset_name": "tbi_pecarn_pred", "data_source": "imodels"},
}

DSET_REGRESSION_KWARGS = {
    # regression
    "bike_sharing": {"dataset_name": 42712, "data_source": "openml"},
    "friedman1": {"dataset_name": "friedman1", "data_source": "synthetic"},
    "friedman2": {"dataset_name": "friedman2", "data_source": "synthetic"},
    "friedman3": {"dataset_name": "friedman3", "data_source": "synthetic"},
    "diabetes_regr": {"dataset_name": "diabetes", "data_source": "sklearn"},
    "abalone": {"dataset_name": 183, "data_source": "openml"},
    "echo_months": {"dataset_name": "1199_BNG_echoMonths", "data_source": "pmlb"},
    "satellite_image": {"dataset_name": "294_satellite_image", "data_source": "pmlb"},
    "california_housing": {
        "dataset_name": "california_housing",
        "data_source": "sklearn",
    },
    # 'breast_tumor': {'dataset_name': '1201_BNG_breastTumor', 'data_source': 'pmlb' # v big
}
DSET_CLASSIFICATION_MULTITASK_NAMES = [
    '3s-bbc1000', '3s-guardian1000', '3s-inter3000', '3s-reuters1000',
    'birds', 'cal500', 'chd_49', 'corel16k001', 'corel16k002',
    'corel16k003', 'corel16k004', 'corel16k005', 'corel16k006',
    'corel16k007', 'corel16k008', 'corel16k009', 'corel16k010',
    'corel5k', 'emotions', 'flags', 'foodtruck', 'genbase', 'image',
    'mediamill', 'scene', 'stackex_chemistry', 'stackex_chess',
    'stackex_cooking', 'stackex_cs', 'water-quality', 'yeast', 'yelp']
DSET_CLASSIFICATION_MULTITASK_KWARGS = {
    name + '_multitask': {"dataset_name": name, "data_source": "imodels-multitask"}
    for name in DSET_CLASSIFICATION_MULTITASK_NAMES
}
DSET_KWARGS = {
    **DSET_CLASSIFICATION_KWARGS, **DSET_REGRESSION_KWARGS,
    **DSET_CLASSIFICATION_MULTITASK_KWARGS}


def get_clean_dataset(
    dataset_name: str,
    data_source: str = "imodels",
    data_path=os.path.expanduser("~/cache_imodels_data"),
    convertna: bool = True,
    test_size: float = None,
    random_state: int = 42,
    verbose=True,
    return_target_col_names: bool = False,
    override_cache: bool = False,
) -> Tuple[np.ndarray, np.ndarray, list]:
    """Fetch clean data (as numpy arrays) from various sources including imodels, pmlb, openml, and sklearn.
    If data is not downloaded, will download and cache. Otherwise will load locally.
    Cleans features so that they are type float and features names don't start with a digit.

    Parameters
    ----------
    dataset_name: str
        Checks for unique identifier in imodels.util.data_util.DSET_KWARGS
        Otherwise, unique dataset identifier (see https://github.com/csinva/imodels-data for unique identifiers)
    data_source: str
        options: 'imodels', 'pmlb', 'sklearn', 'openml', 'synthetic'
    data_path: str
        path to load/save data (default: 'data')
    test_size: float, optional
        if not None, will split data into train and test sets (with fraction test_size in test set)
        & change the return signature to `X_train, X_test, y_train, y_test, feature_names`
    random_state: int, optional
        if test_size is not None, will use this random state to split data
    return_target_col_names: bool, optional
        if True, will return target columns for multitask datasets as final return value
    override_cache: bool, False
        if True, will override the downloaded cache for a dataset


    Returns
    -------
    X: np.ndarray
        features
    y: np.ndarray
        outcome
    feature_names: list
    (if passing test_size, will return more outputs)
    (if multitask dataset, will return target_col_names as well)

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
    if dataset_name in DSET_KWARGS:
        if verbose:
            data_source = DSET_KWARGS[dataset_name]["data_source"]
            dataset_name = DSET_KWARGS[dataset_name]["dataset_name"]
            print(f"fetching {dataset_name} from {data_source}")
    assert data_source in ["imodels", "pmlb", "imodels-multitask", "sklearn", "openml", "synthetic"], (
        data_source + " not correct"
    )
    if test_size is not None:

        def _split(X, y, feature_names):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            return X_train, X_test, y_train, y_test, feature_names

    else:

        def _split(X, y, feature_names):
            return X, y, feature_names

    if data_source == "imodels":
        if not dataset_name.endswith("csv"):
            dataset_name = dataset_name + ".csv"
        if not os.path.isfile(dataset_name) or override_cache:
            _download_imodels_dataset(dataset_name, data_path)
        df = pd.read_csv(oj(data_path, "imodels_data", dataset_name))
        X, y = df.iloc[:, :-1].values, df.iloc[:, -1].values
        feature_names = df.columns.values[:-1]
        if convertna:
            X = np.nan_to_num(X.astype("float32"))
        return _split(X, y, _clean_feat_names(feature_names))
    elif data_source == 'imodels-multitask':
        if not dataset_name.endswith("csv"):
            dataset_name = dataset_name + ".csv"
        if not os.path.isfile(dataset_name) or override_cache:
            _download_imodels_multitask_dataset(dataset_name, data_path)
        df = pd.read_csv(oj(data_path, "imodels_multitask_data", dataset_name))
        target_cols = [col for col in df.columns if col.endswith('__target')]
        feature_names = [col for col in df.columns if col not in target_cols]
        X, y = df[feature_names].values, df[target_cols].values
        if convertna:
            X = np.nan_to_num(X.astype("float32"))
        if return_target_col_names:
            return *(_split(X, y, _clean_feat_names(feature_names))), _clean_feat_names(target_cols)
        else:
            return _split(X, y, _clean_feat_names(feature_names))

    elif data_source == "pmlb":
        from pmlb import fetch_data

        feature_names = list(
            fetch_data(
                dataset_name,
                return_X_y=False,
                local_cache_dir=oj(data_path, "pmlb_data"),
            ).columns
        )
        feature_names.remove("target")
        X, y = fetch_data(
            dataset_name, return_X_y=True, local_cache_dir=oj(data_path, "pmlb_data")
        )
        if (
            np.unique(y).size == 2
        ):  # if binary classification, ensure that the classes are 0 and 1
            y -= np.min(y)
        return _split(_clean_features(X), y, _clean_feat_names(feature_names))
    elif data_source == "sklearn":
        if dataset_name == "diabetes":
            data = sklearn.datasets.load_diabetes()
        elif dataset_name == "california_housing":
            data = sklearn.datasets.fetch_california_housing(
                data_home=oj(data_path, "sklearn_data")
            )
        elif dataset_name == "breast_cancer":
            data = sklearn.datasets.load_breast_cancer()
        return data["data"], data["target"], _clean_feat_names(data["feature_names"])
    elif (
        data_source == "openml"
    ):  # note this api might change in newer sklearn - should give dataset-id not name
        data = sklearn.datasets.fetch_openml(
            data_id=dataset_name, data_home=oj(data_path, "openml_data"), parser="auto"
        )
        X, y, feature_names = (
            data["data"],
            data["target"],
            _clean_feat_names(data["feature_names"]),
        )
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        y = _define_openml_outcomes(y, dataset_name)
        return _split(_clean_features(X), y, _clean_feat_names(feature_names))
    elif data_source == "synthetic":
        if dataset_name == "friedman1":
            X, y = sklearn.datasets.make_friedman1(
                n_samples=200, n_features=10)
        elif dataset_name == "friedman2":
            X, y = sklearn.datasets.make_friedman2(n_samples=200)
        elif dataset_name == "friedman3":
            X, y = sklearn.datasets.make_friedman3(n_samples=200)
        elif dataset_name == "radchenko_james":
            X, y = make_rj()
        elif dataset_name == "vo_pati":
            X, y = make_vp()
        return _split(X, y, ["X_" + str(i + 1) for i in range(X.shape[1])])


def _define_openml_outcomes(y, data_id: str):
    if data_id == "59":  # ionosphere, positive is "good" class
        y = (y == "g").astype(int)
    if data_id == "183":  # abalone, need to convert strings to floats
        y = y.astype(float)
    if data_id == "1182":  # adult, positive is ">50K"
        y = (y == ">50K").astype(int)
    return y


def _clean_feat_names(feature_names):
    # shouldn't start with a digit
    feature_names = ["X_" + x if x[0].isdigit() else x for x in feature_names]
    # shouldn't end with __target
    feature_names = [x if not x.endswith(
        "__target") else x[:-8] for x in feature_names]
    return feature_names


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


def _download_imodels_dataset(dataset_fname, data_path: str):
    dataset_fname = dataset_fname.split(
        "/")[-1]  # remove anything about the path
    download_path = f"https://raw.githubusercontent.com/csinva/imodels-data/master/data_cleaned/{dataset_fname}"
    r = requests.get(download_path)
    if r.status_code == 404:
        raise Exception(
            f"404 Error for dataset {dataset_fname} (see valid files at https://github.com/csinva/imodels-data/tree/master/data_cleaned)"
        )

    os.makedirs(oj(data_path, "imodels_data"), exist_ok=True)
    with open(oj(data_path, "imodels_data", dataset_fname), "w") as f:
        f.write(r.text)


def _download_imodels_multitask_dataset(dataset_fname, data_path: str):
    dataset_fname = dataset_fname.split(
        "/")[-1]  # remove anything about the path
    download_path = f"https://huggingface.co/datasets/imodels/multitask-tabular-datasets/raw/main/{dataset_fname}"
    download_path_large = f'https://huggingface.co/datasets/imodels/multitask-tabular-datasets/resolve/main/{dataset_fname}'
    r = requests.get(download_path)
    if r.status_code == 404:
        raise Exception(
            f"404 Error for dataset {dataset_fname} (see valid files at https://huggingface.co/datasets/imodels/multitask-tabular-datasets)"
        )
    elif 'git-lfs' in r.text:
        r = requests.get(download_path_large)

    os.makedirs(oj(data_path, "imodels_multitask_data"), exist_ok=True)
    with open(oj(data_path, "imodels_multitask_data", dataset_fname), "w") as f:
        f.write(r.text)


def encode_categories(X, features, encoder=None):
    columns_to_keep = list(set(X.columns).difference(features))
    X_encoded = X.loc[:, columns_to_keep]
    X_cat = pd.DataFrame({f: X.loc[:, f] for f in features})
    
    if encoder is None:
        one_hot_encoder = OneHotEncoder(categories="auto", sparse_output=False)
        X_one_hot = pd.DataFrame(one_hot_encoder.fit_transform(X_cat))
    else:
        one_hot_encoder = encoder
        X_one_hot = pd.DataFrame(one_hot_encoder.transform(X_cat))
    X_one_hot.columns = one_hot_encoder.get_feature_names_out(features)
    X_encoded = pd.concat([X_encoded, X_one_hot], axis=1)
    if encoder is not None:
        return X_encoded
    return X_encoded, one_hot_encoder


if __name__ == "__main__":
    import imodels

    # X, y, feature_names = imodels.get_clean_dataset('compas_two_year_clean', data_source='imodels', test_size=0.5)
    X_train, X_test, y_train, y_test, feature_names = imodels.get_clean_dataset(
        "compas_two_year_clean", data_source="imodels", test_size=0.5
    )
    print(X_train.shape, y_train.shape)
