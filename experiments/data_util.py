from typing import Tuple

import numpy as np
import pandas as pd
from scipy.sparse import issparse
from sklearn.datasets import fetch_openml


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


def get_clean_dataset(path: str) -> Tuple[np.array]:
    df = pd.read_csv(path)
    X, y = df.iloc[:, :-1].values, df.iloc[:, -1].values
    feature_names = df.columns.values[:-1]
    return np.nan_to_num(X.astype('float32')), y, feature_names
