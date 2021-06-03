import os
import pickle as pkl
import warnings
from functools import partial
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import issparse
from sklearn.base import BaseEstimator
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score


MODEL_COMPARISON_PATH = os.path.dirname(os.path.realpath(__file__)) + "/comparison_data/"

DATASET_PATH = os.path.dirname(os.path.realpath(__file__)) + "/data/"


class Model:
    def __init__(self, name: str, cls, vary_param: str, vary_param_val: Any, 
                 fixed_param: str = None, fixed_param_val: Any = None,
                 other_params: Dict[str, Any] = {}):
        self.name = name
        self.cls = cls
        self.fixed_param = fixed_param
        self.fixed_param_val = fixed_param_val
        self.vary_param = vary_param
        self.vary_param_val = vary_param_val
        self.kwargs = {self.vary_param: self.vary_param_val}
        if self.fixed_param is not None:
            self.kwargs[self.fixed_param] = self.fixed_param_val
        self.kwargs = {**self.kwargs, **other_params}
    
    def __repr__(self):
        return self.name


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


def get_best_accuracy(ytest, yscore):
    thrs = np.unique(yscore)
    accs = []
    for thr in thrs:
        accs.append(accuracy_score(ytest, yscore > thr))
    return np.max(accs)


def get_comparison_result(path: str, estimator_name: str, dataset: str, prefix='val', low_data=False) -> Dict[str, Any]:
    path += 'low_data/' if low_data else 'reg_data/'
    path += f'{dataset}/'
    if prefix == 'test':
        result_file = path + 'test/' + f'{estimator_name}_test_comparisons.pkl'
    elif prefix == 'cv':
        result_file = path + 'cv/' + f'{estimator_name}_comparisons.pkl'
    else:
        result_file = path + 'val/' + f'{estimator_name}_comparisons.pkl'
    return pkl.load(open(result_file, 'rb'))


def get_best_model_under_complexity(c: int, model_name: str, 
                                    model_cls: BaseEstimator,
                                    dataset: str,
                                    curve_params: list = None,
                                    metric: str = 'mean_rocauc',
                                    kwargs: dict = {},
                                    prefix: str = 'val',
                                    easy: bool = False) -> BaseEstimator:
    # init_models = []
    # for m_name, m_cls in models:
    result = get_comparison_result(MODEL_COMPARISON_PATH, model_name, dataset=dataset, prefix=prefix)
    df, auc_metric = result['df'], result['meta_auc_df'][f'{metric}_auc']

    if curve_params:
        # specify which curve to use
        if type(df.iloc[:, 1][0]) is partial:
            df_best_curve = df[df.iloc[:, 1].apply(lambda x: x.keywords['min_samples_split']).isin(curve_params)]
        else:
            df_best_curve = df[df.iloc[:, 1].isin(curve_params)]

    else:
        # detect which curve to use
        df_best_curve = df[df.index == auc_metric.idxmax()]
    
    df_under_c = df_best_curve[df_best_curve['mean_complexity'] < c]
    if df_under_c.shape[0] == 0:
        warnings.warn(f'{model_name} skipped for complexity limit {c}')
        return None

    best_param = df_under_c.iloc[:, 0][df_under_c[metric].argmax()]
    kwargs[df_under_c.columns[0]] = best_param

    # if there is a second param that was varied
    if auc_metric.shape[0] > 1:
        kwargs[df_under_c.columns[1]] = df_under_c.iloc[0, 1]

    return model_cls(**kwargs)


def remove_x_axis_duplicates(x: np.array, y: np.array) -> Tuple[np.array, np.array]:
    unique_arr, inds, counts = np.unique(x, return_index=True, return_counts=True)

    y_for_unique_x = []
    for i, ind in enumerate(inds):
        y_for_unique_x.append(y[ind:ind+counts[i]].max())

    return unique_arr, np.array(y_for_unique_x)


def merge_overlapping_curves(test_mul_curves, y_col):
    final_x = []
    final_y = []
    curves = test_mul_curves.index.unique()
    
    start_compl = 0
    for i in range(curves.shape[0]):
        curr_x = test_mul_curves[test_mul_curves.index == curves[i]]['mean_complexity']
        curr_y = test_mul_curves[test_mul_curves.index == curves[i]][y_col]
        curr_x, curr_y = curr_x[curr_x.argsort()], curr_y[curr_x.argsort()]
        curr_x, curr_y = remove_x_axis_duplicates(curr_x, curr_y)
        curr_x, curr_y = curr_x[curr_x >= start_compl], curr_y[curr_x >= start_compl]
        
        if i != curves.shape[0] - 1:

            next_x = test_mul_curves[test_mul_curves.index == curves[i+1]]['mean_complexity']
            next_y = test_mul_curves[test_mul_curves.index == curves[i+1]][y_col]
            next_x, next_y = next_x[next_x.argsort()], next_y[next_x.argsort()]
            next_x, next_y = remove_x_axis_duplicates(next_x, next_y)

        found_switch_point = False
        for j in range(curr_x.shape[0] - 1):     
            
            final_x.append(curr_x[j])
            final_y.append(curr_y[j])
            
            if i != curves.shape[0] - 1:
            
                next_x_next_val = next_x[next_x > curr_x[j]][0]
                next_y_next_val = next_y[next_x > curr_x[j]][0]
                curr_x_next_val = curr_x[j+1]
                curr_y_next_val = curr_y[j+1]
            
                if next_y_next_val > curr_y_next_val and next_x_next_val - curr_x_next_val <= 5:
                    start_compl = next_x_next_val
                    found_switch_point = True
                    break
        
        if not found_switch_point:
            return np.array(final_x), np.array(final_y)
    
    return np.array(final_x), np.array(final_y)
