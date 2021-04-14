import os
import pickle as pkl
from typing import Any, Dict, List, Tuple

from sklearn.base import BaseEstimator


MODEL_COMPARISON_PATH = os.path.dirname(os.path.realpath(__file__)) + "/comparison_data/"


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


def get_comparison_result(path: str, estimator_name: str, test=False) -> Dict[str, Any]:
    if test:
        result_file = path + 'test/' + f'{estimator_name}_test_comparisons.pkl'
    else:
        result_file = path + 'val/' + f'{estimator_name}_comparisons.pkl'
    return pkl.load(open(result_file, 'rb'))


def get_best_models_under_complexity(c: int, models: List[Tuple[str, BaseEstimator]], 
                                     metric: str = 'mean_PRAUC'):
    init_models = []
    for m_name, m_cls in models:
        result = get_comparison_result(MODEL_COMPARISON_PATH, m_name)
        df, auc_df = result['df'], result['auc_of_auc']
        df_best_curve = df[df.index == auc_df.idxmax()]
        df_under_c = df_best_curve[df_best_curve['mean_complexity'] < c]
        best_param = df_under_c.iloc[:, 0][df_under_c[metric].argmax()]
        kwargs = {df_under_c.columns[0]: best_param}
        if auc_df.shape[0] > 1:
            kwargs[df_under_c.columns[1]] = int(df_under_c.iloc[0, 1])
        init_models.append(m_cls(**kwargs))
    return init_models
