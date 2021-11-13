from collections import defaultdict
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from sklearn.metrics import accuracy_score

from experiments.util import remove_x_axis_duplicates


def compute_meta_auc(result_data: pd.DataFrame,
                     prefix: str = '',
                     max_allowable_complexity: int = 30,
                     max_start_complexity: int = 10) -> Tuple[pd.DataFrame, Tuple[float]]:
    """

    Parameters
    ----------
    result_data
    prefix
    max_allowable_complexity
        complexity score under which a model is considered interpretable
    max_start_complexity
        min complexity of curves included in the AUC-of-AUC comparison must be below this value

    Returns
    -------

    """

    # x_column = f'{prefix}_mean_complexity'
    x_column = f'mean_complexity'
    compute_columns = result_data.columns[result_data.columns.str.contains('mean')]
    estimators = np.unique(result_data.index)
    xs = np.empty(len(estimators), dtype=object)
    ys = xs.copy()

    for i, est in enumerate(estimators):
        est_result_df = result_data[result_data.index.str.fullmatch(est)]
        complexities_unsorted = est_result_df[x_column]
        complexity_sort_indices = complexities_unsorted.argsort()
        complexities = complexities_unsorted[complexity_sort_indices]

        roc_aucs = est_result_df.iloc[complexity_sort_indices][compute_columns]
        xs[i] = complexities.values
        ys[i] = roc_aucs.values

    # filter out curves which start too complex
    start_under_10 = list(map(lambda x: min(x) < max_start_complexity, xs))

    # find overlapping complexity region for roc-of-roc comparison
    meta_auc_lb = max([x[0] for x in xs])
    endpts = np.array([x[-1] for x in xs])
    meta_auc_ub = min(endpts[endpts > meta_auc_lb])
    meta_auc_ub = min(meta_auc_ub, max_allowable_complexity)

    # handle non-overlapping curves
    endpt_after_lb = endpts > meta_auc_lb
    eligible = start_under_10 & endpt_after_lb

    # compute AUC of interpolated curves in overlap region
    meta_aucs = defaultdict(lambda: [])
    for i in range(len(xs)):
        for c, col in enumerate(compute_columns):
            if eligible[i]:
                x, y = remove_x_axis_duplicates(xs[i], ys[i][:, c])
                f_curve = interp1d(x, y)
                x_interp = np.linspace(meta_auc_lb, meta_auc_ub, 100)
                y_interp = f_curve(x_interp)
                auc_value = np.trapz(y_interp, x=x_interp)
            else:
                auc_value = 0
            meta_aucs[col + '_auc'].append(auc_value)

    meta_auc_df = pd.DataFrame(meta_aucs, index=estimators)
    meta_auc_df[f'{x_column}_lb'] = meta_auc_lb
    meta_auc_df[f'{x_column}_ub'] = meta_auc_ub
    return meta_auc_df


def get_best_accuracy(ytest, yscore):
    thrs = np.unique(yscore)
    accs = []
    for thr in thrs:
        accs.append(accuracy_score(ytest, yscore > thr))
    return np.max(accs)
