import math
from typing import List, Dict, Any, Union

import dvu
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
dvu.set_style()
mpl.rcParams['figure.dpi'] = 250


def get_x_and_y(result_data: pd.Series, x_col: str, y_col: str) -> (pd.Series, pd.Series):
    complexities = result_data[x_col]
    rocs = result_data[y_col]
    complexity_sort_indices = complexities.argsort()
    return complexities[complexity_sort_indices], rocs[complexity_sort_indices]


def viz_comparison_val_average(result: Dict[str, Any], y_column: str = 'PRAUC') -> None:
    '''Plot dataset-averaged y_column vs dataset-averaged complexity for different hyperparameter settings
    of a single model, including zoomed-in plot of overlapping region
    '''
    result_data = result['df']
    result_estimators = result['estimators']
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
    for est in np.unique(result_estimators):

        est_result_data = result_data[result_data.index.str.contains(est)]
        x, y = get_x_and_y(est_result_data, 'mean_complexity', f'mean_{y_column}')
        axes[0].plot(x, y, marker='o', markersize=4, label=est.replace('_', ' '))

        if est in result['auc_of_auc'].index:
            area = result['auc_of_auc'][est]
            label = est.split(' - ')[1] + f' AUC: {area:.3f}'
            axes[1].plot(x, y, marker='o', markersize=4, label=label.replace('_', ' '))

    axes[0].set_title(f'average {y_column} across all comparison datasets')
    axes[1].set_xlim(result['auc_of_auc_lb'], result['auc_of_auc_ub'])
    axes[1].set_title('Overlapping, low (<30) complexity region only')

    for ax in axes:
        ax.set_xlabel('complexity score')
        ax.set_ylabel(y_column)
        ax.legend(frameon=False, handlelength=1)
        # dvu.line_legend(fontsize=10, ax=ax)    
    plt.tight_layout()


def viz_comparison_test_average(results: List[Dict[str, Any]], y_column: str = 'PRAUC', line_legend: bool = False) -> None:
    '''Plot dataset-averaged y_column vs dataset-averaged complexity for different models
    '''
    for result in results:
        result_data = result['df']
        est = result['estimators'][0]
        x, y = get_x_and_y(result_data, 'mean_complexity', f'mean_{y_column}')
        plt.plot(x, y, marker='o', markersize=2, linewidth=1, label=est.replace('_', ' '))
    plt.xlim(0, 30)
    plt.xlabel('complexity score', size=8)
    plt.ylabel(y_column, size=8)
    plt.title(f'average {y_column} across all comparison datasets', size=8)
    if line_legend:
        dvu.line_legend(fontsize=8, adjust_text_labels=True)
    else:
        plt.legend(frameon=False, handlelength=1, fontsize=8)


def viz_comparison_datasets(result: Union[dict[str, Any], list[dict[str, Any]]],
                            y_column: str = 'PRAUC',
                            cols=3, 
                            figsize=(14, 10), 
                            test=False) -> None:
    '''Plot y_column vs complexity for different datasets and models (not averaged)
    '''
    if test:
        results_data = pd.concat([r['df'] for r in result])
        results_estimators = [r['estimators'][0] for r in result]
        results_datasets = result[0]['comparison_datasets']
    else:
        results_data = result['df']
        results_estimators = np.unique(result['estimators'])
        results_datasets = result['comparison_datasets']

    datasets = list(map(lambda x: x[0], results_datasets))
    n_rows = int(math.ceil(len(datasets) / cols))
    plt.figure(figsize=figsize)
    for i, dataset in enumerate(datasets):
        plt.subplot(n_rows, cols, i + 1)

        for est in np.unique(results_estimators):
            est_result_data = results_data[results_data.index.str.contains(est)]
            x, y = get_x_and_y(est_result_data, dataset + '_complexity', dataset + f'_{y_column}')
            plt.plot(x, y, marker='o', markersize=4, label=est.replace('_', ' '))

        plt.xlim(0, 30)
        plt.xlabel('complexity score')
        plt.ylabel(y_column)
#             plt.legend()
        dvu.line_legend(fontsize=10,
                        adjust_text_labels=False,
                        xoffset_spacing=0,
                        extra_spacing=0)
        plt.title(f'dataset {dataset}')
#         plt.legend(frameon=False, handlelength=1)
#     plt.subplot(n_rows, cols, 1)
    plt.tight_layout()
