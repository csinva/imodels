import itertools 
import math
import os
import pickle as pkl
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.datasets import fetch_openml
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['figure.dpi'] = 250
import dvu # for visualization
dvu.set_style()

def get_x_and_y(result_data: pd.Series) -> (pd.Series, pd.Series):
    complexities = result_data[result_data.index.str.contains('complexity')]
    rocs = result_data[result_data.index.str.contains('ROC')]
    complexity_sort_indices = complexities.argsort()    
    return complexities[complexity_sort_indices], rocs[complexity_sort_indices]

def viz_comparison_val_average(result: Dict[str, Any]) -> None:
    '''Plot dataset-averaged ROC AUC vs dataset-averaged complexity for different hyperparameter settings
    of a single model, including zoomed-in plot of overlapping region
    '''
    result_data = result['df']['mean']
    result_estimators = result['estimators']
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
    for est in np.unique(result_estimators):
        
        est_result_data = result_data[result_data.index.str.contains(est)]
        x, y = get_x_and_y(est_result_data)  
        axes[0].plot(x, y, marker='o', markersize=4, label=est.replace('_', ' '))
        
        if est in result['auc_of_auc'].index:
            area = result['auc_of_auc'][est]
            label = est.split(' - ')[1] + f' AUC: {area:.3f}'
            axes[1].plot(x, y, marker='o', markersize=4, label=label.replace('_', ' '))

    axes[0].set_title('average ROC AUC across all comparison datasets')
    axes[1].set_xlim(result['auc_of_auc_lb'], result['auc_of_auc_ub'])
    axes[1].set_title('Overlapping, low (<30) complexity region only')
    
    for ax in axes:
        ax.set_xlabel('complexity score')
        ax.set_ylabel('ROC AUC')
        # ax.legend(frameon=False, handlelength=1)
        dvu.line_legend(fontsize=10, ax=ax)    
    plt.tight_layout()

def viz_comparison_test_average(results: List[Dict[str, Any]]) -> None:
    '''Plot dataset-averaged ROC AUC vs dataset-averaged complexity for different models
    '''
    for result in results:
        mean_result = result['df']['mean']
        est = result['estimators'][0]
        x, y = get_x_and_y(mean_result)  
        plt.plot(x, y, marker='o', markersize=2, linewidth=1, label=est.replace('_', ' '))
    plt.xlim(0, 30)
    plt.xlabel('complexity score', size=8)
    plt.ylabel('ROC AUC', size=8)
    plt.title('average ROC AUC across all comparison datasets', size=8)
#     plt.legend(frameon=False, handlelength=1, fontsize=8)
    dvu.line_legend(fontsize=8, adjust_text_labels=True)

def viz_comparison_datasets(result: Dict[str, Any], cols=3, figsize=(14, 10), test=False) -> None:
    '''Plot ROC AUC vs complexity for different datasets and models (not averaged)
    '''
    if test:
        results_df = pd.concat([r['df'] for r in result])
        results_estimators = [r['estimators'][0] for r in result]
    else:
        results_df = result['df']
        results_estimators = np.unique(result['estimators'])

    datasets = list(results_df.columns)[:-2]
    n_rows = int(math.ceil(len(datasets) / cols))
    plt.figure(figsize=figsize)
    for i, dataset in enumerate(datasets):
#         curr_ax = axes[i // cols, i % cols]
        plt.subplot(n_rows, cols, i + 1)
        results_data = results_df[dataset]
        for est in np.unique(results_estimators):
            results_data_est = results_data[results_data.index.str.contains(est)]
            x, y = get_x_and_y(results_data_est)
            plt.plot(x, y, marker='o', markersize=4, label=est.replace('_', ' '))
        plt.xlim(0, 30)
        plt.xlabel('complexity score')
        plt.ylabel('ROC AUC')
#             plt.legend()
        dvu.line_legend(fontsize=8,
                        adjust_text_labels=False,
                        xoffset_spacing=0,
                        extra_spacing=0)
        plt.title(f'dataset {dataset}')
#         plt.legend(frameon=False, handlelength=1)
#     plt.subplot(n_rows, cols, 1)
#     
    plt.tight_layout()