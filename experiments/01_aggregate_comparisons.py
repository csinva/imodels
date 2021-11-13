import argparse
import glob
import os.path
import pickle as pkl
import warnings
from os.path import join as oj

import numpy as np
import pandas as pd

from experiments.config.datasets import DATASETS
from experiments.util import get_results_path_from_args
from experiments.validate import compute_meta_auc


def combine_comparisons(path: str):
    """Combines comparisons output after running
    Parameters
    ----------
    path: str
        path to directory containing pkl files to combine
    """
    all_files = glob.glob(oj(path, '*'))
    model_files = [f for f in all_files
                   if '_comparisons' in f]

    results_sorted = [pkl.load(open(f, 'rb')) for f in model_files]

    df = pd.concat([r['df'] for r in results_sorted])
    estimators = []
    for r in results_sorted:
        estimators += np.unique(r['estimators']).tolist()

    output_dict = {
        'estimators': estimators,
        'comparison_datasets': results_sorted[0]['comparison_datasets'],
        'metrics': results_sorted[0]['metrics'],
        'df': df,
    }

    if 'rule_df' in results_sorted[0]:
        rule_df = pd.concat([r['rule_df'] for r in results_sorted])
        output_dict['rule_df'] = rule_df

    # for curr_df, prefix in level_dfs:
    try:
        meta_auc_df = compute_meta_auc(df)
    except Exception as e:
        warnings.warn(f'bad complexity range')
        # warnings.warn(e)
        meta_auc_df = None

    output_dict['meta_auc_df'] = meta_auc_df

    # combined_filename = '.'.join(model_files_sorted[0].split('_0.'))
    # pkl.dump(output_dict, open(combined_filename, 'wb'))

    combined_filename = oj(path, 'combined.pkl')
    pkl.dump(output_dict, open(combined_filename, 'wb'))

    # for f in model_files_sorted:
    #     os.remove(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--low_data', action='store_true')
    parser.add_argument('--results_path', type=str,
                        default=oj(os.path.dirname(os.path.realpath(__file__)), 'results'))
    parser.add_argument('--splitting_strategy', type=str, default="train-test")
    args = parser.parse_args()

    datasets = DATASETS
    if args.dataset:
        datasets = list(filter(lambda x: args.dataset == x[0], DATASETS))

    for dataset in datasets:
        path = get_results_path_from_args(args, dataset[0])
        print('path', path)
        combine_comparisons(path)
