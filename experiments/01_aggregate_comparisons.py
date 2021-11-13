import argparse
import glob
import os.path
import pickle as pkl
import warnings
from os.path import join as oj

import numpy as np
import pandas as pd

from experiments.util import get_results_path_from_args
from experiments.validate import compute_meta_auc


def combine_comparisons(path, model, test):
    """Combines comparisons output after running

    Parameters
    ----------
    path
    model
    test

    Returns
    -------

    """
    all_files = glob.glob(path + '*')
    model_files = list(filter(lambda x: (model in x) and ('comparisons_' in x), all_files))
    model_files_sorted = sorted(model_files, key=lambda x: int(x.split('_')[-1][:-4]))
    results_sorted = [pkl.load(open(f, 'rb')) for f in model_files_sorted]

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

    if not test:
        # easy_df = df.loc[:, ['easy' in col for col in df.columns]].copy()
        # med_df = df.loc[:, ['med' in col for col in df.columns]].copy()
        # hard_df = df.loc[:, ['hard' in col for col in df.columns]].copy()
        # all_df = df.loc[:, ['all' in col for col in df.columns]].copy()
        # level_dfs = (med_df, 'med'), (hard_df, 'hard'), (all_df, 'all')

        # for curr_df, prefix in level_dfs:
        try:
            meta_auc_df = compute_meta_auc(df)
        except ValueError as e:
            warnings.warn(f'bad complexity range')
            warnings.warn(e)
            meta_auc_df = None

        output_dict['meta_auc_df'] = meta_auc_df

    combined_filename = '.'.join(model_files_sorted[0].split('_0.'))
    pkl.dump(output_dict, open(combined_filename, 'wb'))

    # for f in model_files_sorted:
    #     os.remove(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--cv', action='store_true')
    parser.add_argument('--low_data', action='store_true')
    parser.add_argument('--model_comparison_path', type=str,
                        default=oj(os.path.dirname(os.path.realpath(__file__)), 'comparison_data'))
    parser.add_argument('--dataset_path', type=str,
                        default=oj(os.path.dirname(os.path.realpath(__file__)), 'data'))
    args = parser.parse_args()

    path = get_results_path_from_args(args)

    combine_comparisons(path, args.model, args.test)
