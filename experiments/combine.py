import argparse
import glob
import os
import pickle as pkl

import pandas as pd

from experiments.compare_models import compute_auc_of_auc, MODEL_COMPARISON_PATH


def combine_comparisons(path, model, test):
    all_files = glob.glob(path + '*')
    model_files = list(filter(lambda x: (model in x) and ('comparisons_' in x), all_files))
    model_files_sorted = sorted(model_files, key=lambda x: int(x.split('_')[-1][:-4]))
    results_sorted = [pkl.load(open(f, 'rb')) for f in model_files_sorted]

    df = pd.concat([r['df'] for r in results_sorted])
    estimators = [r['estimators'][0] for r in results_sorted]

    output_dict = {
        'estimators': estimators,
        'comparison_datasets': results_sorted[0]['comparison_datasets'],
        'metrics': results_sorted[0]['metrics'],
        'df': df,
    }
    if not test:
        compute_auc_of_auc(output_dict, column='mean_PRAUC')

    combined_filename = '.'.join(model_files_sorted[0].split('_0.'))
    pkl.dump(output_dict, open(combined_filename, 'wb'))

    for f in model_files_sorted:
        os.remove(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--low_data', action='store_true')
    args = parser.parse_args()

    path = MODEL_COMPARISON_PATH
    path += 'low_data/' if args.low_data else ''
    path += 'test/' if args.test else 'val/'
    combine_comparisons(path, args.model, args.test)
