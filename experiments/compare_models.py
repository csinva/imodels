'''Compare different estimators on public datasets
Code modified from https://github.com/tmadl/sklearn-random-bits-forest
'''
import argparse
import glob
import os
import pickle as pkl
import time
from collections import defaultdict
from typing import Dict, Any

import numpy as np
import pandas as pd
from imodels import (
    SkopeRulesClassifier as skope, RuleFitClassifier as rfit, FPLassoClassifier as fpl, 
    FPSkopeClassifier as fps, BayesianRuleListClassifier as brl, GreedyRuleListClassifier as grl,
    OneRClassifier as oner, BoostedRulesClassifier as brs
)
from scipy.interpolate import interp1d
from scipy.sparse import issparse
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier as rf, GradientBoostingClassifier as gb
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import KFold, train_test_split
from tqdm import tqdm


COMPARISON_DATASETS = [
    ("breast-cancer", 13),
    ("breast-w", 15),
    ("credit-g", 31),
    ("haberman", 43),
    ("heart", 1574),
    ("labor", 4),
    ("vote", 56),
]

METRICS = [
    ('accuracy', accuracy_score),
    ('ROCAUC', roc_auc_score),
    ('time', None),
    ('complexity', None)
]

# complexity score under which a model is considered interpretable
LOW_COMPLEXITY_CUTOFF = 30

# min complexity of curves included in the AUC-of-AUC comparison must be below this value
MAX_START_COMPLEXITY = 10


class Model:
    def __init__(self, name: str, cls, vary_param: str, vary_param_val: Any, 
                 fixed_param: str = None, fixed_param_val: Any = None):
        self.name = name
        self.cls = cls
        self.fixed_param = fixed_param
        self.fixed_param_val = fixed_param_val
        self.vary_param = vary_param
        self.vary_param_val = vary_param_val
        self.kwargs = {self.vary_param: self.vary_param_val}
        if self.fixed_param is not None:
            self.kwargs[self.fixed_param] = self.fixed_param_val


BEST_ESTIMATORS = [
    [Model('random_forest', rf, 'n_estimators', n, 'max_depth', 1) for n in np.arange(1, 15)],
    [Model('gradient_boosting', gb, 'n_estimators', n, 'max_depth', 1) for n in np.linspace(1, 20, 10, dtype=int)],
    [Model('skope_rules', skope, 'n_estimators', n, 'max_depth', 1) for n in np.linspace(2, 200, 10, dtype=int)],
    [Model('rulefit', rfit, 'max_rules', n, 'tree_size', 2) for n in np.linspace(2, 100, 10, dtype=int)],
    [Model('fplasso', fpl, 'max_rules', n, 'maxcardinality', 1) for n in np.linspace(2, 100, 10, dtype=int)],
    [Model('fpskope', fps, 'maxcardinality', n, 'max_depth_duplication', 3) for n in np.arange(1, 5)],
    [Model('brl', brl, 'listlengthprior', n, 'maxcardinality', 2) for n in np.linspace(1, 16, 8)],
    [Model('grl', grl, 'max_depth', n) for n in np.arange(1, 6)],
    [Model('oner', oner, 'max_depth', n) for n in np.arange(1, 6)],
    [Model('brs', brs, 'n_estimators', n) for n in np.linspace(1, 32, 10, dtype=int)]
]

ALL_ESTIMATORS = []
ALL_ESTIMATORS.append(
    [Model('random_forest - depth_1', rf, 'n_estimators', n, 'max_depth', 1) for n in np.linspace(1, 40, 10, dtype=int)]
    + [Model('random_forest - depth_2', rf, 'n_estimators', n, 'max_depth', 2) for n in np.linspace(1, 15, 10, dtype=int)]
    + [Model('random_forest - depth_3', rf, 'n_estimators', n, 'max_depth', 3) for n in np.arange(1, 8)]
)
ALL_ESTIMATORS.append(
    [Model('gradient_boosting - depth_1', gb, 'n_estimators', n, 'max_depth', 1) for n in np.linspace(1, 40, 10, dtype=int)]
    + [Model('gradient_boosting - depth_2', gb, 'n_estimators', n, 'max_depth', 2) for n in np.linspace(1, 15, 10, dtype=int)]
    + [Model('gradient_boosting - depth_3', gb, 'n_estimators', n, 'max_depth', 3) for n in np.arange(1, 8)]
)
ALL_ESTIMATORS.append(
    [Model('skope_rules - depth_1', skope, 'n_estimators', n, 'max_depth', 1) for n in np.linspace(2, 200, 10, dtype=int)]
    + [Model('skope_rules - depth_2', skope, 'n_estimators', n, 'max_depth', 2) for n in np.linspace(2, 200, 10, dtype=int)]
    + [Model('skope_rules - depth_3', skope, 'n_estimators', n, 'max_depth', 3) for n in np.linspace(2, 80, 10, dtype=int)]
)
ALL_ESTIMATORS.append(
    [Model('rulefit - depth_1', rfit, 'max_rules', n, 'tree_size', 2) for n in np.linspace(2, 100, 10, dtype=int)]
    + [Model('rulefit - depth_2', rfit, 'max_rules', n, 'tree_size', 4) for n in np.linspace(2, 50, 10, dtype=int)]
    + [Model('rulefit - depth_3', rfit, 'max_rules', n, 'tree_size', 8) for n in np.linspace(2, 50, 10, dtype=int)]
)
ALL_ESTIMATORS.append(
    [Model('fplasso - max_card_1', fpl, 'max_rules', n, 'maxcardinality', 1) for n in np.linspace(2, 100, 10, dtype=int)]
    + [Model('fplasso - max_card_2', fpl, 'max_rules', n, 'maxcardinality', 2) for n in np.linspace(2, 60, 10, dtype=int)]
    + [Model('fplasso - max_card_3', fpl, 'max_rules', n, 'maxcardinality', 3) for n in np.linspace(2, 50, 10, dtype=int)]
)
ALL_ESTIMATORS.append(
    [Model('fpskope - No dedup', fps, 'maxcardinality', n, 'max_depth_duplication', None) for n in [1, 2]]
    + [Model('fpskope - max_dedup_1', fps, 'maxcardinality', n, 'max_depth_duplication', 1) for n in [1, 2, 3, 4]]
    + [Model('fpskope - max_dedup_2', fps, 'maxcardinality', n, 'max_depth_duplication', 2) for n in [1, 2, 3, 4]]
    + [Model('fpskope - max_dedup_3', fps, 'maxcardinality', n, 'max_depth_duplication', 3) for n in [1, 2, 3, 4]]
)
ALL_ESTIMATORS.append(
    [Model('brl - max_card_1', brl, 'listlengthprior', n, 'maxcardinality', 1) for n in np.linspace(1, 20, 10, dtype=int)]
    + [Model('brl - max_card_2', brl, 'listlengthprior', n, 'maxcardinality', 2) for n in np.linspace(1, 16, 8, dtype=int)]
    + [Model('brl - max_card_3', brl, 'listlengthprior', n, 'maxcardinality', 3) for n in np.linspace(1, 16, 8, dtype=int)]
)


def get_complexity(estimator):
    if isinstance(estimator, (rf, gb)):
        complexity = 0
        for tree in estimator.estimators_:
            if type(tree) is np.ndarray:
                tree = tree[0]
            complexity += 2 ** tree.get_depth()
            
            # add 0.5 for every antecedent after the first
            if tree.get_depth() > 1:
                complexity += ((2 ** tree.get_depth()) - 1) * 0.5 
        return complexity
    else:
        return estimator.complexity_


def get_dataset(data_id, onehot_encode_strings=True): 
    dataset = fetch_openml(data_id=data_id, as_frame=False)
    X = dataset.data
    if issparse(X):
        X = X.toarray()
    y = (dataset.target[0] == dataset.target).astype(int)
    return np.nan_to_num(X.astype('float32')), y


def compute_auc_of_auc(result: Dict[str, Any], dpi=83) -> None:
    result_data = result['df']
    estimators = np.unique(result['estimators'])
    xs = np.empty(len(estimators), dtype=object)
    ys = xs.copy()

    for i, est in enumerate(estimators):

        est_result_df = result_data[result_data.index.str.contains(est)]
        complexities_unsorted = est_result_df['mean_complexity']
        complexity_sort_indices = complexities_unsorted.argsort()
        complexities = complexities_unsorted[complexity_sort_indices]

        roc_aucs = est_result_df['mean_ROCAUC'][complexity_sort_indices]
        xs[i] = complexities.values
        ys[i] = roc_aucs.values

    # filter out curves which start too complex
    mask = list(map(lambda x: min(x) < MAX_START_COMPLEXITY, xs))
    xs, ys, estimators = xs[mask], ys[mask], estimators[mask]

    # find overlapping complexity region for roc-of-roc comparison
    auc_of_auc_lb = max([x[0] for x in xs])
    endpts = np.array([x[-1] for x in xs])
    auc_of_auc_ub = min(endpts[endpts > auc_of_auc_lb])
    auc_of_auc_ub = min(auc_of_auc_ub, LOW_COMPLEXITY_CUTOFF)

    # handle non-overlapping curves
    mask = endpts > auc_of_auc_lb
    xs, ys, estimators = xs[mask], ys[mask], estimators[mask]

    # compute AUC of interpolated curves in overlap region
    auc_of_aucs = []
    for i in range(len(xs)):

        f_curve = interp1d(xs[i], ys[i])
        x_interp = np.linspace(auc_of_auc_lb, auc_of_auc_ub, 100)
        y_interp = f_curve(x_interp)
        auc_of_aucs.append(np.trapz(y_interp, x=x_interp))

    result['auc_of_auc'] = (
        pd.Series(auc_of_aucs, index=estimators).sort_values(ascending=False)
    )
    result['auc_of_auc_lb'] = auc_of_auc_lb
    result['auc_of_auc_ub'] = auc_of_auc_ub


def compare_estimators(estimators: list,
                       datasets,
                       metrics: list,
                       n_cv_folds=10, decimals=3, cellsize=22, verbose=True):
    if type(estimators) != list:
        raise Exception("First argument needs to be a list of tuples containing ('name', Estimator pairs)")
    if type(metrics) != list:
        raise Exception("Argument metrics needs to be a list of tuples containing ('name', scoring function pairs)")

    mean_results = defaultdict(lambda: [])
    for e in estimators:
        mean_results[e.vary_param].append(e.vary_param_val)
        if e.fixed_param is not None:
            mean_results[e.fixed_param].append(e.fixed_param_val)

    # loop over datasets
    for d in tqdm(datasets):
        if verbose:
            print("comparing on dataset", d[0])
        X, y = get_dataset(d[1])

        # loop over estimators
        for model in estimators:
            mresults = [[] for i in range(len(metrics))]
            est = model.cls(**model.kwargs)

            # loop over folds
            if n_cv_folds == 1:
                fold_iterator = [train_test_split(np.arange(X.shape[0]), test_size=0.2, random_state=0)]
            else:
                kf = KFold(n_splits=n_cv_folds)
                fold_iterator = kf.split(X)
            for train_idx, test_idx in fold_iterator:
                start = time.time()
                est.fit(X[train_idx, :], y[train_idx])
                y_pred = est.predict(X[test_idx, :])
                y_pred_proba = est.predict_proba(X[test_idx, :])[:, 1]
                end = time.time()

                # loop over metrics
                for i, (met_name, met) in enumerate(metrics):
                    if met_name == 'time':
                        mresults[i].append(end - start)
                    elif met_name == 'complexity':
                        mresults[i].append(get_complexity(est))
                    elif met_name == 'ROCAUC':
                        mresults[i].append(roc_auc_score(y[test_idx], y_pred_proba))
                    else:
                        mresults[i].append(met(y[test_idx], y_pred))

            for i, (met_name, met) in enumerate(metrics):
                colname = d[0] + '_' + met_name
                mean_results[colname].append(np.mean(mresults[i]))

    return mean_results


def run_comparison(path, datasets, metrics, estimators, 
                   parallel_id=None, verbose=False, ignore_cache=False, test=False, cv_folds=4):

    estimator_name = estimators[0].name.split(' - ')[0]
    if test:
        model_comparison_file = path + f'{estimator_name}_test_comparisons.pkl'
    else:
        model_comparison_file = path + f'{estimator_name}_comparisons.pkl'
    if parallel_id is not None:
        model_comparison_file = f'_{parallel_id}.'.join(model_comparison_file.split('.'))

    if os.path.isfile(model_comparison_file) and not ignore_cache:
        print(f'{estimator_name} results already computed and cached. use --ignore_cache to recompute')
        return

    mean_results = compare_estimators(estimators=estimators,
                                      datasets=datasets,
                                      metrics=metrics,
                                      verbose=verbose,
                                      n_cv_folds=cv_folds)

    estimators_list = [e.name for e in estimators]
    metrics_list = [m[0] for m in metrics]
    df = pd.DataFrame.from_dict(mean_results)
    df.index = estimators_list

    for (met_name, met) in metrics:
        met_df = df.loc[:, [met_name in col for col in df.columns]]
        df['mean' + '_' + met_name] = met_df.mean(axis=1)

    output_dict = {
        'estimators': estimators_list,
        'comparison_datasets': datasets,
        'metrics': metrics_list,
        'df': df,
    }
    if parallel_id is None:
        compute_auc_of_auc(output_dict)
    pkl.dump(output_dict, open(model_comparison_file, 'wb'))


def combine_comparisons(path, model):
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
    compute_auc_of_auc(output_dict)

    combined_filename = '.'.join(model_files_sorted[0].split(f'_0.'))
    pkl.dump(output_dict, open(combined_filename, 'wb'))

    for f in model_files_sorted:
        os.remove(f)


def main():

    np.random.seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--ignore_cache', action='store_true')
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--parallel_id', type=int, default=None)
    parser.add_argument('--combine', action='store_true')
    args = parser.parse_args()

    path = os.path.dirname(os.path.realpath(__file__)) + "/comparison_data/"
    path += 'test/' if args.test else 'val/'

    if args.combine:
        combine_comparisons(path, args.model)
        return

    if args.test:
        ests = BEST_ESTIMATORS
    else:
        ests = ALL_ESTIMATORS
    
    if args.model:
        ests = list(filter(lambda x: args.model in x[0].name, ests))
        if args.parallel_id is not None:
            ests = [[est[args.parallel_id]] for est in ests]

    for est in ests:
        run_comparison(path,
                       COMPARISON_DATASETS,
                       METRICS,
                       est,
                       parallel_id=args.parallel_id,
                       verbose=False,
                       ignore_cache=args.ignore_cache,
                       test=args.test,
                       cv_folds=4 if args.test else 1)


if __name__ == "__main__":
    main()
