'''Compare different estimators on public datasets
Code modified from https://github.com/tmadl/sklearn-random-bits-forest
'''
import argparse
import glob
import os
import pickle as pkl
import time
from typing import Dict, Any

import imodels
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.sparse import issparse
from sklearn.datasets import fetch_openml
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
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

BEST_ESTIMATORS = [
    [('random_forest', RandomForestClassifier(n_estimators=n, max_depth=2)) for n in np.arange(1, 8)],
    [('gradient_boosting', GradientBoostingClassifier(n_estimators=n, max_depth=1)) for n in np.linspace(1, 20, 10, dtype=int)],
    [('skope_rules', imodels.SkopeRulesClassifier(n_estimators=n, max_depth=1)) for n in np.linspace(2, 200, 10, dtype=int)],
    [('rulefit', imodels.RuleFitClassifier(max_rules=n, tree_size=2)) for n in np.linspace(2, 100, 10, dtype=int)],
    [('fplasso', imodels.FPLassoClassifier(max_rules=n, maxcardinality=1)) for n in np.linspace(2, 100, 10, dtype=int)],
    [('fpskope', imodels.FPSkopeClassifier(maxcardinality=n, max_depth_duplication=3)) for n in np.arange(1, 5)],
    [('brl', imodels.BayesianRuleListClassifier(listlengthprior=n, maxcardinality=2)) for n in np.linspace(1, 16, 8)],
    [('grl', imodels.GreedyRuleListClassifier(max_depth=n)) for n in np.arange(1, 6)],
    [('oner', imodels.OneRClassifier(max_depth=n)) for n in np.arange(1, 6)],
    [('brs', imodels.BoostedRulesClassifier(n_estimators=n)) for n in np.linspace(1, 32, 10, dtype=int)]
]

ALL_ESTIMATORS = []
ALL_ESTIMATORS.append(
    [('random_forest - depth_1', RandomForestClassifier(n_estimators=n, max_depth=1)) for n in np.linspace(1, 40, 10, dtype=int)]
    + [('random_forest - depth_2', RandomForestClassifier(n_estimators=n, max_depth=2)) for n in np.linspace(1, 15, 10, dtype=int)]
    + [('random_forest - depth_3', RandomForestClassifier(n_estimators=n, max_depth=3)) for n in np.arange(1, 8)]
)
ALL_ESTIMATORS.append(
    [('gradient_boosting - depth_1', GradientBoostingClassifier(n_estimators=n, max_depth=1)) for n in np.linspace(1, 40, 10, dtype=int)]
    + [('gradient_boosting - depth_2', GradientBoostingClassifier(n_estimators=n, max_depth=2)) for n in np.linspace(1, 15, 10, dtype=int)]
    + [('gradient_boosting - depth_3', GradientBoostingClassifier(n_estimators=n, max_depth=3)) for n in np.arange(1, 8)]
)
ALL_ESTIMATORS.append(
    [('skope_rules - depth_1', imodels.SkopeRulesClassifier(n_estimators=n, max_depth=1)) for n in np.linspace(2, 200, 10, dtype=int)]
    + [('skope_rules - depth_2', imodels.SkopeRulesClassifier(n_estimators=n, max_depth=2)) for n in np.linspace(2, 200, 10, dtype=int)]
    + [('skope_rules - depth_3', imodels.SkopeRulesClassifier(n_estimators=n, max_depth=3)) for n in np.linspace(2, 80, 10, dtype=int)]
)
ALL_ESTIMATORS.append(
    [('rulefit - depth_1', imodels.RuleFitClassifier(max_rules=n, tree_size=2)) for n in np.linspace(2, 100, 10, dtype=int)]
    + [('rulefit - depth_2', imodels.RuleFitClassifier(max_rules=n, tree_size=4)) for n in np.linspace(2, 50, 10, dtype=int)]
    + [('rulefit - depth_3', imodels.RuleFitClassifier(max_rules=n, tree_size=8)) for n in np.linspace(2, 50, 10, dtype=int)]
)
ALL_ESTIMATORS.append(
    [('fplasso - max_card_1', imodels.FPLassoClassifier(max_rules=n, maxcardinality=1)) for n in np.linspace(2, 100, 10, dtype=int)]
    + [('fplasso - max_card_2', imodels.FPLassoClassifier(max_rules=n, maxcardinality=2)) for n in np.linspace(2, 60, 10, dtype=int)]
    + [('fplasso - max_card_3', imodels.FPLassoClassifier(max_rules=n, maxcardinality=3)) for n in np.linspace(2, 50, 10, dtype=int)]
)
ALL_ESTIMATORS.append(
    [('fpskope - No dedup', imodels.FPSkopeClassifier(maxcardinality=n,  max_depth_duplication=None)) for n in [1, 2]]
    + [('fpskope - max_dedup_1', imodels.FPSkopeClassifier(maxcardinality=n,  max_depth_duplication=1)) for n in [1, 2, 3, 4]]
    + [('fpskope - max_dedup_2', imodels.FPSkopeClassifier(maxcardinality=n,  max_depth_duplication=2)) for n in [1, 2, 3, 4]]
    + [('fpskope - max_dedup_3', imodels.FPSkopeClassifier(maxcardinality=n,  max_depth_duplication=3)) for n in [1, 2, 3, 4]]
)
ALL_ESTIMATORS.append(
    [('brl - max_card_1', imodels.BayesianRuleListClassifier(listlengthprior=n, maxcardinality=1)) for n in np.linspace(1, 20, 10)]
    + [('brl - max_card_2', imodels.BayesianRuleListClassifier(listlengthprior=n, maxcardinality=2)) for n in np.linspace(1, 16, 8)]
    + [('brl - max_card_3', imodels.BayesianRuleListClassifier(listlengthprior=n, maxcardinality=3)) for n in np.linspace(1, 16, 8)]
)


def get_complexity(estimator):
    if isinstance(estimator, (RandomForestClassifier, GradientBoostingClassifier)):
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
    mean_results = result['df']['mean']
    estimators = np.unique(result['estimators'])
    xs = np.empty(len(estimators), dtype=object)
    ys = xs.copy()

    for i, est in enumerate(estimators):

        est_result_df = mean_results[mean_results.index.str.contains(est)]
        complexities_unsorted = est_result_df[est_result_df.index.str.contains('complexity')]
        complexity_sort_indices = complexities_unsorted.argsort()
        complexities = complexities_unsorted[complexity_sort_indices]

        roc_aucs = (
                est_result_df[est_result_df.index.str.contains('ROC')][complexity_sort_indices]
        )
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
    
    mean_results = {d[0]: [] for d in datasets}
    std_results = {d[0]: [] for d in datasets}
    
    # loop over datasets
    for d in tqdm(datasets):
        if verbose:
            print("comparing on dataset", d[0])
        mean_result = []
        std_result = []
        X, y = get_dataset(d[1])
        
        # loop over estimators
        for (est_name, est) in estimators:
            mresults = [[] for i in range(len(metrics))]
            
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
                end = time.time()
                
                # loop over metrics
                for i, (met_name, met) in enumerate(metrics):
                    if met_name == 'time':
                        mresults[i].append(end - start)
                    elif met_name == 'complexity':
                        mresults[i].append(get_complexity(est))
                    else:
                        mresults[i].append(met(y[test_idx], y_pred))

            for i in range(len(mresults)):
                mean_result.append(np.mean(mresults[i]))
                std_result.append(np.std(mresults[i]) / n_cv_folds)
        
        mean_results[d[0]] = mean_result
        std_results[d[0]] = std_result
        
    return mean_results, std_results


def run_comparison(path, datasets, metrics, estimators, 
                   parallel_id=None, verbose=False, ignore_cache=False, test=False, cv_folds=4):

    estimator_name = estimators[0][0].split(' - ')[0]
    if test:
        model_comparison_file = path + f'{estimator_name}_test_comparisons.pkl'
    else:
        model_comparison_file = path + f'{estimator_name}_comparisons.pkl'
    if parallel_id is not None:
        model_comparison_file = f'_{parallel_id}.'.join(model_comparison_file.split('.'))

    if os.path.isfile(model_comparison_file) and not ignore_cache:
        print(f'{estimator_name} results already computed and cached. use --ignore_cache to recompute')
        return

    mean_results, std_results = compare_estimators(estimators=estimators,
                                                   datasets=datasets,
                                                   metrics=metrics,
                                                   verbose=verbose,
                                                   n_cv_folds=cv_folds)
    
    estimators_list = [e[0] for e in estimators]
    metrics_list = [m[0] for m in metrics]
    column_titles = []
    for estimator in estimators_list:
        for metric in metrics_list:
            column_titles.append(estimator + ' ' + metric)
    df = pd.DataFrame.from_dict(mean_results)
    df.index = column_titles

    df['mean'], df['std'] = df.mean(axis=1), df.std(axis=1)

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
    model_files = list(filter(lambda x: model in x, all_files))
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
        ests = list(filter(lambda x: args.model in x[0][0], ests))
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
