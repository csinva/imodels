'''Compare different estimators on public datasets
Code modified from https://github.com/tmadl/sklearn-random-bits-forest
'''
import argparse
import os
import pickle as pkl
import time
import warnings
from collections import defaultdict, OrderedDict
from typing import Any, Callable, List, Dict, Tuple

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, make_scorer
from sklearn.model_selection import KFold, train_test_split, cross_validate
from tqdm import tqdm

from experiments.config.config_general import DATASETS
from experiments.config.util import get_estimators_for_dataset, get_ensembles_for_dataset
from experiments.util import Model, MODEL_COMPARISON_PATH, get_clean_dataset, get_best_accuracy, remove_x_axis_duplicates

warnings.filterwarnings("ignore", message="Bins whose width")


def get_complexity(estimator: BaseEstimator) -> float:
    if isinstance(estimator, (RandomForestClassifier, GradientBoostingClassifier)):
        complexity = 0
        for tree in estimator.estimators_:
            if type(tree) is np.ndarray:
                tree = tree[0]
            complexity += (2 ** tree.get_depth()) * tree.get_depth()
        return complexity
    else:
        return estimator.complexity_


def compute_meta_auc(result_data: pd.DataFrame,
                     prefix: str = '',
                     low_complexity_cutoff: int = 30,
                     max_start_complexity: int = 10) -> Tuple[pd.DataFrame, Tuple[float]]:

    # LOW_COMPLEXITY_CUTOFF: complexity score under which a model is considered interpretable
    # MAX_START_COMPLEXITY: min complexity of curves included in the AUC-of-AUC comparison must be below this value

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
    meta_auc_ub = min(meta_auc_ub, low_complexity_cutoff)

    # handle non-overlapping curves
    endpt_after_lb = endpts > meta_auc_lb
    eligible = start_under_10 & endpt_after_lb

    # compute AUC of interpolated curves in overlap region
    meta_aucs = defaultdict(lambda:[])
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


def compare_estimators(estimators: List[Model],
                       datasets: List[Tuple],
                       metrics: List[Tuple[str, Callable]],
                       scorers: Dict[str, Callable],
                       n_cv_folds: int,
                       low_data: bool,
                       verbose: bool = True,
                       split_seed: int = 0) -> Tuple[dict, dict]:
    if type(estimators) != list:
        raise Exception("First argument needs to be a list of Models")
    if type(metrics) != list:
        raise Exception("Argument metrics needs to be a list containing ('name', callable) pairs")

    mean_results = defaultdict(lambda: [])
    for e in estimators:
        mean_results[e.vary_param].append(e.vary_param_val)
        if e.fixed_param is not None:
            mean_results[e.fixed_param].append(e.fixed_param_val)

    rules = mean_results.copy()

    # loop over datasets
    for d in datasets:
        if verbose:
            print("comparing on dataset", d[0])
        X, y, feat_names = get_clean_dataset(d[1])
        if low_data:
            test_size = X.shape[0] - 1000
        else:
            test_size = 0.2
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=split_seed)

        # loop over estimators
        for model in tqdm(estimators):
            est = model.cls(**model.kwargs)

            if n_cv_folds > 1:
                fold_iterator = KFold(n_splits=n_cv_folds, random_state=split_seed, shuffle=True)
                cv_scores = cross_validate(est, X_train, y_train, cv=fold_iterator, scoring=scorers)
                metric_results = {k.split('_')[1]: np.mean(v) for k, v in cv_scores.items() if k != 'score_time'}
            else:
                if n_cv_folds == 1:
                    X_fit, X_eval, y_fit, y_eval = train_test_split(X_train, y_train, 
                                                                    test_size=0.2, random_state=split_seed)
                else:
                    X_fit, X_eval, y_fit, y_eval = X_train, X_test, y_train, y_test

                start = time.time()
                if type(est) in [RandomForestClassifier, GradientBoostingClassifier]:
                    est.fit(X_fit, y_fit)
                else:
                    est.fit(X_fit, y_fit, feature_names=feat_names)
                end = time.time()

                y_pred_proba = est.predict_proba(X_eval)[:, 1]

                if hasattr(est, 'rules_'):
                    rules[d[0]].append(est.rules_)
                else:
                    rules[d[0]].append('')

                # loop over metrics
                metric_results = {}
                for i, (met_name, met) in enumerate(metrics):
                    if met is not None:
                        metric_results[met_name] = met(y_eval, y_pred_proba)
                metric_results['best_accuracy'] = get_best_accuracy(y_eval, y_pred_proba)
                metric_results['complexity'] = get_complexity(est)
                metric_results['time'] = end - start

            for met_name, met_val in metric_results.items():
                colname = d[0] + '_' + met_name
                mean_results[colname].append(met_val)

    return mean_results, rules


def run_comparison(path: str, 
                   datasets: List[Tuple], 
                   metrics: List[Tuple[str, Callable]],
                   scorers: Dict[str, Callable],
                   estimators: List[Model], 
                   parallel_id: int = None, 
                   split_seed: int = 0, 
                   verbose: bool = False, 
                   ignore_cache: bool = False, 
                   test: bool = False,
                   low_data: bool = False, 
                   cv_folds: int = 4):

    estimator_name = estimators[0].name.split(' - ')[0]
    if test:
        model_comparison_file = path + f'{estimator_name}_test_comparisons.pkl'
    else:
        model_comparison_file = path + f'{estimator_name}_comparisons.pkl'
    if parallel_id is not None:
        model_comparison_file = f'_{parallel_id[0]}.'.join(model_comparison_file.split('.'))

    if os.path.isfile(model_comparison_file) and not ignore_cache:
        print(f'{estimator_name} results already computed and cached. use --ignore_cache to recompute')
        return

    mean_results, rules = compare_estimators(estimators=estimators,
                                      datasets=datasets,
                                      metrics=metrics,
                                      scorers=scorers,
                                      verbose=verbose,
                                      n_cv_folds=cv_folds,
                                      low_data=low_data,
                                      split_seed=split_seed)

    estimators_list = [e.name for e in estimators]
    metrics_list = [m[0] for m in metrics]
    df = pd.DataFrame.from_dict(mean_results)
    df.index = estimators_list
    rule_df = pd.DataFrame.from_dict(rules)
    rule_df.index = estimators_list

    # easy_df = df.loc[:, [any([d in col for d in EASY_DATASETS]) for col in df.columns]].copy()
    # med_df = df.loc[:, [any([d in col for d in MEDIUM_DATASETS]) for col in df.columns]].copy()
    # hard_df = df.loc[:, [any([d in col for d in HARD_DATASETS]) for col in df.columns]].copy()
    # all_df = df.copy()
    # level_dfs = [(med_df, 'med'), (hard_df, 'hard'), (all_df, 'all')]

    # for curr_df, prefix in level_dfs:
    for (met_name, met) in metrics:
        # colname = f'{prefix}_mean_{met_name}'
        colname = f'mean_{met_name}'
        # met_df = curr_df.loc[:, [met_name in col for col in curr_df.columns]]
        met_df = df.iloc[:, 1:].loc[:, [met_name in col for col in df.iloc[:, 1:].columns]]
        df[colname] = met_df.mean(axis=1)
        # curr_df[colname] = met_df.mean(axis=1)
        # df[colname] = curr_df[colname]
    if parallel_id is None:
        try:
            meta_auc_df = compute_meta_auc(df)
        except ValueError as e:
            warnings.warn(f'bad complexity range')
            meta_auc_df = None

    # meta_auc_df = pd.DataFrame([])
    # if parallel_id is None:
    #     for curr_df, prefix in level_dfs:
    #         try:
    #             curr_meta_auc_df = compute_meta_auc(curr_df, prefix)
    #             meta_auc_df = pd.concat((meta_auc_df, curr_meta_auc_df), axis=1)
    #         except ValueError as e:
    #             warnings.warn(f'bad complexity range for {prefix} datasets')

    output_dict = {
        'estimators': estimators_list,
        'comparison_datasets': datasets,
        'metrics': metrics_list,
        'df': df,
    }
    if parallel_id is None:
        output_dict['meta_auc_df'] = meta_auc_df
    if cv_folds <= 1:
        output_dict['rule_df'] = rule_df
    pkl.dump(output_dict, open(model_comparison_file, 'wb'))


def main():

    metrics = [
        ('rocauc', roc_auc_score),
        ('avg_precision', average_precision_score),
        ('best_accuracy', None),
        ('complexity', None),
        ('time', None)
    ]
    scorers = OrderedDict({
        'accuracy': make_scorer(accuracy_score),
        'ROCAUC': make_scorer(roc_auc_score, needs_proba=True), 
        'PRAUC': make_scorer(average_precision_score, needs_proba=True),
        'complexity': lambda m, x, y: get_complexity(m)
    })

    np.random.seed(1)

    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--cv', action='store_true')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--ignore_cache', action='store_true')
    parser.add_argument('--low_data', action='store_true')
    parser.add_argument('--ensemble', action='store_true', default=False)
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--parallel_id', nargs='+', type=int, default=None)
    parser.add_argument('--split_seed', type=int, default=0)
    args = parser.parse_args()

    path = MODEL_COMPARISON_PATH
    path += 'low_data/' if args.low_data else 'reg_data/'
    path += f'{args.dataset}/'
    if args.test:
        path += 'test/'
    elif args.cv:
        path += 'cv/'
    else:
        path += 'val/'

    if args.test:
        cv_folds = -1
    else:
        cv_folds = 4 if args.cv else 1

    datasets = list(filter(lambda x: args.dataset == x[0], DATASETS))
    if args.ensemble:
        ests = get_ensembles_for_dataset(args.dataset, test=args.test)
    else:
        ests = get_estimators_for_dataset(args.dataset, test=args.test)


    if args.model:
        ests = list(filter(lambda x: args.model in x[0].name, ests))
    
    if args.parallel_id is not None and len(args.parallel_id) > 1:
            ests = [est[args.parallel_id[0]:args.parallel_id[1]+1] for est in ests]
    elif args.parallel_id is not None:
        ests = [[est[args.parallel_id[0]]] for est in ests]


    for est in ests:
        run_comparison(path,
                       datasets,
                       metrics,
                       scorers,
                       est,
                       parallel_id=args.parallel_id,
                       split_seed=args.split_seed,
                       verbose=False,
                       ignore_cache=args.ignore_cache,
                       test=args.test,
                       low_data=args.low_data,
                       cv_folds=cv_folds)


if __name__ == "__main__":
    main()
