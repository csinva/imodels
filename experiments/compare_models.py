'''Compare different estimators on public datasets
Code modified from https://github.com/tmadl/sklearn-random-bits-forest
'''
import argparse
import os
import pickle as pkl
import time
from collections import defaultdict, OrderedDict
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.sparse import issparse
from sklearn.base import BaseEstimator
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, make_scorer
from sklearn.model_selection import KFold, train_test_split, cross_validate
from tqdm import tqdm

from experiments.config import COMPARISON_DATASETS, BEST_ESTIMATORS, ALL_ESTIMATORS, ENSEMBLES
from experiments.util import Model, MODEL_COMPARISON_PATH


def get_complexity(estimator: BaseEstimator) -> float:
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


def get_dataset(data_id: int, onehot_encode_strings: bool = True) -> tuple[np.array, np.array]:
    dataset = fetch_openml(data_id=data_id, as_frame=False)
    X = dataset.data
    if issparse(X):
        X = X.toarray()
    y = (dataset.target[0] == dataset.target).astype(int)
    return np.nan_to_num(X.astype('float32')), y


def compute_auc_of_auc(result: dict[str, Any],
                       low_complexity_cutoff: int = 30,
                       max_start_complexity: int = 10,
                       column: str = 'mean_PRAUC') -> None:

    # LOW_COMPLEXITY_CUTOFF: complexity score under which a model is considered interpretable
    # MAX_START_COMPLEXITY: min complexity of curves included in the AUC-of-AUC comparison must be below this value

    result_data = result['df']
    estimators = np.unique(result['estimators'])
    xs = np.empty(len(estimators), dtype=object)
    ys = xs.copy()

    for i, est in enumerate(estimators):

        est_result_df = result_data[result_data.index.str.contains(est)]
        complexities_unsorted = est_result_df['mean_complexity']
        complexity_sort_indices = complexities_unsorted.argsort()
        complexities = complexities_unsorted[complexity_sort_indices]

        roc_aucs = est_result_df[column][complexity_sort_indices]
        xs[i] = complexities.values
        ys[i] = roc_aucs.values

    # filter out curves which start too complex
    mask = list(map(lambda x: min(x) < max_start_complexity, xs))
    xs, ys, estimators = xs[mask], ys[mask], estimators[mask]

    # find overlapping complexity region for roc-of-roc comparison
    auc_of_auc_lb = max([x[0] for x in xs])
    endpts = np.array([x[-1] for x in xs])
    auc_of_auc_ub = min(endpts[endpts > auc_of_auc_lb])
    auc_of_auc_ub = min(auc_of_auc_ub, low_complexity_cutoff)

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


def compare_estimators(estimators: list[Model],
                       datasets: list[tuple[str, int]],
                       metrics: list[tuple[str, Callable]],
                       scorers: dict[str, Callable],
                       n_cv_folds: int,
                       verbose: bool = True,
                       split_seed: int = 0) -> dict[str, list['float or int metric']]:
    if type(estimators) != list:
        raise Exception("First argument needs to be a list of Models")
    if type(metrics) != list:
        raise Exception("Argument metrics needs to be a list containing ('name', callable) pairs")

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
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=split_seed)

        # loop over estimators
        for model in estimators:
            est = model.cls(**model.kwargs)

            if n_cv_folds > 1:
                fold_iterator = KFold(n_splits=n_cv_folds, random_state=0, shuffle=True)
                cv_scores = cross_validate(est, X_train, y_train, cv=fold_iterator, scoring=scorers)
                metric_results = {k.split('_')[1]: np.mean(v) for k, v in cv_scores.items() if k != 'score_time'}
            else:
                if n_cv_folds == 1:
                    X_fit, X_eval, y_fit, y_eval = train_test_split(X_train, y_train, 
                                                                    test_size=0.2, random_state=0)
                else:
                    X_fit, X_eval, y_fit, y_eval = X_train, X_test, y_train, y_test

                start = time.time()
                est.fit(X_fit, y_fit)
                end = time.time()
                y_pred = est.predict(X_eval)
                y_pred_proba = est.predict_proba(X_eval)[:, 1]

                # loop over metrics
                metric_results = {}
                for i, (met_name, met) in enumerate(metrics):
                    if met is not None:
                        tgt = y_pred if met_name == 'accuracy' else y_pred_proba
                        metric_results[met_name] = met(y_eval, tgt)
                metric_results['complexity'] = get_complexity(est)
                metric_results['time'] = end - start

            for met_name, met_val in metric_results.items():
                colname = d[0] + '_' + met_name
                mean_results[colname].append(met_val)

    return mean_results


def run_comparison(path: str, 
                   datasets: list[tuple[str, int]], 
                   metrics: list[tuple[str, Callable]],
                   scorers: dict[str, Callable],
                   estimators: list[Model], 
                   parallel_id: int = None, 
                   split_seed: int = 0, 
                   verbose: bool = False, 
                   ignore_cache: bool = False, 
                   test: bool = False, 
                   cv_folds: int = 4):

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
                                      scorers=scorers,
                                      verbose=verbose,
                                      n_cv_folds=cv_folds,
                                      split_seed=split_seed)

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
        compute_auc_of_auc(output_dict, column='mean_PRAUC')
    pkl.dump(output_dict, open(model_comparison_file, 'wb'))


def main():

    metrics = [
        ('accuracy', accuracy_score),
        ('ROCAUC', roc_auc_score),
        ('PRAUC', average_precision_score),
        ('complexity', None),
        ('time', None)
    ]
    scorers = OrderedDict({
        'accuracy': make_scorer(accuracy_score),
        'ROCAUC': make_scorer(roc_auc_score, needs_proba=True), 
        'PRAUC': make_scorer(average_precision_score, needs_proba=True),
        'complexity': lambda m, x, y: get_complexity(m)
    })

    np.random.seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--cv', action='store_true')
    parser.add_argument('--ensemble', action='store_true')
    parser.add_argument('--ignore_cache', action='store_true')
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--parallel_id', type=int, default=None)
    parser.add_argument('--split_seed', type=int, default=0)
    args = parser.parse_args()

    path = MODEL_COMPARISON_PATH
    path += 'test/' if args.test else 'val/'

    if args.test:
        ests = BEST_ESTIMATORS
        cv_folds = -1
    elif args.ensemble:
        ests = ENSEMBLES
        cv_folds = -1
    else:
        ests = ALL_ESTIMATORS
        cv_folds = 4 if args.cv else 1

    if args.model:
        ests = list(filter(lambda x: args.model in x[0].name, ests))
        if args.parallel_id is not None:
            ests = [[est[args.parallel_id]] for est in ests]

    for est in ests:
        run_comparison(path,
                       COMPARISON_DATASETS,
                       metrics,
                       scorers,
                       est,
                       parallel_id=args.parallel_id,
                       split_seed=args.split_seed,
                       verbose=False,
                       ignore_cache=args.ignore_cache,
                       test=args.test,
                       cv_folds=cv_folds)


if __name__ == "__main__":
    main()
