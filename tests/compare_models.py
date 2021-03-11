'''Compare different estimators on public datasets
Code modified from https://github.com/tmadl/sklearn-random-bits-forest
'''
import os 
import time
import re, string
import pickle as pkl
import argparse
import logging

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import OneHotEncoder
from scipy.stats.stats import mannwhitneyu, ttest_ind
from tqdm import tqdm
import pandas as pd
import imodels


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
    ('Accuracy', accuracy_score),
    ('ROC Score', roc_auc_score),
    ('Time', None),
    ('Complexity', None)
]

BEST_ESTIMATORS = [
    [('random_forest', RandomForestClassifier(n_estimators=n, max_depth=2)) for n in np.linspace(2, 8, 5, dtype=int)],
    [('gradient_boosting', GradientBoostingClassifier(n_estimators=n, max_depth=2)) for n in np.linspace(2, 8, 5, dtype=int)],
    [('skope_rules', imodels.SkopeRulesClassifier(n_estimators=n, max_depth=2)) for n in np.linspace(2, 200, 5, dtype=int)],
    [('rulefit', imodels.RuleFitClassifier(max_rules=n, tree_size=2)) for n in np.linspace(2, 100, 5, dtype=int)],
    [('fplasso', imodels.FPLassoClassifier(max_rules=n, maxcardinality=2)) for n in np.linspace(2, 60, 5, dtype=int)],
    [('fpskope', imodels.FPSkopeClassifier(maxcardinality=n, max_depth_duplication=3)) for n in np.arange(1, 5)],
    [('brl', imodels.BayesianRuleListClassifier(listlengthprior=n, maxcardinality=2)) for n in [2, 4, 8, 16]],
    [('grl', imodels.GreedyRuleListClassifier(max_depth=n)) for n in np.arange(1, 6)],
    [('oner', imodels.OneRClassifier(max_depth=n)) for n in np.arange(1, 6)],
    [('brs', imodels.BoostedRulesClassifier(n_estimators=n)) for n in np.linspace(2, 32, 5, dtype=int)]
]

ALL_ESTIMATORS = []
ALL_ESTIMATORS.append(
    [('random_forest - depth_1', RandomForestClassifier(n_estimators=n, max_depth=1)) for n in np.linspace(2, 40, 5, dtype=int)]
    + [('random_forest - depth_2', RandomForestClassifier(n_estimators=n, max_depth=2)) for n in np.linspace(2, 15, 5, dtype=int)]
    + [('random_forest - depth_3', RandomForestClassifier(n_estimators=n, max_depth=3)) for n in np.arange(2, 8)]
)
ALL_ESTIMATORS.append(
    [('gradient_boosting - depth_1', GradientBoostingClassifier(n_estimators=n, max_depth=1)) for n in np.linspace(2, 40, 5, dtype=int)]
    + [('gradient_boosting - depth_2', GradientBoostingClassifier(n_estimators=n, max_depth=2)) for n in np.linspace(2, 15, 5, dtype=int)]
    + [('gradient_boosting - depth_3', GradientBoostingClassifier(n_estimators=n, max_depth=3)) for n in np.arange(2, 8)]
)
ALL_ESTIMATORS.append(
    [('skope_rules - depth_1', imodels.SkopeRulesClassifier(n_estimators=n, max_depth=1)) for n in np.linspace(2, 200, 5, dtype=int)]
    + [('skope_rules - depth_2', imodels.SkopeRulesClassifier(n_estimators=n, max_depth=2)) for n in np.linspace(2, 200, 5, dtype=int)]
    + [('skope_rules - depth_3', imodels.SkopeRulesClassifier(n_estimators=n, max_depth=3)) for n in np.linspace(2, 80, 5, dtype=int)]
)
ALL_ESTIMATORS.append(
    [('rulefit - depth_1', imodels.RuleFitClassifier(max_rules=n, tree_size=2)) for n in np.linspace(2, 100, 5, dtype=int)]
    + [('rulefit - depth_2', imodels.RuleFitClassifier(max_rules=n, tree_size=4)) for n in np.linspace(2, 50, 5, dtype=int)]
    + [('rulefit - depth_3', imodels.RuleFitClassifier(max_rules=n, tree_size=8)) for n in np.linspace(2, 50, 5, dtype=int)]
)
ALL_ESTIMATORS.append(
    [('fplasso - max_card_1', imodels.FPLassoClassifier(max_rules=n, maxcardinality=1)) for n in np.linspace(2, 100, 5, dtype=int)]
    + [('fplasso - max_card_2', imodels.FPLassoClassifier(max_rules=n, maxcardinality=2)) for n in np.linspace(2, 60, 5, dtype=int)]
    + [('fplasso - max_card_3', imodels.FPLassoClassifier(max_rules=n, maxcardinality=3)) for n in np.linspace(2, 50, 5, dtype=int)]
)
ALL_ESTIMATORS.append(
    [('fpskope - No dedup', imodels.FPSkopeClassifier(maxcardinality=n,  max_depth_duplication=None)) for n in [1, 2]]
    + [('fpskope - max_dedup_1', imodels.FPSkopeClassifier(maxcardinality=n,  max_depth_duplication=1)) for n in [1, 2, 3, 4]]
    + [('fpskope - max_dedup_2', imodels.FPSkopeClassifier(maxcardinality=n,  max_depth_duplication=2)) for n in [1, 2, 3, 4]]
    + [('fpskope - max_dedup_3', imodels.FPSkopeClassifier(maxcardinality=n,  max_depth_duplication=3)) for n in [1, 2, 3, 4]]
)
ALL_ESTIMATORS.append(
    [('brl - max_card_1', imodels.BayesianRuleListClassifier(listlengthprior=n, maxcardinality=1)) for n in [2, 4, 8, 16, 20]]
    + [('brl - max_card_2', imodels.BayesianRuleListClassifier(listlengthprior=n, maxcardinality=2)) for n in [2, 4, 8, 16]]
    + [('brl - max_card_3', imodels.BayesianRuleListClassifier(listlengthprior=n, maxcardinality=3)) for n in [2, 4, 8, 16]]
)


def dshape(X):
    if len(X.shape) == 1:
        return X.reshape(-1,1)
    else:
        return X if X.shape[0]>X.shape[1] else X.T


def unpack(t):
    while type(t) == list or type(t) == np.ndarray:
        t = t[0]
    return t


def to_numeric(lst):
    lbls = {}
    for t in lst.flatten():
        if unpack(t) not in lbls:
            lbls[unpack(t)] = len(lbls.keys())
    return np.array([lbls[unpack(t)] for t in lst.flatten()])


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
    # load
    dataset = fetch_openml(data_id=data_id)
    # get X and y
    X = dshape(dataset.data)
    if type(X) == pd.DataFrame:
        X = X.values
    try:
        target = dshape(dataset.target)
    except:
        print("WARNING: No target found. Taking last column of data matrix as target")
        target = X[:, -1]
        X = X[:, :-1]
    if len(target.shape)>1 and target.shape[1]>X.shape[1]: # some mldata sets are mixed up...
        X = target
        target = dshape(dataset.data)
    if len(X.shape) == 1 or X.shape[1] <= 1:
        for k in dataset.keys():
            if k != 'data' and k != 'target' and len(dataset[k]) == X.shape[1]:
                X = np.hstack((X, dshape(dataset[k])))
    # one-hot for categorical values
    if onehot_encode_strings:
        cat_ft=[i for i in range(X.shape[1]) if 'str' in str(type(unpack(X[0,i]))) or 'unicode' in str(type(unpack(X[0,i])))]
        if len(cat_ft):
            X = OneHotEncoder().fit_transform(X)
    # if sparse, make dense
    try:
        X = X.toarray()
    except:
        pass
    # convert y to monotonically increasing ints
    y = to_numeric(target).astype(int)
    return np.nan_to_num(X.astype(float)),y


def compare_estimators(estimators: list,
                       datasets,
                       metrics: list,
                       n_cv_folds = 10, decimals = 3, cellsize = 22, verbose = True):
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
                    if met_name == 'Time':
                        mresults[i].append(end - start)
                    elif met_name == 'Complexity':
                        if est_name != 'MLPClassifier (sklearn)':
                            mresults[i].append(get_complexity(est))
                    else:
                        try:
                            mresults[i].append(met(y[test_idx], y_pred))
                        except:
                            mresults[i].append(met(to_numeric(y[test_idx]), to_numeric(y_pred)))

            for i in range(len(mresults)):
                mean_result.append(np.mean(mresults[i]))
                std_result.append(np.std(mresults[i]) / n_cv_folds)
        
        mean_results[d[0]] = mean_result
        std_results[d[0]] = std_result
        
    return mean_results, std_results


def run_comparison(path, datasets, metrics, estimators, average=True, verbose=False, ignore_cache=False, test=False, cv_folds=4):

    estimator_name = estimators[0][0].split(' - ')[0]
    if test:
        model_comparison_file = path + f'{estimator_name}_test_comparisons.pkl'
    else:
        model_comparison_file = path + f'{estimator_name}_comparisons.pkl'
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

    if average:
        df = df.mean(axis=1)
    
    output_dict = {
        'estimators': estimators_list,
        'comparison_datasets': datasets,
        'mean_results': mean_results,
        'std_results': std_results,
        'metrics': metrics_list,
        'df': df,
    }
    pkl.dump(output_dict, open(model_comparison_file, 'wb'))

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--val', action='store_true')
    parser.add_argument('--ignore_cache', action='store_true')
    parser.add_argument('--model', type=str, default=None)
    args = parser.parse_args()

    path = os.path.dirname(os.path.realpath(__file__)) + "/test_data/comparison_data/"

    if args.test:
        ests = BEST_ESTIMATORS
    else:
        ests = ALL_ESTIMATORS
    
    if args.model:
        ests = list(filter(lambda x: args.model in x[0][0], ests))

    for est in ests:
        run_comparison(path, 
                       COMPARISON_DATASETS,
                       METRICS, 
                       est, 
                       average=True,
                       verbose=False,
                       ignore_cache=args.ignore_cache,
                       test=args.test,
                       cv_folds=1 if args.val else 4)


if __name__ == "__main__":
    main()
