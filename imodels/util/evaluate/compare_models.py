'''Compare different estimators on public datasets
Code modified from https://github.com/tmadl/sklearn-random-bits-forest
'''
import numpy as np
import imodels
from sklearn.ensemble.forest import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics.classification import accuracy_score, f1_score
import re, string
from sklearn.model_selection import KFold
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import OneHotEncoder
from scipy.stats.stats import mannwhitneyu, ttest_ind
import pickle as pkl
import os 
from tqdm import tqdm
import time
import pandas as pd

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

def get_dataset(datasetname, onehot_encode_strings=True):
    # load
    dataset = fetch_openml(datasetname)
    # get X and y
    X = dshape(dataset.data)
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
            for i in cat_ft:
                X[:,i] = to_numeric(X[:,i]) 
            X = OneHotEncoder(categorical_features=cat_ft).fit_transform(X)
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
                       n_cv_folds = 10, decimals = 3, cellsize = 22):
    if type(estimators) != list:
        raise Exception("First argument needs to be a list of tuples containing ('name', Estimator pairs)")
    if type(metrics) != list:
        raise Exception("Argument metrics needs to be a list of tuples containing ('name', scoring function pairs)")
    
    mean_results = {d: [] for d in datasets}
    std_results = {d: [] for d in datasets}
    
    # loop over datasets
    for d in tqdm(datasets):
#         print("comparing on dataset", d)
        mean_result = []
        std_result = []
        X, y = get_dataset(d)
        
        # loop over estimators
        for (est_name, est) in estimators:
            mresults = [[] for i in range(len(metrics))]
            
            # loop over folds
            kf = KFold(n_splits=n_cv_folds)
            for train_idx, test_idx in kf.split(X):
                start = time.time()
                est.fit(X[train_idx, :], y[train_idx])
                y_pred = est.predict(X[test_idx, :])
                end = time.time()
                
                # loop over metrics
                for i, (met_name, met) in enumerate(metrics):
                    if met_name == 'Time':
                        mresults[i].append(end - start)
                    else:
                        try:
                            mresults[i].append(met(y[test_idx], y_pred))
                        except:
                            mresults[i].append(met(to_numeric(y[test_idx]), to_numeric(y_pred)))

            for i in range(len(metrics)):
                mean_result.append(np.mean(mresults[i]))
                std_result.append(np.std(mresults[i]) / n_cv_folds)
        
        mean_results[d] = mean_result
        std_results[d] = std_result
        
    return mean_results, std_results

if __name__ == '__main__':


    comparison_datasets = [
            "breast-cancer",
    #         "datasets-UCI breast-w",
    #         "datasets-UCI credit-g",
    #         "uci-20070111 haberman",
            "heart",
            "ionosphere",
    #         "uci-20070111 labor",
    #         "liver-disorders",
    #         "uci-20070111 tic-tac-toe",
    #         "datasets-UCI vote"
        ]

    metrics = [
        ('Acc.', accuracy_score),
        ('F1', f1_score),
        ('Time', None)
    ]
    
    
    estimators = [
        ('RandomForest', RandomForestClassifier(n_estimators=200)),
        #                   'ExtraTrees': ExtraTreesClassifier(n_estimators=200),
        ('SkopeRules', imodels.SkopeRulesClassifier()),
#         ('RuleFit', imodels.RuleFitRegressor(max_rules=10)),
    ]
    mean_results, std_results = compare_estimators(estimators=estimators,
                                                   datasets=comparison_datasets,
                                                   metrics=metrics,
                                                   n_cv_folds=3)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    
    estimators_list = [e[0] for e in estimators]
    metrics_list = [m[0] for m in metrics]
    column_titles = []
    for estimator in estimators_list:
        for metric in metrics_list:
            column_titles.append(estimator + ' ' + metric)
    df = pd.DataFrame.from_dict(mean_results)
    df.index = column_titles

    pkl.dump({
        'estimators': estimators_list,
        'comparison_datasets': comparison_datasets,
        'mean_results': mean_results,
        'std_results': std_results,
        'metrics': metrics_list,
        'df': df,
    }, open(os.path.join(dir_path, 'model_comparisons.pkl'), 'wb'))
    