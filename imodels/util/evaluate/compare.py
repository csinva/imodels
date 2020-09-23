'''Compare different estimators on public datasets
Code modified from https://github.com/tmadl/sklearn-random-bits-forest
'''
import numpy as np
from imodels import *
from sklearn.ensemble.forest import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics.classification import accuracy_score, f1_score
import re, string
from sklearn.model_selection import KFold
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import OneHotEncoder
from scipy.stats.stats import mannwhitneyu, ttest_ind



def dshape(X):
    if len(X.shape) == 1:
        return X.reshape(-1,1)
    else:
        return X if X.shape[0]>X.shape[1] else X.T

def unpack(t):
    while type(t) == list or type(t) == np.ndarray:
        t = t[0]
    return t

def tonumeric(lst):
    lbls = {}
    for t in lst.flatten():
        if unpack(t) not in lbls:
            lbls[unpack(t)] = len(lbls.keys())
    return np.array([lbls[unpack(t)] for t in lst.flatten()])

def getdataset(datasetname, onehot_encode_strings=True):
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
                X[:,i] = tonumeric(X[:,i]) 
            X = OneHotEncoder(categorical_features=cat_ft).fit_transform(X)
    # if sparse, make dense
    try:
        X = X.toarray()
    except:
        pass
    # convert y to monotonically increasing ints
    y = tonumeric(target).astype(int)
    return np.nan_to_num(X.astype(float)),y

def shorten(d):
    return "".join(re.findall("[^\W\d_]", d.lower().replace('datasets-', '').replace('uci', '')))

def print_results_table(results, rows, cols, cellsize=20):
    row_format =("{:>"+str(cellsize)+"}") * (len(cols) + 1)
    print(row_format.format("", *cols))
    print("".join(["="]*cellsize*(len(cols)+1)))
    for rh, row in zip(rows, results):
        print(row_format.format(rh, *row))

def compare_estimators(estimators,
                       datasets,
                       metrics,
                       n_cv_folds = 10, decimals = 3, cellsize = 22):
    if type(estimators) != dict:
        raise Exception("First argument needs to be a dict containing 'name': Estimator pairs")
    if type(metrics) != dict:
        raise Exception("Argument metrics needs to be a dict containing 'name': scoring function pairs")
    cols = []
    for e in range(len(estimators)):
        for mname in metrics.keys():
            cols.append(sorted(estimators.keys())[e]+" "+mname)
    
    rows = []
    mean_results = []
    std_results = []
    for d in datasets:
        print("comparing on dataset",d)
        mean_result = []
        std_result = []
        X, y = getdataset(d)
        rows.append(shorten(d)+" (n="+str(len(y))+")")
        for e in range(len(estimators.keys())):
            est = estimators[sorted(estimators.keys())[e]]
            mresults = [[] for i in range(len(metrics))]
            kf = KFold(n_splits=n_cv_folds)
            for train_idx, test_idx in kf.split(X): #(len(y), n_splits=n_cv_folds):
                est.fit(X[train_idx, :], y[train_idx])
                y_pred = est.predict(X[test_idx, :])
                for i, k in enumerate(sorted(metrics)):
                    try:
                        mresults[i].append(metrics[k](y[test_idx], y_pred))
                    except:
                        mresults[i].append(metrics[k](tonumeric(y[test_idx]), tonumeric(y_pred)))

            for i in range(len(metrics)):
                mean_result.append(np.mean(mresults[i]))
                std_result.append(np.std(mresults[i])/n_cv_folds)
        mean_results.append(mean_result)
        std_results.append(std_result)
    
    results = []
    for i in range(len(datasets)):
        result = []
        
        sigstars = ["*"]*(len(estimators)*len(metrics))
        for j in range(len(estimators)):
            for k in range(len(metrics)):
                for l in range(len(estimators)):
                    #if j != l and mean_results[i][j*len(metrics)+k] < mean_results[i][l*len(metrics)+k] + 2*(std_results[i][j*len(metrics)+k] + std_results[i][l*len(metrics)+k]):
                    if j != l and mean_results[i][j*len(metrics)+k] < mean_results[i][l*len(metrics)+k]:
                        sigstars[j*len(metrics)+k] = ""
        
        for j in range(len(estimators)):
            for k in range(len(metrics)):
                result.append((sigstars[j*len(metrics)+k]+"%."+str(decimals)+"f (SE=%."+str(decimals)+"f)") % (mean_results[i][j*len(metrics)+k], std_results[i][j*len(metrics)+k]))
        results.append(result)

    print_results_table(results, rows, cols, cellsize)
        
    return mean_results, std_results, results

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

    metrics = {
               'Acc.': accuracy_score, 
               'F1score': f1_score
            }
    
    
    estimators = {
                  'RandomForest': RandomForestClassifier(n_estimators=200),
#                   'ExtraTrees': ExtraTreesClassifier(n_estimators=200),
                  'SkopeRulesClassifier': SkopeRulesClassifier(),
                }
    compare_estimators(estimators=estimators, datasets=comparison_datasets, metrics=metrics)