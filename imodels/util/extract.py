from typing import Iterable

import numpy as np
from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from imodels.util.convert import tree_to_rules

def extract_fpgrowth():
    pass

def extract_rulefit():
    pass

def extract_skope(X, y, feature_names, 
                  sample_weight=None,
                  n_estimators=10,
                  max_samples=.8,
                  max_samples_features=1.,
                  bootstrap=False,
                  bootstrap_features=False,
                  max_depths=[3], 
                  max_depth_duplication=None,
                  max_features=1.,
                  min_samples_split=2,
                  n_jobs=1,
                  random_state=None,
                  verbose=0):
    
    ensembles = []
    if not isinstance(max_depths, Iterable):
        max_depths = [max_depths]

    for max_depth in max_depths:
        bagging_clf = BaggingRegressor(
            base_estimator= DecisionTreeRegressor(
                max_depth=max_depth,
                max_features=max_features,
                min_samples_split=min_samples_split
            ),
            n_estimators=n_estimators,
            max_samples=max_samples,
            max_features=max_samples_features,
            bootstrap=bootstrap,
            bootstrap_features=bootstrap_features,
            # oob_score=... XXX may be added
            # if selection on tree perf needed.
            # warm_start=... XXX may be added to increase computation perf.
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose
        )
        ensembles.append(bagging_clf)

    y_reg = y
    if sample_weight is not None:
        sample_weight = check_array(sample_weight, ensure_2d=False)
        weights = sample_weight - sample_weight.min()
        contamination = float(sum(y)) / len(y)
        y_reg = (
                pow(weights, 0.5) * 0.5 / contamination * (y > 0) -
                pow((weights).mean(), 0.5) * (y == 0)
        )
        y_reg = 1. / (1 + np.exp(-y_reg))  # sigmoid

    for e in ensembles[:len(ensembles) // 2]:
        e.fit(X, y)

    for e in ensembles[len(ensembles) // 2:]:
        e.fit(X, y_reg)

    estimators_, estimators_samples_, estimators_features_ = [], [], []
    for ensemble in ensembles:
        estimators_ += ensemble.estimators_
        estimators_samples_ += ensemble.estimators_samples_
        estimators_features_ += ensemble.estimators_features_

    extracted_rules = []
    for estimator, features in zip(estimators_, estimators_features_):
        extracted_rules.append(tree_to_rules(estimator, np.array(feature_names)[features]))
    
    return extracted_rules, estimators_, estimators_samples_, estimators_features_
