from typing import Iterable, Tuple, List

import numpy as np
import pandas as pd
from sklearn.ensemble import BaggingClassifier, BaggingRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from mlxtend.frequent_patterns import fpgrowth

from imodels.util.convert import tree_to_rules
from imodels.util.discretization.mdlp import BRLDiscretizer


def extract_fpgrowth(X, y,
                     feature_labels=None,
                     minsupport=0.1,
                     maxcardinality=2,
                     undiscretized_features=[],
                     verbose=False) -> Tuple[List[Tuple], BRLDiscretizer]:

    # deal with pandas data
    if type(X) in [pd.DataFrame, pd.Series]:
        if feature_labels is None:
            feature_labels = X.columns
        X = X.values
    if type(y) in [pd.DataFrame, pd.Series]:
        y = y.values

    if feature_labels is None:
        feature_labels = [f'feature_{i}' for i in range(X.shape[1])]
    
    discretizer = BRLDiscretizer(X, y, feature_labels=feature_labels, verbose=verbose)
    X = discretizer.discretize_mixed_data(X, y, undiscretized_features)
    X_df_onehot = discretizer.onehot_df
    
    # Now find frequent itemsets
    itemsets_df = fpgrowth(X_df_onehot, min_support=minsupport, max_len=maxcardinality)
    itemsets_indices = [tuple(s[1]) for s in itemsets_df.values]
    itemsets = [np.array(X_df_onehot.columns)[list(inds)] for inds in itemsets_indices]
    itemsets = list(map(tuple, itemsets))
    if verbose:
        print(len(itemsets), 'rules mined')

    return itemsets, discretizer


def extract_rulefit(X, y, feature_names,
                    tree_size=4,
                    max_rules=2000,
                    memory_par=0.01,
                    tree_generator=None,
                    exp_rand_tree_size=True,
                    random_state=None) -> List[str]:

    if tree_generator is None:
        n_estimators_default = int(np.ceil(max_rules / tree_size))
        sample_fract_ = min(0.5, (100 + 6 * np.sqrt(X.shape[0])) / X.shape[0])

        tree_generator = GradientBoostingRegressor(n_estimators=n_estimators_default,
                                                    max_leaf_nodes=tree_size,
                                                    learning_rate=memory_par,
                                                    subsample=sample_fract_,
                                                    random_state=random_state,
                                                    max_depth=100)

    if type(tree_generator) not in [GradientBoostingRegressor, RandomForestRegressor]:
        raise ValueError("RuleFit only works with RandomForest and BoostingRegressor")

    ## fit tree generator
    if not exp_rand_tree_size:  # simply fit with constant tree size
        tree_generator.fit(X, y)
    else:  # randomise tree size as per Friedman 2005 Sec 3.3
        np.random.seed(random_state)
        tree_sizes = np.random.exponential(scale=tree_size - 2,
                                            size=int(np.ceil(max_rules * 2 / tree_size)))
        tree_sizes = np.asarray([2 + np.floor(tree_sizes[i_]) for i_ in np.arange(len(tree_sizes))], dtype=int)
        i = int(len(tree_sizes) / 4)
        while np.sum(tree_sizes[0:i]) < max_rules:
            i = i + 1
        tree_sizes = tree_sizes[0:i]
        tree_generator.set_params(warm_start=True)
        curr_est_ = 0
        for i_size in np.arange(len(tree_sizes)):
            size = tree_sizes[i_size]
            tree_generator.set_params(n_estimators=curr_est_ + 1)
            tree_generator.set_params(max_leaf_nodes=size)
            random_state_add = random_state if random_state else 0
            tree_generator.set_params(
                random_state=i_size + random_state_add)  # warm_state=True seems to reset random_state, such that the trees are highly correlated, unless we manually change the random_sate here.
            tree_generator.fit(np.copy(X, order='C'), np.copy(y, order='C'))
            curr_est_ = curr_est_ + 1
        tree_generator.set_params(warm_start=False)

    if isinstance(tree_generator, RandomForestRegressor):
        estimators_ = [[x] for x in tree_generator.estimators_]
    else:
        estimators_ = tree_generator.estimators_

    seen_antecedents = set()
    extracted_rules = [] 
    for estimator in estimators_:
        for rule_value_pair in tree_to_rules(estimator[0], np.array(feature_names), prediction_values=True):
            if rule_value_pair[0] not in seen_antecedents:
                extracted_rules.append(rule_value_pair)
                seen_antecedents.add(rule_value_pair[0])
    
    extracted_rules = sorted(extracted_rules, key=lambda x: x[1])
    extracted_rules = list(map(lambda x: x[0], extracted_rules))
    return extracted_rules


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
                  verbose=0) -> Tuple[List[str], List[np.array], List[np.array]]:
    
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
    
    return extracted_rules, estimators_samples_, estimators_features_
