from typing import Iterable, Tuple, List

import numpy as np
import pandas as pd
from mlxtend import frequent_patterns as mlx
from sklearn.ensemble import BaggingRegressor, GradientBoostingRegressor, RandomForestRegressor, \
    GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils.validation import check_array
import inspect
from imodels.util import rule, convert


def extract_fpgrowth(X,
                     minsupport=0.1,
                     maxcardinality=2,
                     verbose=False) -> List[Tuple]:

    itemsets_df = mlx.fpgrowth(
        X, min_support=minsupport, max_len=maxcardinality)
    itemsets_indices = [tuple(s[1]) for s in itemsets_df.values]
    itemsets = [np.array(X.columns)[list(inds)] for inds in itemsets_indices]
    itemsets = list(map(tuple, itemsets))
    if verbose:
        print(len(itemsets), 'rules mined')

    return itemsets


def extract_rulefit(X, y, feature_names,
                    n_estimators=10,
                    tree_size=4,
                    memory_par=0.01,
                    tree_generator=None,
                    exp_rand_tree_size=True,
                    random_state=None) -> List[str]:
    if tree_generator is None:
        sample_fract_ = min(0.5, (100 + 6 * np.sqrt(X.shape[0])) / X.shape[0])

        tree_generator = GradientBoostingRegressor(n_estimators=n_estimators,
                                                   max_leaf_nodes=tree_size,
                                                   learning_rate=memory_par,
                                                   subsample=sample_fract_,
                                                   random_state=random_state,
                                                   max_depth=100)

    if type(tree_generator) not in [GradientBoostingClassifier, GradientBoostingRegressor,
                                    RandomForestRegressor, RandomForestClassifier]:
        raise ValueError(
            "RuleFit only works with GradientBoostingClassifier(), GradientBoostingRegressor(), "
            "RandomForestRegressor() or RandomForestClassifier()")

    # fit tree generator
    if not exp_rand_tree_size:  # simply fit with constant tree size
        tree_generator.fit(X, y)
    else:  # randomise tree size as per Friedman 2005 Sec 3.3
        np.random.seed(random_state)
        tree_sizes = np.random.exponential(
            scale=tree_size - 2, size=n_estimators)
        tree_sizes = np.asarray([2 + np.floor(tree_sizes[i_])
                                for i_ in np.arange(len(tree_sizes))], dtype=int)
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

    if isinstance(tree_generator, RandomForestRegressor) or isinstance(tree_generator, RandomForestClassifier):
        estimators_ = [[x] for x in tree_generator.estimators_]
    else:
        estimators_ = tree_generator.estimators_

    seen_rules = set()
    extracted_rules = []
    for estimator in estimators_:
        for rule_value_pair in convert.tree_to_rules(estimator[0], np.array(feature_names), prediction_values=True):

            rule_obj = rule.Rule(rule_value_pair[0])

            if rule_obj not in seen_rules:
                extracted_rules.append(rule_value_pair)
                seen_rules.add(rule_obj)

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
                  max_features=1.,
                  min_samples_split=2,
                  n_jobs=1,
                  random_state=None,
                  verbose=0) -> Tuple[List[str], List[np.array], List[np.array]]:
    ensembles = []
    if not isinstance(max_depths, Iterable):
        max_depths = [max_depths]

    for max_depth in max_depths:

        # pass different key based on sklearn version
        estimator = DecisionTreeRegressor(
            max_depth=max_depth,
            max_features=max_features,
            min_samples_split=min_samples_split,

        )
        init_signature = inspect.signature(BaggingRegressor.__init__)
        estimator_key = 'estimator' if 'estimator' in init_signature.parameters.keys(
        ) else 'base_estimator'
        kwargs = {
            estimator_key: estimator,
        }
        bagging_clf = BaggingRegressor(
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
            verbose=verbose,
            **kwargs
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
        extracted_rules.append(convert.tree_to_rules(
            estimator, np.array(feature_names)[features]))

    return extracted_rules, estimators_samples_, estimators_features_


def extract_marginal_curves(clf, X, max_evals=100):
    """Uses predict_proba to compute marginal curves.
    Assumes clf is a classifier with a predict_proba method and that classifier is additive across features
    For GAM, this returns the shape functions

    Params
    ------
    clf : classifier
        A classifier with a predict_proba method
    X : array-like
        The data to compute the marginal curves on (used to calculate unique feature vals)
    max_evals : int
        The maximum number of evaluations to make for each feature

    Returns
    -------
    feature_vals_list : list of arrays
        The values of each feature for which the shape function is evaluated.
    shape_function_vals_list : list of arrays
        The shape function evaluated at each value of the corresponding feature.
    """
    p = X.shape[1]
    dummy_input = np.zeros((1, p))
    base = clf.predict_proba(dummy_input)[:, 1][0]
    feature_vals_list = []
    shape_function_vals_list = []
    for feat_num in range(p):
        feature_vals = sorted(np.unique(X[:, feat_num]))
        while len(feature_vals) > max_evals:
            feature_vals = feature_vals[::2]
        dummy_input = np.zeros((len(feature_vals), p))
        dummy_input[:, feat_num] = feature_vals
        shape_function_vals = clf.predict_proba(dummy_input)[:, 1] - base
        feature_vals_list.append(feature_vals)
        shape_function_vals_list.append(shape_function_vals.tolist())
    return feature_vals_list, shape_function_vals_list


if __name__ == '__main__':
    init_signature = inspect.signature(BaggingRegressor.__init__)
    print('estimator' in init_signature.parameters.keys())
