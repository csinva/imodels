from collections import Counter
from typing import List, Mapping, Union, Optional

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from bartpy.runner import run_models
from bartpy.sklearnmodel import SklearnModel

ImportanceMap = Mapping[int, float]
ImportanceDistributionMap = Mapping[int, List[float]]


def feature_split_proportions(model: SklearnModel, columns: Optional[List[int]]=None) -> Mapping[int, float]:

    split_variables = []
    for sample in model.model_samples:
        for tree in sample.trees:
            for node in tree.nodes:
                splitting_var = node.split.splitting_variable
                split_variables.append(splitting_var)
    counter = Counter(split_variables)
    if columns is None:
        columns = sorted(list([x for x in counter.keys() if x is not None]))

    proportions = {}
    for column in columns:
        if column in counter.keys():
            proportions[column] = counter[column] / len(split_variables)
        else:
            proportions[column] = 0.0

    return proportions


def plot_feature_split_proportions(model: SklearnModel, ax=None):
    if ax is None:
        _, ax = plt.subplots(1, 1)
    proportions = feature_split_proportions(model)

    y_pos = np.arange(len(proportions))
    name, count = list(proportions.keys()), list(proportions.values())
    props = pd.DataFrame({"name": name, "counts": count}).sort_values("name", ascending=True)
    plt.barh(y_pos, props.counts, align='center', alpha=0.5)
    plt.yticks(y_pos, props.name)
    plt.xlabel('Proportion of all splits')
    plt.ylabel('Feature')
    plt.title('Proportion of Splits Made on Each Variable')
    return ax


def null_feature_split_proportions_distribution(model: SklearnModel,
                                                X: Union[pd.DataFrame, np.ndarray],
                                                y: np.ndarray,
                                                n_permutations: int=10) -> Mapping[int, List[float]]:
    """
    Calculate a null distribution of proportion of splits on each variable in X

    Works by randomly permuting y to remove any true dependence of y on X and calculating feature importance

    Parameters
    ----------
    model: SklearnModel
        Model specification to work with
    X: np.ndarray
        Covariate matrix
    y: np.ndarray
        Target data
    n_permutations: int
        How many permutations to run
        The higher the number of permutations, the more accurate the null distribution, but the longer it will take to run
    Returns
    -------
    Mapping[int, List[float]]
        A list of inclusion proportions for each variable in X
    """

    inclusion_dict = {x: [] for x in range(X.shape[1])}

    y_s = [np.random.permutation(y) for _ in range(n_permutations)]
    X_s = [X for _ in y_s]

    fit_models = run_models(model, X_s, y_s)

    for model in fit_models:
        splits_run = feature_split_proportions(model, list(range(X.shape[1])))
        for key, value in splits_run.items():
            inclusion_dict[key].append(value)

    return inclusion_dict


def plot_null_feature_importance_distributions(null_distributions: Mapping[int, List[float]], ax=None) -> None:
    if ax is None:
        _, ax = plt.subplots(1, 1)
    df = pd.DataFrame(null_distributions)
    df = pd.DataFrame(df.unstack()).reset_index().drop("level_1", axis=1)
    df.columns = ["variable", "p"]
    sns.boxplot(x="variable", y="p", data=df, ax=ax)
    ax.set_title("Null Feature Importance Distribution")
    return ax


def local_thresholds(null_distributions: ImportanceDistributionMap, percentile: float) -> Mapping[int, float]:
    """
    Calculate the required proportion of splits to be selected by variable

    Creates a null distribution for each variable based on the % of splits including that variable in each of the permuted models

    Each variable has its own threshold that is independent of the other variables

    Note - this is significantly less stringent than the global threshold

    Parameters
    ----------
    null_distributions: ImportanceDistributionMap
        A mapping from variable to distribution of split inclusion proportions under the null
    percentile: float
        The percentile of the null distribution to use as a cutoff.
        The closer to 1.0, the more stringent the threshold

    Returns
    -------
    Mapping[int, float]
        A lookup from column to % inclusion threshold
    """
    return {feature: np.percentile(null_distributions[feature], percentile) for feature in null_distributions}


def global_thresholds(null_distributions: ImportanceDistributionMap, percentile: float) -> Mapping[int, float]:
    """
    Calculate the required proportion of splits to be selected by variable

    Creates a distribution of the _highest_ inclusion percentage of any variable in each of the permuted models
    Threshold is set as a percentile of this distribution

    All variables have the same threshold

    Note that this is significantly more stringent than the local threshold

    Parameters
    ----------
    null_distributions: ImportanceDistributionMap
        A mapping from variable to distribution of split inclusion proportions under the null
    percentile: float
        The percentile of the null distribution to use as a cutoff.
        The closer to 1.0, the more stringent the threshold

    Returns
    -------
    Mapping[int, float]
        A lookup from column to % inclusion threshold
    """
    q_s = []
    df = pd.DataFrame(null_distributions)
    for row in df.iter_rows():
        q_s.append(np.max(row))
    threshold = np.percentile(q_s, percentile)
    return {feature: threshold for feature in null_distributions}


def kept_features(feature_proportions: Mapping[int, float], thresholds: Mapping[int, float]) -> List[int]:
    """
    Extract the features to keep

    Parameters
    ----------
    feature_proportions: Mapping[int, float]
        Lookup from variable to % of splits in the model that use that variable
    thresholds:  Mapping[int, float]
        Lookup from variable to required % of splits in the model to be kept

    Returns
    -------
    List[int]
        Variable selected for inclusion in the final model
    """
    return [x[0] for x in zip(sorted(feature_proportions.keys()), is_kept(feature_proportions, thresholds)) if x[1]]


def is_kept(feature_proportions: Mapping[int, float], thresholds: Mapping[int, float]) -> List[bool]:
    """
    Determine whether each variable should be kept after selection

    Parameters
    ----------
    feature_proportions: Mapping[int, float]
        Lookup from variable to % of splits in the model that use that variable
    thresholds:  Mapping[int, float]
        Lookup from variable to required % of splits in the model to be kept

    Returns
    -------
    List[bool]
        An array of length equal to the width of the covariate matrix
        True if the variable should be kept, False otherwise
    """
    print(sorted(list(feature_proportions.keys())))
    return [feature_proportions[feature] > thresholds[feature] for feature in sorted(list(feature_proportions.keys()))]


def partition_into_passed_and_failed_features(feature_proportions, thresholds):
    kept = kept_features(feature_proportions, thresholds)
    passed_features = {x[0]: x[1] for x in feature_proportions.items() if x[0] in kept}
    failed_features = {x[0]: x[1] for x in feature_proportions.items() if x[0] not in kept}
    return passed_features, failed_features


def plot_feature_proportions_against_thresholds(feature_proportions, thresholds, ax=None):
    if ax is None:
        _, ax = plt.subplots(1, 1)
    passed_features, failed_features = partition_into_passed_and_failed_features(feature_proportions, thresholds)

    ax.bar(thresholds.keys(), [x * 100 for x in thresholds.values()], width=0.01, color="black", alpha=0.5)
    ax.scatter(passed_features.keys(), [x * 100 for x in passed_features.values()], c="g")
    ax.scatter(failed_features.keys(), [x * 100 for x in failed_features.values()], c="r")
    ax.set_title("Feature Importance Compared to Threshold")
    ax.set_xlabel("Feature")
    ax.set_ylabel("% Splits")
    return ax
