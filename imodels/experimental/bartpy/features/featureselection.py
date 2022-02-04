from copy import deepcopy

import numpy as np
from matplotlib import pyplot as plt
from sklearn.base import BaseEstimator
from sklearn.feature_selection.base import SelectorMixin

from bartpy.diagnostics.features import null_feature_split_proportions_distribution, \
    local_thresholds, global_thresholds, is_kept, feature_split_proportions, plot_feature_proportions_against_thresholds, plot_null_feature_importance_distributions, \
    plot_feature_split_proportions
from bartpy.sklearnmodel import SklearnModel


class SelectSplitProportionThreshold(BaseEstimator, SelectorMixin):

    def __init__(self,
                 model: SklearnModel,
                 percentile: float=0.2):
        self.model = deepcopy(model)
        self.percentile = percentile

    def fit(self, X, y):
        self.model.fit(X, y)
        self.X, self.y = X, y
        self.feature_proportions = feature_split_proportions(self.model)
        return self

    def _get_support_mask(self):
        return np.array([proportion > self.percentile for proportion in self.feature_proportions.values()])

    def plot(self):
        plot_feature_split_proportions(self.model)
        plt.show()


class SelectNullDistributionThreshold(BaseEstimator, SelectorMixin):

    def __init__(self,
                 model: SklearnModel,
                 percentile: float=0.95,
                 method="local",
                 n_permutations=10,
                 n_trees=None):
        if method == "local":
            self.method = local_thresholds
        elif method == "global":
            self.method = global_thresholds
        else:
            raise NotImplementedError("Currently only local and global methods are supported, found {}".format(self.method))
        self.model = deepcopy(model)
        if n_trees is not None:
            self.model.n_trees = n_trees
        self.percentile = percentile
        self.n_permutations = n_permutations

    def fit(self, X, y):
        self.model.fit(X, y)
        self.X, self.y = X, y
        self.null_distribution = null_feature_split_proportions_distribution(self.model, X, y, self.n_permutations)
        self.thresholds = self.method(self.null_distribution, self.percentile)
        self.feature_proportions = feature_split_proportions(self.model)
        return self

    def _get_support_mask(self):
        return np.array(is_kept(self.feature_proportions, self.thresholds))

    def plot(self):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        plot_feature_proportions_against_thresholds(self.feature_proportions, self.thresholds, ax1)
        plot_null_feature_importance_distributions(self.null_distribution, ax2)
        plt.show()
