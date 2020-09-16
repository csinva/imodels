'''Shared transforms between different interpretable models
'''

import numpy as np


class Winsorizer():
    """Performs Winsorization 1->1*

    Warning: this class should not be used directly.
    """

    def __init__(self, trim_quantile=0.0):
        self.trim_quantile = trim_quantile
        self.winsor_lims = None

    def train(self, X):
        # get winsor limits
        self.winsor_lims = np.ones([2, X.shape[1]]) * np.inf
        self.winsor_lims[0, :] = -np.inf
        if self.trim_quantile > 0:
            for i_col in np.arange(X.shape[1]):
                lower = np.percentile(X[:, i_col], self.trim_quantile * 100)
                upper = np.percentile(X[:, i_col], 100 - self.trim_quantile * 100)
                self.winsor_lims[:, i_col] = [lower, upper]

    def trim(self, X):
        X_ = X.copy()
        X_ = np.where(X > self.winsor_lims[1, :], np.tile(self.winsor_lims[1, :], [X.shape[0], 1]),
                      np.where(X < self.winsor_lims[0, :], np.tile(self.winsor_lims[0, :], [X.shape[0], 1]), X))
        return X_


class FriedScale():
    """Performs scaling of linear variables according to Friedman et al. 2005 Sec 5

    Each variable is first Winsorized l->l*, then standardised as 0.4 x l* / std(l*)
    Warning: this class should not be used directly.
    """

    def __init__(self, winsorizer=None):
        self.scale_multipliers = None
        self.winsorizer = winsorizer

    def train(self, X):
        # get multipliers
        if self.winsorizer != None:
            X_trimmed = self.winsorizer.trim(X)
        else:
            X_trimmed = X

        scale_multipliers = np.ones(X.shape[1])
        for i_col in np.arange(X.shape[1]):
            num_uniq_vals = len(np.unique(X[:, i_col]))
            if num_uniq_vals > 2:  # don't scale binary variables which are effectively already rules
                scale_multipliers[i_col] = 0.4 / (1.0e-12 + np.std(X_trimmed[:, i_col]))
        self.scale_multipliers = scale_multipliers

    def scale(self, X):
        if self.winsorizer != None:
            return self.winsorizer.trim(X) * self.scale_multipliers
        else:
            return X * self.scale_multipliers
