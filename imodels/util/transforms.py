'''Shared transforms between different interpretable models
'''

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


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
                upper = np.percentile(
                    X[:, i_col], 100 - self.trim_quantile * 100)
                self.winsor_lims[:, i_col] = [lower, upper]

    def trim(self, X):
        X_ = X.copy()
        X_ = np.where(X > self.winsor_lims[1, :], np.tile(self.winsor_lims[1, :], [X.shape[0], 1]),
                      np.where(X < self.winsor_lims[0, :], np.tile(self.winsor_lims[0, :], [X.shape[0], 1]), X))
        return X_


class FriedScale():
    """Performs scaling of linear variables according to Friedman et alpha_l. 2005 Sec 5

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
                scale_multipliers[i_col] = 0.4 / \
                    (1.0e-12 + np.std(X_trimmed[:, i_col]))
        self.scale_multipliers = scale_multipliers

    def scale(self, X):
        if self.winsorizer != None:
            return self.winsorizer.trim(X) * self.scale_multipliers
        else:
            return X * self.scale_multipliers


class CorrelationScreenTransformer(BaseEstimator, TransformerMixin):
    '''Finds correlated features above a magnitude threshold
    and zeros out all but the first of them
    '''

    def __init__(self, threshold=1.0):
        # Initialize with a correlation threshold
        self.threshold = threshold
        self.correlated_feature_sets = []

    def fit(self, X, y=None):
        # Check if X is a pandas DataFrame; if not, convert it to DataFrame
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # Calculate the correlation matrix
        corr_matrix = X.corr().abs()

        # Identify the features that are correlated based on the threshold
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if corr_matrix.iloc[i, j] >= self.threshold or corr_matrix.iloc[i, j] <= -self.threshold:
                    # Find the set this feature belongs to
                    found_set = False
                    for feature_set in self.correlated_feature_sets:
                        if i in feature_set or j in feature_set:
                            feature_set.update([i, j])
                            found_set = True
                            break
                    if not found_set:
                        self.correlated_feature_sets.append(set([i, j]))

        # Convert the sets to list of lists where each sublist has indexes to keep and to remove
        self.to_keep_remove = []
        for feature_set in self.correlated_feature_sets:
            feature_list = list(feature_set)
            # keep the first, remove the rest
            self.to_keep_remove.append((feature_list[0], feature_list[1:]))

        return self

    def transform(self, X):
        # Again, check if X is a pandas DataFrame; if not, convert it
        input_type = type(X)
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # Set the identified correlated features (except the first) to 0
        X_transformed = X.copy()
        for keep, to_remove in self.to_keep_remove:
            X_transformed.iloc[:, to_remove] = 0

        if input_type == np.ndarray:
            X_transformed == X_transformed.values

        return X_transformed


if __name__ == '__main__':
    X = np.random.randn(5, 5)
    X[:, 0] = [1, 1, 0, 1, 1]
    X[:, 1] = X[:, 0]

    transformer = CorrelationScreenTransformer()
    print(X)
    X_transformed = transformer.fit_transform(X)
    print(X_transformed)
