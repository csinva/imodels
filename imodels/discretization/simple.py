import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer


class SimpleDiscretizer:

    def __init__(self, n_bins: int = 8, strategy: str = 'uniform'):
        self.n_bins = n_bins
        self.strategy = strategy

    def fit(self, X: np.array, feature_labels: np.array):
        self.is_categorical = np.array([set(np.unique(X[:, i])).issubset({0, 1}) for i in np.arange(X.shape[1])])

        if False not in self.is_categorical:
            self.feature_labels = feature_labels
            self.discretizer = None
            return

        if isinstance(feature_labels, list):
            feature_labels = np.array(feature_labels)

        # X_categorical = X[:, self.is_categorical]
        X_categorical_columns = feature_labels[self.is_categorical]
        # X_numeric = X[:, ~self.is_categorical]
        X_numeric_columns = feature_labels[~self.is_categorical]

        self.discretizer = KBinsDiscretizer(n_bins=self.n_bins, encode='onehot', strategy=self.strategy)
        # X_numeric_discretized = self.discretizer.fit(X_numeric)

        discretized_featnames = []
        for feat_name, bin_edges in zip(X_numeric_columns, self.discretizer.bin_edges_):
            be_str = bin_edges.astype(str)
            discretized_featnames += (
                [f'{feat_name}_' + '_to_'.join([be_str[i], be_str[i + 1]]) for i in range(bin_edges.shape[0] - 1)]
            )
        self.featnames_after_disc = np.append(discretized_featnames, X_categorical_columns)

    def transform(self, X: np.array):
        if self.discretizer is None:
            return pd.DataFrame(X, columns=self.feature_labels)

        X_categorical = X[:, self.is_categorical]
        X_numeric = X[:, ~self.is_categorical]

        X_numeric_discretized = self.discretizer.transform(X_numeric).toarray()
        X_concat = np.concatenate((X_numeric_discretized, X_categorical), axis=1)
        X_df_onehot = pd.DataFrame(X_concat, columns=self.featnames_after_disc)

        return X_df_onehot

    def fit_transform(self, X: np.array, feature_labels: np.array):
        self.fit(X, feature_labels)
        return self.transform(X)
