import numbers

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder
from sklearn.utils.validation import check_is_fitted, check_array

"""
The classes below (BasicDiscretizer and RFDiscretizer) provide 
additional functionalities and wrappers around KBinsDiscretizer 
from sklearn. In particular, the following AbstractDiscretizer classes
    - take a data frame as input and output a data frame
    - allow for discretization of a subset of columns in the data 
      frame and returns the full data frame with both the 
      discretized and non-discretized columns
    - allow quantile bins to be a single point if necessary
"""


class AbstractDiscretizer(TransformerMixin, BaseEstimator):
    """
    Discretize numeric data into bins. Base class.

    Params
    ------
    n_bins : int or array-like of shape (len(dcols),), default=2
        Number of bins to discretize each feature into.

    dcols : list of strings
        The names of the columns to be discretized; by default,
        discretize all float and int columns in X.

    encode : {‘onehot’, ‘ordinal’}, default=’onehot’
        Method used to encode the transformed result.

        onehot
            Encode the transformed result with one-hot encoding and
            return a dense array.
        ordinal
            Return the bin identifier encoded as an integer value.

    strategy : {‘uniform’, ‘quantile’, ‘kmeans’}, default=’quantile’
        Strategy used to define the widths of the bins.

        uniform
            All bins in each feature have identical widths.
        quantile
            All bins in each feature have the same number of points.
        kmeans
            Values in each bin have the same nearest center of a 1D
            k-means cluster.

    onehot_drop : {‘first’, ‘if_binary’} or a array-like of shape (len(dcols),), default='if_binary'
        Specifies a methodology to use to drop one of the categories
        per feature when encode = "onehot".

        None
            Retain all features (the default).
        ‘first’
            Drop the first y_str in each feature. If only one y_str
            is present, the feature will be dropped entirely.
        ‘if_binary’
            Drop the first y_str in each feature with two categories.
            Features with 1 or more than 2 categories are left intact.
    """

    def __init__(self, n_bins=2, dcols=[],
                 encode='onehot', strategy='quantile',
                 onehot_drop='if_binary'):
        self.n_bins = n_bins
        self.encode = encode
        self.strategy = strategy
        self.dcols = dcols
        if encode == 'onehot':
            self.onehot_drop = onehot_drop

    def _validate_n_bins(self):
        """
        Check if n_bins argument is valid.
        """
        orig_bins = self.n_bins
        n_features = len(self.dcols_)
        if isinstance(orig_bins, numbers.Number):
            if not isinstance(orig_bins, numbers.Integral):
                raise ValueError(
                    "{} received an invalid n_bins type. "
                    "Received {}, expected int.".format(
                        AbstractDiscretizer.__name__, type(orig_bins).__name__
                    )
                )
            if orig_bins < 2:
                raise ValueError(
                    "{} received an invalid number "
                    "of bins. Received {}, expected at least 2.".format(
                        AbstractDiscretizer.__name__, orig_bins
                    )
                )
            self.n_bins = np.full(n_features, orig_bins, dtype=int)
        else:
            n_bins = check_array(orig_bins, dtype=int, copy=True, ensure_2d=False)

            if n_bins.ndim > 1 or n_bins.shape[0] != n_features:
                raise ValueError("n_bins must be a scalar or array of shape (n_features,).")

            bad_nbins_value = (n_bins < 2) | (n_bins != orig_bins)

            violating_indices = np.where(bad_nbins_value)[0]
            if violating_indices.shape[0] > 0:
                indices = ", ".join(str(i) for i in violating_indices)
                raise ValueError(
                    "{} received an invalid number "
                    "of bins at indices {}. Number of bins "
                    "must be at least 2, and must be an int.".format(
                        AbstractDiscretizer.__name__, indices
                    )
                )
            self.n_bins = n_bins

    def _validate_dcols(self, X):
        """
        Check if dcols argument is valid.
        """
        for col in self.dcols_:
            if col not in X.columns:
                raise ValueError("{} is not a column in X.".format(col))
            if not is_numeric_dtype(X[col].dtype):
                raise ValueError("Cannot discretize non-numeric columns.")

    def _validate_args(self):
        """
        Check if encode, strategy arguments are valid.
        """

        valid_encode = ('onehot', 'ordinal')
        if self.encode not in valid_encode:
            raise ValueError("Valid options for 'encode' are {}. Got encode={!r} instead." \
                             .format(valid_encode, self.encode))

        valid_strategy = ('uniform', 'quantile', 'kmeans')
        if (self.strategy not in valid_strategy):
            raise ValueError("Valid options for 'strategy' are {}. Got strategy={!r} instead." \
                             .format(valid_strategy, self.strategy))

    def _discretize_to_bins(self, x, bin_edges,
                            keep_pointwise_bins=False):
        """
        Discretize data into bins of the form [a, b) given bin
        edges/boundaries

        Parameters
        ----------
        x : array-like of shape (n_samples,)
            Data vector to be discretized.

        bin_edges : array-like
            Values to serve as bin edges; should include min and
            max values for the range of x

        keep_pointwise_bins : boolean
            If True, treat duplicate bin_edges as a pointiwse bin,
            i.e., [a, a]. If False, these bins are in effect ignored.

        Returns
        -------
        xd: array of shape (n_samples,) where x has been
            transformed to the binned space
        """

        # ignore min and max values in bin generation
        unique_edges = np.unique(bin_edges[1:-1])

        if keep_pointwise_bins:
            # note: min and max values are used to define pointwise bins
            pointwise_bins = np.unique(bin_edges[pd.Series(bin_edges).duplicated()])
        else:
            pointwise_bins = np.array([])

        xd = np.zeros_like(x)
        i = 1
        for idx, split in enumerate(unique_edges):
            if idx == (len(unique_edges) - 1):  # uppermost bin
                if (idx == 0) & (split in pointwise_bins):
                    indicator = x > split  # two bins total: (-inf, a], (a, inf)
                else:
                    indicator = x >= split  # uppermost bin: [a, inf)
            else:
                if split in pointwise_bins:
                    # create two bins: [a, a], (a, b)
                    indicator = (x > split) & (x < unique_edges[idx + 1])  #
                    if idx != 0:
                        xd[x == split] = i
                        i += 1
                else:
                    # create bin: [a, b)
                    indicator = (x >= split) & (x < unique_edges[idx + 1])
            xd[indicator] = i
            i += 1

        return xd.astype(int)

    def _fit_preprocessing(self, X):
        """
        Initial checks before fitting the estimator.

        Parameters
        ----------
        X : data frame of shape (n_samples, n_features)
            (Training) data to be discretized.

        Returns
        -------
        self
        """

        # by default, discretize all numeric columns
        if len(self.dcols) == 0:
            numeric_cols = [col for col in X.columns if is_numeric_dtype(X[col].dtype)]
            self.dcols_ = numeric_cols

        # error checking
        self._validate_n_bins()
        self._validate_args()
        self._validate_dcols(X)

    def _transform_postprocessing(self, discretized_df, X):
        """
        Final processing in transform method. Does one-hot encoding
        (if specified) and joins discretized columns to the
        un-transformed columns in X.

        Parameters
        ----------
        discretized_df : data frame of shape (n_sample, len(dcols))
            Discretized data in the transformed bin space.

        X : data frame of shape (n_samples, n_features)
            Data to be discretized.

        Returns
        -------
        X_discretized : data frame
            Data with features in dcols transformed to the
            binned space. All other features remain unchanged.
            Encoded either as ordinal or one-hot.
        """

        discretized_df = discretized_df[self.dcols_]

        # return onehot encoded X if specified
        if self.encode == "onehot":
            colnames = [str(col) for col in self.dcols_]
            try:
                onehot_col_names = self.onehot_.get_feature_names_out(colnames)
            except:
                onehot_col_names = self.onehot_.get_feature_names(colnames)  # older versions of sklearn
            discretized_df = self.onehot_.transform(discretized_df.astype(str))
            discretized_df = pd.DataFrame(discretized_df,
                                          columns=onehot_col_names,
                                          index=X.index).astype(int)

        # join discretized columns with rest of X
        cols = [col for col in X.columns if col not in self.dcols_]
        X_discretized = pd.concat([discretized_df, X[cols]], axis=1)

        return X_discretized


class ExtraBasicDiscretizer(TransformerMixin):
    """
    Discretize provided columns into bins and return in one-hot format. 
    Generates meaningful column names based on bin edges.
    Wraps KBinsDiscretizer from sklearn.

    Params
    ------
    dcols : list of strings
        The names of the columns to be discretized.

    n_bins : int or array-like of shape (len(dcols),), default=4
        Number of bins to discretize each feature into.

    strategy : {'uniform', 'quantile', 'kmeans'}, default='quantile'
        Strategy used to define the widths of the bins.

        uniform
            All bins in each feature have identical widths.
        quantile
            All bins in each feature have the same number of points.
        kmeans
            Values in each bin have the same nearest center of a 1D
            k-means cluster.

    onehot_drop : {'first', 'if_binary'} or a array-like of shape  (len(dcols),), default='if_binary'
        Specifies a methodology to use to drop one of the categories
        per feature when encode = "onehot".

        None
            Retain all features (the default).
        'first'
            Drop the first y_str in each feature. If only one y_str
            is present, the feature will be dropped entirely.
        'if_binary'
            Drop the first y_str in each feature with two categories.
            Features with 1 or more than 2 categories are left intact.

    Attributes
    ----------
    discretizer_ : object of class KBinsDiscretizer()
        Primary discretization method used to bin numeric data

    Examples
    --------
    """

    def __init__(self,
                 dcols,
                 n_bins=4,
                 strategy='quantile',
                 onehot_drop='if_binary'):
        self.dcols = dcols
        self.n_bins = n_bins
        self.strategy = strategy
        self.onehot_drop = onehot_drop

    def fit(self, X, y=None):
        """
        Fit the estimator.

        Parameters
        ----------
        X : data frame of shape (n_samples, n_features)
            (Training) data to be discretized.

        y : Ignored. This parameter exists only for compatibility with
            :class:`~sklearn.pipeline.Pipeline` and fit_transform method

        Returns
        -------
        self
        """

        # Fit KBinsDiscretizer to the selected columns
        discretizer = KBinsDiscretizer(
            n_bins=self.n_bins, strategy=self.strategy, encode='ordinal')
        discretizer.fit(X[self.dcols])
        self.discretizer_ = discretizer

        # Fit OneHotEncoder to the ordinal output of KBinsDiscretizer
        disc_ordinal_np = discretizer.transform(X[self.dcols])
        disc_ordinal_df = pd.DataFrame(disc_ordinal_np, columns=self.dcols)
        disc_ordinal_df_str = disc_ordinal_df.astype(int).astype(str)

        encoder = OneHotEncoder(drop=self.onehot_drop, sparse=False)
        encoder.fit(disc_ordinal_df_str)
        self.encoder_ = encoder

        return self

    def transform(self, X):
        """
        Discretize the data.

        Parameters
        ----------
        X : data frame of shape (n_samples, n_features)
            Data to be discretized.

        Returns
        -------
        X_discretized : data frame
            Data with features in dcols transformed to the
            binned space. All other features remain unchanged.
        """

        # Apply discretizer transform to get ordinally coded DF
        disc_ordinal_np = self.discretizer_.transform(X[self.dcols])
        disc_ordinal_df = pd.DataFrame(disc_ordinal_np, columns=self.dcols)
        disc_ordinal_df_str = disc_ordinal_df.astype(int).astype(str)

        # One-hot encode the ordinal DF
        disc_onehot_np = self.encoder_.transform(disc_ordinal_df_str)
        disc_onehot = pd.DataFrame(disc_onehot_np, columns=self.encoder_.get_feature_names_out())

        # Name columns after the interval they represent (e.g. 0.1_to_0.5)
        for col, bin_edges in zip(self.dcols, self.discretizer_.bin_edges_):
            bin_edges = bin_edges.astype(str)

            for ordinal_value in disc_ordinal_df_str[col].unique():
                bin_lb = bin_edges[int(ordinal_value)]
                bin_ub = bin_edges[int(ordinal_value) + 1]
                interval_string = f'{bin_lb}_to_{bin_ub}'

                disc_onehot = disc_onehot.rename(
                    columns={f'{col}_{ordinal_value}': f'{col}_' + interval_string})

        # Join discretized columns with rest of X
        non_dcols = [col for col in X.columns if col not in self.dcols]
        X_discretized = pd.concat([disc_onehot, X[non_dcols]], axis=1)

        return X_discretized


class BasicDiscretizer(AbstractDiscretizer):
    """
    Discretize numeric data into bins. Provides a wrapper around
    KBinsDiscretizer from sklearn

    Params
    ------
    n_bins : int or array-like of shape (len(dcols),), default=2
        Number of bins to discretize each feature into.

    dcols : list of strings
        The names of the columns to be discretized; by default,
        discretize all float and int columns in X.

    encode : {'onehot', 'ordinal'}, default='onehot'
        Method used to encode the transformed result.

        onehot
            Encode the transformed result with one-hot encoding and
            return a dense array.
        ordinal
            Return the bin identifier encoded as an integer value.

    strategy : {'uniform', 'quantile', 'kmeans'}, default='quantile'
        Strategy used to define the widths of the bins.

        uniform
            All bins in each feature have identical widths.
        quantile
            All bins in each feature have the same number of points.
        kmeans
            Values in each bin have the same nearest center of a 1D
            k-means cluster.

    onehot_drop : {‘first’, ‘if_binary’} or a array-like of shape  (len(dcols),), default='if_binary'
        Specifies a methodology to use to drop one of the categories
        per feature when encode = "onehot".

        None
            Retain all features (the default).
        ‘first’
            Drop the first y_str in each feature. If only one y_str
            is present, the feature will be dropped entirely.
        ‘if_binary’
            Drop the first y_str in each feature with two categories.
            Features with 1 or more than 2 categories are left intact.

    Attributes
    ----------
    discretizer_ : object of class KBinsDiscretizer()
        Primary discretization method used to bin numeric data

    manual_discretizer_ : dictionary
        Provides bin_edges to feed into _quantile_discretization()
        and do quantile discreization manually for features where
        KBinsDiscretizer() failed. Ignored if strategy != 'quantile'
        or no errors in KBinsDiscretizer().

    onehot_ : object of class OneHotEncoder()
        One hot encoding fit. Ignored if encode != 'onehot'

    Examples
    --------
    """

    def __init__(self, n_bins=2, dcols=[],
                 encode='onehot', strategy='quantile',
                 onehot_drop='if_binary'):
        super().__init__(n_bins=n_bins, dcols=dcols,
                         encode=encode, strategy=strategy,
                         onehot_drop=onehot_drop)

    def fit(self, X, y=None):
        """
        Fit the estimator.

        Parameters
        ----------
        X : data frame of shape (n_samples, n_features)
            (Training) data to be discretized.

        y : Ignored. This parameter exists only for compatibility with
            :class:`~sklearn.pipeline.Pipeline` and fit_transform method

        Returns
        -------
        self
        """

        # initalization and error checking
        self._fit_preprocessing(X)

        # apply KBinsDiscretizer to the selected columns
        discretizer = KBinsDiscretizer(n_bins=self.n_bins,
                                       encode='ordinal',
                                       strategy=self.strategy)

        discretizer.fit(X[self.dcols_])
        self.discretizer_ = discretizer

        if (self.encode == 'onehot') | (self.strategy == 'quantile'):
            discretized_df = discretizer.transform(X[self.dcols_])

            discretized_df = pd.DataFrame(discretized_df,
                                          columns=self.dcols_,
                                          index=X.index).astype(int)

        # fix KBinsDiscretizer errors if any when strategy = "quantile"
        if self.strategy == "quantile":
            err_idx = np.where(discretized_df.nunique() != self.n_bins)[0]
            self.manual_discretizer_ = dict()
            for idx in err_idx:
                col = self.dcols_[idx]
                if X[col].nunique() > 1:
                    q_values = np.linspace(0, 1, self.n_bins[idx] + 1)
                    bin_edges = np.quantile(X[col], q_values)
                    discretized_df[col] = self._discretize_to_bins(X[col], bin_edges,
                                                                   keep_pointwise_bins=True)
                    self.manual_discretizer_[col] = bin_edges

        # fit onehot encoded X if specified
        if self.encode == "onehot":
            onehot = OneHotEncoder(drop=self.onehot_drop, sparse=False)
            onehot.fit(discretized_df.astype(str))
            self.onehot_ = onehot

        return self

    def transform(self, X):
        """
        Discretize the data.

        Parameters
        ----------
        X : data frame of shape (n_samples, n_features)
            Data to be discretized.

        Returns
        -------
        X_discretized : data frame
            Data with features in dcols transformed to the
            binned space. All other features remain unchanged.
        """

        check_is_fitted(self)

        # transform using KBinsDiscretizer
        discretized_df = self.discretizer_.transform(X[self.dcols_]).astype(int)
        discretized_df = pd.DataFrame(discretized_df,
                                      columns=self.dcols_,
                                      index=X.index)

        # fix KBinsDiscretizer errors (if any) when strategy = "quantile"
        if self.strategy == "quantile":
            for col in self.manual_discretizer_.keys():
                bin_edges = self.manual_discretizer_[col]
                discretized_df[col] = self._discretize_to_bins(X[col], bin_edges,
                                                               keep_pointwise_bins=True)

        # return onehot encoded data if specified and
        # join discretized columns with rest of X
        X_discretized = self._transform_postprocessing(discretized_df, X)

        return X_discretized


class RFDiscretizer(AbstractDiscretizer):
    """
    Discretize numeric data into bins using RF splits.

    Parameters
    ----------
    rf_model : RandomForestClassifer() or RandomForestRegressor()
        RF model from which to extract splits for discretization.
        Default is RandomForestClassifer(n_estimators = 500) or
        RandomForestRegressor(n_estimators = 500)

    classification : boolean; default=False
        Used only if rf_model=None. If True,
        rf_model=RandomForestClassifier(n_estimators = 500).
        Else, rf_model=RandomForestRegressor(n_estimators = 500)

    n_bins : int or array-like of shape (len(dcols),), default=2
        Number of bins to discretize each feature into.

    dcols : list of strings
        The names of the columns to be discretized; by default,
        discretize all float and int columns in X.

    encode : {‘onehot’, ‘ordinal’}, default=’onehot’
        Method used to encode the transformed result.

        onehot - Encode the transformed result with one-hot encoding and
            return a dense array.
        ordinal - Return the bin identifier encoded as an integer value.

    strategy : {‘uniform’, ‘quantile’}, default=’quantile’
        Strategy used to choose RF split points.
        uniform - RF split points chosen to be uniformly spaced out.
        quantile - RF split points chosen based on equally-spaced quantiles.

    backup_strategy : {‘uniform’, ‘quantile’, ‘kmeans’}, default=’quantile’
        Strategy used to define the widths of the bins if no rf splits exist for
        that feature. Used in KBinsDiscretizer.
        uniform
            All bins in each feature have identical widths.
        quantile
            All bins in each feature have the same number of points.
        kmeans
            Values in each bin have the same nearest center of a 1D
            k-means cluster.

    onehot_drop : {‘first’, ‘if_binary’} or array-like of shape  (len(dcols),), default='if_binary'
        Specifies a methodology to use to drop one of the categories
        per feature when encode = "onehot".
        None
            Retain all features (the default).
        ‘first’
            Drop the first y_str in each feature. If only one y_str
            is present, the feature will be dropped entirely.
        ‘if_binary’
            Drop the first y_str in each feature with two categories.
            Features with 1 or more than 2 categories are left intact.

    Attributes
    ----------
    rf_splits : dictionary where
        key = feature name
        value = array of all RF split threshold values

    bin_edges_ : dictionary where
        key = feature name
        value = array of bin edges used for discretization, taken from
            RF split values

    missing_rf_cols_ : array-like
        List of features that were not used in RF

    backup_discretizer_ : object of class BasicDiscretizer()
        Discretization method used to bin numeric data for features
        in missing_rf_cols_

    onehot_ : object of class OneHotEncoder()
        One hot encoding fit. Ignored if encode != 'onehot'

    """

    def __init__(self, rf_model=None, classification=False,
                 n_bins=2, dcols=[], encode='onehot',
                 strategy='quantile', backup_strategy='quantile',
                 onehot_drop='if_binary'):
        super().__init__(n_bins=n_bins, dcols=dcols,
                         encode=encode, strategy=strategy,
                         onehot_drop=onehot_drop)
        self.backup_strategy = backup_strategy
        self.rf_model = rf_model
        if rf_model is None:
            self.classification = classification

    def _validate_args(self):
        """
        Check if encode, strategy, backup_strategy arguments are valid.
        """
        super()._validate_args()
        valid_backup_strategy = ('uniform', 'quantile', 'kmeans')
        if (self.backup_strategy not in valid_backup_strategy):
            raise ValueError("Valid options for 'strategy' are {}. Got strategy={!r} instead." \
                             .format(valid_backup_strategy, self.backup_strategy))

    def _get_rf_splits(self, col_names):
        """
        Get all splits in random forest ensemble

        Parameters
        ----------
        col_names : array-like of shape (n_features,)
            Column names for X used to train rf_model

        Returns
        -------
        rule_dict : dictionary where
            key = feature name
            value = array of all RF split threshold values
        """

        rule_dict = {}
        for model in self.rf_model.estimators_:
            tree = model.tree_
            tree_it = enumerate(zip(tree.children_left,
                                    tree.children_right,
                                    tree.feature,
                                    tree.threshold))
            for node_idx, data in tree_it:
                left, right, feature, th = data
                if (left != -1) | (right != -1):
                    feature = col_names[feature]
                    if feature in rule_dict:
                        rule_dict[feature].append(th)
                    else:
                        rule_dict[feature] = [th]
        return rule_dict

    def _fit_rf(self, X, y=None):
        """
        Fit random forest (if necessary) and obtain RF split thresholds

        Parameters
        ----------
        X : data frame of shape (n_samples, n_fatures)
            Training data used to fit RF

        y : array-like of shape (n_samples,)
            Training response vector used to fit RF

        Returns
        -------
        rf_splits : dictionary where
            key = feature name
            value = array of all RF split threshold values
        """

        # If no rf_model given, train default random forest model
        if self.rf_model is None:
            if y is None:
                raise ValueError("Must provide y if rf_model is not given.")
            if self.classification:
                self.rf_model = RandomForestClassifier(n_estimators=500)
            else:
                self.rf_model = RandomForestRegressor(n_estimators=500)
            self.rf_model.fit(X, y)

        else:
            # provided rf model has not yet been trained
            if not check_is_fitted(self.rf_model):
                if y is None:
                    raise ValueError("Must provide y if rf_model has not been trained.")
                self.rf_model.fit(X, y)

        # get all random forest split points
        self.rf_splits = self._get_rf_splits(list(X.columns))

    def reweight_n_bins(self, X, y=None, by="nsplits"):
        """
        Reallocate number of bins per feature.

        Parameters
        ----------
        X : data frame of shape (n_samples, n_features)
            (Training) data to be discretized.

        y : array-like of shape (n_samples,)
            (Training) response vector. Required only if
            rf_model = None or rf_model has not yet been fitted

        by : {'nsplits'}, default='nsplits'
            Specifies how to reallocate number of bins per feature.

            nsplits
                Reallocate number of bins so that each feature
                in dcols get at a minimum of 2 bins with the
                remaining bins distributed proportionally to the
                number of RF splits using that feature

        Returns
        -------
        self.n_bins : array of shape (len(dcols),)
            number of bins per feature reallocated according to
            'by' argument
        """
        # initialization and error checking
        self._fit_preprocessing(X)

        # get all random forest split points
        self._fit_rf(X=X, y=y)

        # get total number of bins to reallocate
        total_bins = self.n_bins.sum()

        # reweight n_bins
        if by == "nsplits":
            # each col gets at least 2 bins; remaining bins get
            # reallocated based on number of RF splits using that feature
            n_rules = np.array([len(self.rf_splits[col]) for col in self.dcols_])
            self.n_bins = np.round(n_rules / n_rules.sum() * \
                                   (total_bins - 2 * len(self.dcols_))) + 2
        else:
            valid_by = ('nsplits')
            raise ValueError("Valid options for 'by' are {}. Got by={!r} instead." \
                             .format(valid_by, by))

    def fit(self, X, y=None):
        """
        Fit the estimator.

        Parameters
        ----------
        X : data frame of shape (n_samples, n_features)
            (Training) data to be discretized.

        y : array-like of shape (n_samples,)
            (Training) response vector. Required only if
            rf_model = None or rf_model has not yet been fitted

        Returns
        -------
        self
        """
        # initialization and error checking
        self._fit_preprocessing(X)

        # get all random forest split points
        self._fit_rf(X=X, y=y)

        # features that were not used in the rf but need to be discretized
        self.missing_rf_cols_ = list(set(self.dcols_) - \
                                     set(self.rf_splits.keys()))
        if len(self.missing_rf_cols_) > 0:
            print("{} did not appear in random forest so were discretized via {} discretization" \
                  .format(self.missing_rf_cols_, self.strategy))
            missing_n_bins = np.array([self.n_bins[np.array(self.dcols_) == col][0] \
                                       for col in self.missing_rf_cols_])

            backup_discretizer = BasicDiscretizer(n_bins=missing_n_bins,
                                                  dcols=self.missing_rf_cols_,
                                                  encode='ordinal',
                                                  strategy=self.backup_strategy)
            backup_discretizer.fit(X[self.missing_rf_cols_])
            self.backup_discretizer_ = backup_discretizer
        else:
            self.backup_discretizer_ = None

        if self.encode == 'onehot':
            if len(self.missing_rf_cols_) > 0:
                discretized_df = backup_discretizer.transform(X[self.missing_rf_cols_])
            else:
                discretized_df = pd.DataFrame({}, index=X.index)

        # do discretization based on rf split thresholds
        self.bin_edges_ = dict()
        for col in self.dcols_:
            if col in self.rf_splits.keys():
                b = self.n_bins[np.array(self.dcols_) == col]
                if self.strategy == "quantile":
                    q_values = np.linspace(0, 1, int(b) + 1)
                    bin_edges = np.quantile(self.rf_splits[col], q_values)
                elif strategy == "uniform":
                    width = (max(self.rf_splits[col]) - min(self.rf_splits[col])) / b
                    bin_edges = width * np.arange(0, b + 1) + min(self.rf_splits[col])
                self.bin_edges_[col] = bin_edges
                if self.encode == 'onehot':
                    discretized_df[col] = self._discretize_to_bins(X[col], bin_edges)

        # fit onehot encoded X if specified
        if self.encode == "onehot":
            onehot = OneHotEncoder(drop=self.onehot_drop, sparse=False)
            onehot.fit(discretized_df[self.dcols_].astype(str))
            self.onehot_ = onehot

        return self

    def transform(self, X):
        """
        Discretize the data.

        Parameters
        ----------
        X : data frame of shape (n_samples, n_features)
            Data to be discretized.

        Returns
        -------
        X_discretized : data frame
            Data with features in dcols transformed to the
            binned space. All other features remain unchanged.
        """

        check_is_fitted(self)

        # transform features that did not appear in RF
        if len(self.missing_rf_cols_) > 0:
            discretized_df = self.backup_discretizer_.transform(X[self.missing_rf_cols_])
            discretized_df = pd.DataFrame(discretized_df,
                                          columns=self.missing_rf_cols_,
                                          index=X.index)
        else:
            discretized_df = pd.DataFrame({}, index=X.index)

        # do discretization based on rf split thresholds
        for col in self.bin_edges_.keys():
            discretized_df[col] = self._discretize_to_bins(X[col], self.bin_edges_[col])

        # return onehot encoded data if specified and
        # join discretized columns with rest of X
        X_discretized = self._transform_postprocessing(discretized_df, X)

        return X_discretized
