import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    
class BasicDiscretizer():
    """
    Discretize numeric data into bins.
    
    Parameters
    ----------  
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
    
    onehot_drop : {‘first’, ‘if_binary’} or a array-like of shape 
    (len(dcols),), default='if_binary'
        Specifies a methodology to use to drop one of the categories 
        per feature when encode = "onehot".
        
        None
            Retain all features (the default).
        ‘first’
            Drop the first category in each feature. If only one category 
            is present, the feature will be dropped entirely.
        ‘if_binary’
            Drop the first category in each feature with two categories.
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
    
    def __init__(self, n_bins = 2, dcols = [],
                 encode = 'onehot', strategy = 'quantile',
                 onehot_drop = 'if_binary'):
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
        if isinstance(self.n_bins, int):
            if self.n_bins < 2:
                raise ValueError("Invalid number of bins. n_bins must be at least 2.");
            self.n_bins = np.repeat(self.n_bins, len(self.dcols))
        elif len(self.n_bins) > 1:
            if any(self.n_bins < 2):
                raise ValueError("Invalid number of bins. n_bins must be at least 2.");
            elif len(self.n_bins) != len(self.dcols):
                raise ValueError("n_bins must be an int or array-like of shape (len(dcols),)")
        
        
    def _validate_dcols(self, X):
        """
        Check if dcols argument is valid.
        """
        for col in self.dcols:
            if col not in X.columns:
                raise ValueError("{} is not a column in X.".format(col))
            if X[col].dtype not in ['float', 'int']:
                raise ValueError("Cannot discretize non-numeric columns.")
        
        
    def _validate_args(self):
        """
        Check if encode, strategy arguments are valid.
        """
            
        valid_encode = ('onehot', 'ordinal')
        if self.encode not in valid_encode:
            raise ValueError("Valid options for 'encode' are {}. Got encode={!r} instead."\
                             .format(valid_encode, self.encode))

        valid_strategy = ('uniform', 'quantile', 'kmeans')
        if (self.strategy not in valid_strategy):
            raise ValueError("Valid options for 'strategy' are {}. Got strategy={!r} instead."\
                             .format(valid_strategy, self.strategy))
            
            
    def _quantile_discretization(self, x, bin_edges):
        """
        Perform quantile discretization manually
        
        Parameters
        ----------
        x : array-like of shape (n_samples,)
            Data vector to be discretized.
        bin_edges: array-like
            Values to serve as bin edges
          
        Returns
        -------
        xd: array of shape (n_samples,) where x has been 
            transformed to the binned space
        """
        
        unique_edges = np.unique(bin_edges[1:-1])
        pointwise_bins = np.unique(bin_edges[pd.Series(bin_edges).duplicated()])
    
        xd = np.zeros_like(x)
        i = 1
        for idx, split in enumerate(unique_edges):
            if idx == (len(unique_edges) - 1):
                if (idx == 0) & (split in pointwise_bins):
                    indicator = x > split
                else:
                    indicator = x >= split
            else:
                if split in pointwise_bins:
                    indicator = (x > split) & (x < unique_edges[idx + 1])
                    if idx != 0:
                        xd[x == split] = i
                        i += 1
                else:
                    indicator = (x >= split) & (x < unique_edges[idx + 1])
            xd[indicator] = i
            i += 1
        
        return xd.astype(int)
    
    
    def fit(self, X):
        """
        Fit the estimator.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            (Training) data to be discretized.
            
        Returns
        -------
        self
        """
        
        # by default, discretize all numeric columns
        if len(self.dcols) == 0:
            for col in X.columns:
                if X[col].dtype in ['float', 'int']:
                    self.dcols.append(col)
                    
        # error checking
        self._validate_n_bins()
        self._validate_args()
        self._validate_dcols(X)
        
        # apply KBinsDiscretizer to the selected columns
        discretizer = KBinsDiscretizer(n_bins = self.n_bins,
                                       encode = 'ordinal', 
                                       strategy = self.strategy)
        discretizer.fit(X[self.dcols])
        self.discretizer_ = discretizer
        
        if (self.encode == 'onehot') | (self.strategy == 'quantile'):
            discretized_df = discretizer.transform(X[self.dcols])
            discretized_df = pd.DataFrame(discretized_df, 
                                          columns = self.dcols,
                                          index = X.index).astype(int)

        # fix KBinsDiscretizer errors if any when strategy = "quantile"
        if self.strategy == "quantile":
            err_idx = np.where(discretized_df.nunique() != self.n_bins)[0]
            self.manual_discretizer_ = dict()
            for idx in err_idx:
                col = self.dcols[idx]
                if X[col].nunique() > 1:
                    q_values = np.linspace(0, 1, self.n_bins[idx]+1)
                    quantiles = np.quantile(X[col], q_values)
                    discretized_df[col] = self._quantile_discretization(X[col], 
                                                                        quantiles)
                    self.manual_discretizer_[col] = quantiles
                    
        # fit onehot encoded X if specified
        if self.encode == "onehot":
            onehot = OneHotEncoder(drop = self.onehot_drop, sparse = False)
            onehot.fit(discretized_df.astype(str))
            self.onehot_ = onehot
        
        
    def transform(self, X):
        """
        Discretize the data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to be discretized.
        
        Returns
        -------
        X_discretized : data frame
            Data with features in dcols transformed to the 
            binned space. All other features remain unchanged.
        """
        
        # transform using KBinsDiscretizer
        discretized_df = self.discretizer_.transform(X[self.dcols]).astype(int)
        discretized_df = pd.DataFrame(discretized_df, 
                                      columns = self.dcols,
                                      index = X.index)

        # fix KBinsDiscretizer errors (if any) when strategy = "quantile"
        if self.strategy == "quantile":
            for col in self.manual_discretizer_.keys():
                bin_edges = self.manual_discretizer_[col]
                discretized_df[col] = self._quantile_discretization(X[col],
                                                                    bin_edges)

        # return onehot encoded X if specified
        if self.encode == "onehot":
            colnames = [str(col) for col in self.dcols]
            onehot_col_names = self.onehot_.get_feature_names(colnames)
            discretized_df = self.onehot_.transform(discretized_df.astype(str))
            discretized_df = pd.DataFrame(discretized_df, 
                                          columns = onehot_col_names, 
                                          index = X.index).astype(int)

        # join discretized columns with rest of X
        cols = [col for col in X.columns if col not in self.dcols]
        X_discretized = pd.concat([discretized_df, X[cols]], axis = 1)

        return X_discretized
        
        
class RFDiscretizer():
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
        
        onehot
            Encode the transformed result with one-hot encoding and
            return a dense array.
        ordinal
            Return the bin identifier encoded as an integer value.
            
    strategy : {‘uniform’, ‘quantile’}, default=’quantile’
        Strategy used to choose RF split points.
        
        uniform
            RF split points chosen to be uniformly spaced out.
        quantile
            RF split points chosen based on equally-spaced quantiles.
    
    backup_strategy : {‘uniform’, ‘quantile’, ‘kmeans’}, default=’quantile’
        Strategy used to define the widths of the bins if no rf splits exist for 
        that feature.
        
        uniform
            All bins in each feature have identical widths.
        quantile
            All bins in each feature have the same number of points.
        kmeans
            Values in each bin have the same nearest center of a 1D
            k-means cluster.
    
    onehot_drop : {‘first’, ‘if_binary’} or a array-like of shape 
    (len(dcols),), default='if_binary'
        Specifies a methodology to use to drop one of the categories
        per feature when encode = "onehot".
        
        None
            Retain all features (the default).
        ‘first’
            Drop the first category in each feature. If only one category 
            is present, the feature will be dropped entirely.
        ‘if_binary’
            Drop the first category in each feature with two categories.
            Features with 1 or more than 2 categories are left intact.
    
    Attributes
    ----------
    bin_edges_ : dictionary where
        key = feature name
        value = array of bin edges used for discretization, taken from 
            RF split values
        
    rf_splits_ : dictionary where
        key = feature name
        value = array of all RF split threshold values
    
    missing_rf_cols_ : array-like
        List of features that were not used in RF
    
    backup_discretizer_ : object of class BasicDiscretizer()
        Discretization method used to bin numeric data for features
        in missing_rf_cols_
        
    onehot_ : object of class OneHotEncoder()
        One hot encoding fit. Ignored if encode != 'onehot'
        
    Examples
    --------
    """
    
    def __init__(self, rf_model = None, classification = False,
                 n_bins = 2, dcols = [], encode = 'onehot', 
                 strategy = 'quantile', backup_strategy = 'quantile', 
                 onehot_drop = 'if_binary'):
        
        self.rf_model = rf_model
        if rf_model is None:
            self.classification = classification
        self.n_bins = n_bins
        self.encode = encode
        self.strategy = strategy
        self.backup_strategy = backup_strategy
        self.dcols = dcols
        if encode == 'onehot':
            self.onehot_drop = onehot_drop
        
        
    def _validate_n_bins(self):
        """
        Check if n_bins argument is valid.
        """
        if isinstance(self.n_bins, int):
            if self.n_bins < 2:
                raise ValueError("Invalid number of bins. n_bins must be at least 2.");
            self.n_bins = np.repeat(self.n_bins, len(self.dcols))
        elif len(self.n_bins) > 1:
            if any(self.n_bins < 2):
                raise ValueError("Invalid number of bins. n_bins must be at least 2.");
            elif len(self.n_bins) != len(self.dcols):
                raise ValueError("n_bins must be an int or array-like of shape (len(dcols),)")
        
        
    def _validate_dcols(self, X):
        """
        Check if dcols argument is valid.
        """
        for col in self.dcols:
            if col not in X.columns:
                raise ValueError("{} is not a column in X.".format(col))
            if X[col].dtype not in ['float', 'int']:
                raise ValueError("Cannot discretize non-numeric columns.")
        
        
    def _validate_args(self):
        """
        Check if encode, strategy, backup_strategy arguments are valid.
        """
        valid_encode = ('onehot', 'ordinal')
        if self.encode not in valid_encode:
            raise ValueError("Valid options for 'encode' are {}. Got encode={!r} instead."\
                             .format(valid_encode, self.encode))

        valid_strategy = ('uniform', 'quantile')
        if (self.strategy not in valid_strategy):
            raise ValueError("Valid options for 'strategy' are {}. Got strategy={!r} instead."\
                             .format(valid_strategy, self.strategy))
        
        valid_backup_strategy = ('uniform', 'quantile', 'kmeans')
        if (self.backup_strategy not in valid_backup_strategy):
            raise ValueError("Valid options for 'strategy' are {}. Got strategy={!r} instead."\
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
            
        
    def _fit_rf(self, X, y = None):
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
        rf_splits_ : dictionary where
            key = feature name
            value = array of all RF split threshold values
        """
        
        # If no rf_model given, train default random forest model
        if self.rf_model is None:
            if y is None:
                raise ValueError("Must provide y if rf_model is not given.")
            if self.classification:
                self.rf_model = RandomForestClassifier(n_estimators = 500)
            else:
                self.rf_model = RandomForestRegressor(n_estimators = 500)
            self.rf_model.fit(X, y)

        else:
            # provided rf model has not yet been trained
            if not hasattr(self.rf_model, "estimators_"):
                if y is None:
                    raise ValueError("Must provide y if rf_model has not been trained.")
                self.rf_model.fit(X, y)
                
        # get all random forest split points
        self.rf_splits_ = self._get_rf_splits(list(X.columns))
            
            
    def _discretize_by_threshold(self, x, bin_edges):
        """
        Discretize data given RF split thresholds
        
        Parameters
        ----------
        x : array-like of shape (n_samples,)
            Data vector to be discretized.
        bin_edges: array-like
            Values to serve as bin edges
        
        Returns
        -------
        xd: array of shape (n_samples,) where x has been transformed to the binned space
        """
        
        xd = np.zeros_like(x)
        for idx, split in enumerate(bin_edges):
            if idx == (len(bin_edges) - 1):
                indicator = x >= split
            else:
                indicator = (x >= split) & (x < bin_edges[idx + 1])
            xd[indicator] = idx + 1
        return xd.astype(int)
        
        
    def reweight_n_bins(self, X, y = None, by = "nsplits"):
        """
        Reallocate number of bins per feature.

        Parameters
        ----------  
        X : array-like of shape (n_samples, n_features)
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
        
        
        # by default, discretize all numeric columns
        if len(self.dcols) == 0:
            for col in X.columns:
                if X[col].dtype in ['float', 'int']:
                    self.dcols.append(col)
                    
        # error checking
        self._validate_n_bins()
        self._validate_args()
        self._validate_dcols(X)
        
        # get all random forest split points
        self._fit_rf(X = X, y = y)
        
        # get total number of bins to reallocate
        total_bins = self.n_bins.sum()

        # reweight n_bins
        if by == "nsplits":
            # each col gets at least 2 bins; remaining bins get 
            # reallocated based on number of RF splits using that feature
            n_rules = np.array([len(self.rf_splits_[col]) for col in self.dcols])
            self.n_bins = np.round(n_rules / n_rules.sum() *\
                                   (total_bins - 2 * len(self.dcols))) + 2
        else:
            valid_by = ('nsplits')
            raise ValueError("Valid options for 'by' are {}. Got by={!r} instead."\
                             .format(valid_by, by))
        
        
    def fit(self, X, y = None):
        """
        Fit the estimator.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            (Training) data to be discretized.
            
        y : array-like of shape (n_samples,)
            (Training) response vector. Required only if 
            rf_model = None or rf_model has not yet been fitted
            
        Returns
        -------
        self
        """
        
        # by default, discretize all numeric columns
        if len(self.dcols) == 0:
            for col in X.columns:
                if X[col].dtype in ['float', 'int']:
                    self.dcols.append(col)
                    
        # error checking
        self._validate_n_bins()
        self._validate_args()
        self._validate_dcols(X)
        
        # get all random forest split points
        self._fit_rf(X = X, y = y)
        
        # features that were not used in the rf but need to be discretized
        self.missing_rf_cols_ = list(set(self.dcols) -\
                                     set(self.rf_splits_.keys()))
        if len(self.missing_rf_cols_) > 0:
            print("{} did not appear in random forest so were discretized via {} discretization"\
                  .format(self.missing_rf_cols_, self.strategy))
            missing_n_bins = np.array([self.n_bins[np.array(self.dcols) == col][0]\
                                       for col in self.missing_rf_cols_])
            
            backup_discretizer = BasicDiscretizer(n_bins = missing_n_bins,
                                                  dcols = self.missing_rf_cols_, 
                                                  encode = 'ordinal', 
                                                  strategy = self.backup_strategy)
            backup_discretizer.fit(X[self.missing_rf_cols_])
            self.backup_discretizer_ = backup_discretizer
        else:
            self.backup_discretizer_ = None
            
        if self.encode == 'onehot':
            if len(self.missing_rf_cols_) > 0:
                discretized_df = backup_discretizer.transform(X[self.missing_rf_cols_])
            else:
                discretized_df = pd.DataFrame({}, index = X.index)

        # do discretization based on rf split thresholds
        self.bin_edges_ = dict()
        for col in self.dcols:
            if col in self.rf_splits_.keys():
                b = self.n_bins[np.array(self.dcols) == col]
                if self.strategy == "quantile":
                    q_values = np.linspace(0, 1, int(b)+1)[1:-1]
                    splits = np.quantile(self.rf_splits_[col], q_values)
                elif strategy == "uniform":
                    width = (max(self.rf_splits_[col]) - min(self.rf_splits_[col])) / b
                    splits = width * np.arange(1, b) + min(self.rf_splits_[col])
                self.bin_edges_[col] = splits
                if self.encode == 'onehot':
                    discretized_df[col] = self._discretize_by_threshold(X[col], splits)
        
        # fit onehot encoded X if specified
        if self.encode == "onehot":
            onehot = OneHotEncoder(drop = self.onehot_drop, sparse = False)
            onehot.fit(discretized_df[self.dcols].astype(str))
            self.onehot_ = onehot
        
        
    def transform(self, X):
        """
        Discretize the data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to be discretized.
        
        Returns
        -------
        X_discretized : data frame
            Data with features in dcols transformed to the 
            binned space. All other features remain unchanged.
        """
        
        # transform features that did not appear in RF
        if len(self.missing_rf_cols_) > 0:
            discretized_df = self.backup_discretizer_.transform(X[self.missing_rf_cols_])
            discretized_df = pd.DataFrame(discretized_df, 
                                          columns = self.missing_rf_cols_,
                                          index = X.index)
        else:
            discretized_df = pd.DataFrame({}, index = X.index)

        # do discretization based on rf split thresholds
        for col in self.bin_edges_.keys():
            discretized_df[col] = self._discretize_by_threshold(X[col], 
                                                                self.bin_edges_[col])
        
        discretized_df = discretized_df[self.dcols]

        # return onehot encoded X if specified
        if self.encode == "onehot":
            colnames = [str(col) for col in self.dcols]
            onehot_col_names = self.onehot_.get_feature_names(colnames)
            discretized_df = self.onehot_.transform(discretized_df.astype(str))
            discretized_df = pd.DataFrame(discretized_df, 
                                          columns = onehot_col_names, 
                                          index = X.index).astype(int)

        # join discretized columns with rest of X
        cols = [col for col in X.columns if col not in self.dcols]
        X_discretized = pd.concat([discretized_df, X[cols]], axis = 1)

        return X_discretized