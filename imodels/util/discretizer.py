import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    
def discretize_features(X, Xtest = None, cols_to_discretize = [], n_bins = 2, 
                        encode = "onehot", strategy = "quantile", 
                        onehot_drop = "if_binary"):
    """
    Function to discretize the numeric features of a data frame.
    
    Parameters
    ----------
    X: data frame of shape (n_samples, n_features)
        Data frame to be discretized; training data
    
    Xtest: optional data frame of shape (n_samples_test, n_features)
        Data frame to be discretized; test data
        
    cols_to_discretize: list of strings
        The names of the columns to be discretized; by default, 
        discretize all float and int columns in X
        
    n_bins: int or array-like of shape (len(cols_to_discretize),), default=2
        Number of bins to discretize each feature into
        
    encode: {‘onehot’, ‘ordinal’}, default=’onehot’
        Method used to encode the transformed result.
        onehot:
            Encode the transformed result with one-hot encoding and
            return a dense array.
        ordinal:
            Return the bin identifier encoded as an integer value
            
    strategy: {‘uniform’, ‘quantile’, ‘kmeans’}, default=’quantile’
        Strategy used to define the widths of the bins.
        uniform:
            All bins in each feature have identical widths.
        quantile:
            All bins in each feature have the same number of points.
        kmeans:
            Values in each bin have the same nearest center of a 1D
            k-means cluster.
    
    onehot_drop: {‘first’, ‘if_binary’} or a array-like of shape 
    (len(cols_to_discretize),), default='if_binary'
        Specifies a methodology to use to drop one of the categories per feature 
        when encode = "onehot".
        None:
            retain all features (the default).
        ‘first’: 
            drop the first category in each feature. If only one category 
            is present, the feature will be dropped entirely.
        ‘if_binary’ : 
            drop the first category in each feature with two categories.
            Features with 1 or more than 2 categories are left intact.
    
    Returns
    -------
    X_discretized: training data frame where selected columns have been discretized
    Xtest_discretized: test data frame where selected columns have been discretized
    
    """
    
    # error checking
    if isinstance(n_bins, int):
        if n_bins < 2:
            raise ValueError("Invalid number of bins. n_bins must be at least 2.");
    elif len(n_bins) > 1:
        if any(n_bins < 2):
            raise ValueError("Invalid number of bins. n_bins must be at least 2.");
        
    valid_encode = ('onehot', 'ordinal')
    if encode not in valid_encode:
        raise ValueError("Valid options for 'encode' are {}. Got encode={!r} instead."\
                         .format(valid_encode, encode))
    
    valid_strategy = ('uniform', 'quantile', 'kmeans')
    if (strategy not in valid_strategy):
        raise ValueError("Valid options for 'strategy' are {}. Got strategy={!r} instead."\
                         .format(valid_strategy, strategy))
        
    # get all numeric columns in X
    numeric_cols = []
    for col in X.columns:
        if X[col].dtype in ["float", "int"]:
            numeric_cols.append(col)
            
    # If no columns given, discretize all columns of type float or int
    if len(cols_to_discretize) == 0:
        cols_to_discretize = numeric_cols
    else:
        for col in cols_to_discretize:
            if col not in X.columns:
                raise ValueError("{} is not a column in X.".format(col))
            if col not in numeric_cols:
                raise ValueError("Cannot discretize non-numeric columns.")
    
    if not isinstance(n_bins, int):
        if len(n_bins) != len(cols_to_discretize):
            raise ValueError("n_bins must be an int or array-like of shape (len(cols_to_discretize),)")
            
    # Apply KBinsDiscretizer to the selected columns
    discretizer = KBinsDiscretizer(n_bins = n_bins,
                                   encode = "ordinal", 
                                   strategy = strategy)
    discretizer.fit(X[cols_to_discretize])
    discretized_df = discretizer.transform(X[cols_to_discretize]).astype(int)
    discretized_df = pd.DataFrame(discretized_df, 
                                  columns = cols_to_discretize,
                                  index = X.index)
    if Xtest is not None:
        test_discretized_df = discretizer.transform(Xtest[cols_to_discretize]).astype(int)
        test_discretized_df = pd.DataFrame(test_discretized_df, 
                                           columns = cols_to_discretize, 
                                           index = Xtest.index)
    
    # fix KBinsDiscretizer errors if any when strategy = "quantile"
    if strategy == "quantile":
        for col in cols_to_discretize:
            if (discretized_df[col].nunique() != n_bins) & (X[col].nunique() > 1):
                # KBinsDiscretizer ran error, so do discretization manually
                if isinstance(n_bins, int):
                    b = n_bins
                else:
                    b = n_bins[cols_to_discretize == col]
                discretized_df[col] = quantile_discretization(X[col], b)
                if Xtest is not None:
                    test_discretized_df[col] = quantile_discretization(Xtest[col], b)
    
    # return onehot encoded X if specified
    if encode == "onehot":
        onehot = OneHotEncoder(drop = onehot_drop, sparse = False)
        onehot.fit(discretized_df.astype(str))
        onehot_col_names = onehot.get_feature_names(cols_to_discretize)
        discretized_df = onehot.transform(discretized_df.astype(str))
        discretized_df = pd.DataFrame(discretized_df, 
                                      columns = onehot_col_names,
                                      index = X.index).astype(int)
        if Xtest is not None:
            test_discretized_df = onehot.transform(test_discretized_df.astype(str))
            test_discretized_df = pd.DataFrame(test_discretized_df, 
                                               columns = onehot_col_names,
                                               index = Xtest.index).astype(int)
    
    # join discretized columns with rest of X
    cols = [col for col in X.columns if col not in cols_to_discretize]
    X_discretized = pd.concat([discretized_df, X[cols]], axis = 1)
    if Xtest is not None:
        Xtest_discretized = pd.concat([test_discretized_df, Xtest[cols]], axis = 1)
        return X_discretized, Xtest_discretized
            
    return X_discretized


def discretize_features_rf(X, y = None, Xtest = None, 
                           rf_model = None, classification = False,
                           cols_to_discretize = [], n_bins = 2, 
                           encode = "onehot", strategy = "quantile", 
                           backup_strategy = "quantile",
                           onehot_drop = "if_binary"):
    """
    Function to discretize the numeric features of a data frame using RF splits.
    
    Parameters
    ----------
    X: data frame of shape (n_samples, n_features)
        Data frame to be discretized; training data
    
    y: series of shape (n_samples,)
        Response for fitting random forest if needed
    
    Xtest: optional data frame of shape (n_samples_test, n_features)
        Data frame to be discretized; test data
        
    rf_model: RandomForestClassifer() or RandomForestRegressor()
        RF model from which to extract splits for discretization
        
    cols_to_discretize: list of strings
        The names of the columns to be discretized; by default, 
        discretize all float and int columns in X
        
    n_bins: int or array-like of shape (len(cols_to_discretize),), default=2
        Number of bins to discretize each feature into
        
    encode: {‘onehot’, ‘ordinal’}, default=’onehot’
        Method used to encode the transformed result.
        onehot:
            Encode the transformed result with one-hot encoding and
            return a dense array.
        ordinal:
            Return the bin identifier encoded as an integer value
            
    strategy: {‘uniform’, ‘quantile’}, default=’quantile’
        Strategy used to choose RF split points.
        uniform:
            RF split points chosen to be uniformly spaced out.
        quantile:
            RF split points chosen to be uniform with respect to its quantilization.
    
    backup_strategy: {‘uniform’, ‘quantile’, ‘kmeans’}, default=’quantile’
        Strategy used to define the widths of the bins if no rf splits exist for 
        that feature.
        uniform:
            All bins in each feature have identical widths.
        quantile:
            All bins in each feature have the same number of points.
        kmeans:
            Values in each bin have the same nearest center of a 1D
            k-means cluster.
    
    onehot_drop: {‘first’, ‘if_binary’} or a array-like of shape 
    (len(cols_to_discretize),), default='if_binary'
        Specifies a methodology to use to drop one of the categories per feature 
        when encode = "onehot".
        None:
            retain all features (the default).
        ‘first’: 
            drop the first category in each feature. If only one category 
            is present, the feature will be dropped entirely.
        ‘if_binary’ : 
            drop the first category in each feature with two categories.
            Features with 1 or more than 2 categories are left intact.
    
    Returns
    -------
    X_discretized: training data frame where selected columns have been discretized
    Xtest_discretized: test data frame where selected columns have been discretized
    
    """
    
    # error checking
    if isinstance(n_bins, int):
        if n_bins < 2:
            raise ValueError("Invalid number of bins. n_bins must be at least 2.");
    elif len(n_bins) > 1:
        if any(n_bins < 2):
            raise ValueError("Invalid number of bins. n_bins must be at least 2.");
        
    valid_encode = ('onehot', 'ordinal')
    if encode not in valid_encode:
        raise ValueError("Valid options for 'encode' are {}. Got encode={!r} instead."\
                         .format(valid_encode, encode))
    
    valid_strategy = ('uniform', 'quantile', 'kmeans')
    if (strategy not in valid_strategy):
        raise ValueError("Valid options for 'strategy' are {}. Got strategy={!r} instead."\
                         .format(valid_strategy, strategy))
        
    # get all numeric columns in X
    numeric_cols = []
    for col in X.columns:
        if X[col].dtype in ["float", "int"]:
            numeric_cols.append(col)
            
    # If no columns given, discretize all columns of type float or int
    if len(cols_to_discretize) == 0:
        cols_to_discretize = numeric_cols
    else:
        for col in cols_to_discretize:
            if col not in X.columns:
                raise ValueError("{} is not a column in X.".format(col))
            if col not in numeric_cols:
                raise ValueError("Cannot discretize non-numeric columns.")
    
    if not isinstance(n_bins, int):
        if len(n_bins) != len(cols_to_discretize):
            raise ValueError("n_bins must be an int or array-like of shape (len(cols_to_discretize),)")
    else:
        n_bins = np.repeat(n_bins, len(cols_to_discretize))
    
    # If no rf_model given, train default random forest model
    if rf_model is None:
        if y is None:
            raise ValueError("Must provide y if rf_model is not given.")
        if classification:
            rf_model = RandomForestClassifier(n_estimators = 500)
        else:
            rf_model = RandomForestRegressor(n_estimators = 500)
        
        print("Fitting random forest...")
        rf_model.fit(X, y)
        
    else:
        # given rf model has not yet been trained
        if not hasattr(rf_model, "estimators_"):
            if y is None:
                raise ValueError("Must provide y if rf_model has not been trained.")
                
            print("Fitting random forest...")
            rf_model.fit(X, y)
    
    # get all random forest split points
    rule_dict = get_rf_splits(rf_model, list(X.columns))
    
    # features that were not used in the rf but need to be discretized
    missing_cols = list(set(cols_to_discretize) - set(rule_dict.keys()))
    if len(missing_cols) > 0:
        print("{} did not appear in random forest so were discretized via {} discretization".format(missing_cols, strategy))
        missing_n_bins = np.array([n_bins[np.array(cols_to_discretize) == col][0]\
                                   for col in missing_cols])
        if Xtest is not None:
            discretized_df, \
            test_discretized_df = discretize_features(X = X, Xtest = Xtest,
                                                      cols_to_discretize = missing_cols,
                                                      n_bins = missing_n_bins, 
                                                      encode = "ordinal",
                                                      strategy = backup_strategy)
            test_discretized_df = test_discretized_df.loc[:, missing_cols]
        else:
            discretized_df = discretize_features(X = X,
                                                 cols_to_discretize = missing_cols,
                                                 n_bins = missing_n_bins, 
                                                 encode = "ordinal",
                                                 strategy = backup_strategy)
        discretized_df = discretized_df.loc[:, missing_cols]
    else:
        discretized_df = pd.DataFrame({}, index = X.index)
        if Xtest is not None:
            test_discretized_df = pd.DataFrame({}, index = Xtest.index)
    
    # do discretization based on rf split thresholds
    for col in cols_to_discretize:
        if col in rule_dict.keys():
            b = n_bins[np.array(cols_to_discretize) == col]
            if strategy == "quantile":
                q_values = np.linspace(0, 1, int(b)+1)[1:-1]
                splits = np.quantile(rule_dict[col], q_values)
            elif strategy == "uniform":
                width = (max(rule_dict[col]) - min(rule_dict[col])) / b
                splits = width * np.arange(1, b) + min(rule_dict[col])
        discretized_df[col] = discretize_by_threshold(X[col], splits)
        if Xtest is not None:
            test_discretized_df[col] = discretize_by_threshold(Xtest[col], splits)
    
    # return onehot encoded X if specified
    if encode == "onehot":
        onehot = OneHotEncoder(drop = onehot_drop, sparse = False)
        onehot.fit(discretized_df.astype(str))
        onehot_col_names = onehot.get_feature_names(cols_to_discretize)
        discretized_df = onehot.transform(discretized_df.astype(str))
        discretized_df = pd.DataFrame(discretized_df, 
                                      columns = onehot_col_names,
                                      index = X.index).astype(int)
        if Xtest is not None:
            test_discretized_df = onehot.transform(test_discretized_df.astype(str))
            test_discretized_df = pd.DataFrame(test_discretized_df, 
                                               columns = onehot_col_names,
                                               index = Xtest.index).astype(int)
    
    # join discretized columns with rest of X
    cols = [col for col in X.columns if col not in cols_to_discretize]
    X_discretized = pd.concat([discretized_df, X[cols]], axis = 1)
    if Xtest is not None:
        Xtest_discretized = pd.concat([test_discretized_df, Xtest[cols]], axis = 1)
        return X_discretized, Xtest_discretized
            
    return X_discretized
    

def quantile_discretization(x, n_bins):
    
    q_values = np.linspace(0, 1, n_bins+1)
    quantiles = np.quantile(x, q_values)
    unique_quantiles = np.unique(quantiles)
    duplicated_quantiles = np.unique(quantiles[pd.Series(quantiles).duplicated()])
    
    discretized_values = np.zeros_like(x)
    i = 0
    for q_idx, q in enumerate(unique_quantiles[:-1]):
        indicator = (x >= q) & (x < unique_quantiles[q_idx+1])
        if q in duplicated_quantiles:
            discretized_values[x == q] = i
            indicator = (x > q) & (x < unique_quantiles[q_idx+1])
            i += 1
        discretized_values[indicator] = i
        i += 1
    if unique_quantiles[-1] in duplicated_quantiles:
        discretized_values[x == unique_quantiles[-1]] = i
    else:
        discretized_values[x == unique_quantiles[-1]] = i-1
    
    return discretized_values.astype(int)


def discretize_by_threshold(x, splits):
    
    discretized_values = np.zeros_like(x)
    i = 0
    for idx, split in enumerate(splits):
        if idx == 0:
            indicator = x < split
            discretized_values[indicator] = i
            i += 1
        if idx == (len(splits) - 1):
            indicator = x >= split
            discretized_values[indicator] = i
            i += 1
        if (idx != 0) & (idx != (len(splits) - 1)):
            indicator =  (x >= split) & (x < splits[idx + 1])
            discretized_values[indicator] = i
            i += 1
    
    return discretized_values.astype(int)


def get_rf_splits(rf_model, col_names):
    
    rule_dict = {}
    for model in rf_model.estimators_:
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