import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array
from sklearn.utils.multiclass import check_classification_targets
import scipy.sparse

def check_fit_arguments(model, X, y, feature_names, multi_output=False, is_classmixin=True):
    """Process arguments for fit and predict methods.
    """
    if isinstance(model, ClassifierMixin) and is_classmixin:
        model.classes_, y = np.unique(y, return_inverse=True)  # deals with str inputs
        check_classification_targets(y)

    if feature_names is None:
        if isinstance(X, pd.DataFrame):
            model.feature_names_ = X.columns
        elif isinstance(X, list):
            model.feature_names_ = ['X' + str(i) for i in range(len(X[0]))]
        else:
            model.feature_names_ = ['X' + str(i) for i in range(X.shape[1])]
    else:
        model.feature_names_ = feature_names
    if scipy.sparse.issparse(X):
        X = X.toarray()
    X, y = check_X_y(X=X, y=y, multi_output=multi_output)
    _, model.n_features_in_ = X.shape
    assert len(model.feature_names_) == model.n_features_in_, 'feature_names should be same size as X.shape[1]'
    y = y.astype(float)
    return X, y, model.feature_names_

def check_fit_X(X):
    """Process X argument for fit and predict methods.
    """
    if scipy.sparse.issparse(X):
        X = X.toarray()
    X = check_array(X)
    return X
