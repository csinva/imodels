import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.utils.validation import check_X_y
from sklearn.utils.multiclass import check_classification_targets


def check_fit_arguments(model, X, y, feature_names):
    """Process arguments for fit and predict methods.
    """
    if isinstance(model, ClassifierMixin):
        model.classes_, y = np.unique(y, return_inverse=True)  # deals with str inputs
        check_classification_targets(y)

    if feature_names is None:
        if isinstance(X, pd.DataFrame):
            model.feature_names_ = X.columns
        else:
            model.feature_names_ = ['X' + str(i) for i in range(len(X[0]))]
    else:
        model.feature_names_ = feature_names
    X, y = check_X_y(X, y)
    _, model.n_features_in_ = X.shape
    assert len(model.feature_names_) == model.n_features_in_, 'feature_names should be same size as X.shape[1]'
    y = y.astype(float)
    return X, y, model.feature_names_
    # if sample_weight is not None:
        # sample_weight = _check_sample_weight(sample_weight, X)


