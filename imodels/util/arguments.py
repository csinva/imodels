from inspect import signature

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array
from sklearn.utils.multiclass import check_classification_targets
import scipy.sparse


def check_fit_arguments(model, X, y, feature_names, multi_output=False, is_classmixin=True,
                        allow_nan=False):
    """Process arguments for fit and predict methods.

    For classifiers, sets ``model.classes_`` and encodes y as integers 0..n_classes-1
    (use `decode_labels` to map predictions back onto the original labels).
    Always sets ``model.feature_names_`` and ``model.n_features_in_``, and sets the
    sklearn-standard ``model.feature_names_in_`` when X carries column names.

    allow_nan: pass missing values in X through instead of rejecting them, for models
        that delegate to an estimator able to handle them (e.g. an sklearn decision
        tree). That estimator still raises if it cannot.
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

    set_feature_names_in(model, X)

    if scipy.sparse.issparse(X):
        X = X.toarray()
    X, y = check_X_y(X=X, y=y, multi_output=multi_output,
                     **_finite_check_kwarg(allow_nan))
    _, model.n_features_in_ = X.shape
    assert len(model.feature_names_) == model.n_features_in_, 'feature_names should be same size as X.shape[1]'
    y = y.astype(float)
    return X, y, model.feature_names_


def _finite_check_kwarg(allow_nan):
    """Spell the "allow NaN" option the way the installed sklearn expects.

    force_all_finite was renamed to ensure_all_finite in sklearn 1.6.
    """
    if not allow_nan:
        return {}
    name = ("ensure_all_finite"
            if "ensure_all_finite" in signature(check_X_y).parameters
            else "force_all_finite")
    return {name: "allow-nan"}


def check_predict_X(model, X):
    """Check X at predict time against the data the model was fitted on.

    Models that index X by a stored feature number silently accept extra
    columns, returning predictions that look reasonable but ignore part of the
    input. sklearn raises instead, and so should we.
    """
    n_features = getattr(model, 'n_features_in_', None)
    if n_features is not None and np.shape(X)[1] != n_features:
        raise ValueError(
            f"X has {np.shape(X)[1]} features, but "
            f"{type(model).__name__} is expecting {n_features} features as input."
        )
    return X


def set_feature_names_in(model, X):
    """Set the sklearn-standard ``feature_names_in_`` if X carries string column names.

    sklearn deletes this attribute when a model is subsequently fit on a plain array,
    so wrappers that forward a numpy array to a parent estimator should call this
    again afterwards.
    """
    if hasattr(X, 'columns') and all(isinstance(c, str) for c in X.columns):
        model.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return True
    return False


def check_fit_X(X):
    """Process X argument for fit and predict methods.
    """
    if scipy.sparse.issparse(X):
        X = X.toarray()
    X = check_array(X)
    return X


def decode_labels(model, preds):
    """Map integer-encoded predictions back onto the original labels seen during fit.

    `check_fit_arguments` encodes y as 0..n_classes-1, so classifiers must decode
    their predictions to honor the sklearn contract that predict returns labels
    drawn from ``classes_`` (which may be strings, or ints that aren't 0/1).
    """
    preds = np.asarray(preds)
    if not hasattr(model, 'classes_'):
        return preds
    classes = np.asarray(model.classes_)
    return classes[preds.astype(int)]
