import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y

import imodels


def explain_classification_errors(X, predictions, y,
                                  feature_names: list = None,
                                  target_name: str = None,
                                  classifier: BaseEstimator = imodels.GreedyTreeClassifier(),
                                  target_one_hot_encode: bool = False,
                                  print_rules: bool = True):
    """Explains the classification errors of a model by fitting an interpretable model to them.
    Currently only supports binary classification.

    Parameters
    ----------
    X: array_like
        (n, n_features)
    predictions: array_like
        (n, 1) predictions
    y
        (n, 1) targets with integer values representing class
    feature_names
        n_features

    Returns
    -------
    model: BaseEstimator
    """
    # deal with names
    if feature_names is None:
        if isinstance(X, pd.DataFrame):
            feature_names = X.columns.tolist()
        else:
            feature_names = [f'X{i + 1}' for i in range(X.shape[1])]

    if target_name is None:
        if isinstance(y, pd.DataFrame):
            target_name = y.columns[0]
        elif isinstance(y, pd.Series):
            target_name = y.name
        else:
            target_name = 'target'
    if isinstance(predictions, pd.Series) or isinstance(predictions, pd.DataFrame):
        predictions = predictions.values

    X, y = check_X_y(X, y)  # converts to np
    if len(y.shape) == 1:
        y = y.reshape(-1, 1)
    if len(predictions.shape) == 1:
        predictions = predictions.reshape(-1, 1)

    errors = np.array(predictions != y).astype(int)

    features = pd.DataFrame(np.hstack((X, y)))
    features.columns = [*feature_names, target_name]
    classifier.fit(features, errors.flatten())
    if print_rules:
        print(classifier)
    return classifier, features.columns
