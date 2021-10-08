import numpy as np
from sklearn.base import BaseEstimator

import imodels


def explain_classification_errors(X, predictions, targets,
                                  classifier: BaseEstimator = imodels.CorelsRuleListClassifier):
    """Explains the classification errors of a model by fitting an interpretable model to them.
    Parameters
    ----------
    X: array_like
        (n, n_features)
    predictions: array_like
        (n, 1) predictions
    targets
        (n, 1) targets

    Returns
    -------
    model: BaseEstimator
    """
    if len(targets.shape) == 1:
        targets = targets.reshape(-1, 1)
    if len(predictions.shape) == 1:
        predictions = predictions.reshape(-1, 1)

    errors = np.array(targets != predictions).astype(int)

    features = np.hstack((X, targets))
    cls = classifier()
    cls.fit(features, errors.flatten())
    print(cls)
    return cls
