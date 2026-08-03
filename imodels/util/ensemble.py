from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import numpy as np


class ResidualBoostingRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, estimator, n_estimators=10):
        """
        A meta-estimator that fits a base estimator to the residuals of the
        previous estimators.

        Parameters:
        - estimator: The estimator to fit on the residual of the previous step.
        - n_estimators: The number of estimators to fit.
        """
        self.estimator = estimator
        self.n_estimators = n_estimators

    def fit(self, X, y):
        """
        Fit the ensemble of base estimators on the training data.

        Parameters:
        - X: array-like of shape (n_samples, n_features)
            Training data.
        - y: array-like of shape (n_samples,)
            Target values.

        Returns:
        - self: object
        """
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)

        self.estimators_ = []
        current_prediction = np.zeros(y.shape)

        for _ in range(self.n_estimators):
            residual = y - current_prediction
            estimator = clone(self.estimator)
            estimator.fit(X, residual)
            self.estimators_.append(estimator)
            current_prediction += estimator.predict(X)

        return self

    def predict(self, X):
        """
        Predict regression target for X.

        Parameters:
        - X: array-like of shape (n_samples, n_features)
            The input samples.

        Returns:
        - y_pred: ndarray of shape (n_samples,)
            The predicted values.
        """
        # Check is fit had been called
        check_is_fitted(self)
        # Input validation
        X = check_array(X)

        predictions = sum(estimator.predict(X)
                          for estimator in self.estimators_)
        return predictions


class SimpleBaggingRegressor:
    def __init__(self, estimator, n_estimators=10, random_state=None):
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.random_state = random_state

    def fit(self, X, y):
        np.random.seed(self.random_state)
        self.estimators_ = []

        rng = np.random.default_rng(self.random_state)
        for _ in range(self.n_estimators):
            # Simple bootstrap sampling
            # sample_indices = np.random.choice(
            # range(X.shape[0]), size=X.shape[0], replace=True)
            sample_indices = rng.choice(
                range(X.shape[0]), size=X.shape[0], replace=True)
            X_sample = X[sample_indices]
            y_sample = y[sample_indices]

            # Fit a base estimator
            # estimator = DecisionTreeRegressor()
            estimator = clone(self.estimator)
            estimator.fit(X_sample, y_sample)
            self.estimators_.append(estimator)

    def predict(self, X):
        # Collect predictions from each base estimator
        predictions = np.array([estimator.predict(X)
                               for estimator in self.estimators_])

        # Aggregate predictions
        return np.mean(predictions, axis=0)
