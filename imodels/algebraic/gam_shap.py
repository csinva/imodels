from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import resample
import numpy as np


class ShapGAMClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=10, feature_fraction=0.7, random_state=None, ebm_kwargs: dict = {}):
        """
        Initialize the ensemble EBM classifier.

        Parameters:
        - n_estimators: Number of EBM classifiers to create with different random subsets of features.
        - feature_fraction: Fraction of features to use for each EBM classifier.
        - random_state: Seed for random number generator to ensure reproducibility.
        """
        self.n_estimators = n_estimators
        self.feature_fraction = feature_fraction
        self.random_state = random_state
        self.models = []
        self.feature_subsets = []
        self.ebm_kwargs = ebm_kwargs

    def fit(self, X, y):
        """
        Fit the ensemble of EBM classifiers on random feature subsets.
        """
        rng = np.random.default_rng(self.random_state)
        n_features = X.shape[1]

        for _ in range(self.n_estimators):
            # Randomly select a subset of features
            if isinstance(self.feature_fraction, float):
                n_features = max(1, int(self.feature_fraction * n_features))
            elif self.feature_fraction == 'uniform':
                n_features = rng.integers(1, n_features + 1)
            feature_subset = rng.choice(
                n_features,
                size=n_features,
                replace=False
            )
            self.feature_subsets.append(feature_subset)

            # Create an EBM with the selected feature subset
            ebm = ExplainableBoostingClassifier(
                random_state=self.random_state, **self.ebm_kwargs)
            X_subset = X[:, feature_subset]
            ebm.fit(X_subset, y)
            self.models.append(ebm)

        return self

    def predict_proba(self, X):
        """
        Predict probabilities by averaging the predictions of all models.
        """
        # Aggregate predictions from all models
        probs = np.zeros((X.shape[0], 2))

        for ebm, feature_subset in zip(self.models, self.feature_subsets):
            probs_ = ebm.predict_proba(X[:, feature_subset])
            probs += probs_

        # Average the probabilities across all models
        probs /= self.n_estimators
        return probs

    def predict(self, X):
        """
        Predict class labels by averaging the shape function outputs and taking the argmax.
        """
        return np.argmax(self.predict_proba(X), axis=1)

    def get_params(self, deep=True):
        return {
            "n_estimators": self.n_estimators,
            "feature_fraction": self.feature_fraction,
            "random_state": self.random_state
        }

    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        return self


if __name__ == '__main__':
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    # Load data
    X, y = load_iris(return_X_y=True)

    # make binary classification
    X = X[y < 2]
    y = y[y < 2]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    # Create and fit ensemble EBM
    ebm_ensemble = ShapGAMClassifier(
        n_estimators=5, feature_fraction=0.8, random_state=42)
    ebm_ensemble.fit(X_train, y_train)

    # Evaluate
    y_pred = ebm_ensemble.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Ensemble EBM Accuracy: {accuracy}")
