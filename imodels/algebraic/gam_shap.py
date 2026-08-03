from interpret.glassbox import ExplainableBoostingClassifier, ExplainableBoostingRegressor
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
import numpy as np
from imodels.util.arguments import explicit_get_params, explicit_set_params


class ShapGAM(BaseEstimator):
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
            cls = ExplainableBoostingClassifier if isinstance(
                self, ShapGAMClassifier) else ExplainableBoostingRegressor
            ebm = cls(
                random_state=self.random_state, **self.ebm_kwargs)
            X_subset = X[:, feature_subset]
            ebm.fit(X_subset, y)
            self.models.append(ebm)

        return self

    def predict_proba(self, X):
        """
        Predict probabilities by averaging the shaping functions for each feature across models then predicting
        """
        if isinstance(self, ShapGAMRegressor):
            raise NotImplementedError(
                "This method is not implemented for regression tasks.")
        # Aggregate predictions from all models
        probs = np.zeros((X.shape[0], 2))

        for feat_num in range(X.shape[1]):
            X_ = np.zeros_like(X)
            X_[:, feat_num] = X[:, feat_num]
            feat_num_counts = 0
            probs_feat = np.zeros((X.shape[0], 2))
            for ebm, feature_subset in zip(self.models, self.feature_subsets):
                if feat_num in feature_subset:
                    # todo: there is an error here -- aggregating probas is not the same as aggregating logits!
                    raise NotImplementedError(
                        "This is not the same as aggregating logits Look at the predict function for smth more accurate!")
                    probs_ = ebm.predict_proba(X[:, feature_subset])
                    probs_feat += probs_
                    feat_num_counts += 1
            if feat_num_counts > 0:
                probs_feat /= feat_num_counts
                probs += probs_feat

                # for ebm, feature_subset in zip(self.models, self.feature_subsets):
                # probs_ = ebm.predict_proba(X[:, feature_subset])
                # probs += probs_

                # Average the probabilities across all models
                # probs /= self.n_estimators
        return probs

    def predict(self, X):
        """
        Predict class labels by averaging the shape function outputs and taking the argmax.
        """
        if isinstance(self, ShapGAMClassifier):
            return np.argmax(self.predict_proba(X), axis=1)
        else:
            # naively averaging the predictions across all models
            # preds = np.zeros(X.shape[0])
            # for ebm, feature_subset in zip(self.models, self.feature_subsets):
            #     preds += ebm.predict(X[:, feature_subset])
            # preds /= self.n_estimators
            # return preds

            # averaging the predictions across all models for each feature
            preds = np.zeros(len(X))
            for ex_num in range(len(X)):
                for feat_num in range(X.shape[1]):
                    feat_count = 0
                    pred_feat = 0
                    for ebm, feature_subset in zip(self.models, self.feature_subsets):
                        if feat_num in feature_subset:
                            expl_local = ebm.explain_local(
                                X[ex_num, feature_subset])
                            feat_num_in_subset = np.where(
                                feature_subset == feat_num)[0][0]
                            pred_feat += expl_local.data(
                                0)['scores'][feat_num_in_subset]
                            feat_count += 1
                    if feat_count > 0:
                        pred_feat /= feat_count
                    preds[ex_num] += pred_feat

            return preds

    _PARAM_NAMES = ("n_estimators", "feature_fraction", "random_state")

    def get_params(self, deep=True):
        return explicit_get_params(self, self._PARAM_NAMES, deep=deep)

    def set_params(self, **params):
        return explicit_set_params(self, self._PARAM_NAMES, **params)


class ShapGAMRegressor(ShapGAM, RegressorMixin):
    ...


class ShapGAMClassifier(ShapGAM, ClassifierMixin):
    ...
