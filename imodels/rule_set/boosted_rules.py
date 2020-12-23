import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from copy import deepcopy
from functools import partial
from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeClassifier

from imodels.rule_set.rule_set import RuleSet
from imodels.util.convert import tree_to_code


class BoostedRulesClassifier(BaseEstimator, RuleSet):
    '''An easy-interpretable classifier optimizing simple logical rules.
    Currently limited to only binary classification.
    '''

    def __init__(self, n_estimators=10, base_estimator=partial(DecisionTreeClassifier, max_depth=1)):
        self.n_estimators = n_estimators
        self.base_estimator_ = base_estimator

    def fit(self, X, y, feature_names=None, sample_weight=None):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape (n_samples,)
            Target vector relative to X. Has to follow the convention 0 for
            normal data, 1 for anomalies.

        sample_weight : array-like, shape (n_samples,) optional
            Array of weights that are assigned to individual samples, typically
            the amount in case of transactions data. Used to grow regression
            trees producing further rules to be tested.
            If not provided, then each sample is given unit weight.

        Returns
        -------
        self : object
            Returns self.
        """

        # Initialize weights
        n_train = y.shape[0]
        w = np.ones(n_train) / n_train
        self.estimators_ = []
        self.estimator_weights_ = []
        self.estimator_errors_ = []
        self.feature_names = feature_names
        for _ in range(self.n_estimators):
            # Fit a classifier with the specific weights
            clf = self.base_estimator_()
            clf.fit(X, y, sample_weight=w)  # uses w as the sampling weight!
            preds = clf.predict(X)

            # Indicator function
            miss = [int(x)
                    for x in (preds != y)]

            # Equivalent with 1/-1 to update weights
            miss2 = [x if x == 1 else -1
                     for x in miss]

            # Error
            err_m = np.dot(w, miss) / sum(w)
            if err_m < 1e-3:
                return self

            # Alpha
            alpha_m = 0.5 * np.log((1 - err_m) / float(err_m))

            # New weights
            w = np.multiply(w, np.exp([float(x) * alpha_m
                                       for x in miss2]))

            self.estimators_.append(deepcopy(clf))
            self.estimator_weights_.append(alpha_m)
            self.estimator_errors_.append(err_m)
        return self

    def predict_proba(self, X):
        '''Predict probabilities for X
        '''

        if type(X) == pd.DataFrame:
            X = X.values.astype(np.float32)

        # Add to prediction
        n_train = X.shape[0]
        n_estimators = len(self.estimators_)
        n_classes = 2  # hard-coded for now!
        preds = np.zeros((n_train, n_classes))
        for i in range(n_estimators):
            preds += self.estimator_weights_[i] * self.estimators_[i].predict_proba(X)
        return preds / n_estimators

    def predict(self, X):
        """Predict outcome for X
        """

        return self.predict_proba(X).argmax(axis=1)

    def __str__(self):
        try:
            s = 'Mined rules:\n'
            for est in self.estimators_:
                s += '\t' + tree_to_code(est, self.feature_names)
            return s
        except:
            return f'BoostedRules with {len(self.estimators_)} estimators'
