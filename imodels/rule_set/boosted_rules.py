from copy import deepcopy
from functools import partial

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, MetaEstimatorMixin
from sklearn.preprocessing import normalize
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.multiclass import check_classification_targets, unique_labels
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from imodels.rule_set.rule_set import RuleSet
from imodels.rule_set.slipper_util import SlipperBaseEstimator
from imodels.util.convert import tree_to_code, tree_to_rules, dict_to_rule
from imodels.util.rule import Rule, get_feature_dict, replace_feature_name


class BoostedRulesClassifier(RuleSet, BaseEstimator, MetaEstimatorMixin, ClassifierMixin):
    '''An easy-interpretable classifier optimizing simple logical rules.
    Currently limited to only binary classification.
    
    Params
    ------
    estimator: object with fit and predict methods
        Defaults to DecisionTreeClassifier with AdaBoost.
        For SLIPPER, should pass estimator=imodels.SlipperBaseEstimator
    '''

    def __init__(self, n_estimators=10, estimator=partial(DecisionTreeClassifier, max_depth=1), random_state=0):
        self.n_estimators = n_estimators
        self.estimator = estimator
        self.random_state = random_state

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

        X, y = check_X_y(X, y)
        check_classification_targets(y)
        self.n_features_ = X.shape[1]
        self.classes_ = unique_labels(y)
        np.random.seed(self.random_state)

        if feature_names is None:
            feature_names = [f'X_{i + 1}' for i in range(X.shape[1])]
        self.feature_dict_ = get_feature_dict(X.shape[1], feature_names)
        self.feature_placeholders = list(self.feature_dict_.keys())
        self.feature_names = list(self.feature_dict_.values())

        n_train = y.shape[0]
        w = np.ones(n_train) / n_train
        self.estimators_ = []
        self.estimator_weights_ = []
        self.estimator_errors_ = []
        self.estimator_mean_prediction_ = [] # this is just for printing
        if feature_names is not None:
            self.feature_names = feature_names
        for _ in range(self.n_estimators):
            # Fit a classifier with the specific weights
            clf = self.estimator()
            clf.fit(X, y, sample_weight=w)  # uses w as the sampling weight!
            preds = clf.predict(X)
            self.estimator_mean_prediction_.append(np.mean(preds)) # just for printing

            # Indicator function
            miss = preds != y

            # Equivalent with 1/-1 to update weights
            miss2 = np.ones(miss.size)
            miss2[~miss] = -1

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

        rules = []

        for est, est_weight in zip(self.estimators_, self.estimator_weights_):
            if type(clf) == DecisionTreeClassifier:
                est_rules_values = tree_to_rules(est, self.feature_placeholders, prediction_values=True)
                est_rules = list(map(lambda x: x[0], est_rules_values))

                # BRS scores are difference between class 1 % and class 0 % in a node
                est_values = np.array(list(map(lambda x: x[1], est_rules_values)))
                rule_scores = (est_values[:, 1] - est_values[:, 0]) / est_values.sum(axis=1)

                compos_score = est_weight * rule_scores
                rules += [Rule(r, args=[w]) for (r, w) in zip(est_rules, compos_score)]

            if type(clf) == SlipperBaseEstimator:
                # SLIPPER uses uniform confidence over in rule observations
                est_rule = dict_to_rule(est.rule, est.feature_dict)
                rules += [Rule(est_rule, args=[est_weight])]

        self.rules_without_feature_names_ = rules
        self.rules_ = [
            replace_feature_name(rule, self.feature_dict_)
            for rule in self.rules_without_feature_names_
        ]
        self.complexity_ = self._get_complexity()
        return self

    def predict_proba(self, X):
        ''' Predict probabilities for X '''
        check_is_fitted(self)
        X = check_array(X)

        # Add to prediction
        n_train = X.shape[0]
        n_estimators = len(self.estimators_)
        n_classes = 2  # hard-coded for now!
        preds = np.zeros((n_train, n_classes))
        for i in range(n_estimators):
            preds += self.estimator_weights_[i] * self.estimators_[i].predict_proba(X)
        pred_values = preds / n_estimators
        return normalize(pred_values, norm='l1')

    def predict(self, X):
        """Predict outcome for X
        """
        check_is_fitted(self)
        X = check_array(X)
        return self._eval_weighted_rule_sum(X) > 0

    def __str__(self):
        try:
            s = 'BoostedRules:\n'
            s += 'Rule \u2192 predicted probability (final prediction is weighted sum of all predictions)\n'
            for i in range(len(self.estimators_)):
                s += f'  If\033[96m {str(self.rules_[i])}\033[00m \u2192 {self.estimator_mean_prediction_[i]:.2f} (weight: {self.estimator_weights_[i]:.2f})\n'
            # for est in self.estimators_:
            #     s += '\t' + tree_to_code(est, self.feature_names)
            return s
        except:
            return f'BoostedRules with {len(self.estimators_)} estimators'
