from copy import deepcopy
from functools import partial

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, MetaEstimatorMixin
from sklearn.preprocessing import normalize
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from imodels.rule_set.rule_set import RuleSet
from imodels.util.convert import tree_to_code, tree_to_rules
from imodels.util.rule import Rule, get_feature_dict, replace_feature_name


class BoostedRulesClassifier(RuleSet, BaseEstimator, MetaEstimatorMixin, ClassifierMixin):
    '''An easy-interpretable classifier optimizing simple logical rules.
    Currently limited to only binary classification.
    '''

    def __init__(self, n_estimators=10, estimator=partial(DecisionTreeClassifier, max_depth=1)):
        self.n_estimators = n_estimators
        self.estimator = estimator

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
        self.feature_dict_ = get_feature_dict(X.shape[1], feature_names)
        self.feature_placeholders = list(self.feature_dict_.keys())
        self.feature_names = list(self.feature_dict_.values())

        n_train = y.shape[0]
        w = np.ones(n_train) / n_train
        self.estimators_ = []
        self.estimator_weights_ = []
        self.estimator_errors_ = []
        self.feature_names = feature_names
        for _ in range(self.n_estimators):
            # Fit a classifier with the specific weights
            clf = self.estimator()
            clf.fit(X, y, sample_weight=w)  # uses w as the sampling weight!
            preds = clf.predict(X)

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
            est_rules = tree_to_rules(est, self.feature_placeholders)
            
            rule_scores = np.max(est.predict_proba(X), axis=0)
            rule_scores = (rule_scores * 2) - 1

            # order scores to correspond to correct rule
            rule_pred_class = np.argmax(est.tree_.value[1:], axis=2).flatten()
            rule_scores = rule_scores[rule_pred_class]

            # weight negatively rules that predict class 0
            rule_scores[rule_pred_class[0]] *= -1

            compos_score = est_weight * rule_scores
            rules += [Rule(r, args=[w]) for (r, w) in zip(est_rules, compos_score)]

        self.rules_without_feature_names_ = rules
        self.rules_ = [
            replace_feature_name(rule, self.feature_dict_) for rule in self.rules_without_feature_names_
        ]
        self.complexity_ = len(self.estimators_)
        return self

    def predict_proba(self, X):
        '''Predict probabilities for X
        '''
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
        return self.eval_weighted_rule_sum(X) > 0

    # def predict(self, X):
    #     """Predict outcome for X
    #     """
    #     check_is_fitted(self)
    #     X = check_array(X)
    #     return self.predict_proba(X).argmax(axis=1)

    def __str__(self):
        try:
            s = 'Mined rules:\n'
            for est in self.estimators_:
                s += '\t' + tree_to_code(est, self.feature_names)
            return s
        except:
            return f'BoostedRules with {len(self.estimators_)} estimators'
