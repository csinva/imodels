import random

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split

from imodels.util.arguments import check_fit_arguments


class SlipperBaseEstimator(BaseEstimator, ClassifierMixin):
    """ An estimator that supports building rules as described in
    A Simple, Fast, and Effective Rule Learner (1999). Intended to be used
    as part of the SlipperRulesClassifier.
    """

    def __init__(self):
        self.Z = None
        self.rule = None
        self.D = None

    def _make_candidate(self, X, y, curr_rule, feat, A_c):
        """ Make candidate rules for grow routine to compare scores"""

        # make candidate rules
        candidates = [curr_rule.copy() for _ in range(3)]
        candidates = [
            x + [{
                'feature': int(feat),
                'operator': operator,
                'pivot': float(A_c)}]
            for x, operator in zip(candidates, ['>', '<', '=='])
        ]

        # pick best condition
        Zs = [self._grow_rule_obj(X, y, r) for r in candidates]
        return candidates[Zs.index(max(Zs))]

    def _condition_classify(self, X, condition):
        """
        Helper function to make classifications for a condition
        in a rule
        """

        logic = 'X[:, {}] {} {}'.format(
            condition['feature'],
            condition['operator'],
            condition['pivot']
        )

        output = np.where(eval(logic))
        return output[0]

    def _rule_predict(self, X, rule):
        """ return all indices for which the passed rule holds on X """

        preds = np.zeros(X.shape[0])
        positive_cases = set(range(X.shape[0]))

        for condition in rule:
            outputs = set(list(self._condition_classify(X, condition)))
            positive_cases = positive_cases.intersection(outputs)

        preds[list(positive_cases)] = 1

        return preds

    def _get_design_matrices(self, X, y, rule):
        """ produce design matrices used in most equations"""
        preds = self._rule_predict(X, rule)

        W_plus_idx = np.where((preds == 1) & (y == 1))
        W_minus_idx = np.where((preds == 1) & (y == 0))

        return np.sum(self.D[W_plus_idx]), np.sum(self.D[W_minus_idx])

    def _grow_rule_obj(self, X, y, rule):
        """ equation to maximize in growing rule
        equation 6 from Cohen & Singer (1999)
        """
        W_plus, W_minus = self._get_design_matrices(X, y, rule)
        # C_R = self._sample_weight(W_plus, W_minus)
        return np.sqrt(W_plus) - np.sqrt(W_minus)

    def _sample_weight(self, plus, minus):
        """ Calculate learner sample weight
        in paper this is C_R, which is confidence of learner
        """
        return 0.5 * np.log((plus + (1 / (2 * len(self.D)))) /
                            (minus + 1 / (2 * len(self.D))))

    def _grow_rule(self, X, y):
        """ Starts with empty conjunction of conditions and
        greedily adds rules to maximize Z_tilde
        """

        stop_condition = False
        features = list(range(X.shape[1]))

        # rule is stored as a list of dictionaries, each dictionary is a condition
        curr_rule = []

        while not stop_condition:
            candidate_rule = curr_rule.copy()
            for feat in features:
                try:
                    pivots = np.percentile(X[:, feat], range(0, 100, 4),
                                       method='linear')
                except:
                    pivots = np.percentile(X[:, feat], range(0, 100, 4), # deprecated
                                           interpolation='midpoint')
                # get a list of possible rules
                feature_candidates = [
                    self._make_candidate(X, y, curr_rule, feat, A_c)
                    for A_c in pivots
                ]

                # get max Z_tilde and update candidate accordingly
                tildes = [self._grow_rule_obj(X, y, r) for r in feature_candidates]
                if max(tildes) > self._grow_rule_obj(X, y, candidate_rule):
                    candidate_rule = feature_candidates[
                        tildes.index(max(tildes))
                    ]

            preds = self._rule_predict(X, candidate_rule)
            negative_coverage = np.where((preds == y) & (y == 0))

            if self._grow_rule_obj(X, y, curr_rule) >= self._grow_rule_obj(X, y, candidate_rule) or \
                    len(negative_coverage) == 0:
                stop_condition = True
            else:
                curr_rule = candidate_rule.copy()

        return curr_rule

    def _prune_rule(self, X, y, rule):
        """ Remove conditions from greedily built rule until
        objective does not improve
        """
        stop_condition = False
        curr_rule = rule.copy()

        while not stop_condition:
            candidate_rules = []

            if len(curr_rule) == 1:
                return curr_rule

            candidate_rules = [
                self._pop_condition(curr_rule, condition)
                for condition in curr_rule
            ]

            prune_objs = [self._prune_rule_obj(X, y, rule) for x in candidate_rules]
            best_prune = candidate_rules[
                prune_objs.index(min(prune_objs))
            ]

            if self._prune_rule_obj(X, y, rule) > self._prune_rule_obj(X, y, rule):
                curr_rule = best_prune.copy()
            else:
                stop_condition = True

        return curr_rule

    def _pop_condition(self, rule, condition):
        """
        Remove a condition from an existing Rule object
        """
        temp = rule.copy()
        temp.remove(condition)
        return temp

    def _make_default_rule(self, X, y):
        """
        Make the default rule that is true for every observation
        of data set. Without default rule a SlipperBaseEstimator would never
        predict negative
        """
        default_rule = []
        features = random.choices(
            range(X.shape[1]),
            k=random.randint(2, 8)
        )

        default_rule.append({
            'feature': str(features[0]),
            'operator': '>',
            'pivot': str(min(X[:, features[0]]))
        })

        for i, x in enumerate(features):
            if i % 2:
                default_rule.append({
                    'feature': x,
                    'operator': '<',
                    'pivot': str(max(X[:, x]))
                })
            else:
                default_rule.append({
                    'feature': x,
                    'operator': '>',
                    'pivot': str(min(X[:, x]))
                })

        return default_rule

    def _prune_rule_obj(self, X, y, rule):
        """
        objective function for prune rule routine
        eq 7 from Cohen & Singer (1999)
        """

        V_plus, V_minus = self._get_design_matrices(X, y, rule)
        C_R = self._sample_weight(V_plus, V_minus)
        return (1 - V_plus - V_minus) + V_plus * np.exp(-C_R) \
               + V_minus * np.exp(C_R)

    def _eq_5(self, X, y, rule):
        """
        equation 5 from Cohen & Singer (1999)
        used to compare the learned rule with a default rule
        """
        W_plus, W_minus = self._get_design_matrices(X, y, rule)
        return 1 - np.square(np.sqrt(W_plus) - np.sqrt(W_minus))

    def _set_rule_or_default(self, X, y, learned_rule):
        """
        Compare output of eq 5 between learned rule and default rule
        return rule that minimizes eq 5
        """

        rules = [self._make_default_rule(X, y), learned_rule]
        scores = [self._eq_5(X, y, rule) for rule in rules]
        self.rule = rules[scores.index(min(scores))]

    def _make_feature_dict(self, num_features, features):
        """
        Map features to place holder names
        """
        if features is None:
            new_feats = ['X_' + str(i) for i in range(num_features)]
        else:
            new_feats = features

        self.feature_dict = {
            old_feat: new_feat for old_feat, new_feat in enumerate(new_feats)
        }

    def predict_proba(self, X):

        proba = self.predict(X)
        proba = proba.reshape(-1, 1)
        proba = np.hstack([
            np.zeros(proba.shape), proba
        ])

        return proba

    def predict(self, X):
        """
        external predict function that returns predictions
        using estimators rule
        """
        return self._rule_predict(X, self.rule)

    def fit(self, X, y, sample_weight=None, feature_names=None):
        """
        Main loop for training
        """
        X, y, feature_names = check_fit_arguments(self, X, y, feature_names) 
        if sample_weight is not None:
            self.D = sample_weight

        X_grow, X_prune, y_grow, y_prune = \
            train_test_split(X, y, test_size=0.33)

        self._make_feature_dict(X.shape[1], feature_names)

        rule = self._grow_rule(X_grow, y_grow)
        rule = self._prune_rule(X_prune, y_prune, rule)
        self._set_rule_or_default(X, y, rule)

        return self
