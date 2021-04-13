import pandas as pd
import numpy as np
import random
import string

from copy import deepcopy

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split

from imodels.util.rule import Rule


class SlipperRule(BaseEstimator, ClassifierMixin):
    """ An estimator that supports building rules as described in
    A Simple, Fast, and Effective Rule Learner (1999). Intended to be used
    as part of the BoostedRulesClassifier.
    """

    def __init__(self, D):
        self.Z = None
        self.rule = Rule('')
        self.D = D
        self._place_holders = None

    def _make_candidate(self, X, y, curr_rule, feat, A_c):
        """ Make candidate rules to explore best option
        greedily

        Parameters
        ----------
          X (np.array): Data matrix
          y (np.array): Response matrix
          D (np.array): Array of distributions
        """

        # make candidate rules
        candidates = []
        for operator in ['>', '<', '==']:
            if curr_rule.rule == '':
                temp_rule = Rule(
                    str(feat) + ' ' +
                    operator + ' ' + str(A_c)
                )
            else:
                temp_rule = Rule(
                    curr_rule.rule + 'and ' + str(feat) +
                    ' ' + operator + ' ' + str(A_c)
                )
            candidates.append(temp_rule)

        # pick best condition
        Zs = [self._grow_rule_obj(X, y, r) for r in candidates]
        return candidates[Zs.index(max(Zs))]

    def _rule_predict(self, X, rule): 
        preds = np.zeros((X.shape[0],))
        df = pd.DataFrame(X)
        idx = df.query(rule.rule).index[0]
        preds[idx] = 1
        return preds

    def _get_design_matrices(self, X, y, rule):
        preds = self._rule_predict(X, rule)

        W_plus_idx = np.where((preds == 1) & (y == 1))
        W_minus_idx = np.where((preds == 1) & (y == 0))

        return np.sum(self.D[W_plus_idx]), np.sum(self.D[W_minus_idx])

    def _grow_rule_obj(self, X, y, rule):
        W_plus, W_minus = self._get_design_matrices(X, y, rule) 
        C_R = self.sample_weight(W_plus, W_minus)
        return np.sqrt(W_plus) - np.sqrt(W_minus)

    def sample_weight(self, plus, minus):
        """ Calculate learner sample weight
        in paper this is C_R, which is confidence of learner 
        """
        return 0.5 * np.log((plus + (1 / (2 * len(self.D)))) /
                            (minus + 1 / (2 * len(self.D))))

    def _grow_rule(self, X, y):
        """ Starts with empty conjunction of conditions and
        greddily adds rules to mazimize Z_tilda

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape (n_samples,)
            Target vector relative to X. Has to follow the convention 0 for
            normal data, 1 for anomalies.

        tol: tolerance for when to end adding conditions to rule

        con_tol: condition tolerance for when to stop
                 tuning condition for feature
        """

        stop_condition = False
        features = list(range(X.shape[1]))
        curr_rule = Rule('')

        while not stop_condition:
            candidate_rule = deepcopy(curr_rule)
            for feat in features:
                pivots = np.percentile(X[:, feat], range(0, 100, 10),
                                       interpolation='midpoint')

                feature_candidates = [
                    self._make_candidate(X, y, curr_rule, feat, A_c)
                    for A_c in pivots
                ]

                # get max Z_tilda and update candidate accordingly
                tildas = [self._grow_rule_obj(X, y, r) for r in feature_candidates]
                if max(tildas) > self._grow_rule_obj(X, y, candidate_rule):
                    candidate_rule = feature_candidates[
                        tildas.index(max(tildas))
                    ]

            preds = candidate_rule.predict(X)
            negative_coverage = np.where((preds == y) & (y == 0))

            if self._grow_rule_obj(X, y, rule) >= self._grow_rule_obj(candidate_rule) or \
                    len(negative_coverage) == 0:
                stop_condition = True
            else:
                curr_rule = deepcopy(candidate_rule)

        return curr_rule

    def _prune_rule(self, X, y):
        stop_condition = False
        curr_rule = deepcopy(rule)
        curr_rule.prune_r

        while not stop_condition:
            candidate_rules = []

            if len(curr_rule.conditions) == 1:
                return curr_rule

            for condition in curr_rule.conditions:
                R_prime = deepcopy(curr_rule)
                R_prime.prune_rule(X, y, condition)
                candidate_rules.append(R_prime)

            prune_objs = [self._prune_rule_obj(pobj) for x in candidate_rules]
            best_prune = candidate_rules[
                prune_objs.index(min(prune_objs))
            ]

            if self._prune_rule_obj(curr_rule) > self._prune_rule_obj(best_prune):
                curr_rule = deepcopy(best_prune)
            else:
                stop_condition = True

        return curr_rule

    def _make_default_rule(self, X, y):
        default_rule = Rule('')
        features = random.choices(
            range(X.shape[1]),
            k=random.randint(2, 8)
        )

        default_rule.rule = str(features[0]) + ' > ' + min(X[:, features[0]]) 

        for i, x in enumerate(features):
            if i % 2:
                default_rule.rule += ' and ' + str(feature) + ' < ' + max(X[:, x])
            else:
                default_rule.rule += ' and ' + str(feature) + ' > ' + min(X[:, x])
        
        return default_rule

    def _prune_rule_obj(self, X, y, rule):
        V_plus, V_minus = self._get_design_matrices(X, y, rule)
        C_R = self.sample_weight(V_plus, V_minus)
        return (1 - V_plus - V_minus) + V_plus * np.exp(-C_R) \
            + V_minus * np.exp(C_R)

    def _eq_5(self, X, y, rule):
        W_plus, W_minus = self._get_design_matrices(X, y, rule)
        return 1 - np.square(np.sqrt(W_plus) - np.sqrt(W_minus))

    def _set_rule_or_default(self, X, y, learned_rule):
        rules = [self._make_default_rule(X, y), learned_rule]
        scores = [self._eq_5(self, X, y, rule) for rule in rules]
        self.rule = rules[rules.index(min(scores))]

    def fit(self, X, y):
        """
        Main loop for training
        """
        X_grow, X_prune, y_grow, y_prune = \
            train_test_split(X, y, test_size=0.33)
        
        rule = self._grow_rule(X_grow, y_grow)
        # rule = self._prune_rule(X_prune, y_prune, rule)

        self._set_rule_or_default(X, y, rule)
