import numpy as np
import pandas as pd
from sklearn.utils.validation import check_array, check_is_fitted
from typing import Tuple, List, Dict


class RuleSet:

    def _extract_rules(self, X, y):
        pass

    def _score_rules(self, X, y, rules):
        pass

    def _prune_rules(self, rules):
        pass

    def eval_weighted_rule_sum(self, X) -> np.ndarray:

        check_is_fitted(self, ['rules_without_feature_names_', 'n_features_', 'feature_placeholders'])
        X = check_array(X)

        if X.shape[1] != self.n_features_:
            raise ValueError("X.shape[1] = %d should be equal to %d, the number of features at training time."
                             " Please reshape your data."
                             % (X.shape[1], self.n_features_))

        df = pd.DataFrame(X, columns=self.feature_placeholders)
        selected_rules = self.rules_without_feature_names_

        scores = np.zeros(X.shape[0])
        for (r, w) in selected_rules:
            scores[list(df.query(r).index)] += w[0]

        return scores

    def _get_complexity(self):
        check_is_fitted(self, ['rules_without_feature_names_'])
        num_rules = len(self.rules_without_feature_names_)
        extra_antecedents = np.sum([(len(rule.agg_dict) - 1) for rule in self.rules_without_feature_names_])
        return num_rules + extra_antecedents * 0.5
