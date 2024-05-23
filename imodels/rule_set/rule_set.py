import numpy as np
import pandas as pd
from sklearn.utils.validation import check_array, check_is_fitted


class RuleSet:

    def _extract_rules(self, X, y):
        pass

    def _score_rules(self, X, y, rules):
        pass

    def _prune_rules(self, rules):
        pass

    def _eval_weighted_rule_sum(self, X) -> np.ndarray:

        check_is_fitted(self, ['rules_without_feature_names_', 'n_features_', 'feature_placeholders'])
        X = check_array(X)

        if X.shape[1] != self.n_features_:
            raise ValueError("X.shape[1] = %d should be equal to %d, the number of features at training time."
                             " Please reshape your data."
                             % (X.shape[1], self.n_features_))

        df = pd.DataFrame(X, columns=self.feature_placeholders)
        selected_rules = self.rules_without_feature_names_

        scores = np.zeros(X.shape[0])
        for r in selected_rules: 
            features_r_uses = list(set(map(lambda x: x[0], r.agg_dict.keys())))
            scores[df[features_r_uses].query(str(r)).index.values] += r.args[0]

        return scores

    def _get_complexity(self):
        check_is_fitted(self, ['rules_without_feature_names_'])
        return sum([len(rule.agg_dict) for rule in self.rules_without_feature_names_]) 
