from typing import Tuple, List, Dict

import numpy as np
import pandas as pd
from sklearn.utils.validation import check_array, check_is_fitted


class RuleSet:

    def _get_tree_ensemble(self):
        pass

    def _fit_tree_ensemble(self, X, y):
        pass

    def _extract_rules(self):
        pass

    def _score_rules(self, X, y, rules):
        pass

    def _prune_rules(self, rules):
        pass

    def eval_weighted_rule_sum(self, X) -> np.ndarray:

        check_is_fitted(self, ['rules_without_feature_names_', 'n_features_', 'feature_names_'])
        X = check_array(X)

        if X.shape[1] != self.n_features_:
            raise ValueError("X.shape[1] = %d should be equal to %d, the number of features at training time."
                             " Please reshape your data."
                             % (X.shape[1], self.n_features_))

        df = pd.DataFrame(X, columns=self.feature_names_)
        selected_rules = self.rules_without_feature_names_

        scores = np.zeros(X.shape[0])
        for (r, w) in selected_rules:
            scores[list(df.query(r).index)] += w[0]

        return scores
