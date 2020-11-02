from typing import List, Tuple
from warnings import warn

import pandas as pd
import numpy as np
from sklearn.utils import indices_to_mask
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LogisticRegressionCV

from imodels.util.rule import Rule


def score_oob(X,
              y,
              rules: List[List[str]],
              samples: List[List[int]],
              features: List[List[int]],
              feature_names: List[str]) -> List[Rule]:

    scored_rules = []

    for curr_rules, curr_samples, curr_features in zip(rules, samples, features):

        # Create mask for OOB samples
        mask = ~indices_to_mask(curr_samples, X.shape[0])
        if sum(mask) == 0:
            warn(
                "OOB evaluation not possible: doing it in-bag. Performance evaluation is likely to be wrong"
                " (overfitting) and selected rules are likely to not perform well! Please use max_samples < 1."
            )
            mask = curr_samples

        # XXX todo: idem without dataframe

        X_oob = pd.DataFrame(
            (X[mask, :])[:, curr_features],
            columns=np.array(feature_names)[curr_features]
        )

        if X_oob.shape[1] <= 1:  # otherwise pandas bug (cf. issue #16363)
            return []

        y_oob = y[mask]
        y_oob = np.array((y_oob != 0))

        # Add OOB performances to rules:
        scored_rules += [
            Rule(r, args=_eval_rule_perf(r, X_oob, y_oob)) for r in set(curr_rules)
        ]

    return scored_rules


def _eval_rule_perf(rule, X, y) -> Tuple[float, float]:
    detected_index = list(X.query(rule).index)
    if len(detected_index) <= 1:
        return (0, 0)
    y_detected = y[detected_index]
    true_pos = y_detected[y_detected > 0].sum()
    if true_pos == 0:
        return (0, 0)
    pos = y[y > 0].sum()
    return y_detected.mean(), float(true_pos) / pos


def score_lasso(X, y, rules: List[str], Cs, cv, random_state) -> Tuple[List[Rule], LassoCV]:
    if Cs is None:
        n_alphas = 100
        alphas = None
    elif hasattr(Cs, "__len__"):
        n_alphas = None
        alphas = 1. / Cs
    else:
        n_alphas = Cs
        alphas = None
    lscv = LassoCV(n_alphas=n_alphas, alphas=alphas, cv=cv, random_state=random_state)
    lscv.fit(X, y)

    rules = [Rule(r, args=[w]) for r, w in zip(rules, lscv.coef_[-len(rules):])]
    return rules, lscv

