from typing import List, Tuple
from warnings import warn

import pandas as pd
import numpy as np
from sklearn.utils import indices_to_mask
from sklearn.linear_model import Lasso
from sklearn.linear_model._coordinate_descent import _alpha_grid
from sklearn.model_selection import KFold

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


def score_lasso(X, y, rules: List[str], alphas=None, cv=3, max_rules=2000, random_state=None) -> Tuple[List[Rule], Lasso]:
    if alphas is None:
        alphas = _alpha_grid(X, y) 

    coef_zero_threshold = 1e-6 / np.mean(np.abs(y))
    mse_cv_scores = []
    nonzero_rule_coefs_count = []
    kf = KFold(cv)
    for alpha in alphas: # alphas are sorted from largest to smallest
        m = Lasso(alpha=alpha, random_state=random_state)
        mse_cv = 0
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            m.fit(X_train, y_train)
            mse_cv += np.mean((m.predict(X_test) - y_test) ** 2)
        
        m.fit(X, y)
        
        rule_count = sum(np.abs(m.coef_) > coef_zero_threshold)
        if rule_count > max_rules:
            break
        nonzero_rule_coefs_count.append(rule_count)
        mse_cv_scores.append(mse_cv / cv)
    
    best_alpha = alphas[np.argmin(mse_cv_scores)]
    lscv = Lasso(alpha=best_alpha, random_state=random_state, max_iter=2000)
    lscv.fit(X, y)

    coefs = list(lscv.coef_[:-len(rules)])
    support = np.sum(X[:, -len(rules):], axis=0) / X.shape[0]

    nonzero_rules = []
    for r, w, s in zip(rules, lscv.coef_[-len(rules):], support):
        if abs(w) > coef_zero_threshold:
            nonzero_rules.append(Rule(r, args=[w], support=s))
            coefs.append(w)
    
    return nonzero_rules, coefs, lscv.intercept_
