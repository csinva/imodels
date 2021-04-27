from typing import List, Tuple
from warnings import warn

import pandas as pd
import numpy as np
from sklearn.utils import indices_to_mask
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.linear_model._coordinate_descent import _alpha_grid
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import accuracy_score

from imodels.util.rule import Rule


def score_precision_recall(X,
                           y,
                           rules: List[List[str]],
                           samples: List[List[int]],
                           features: List[List[int]],
                           feature_names: List[str],
                           oob: bool = True) -> List[Rule]:

    scored_rules = []

    for curr_rules, curr_samples, curr_features in zip(rules, samples, features):

        # Create mask for OOB samples
        mask = ~indices_to_mask(curr_samples, X.shape[0])
        if sum(mask) == 0:
            if oob:
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
            Rule(r, args=_eval_rule_perf(r, X_oob, y_oob))
            for r in set(curr_rules)
        ]

    return scored_rules


def _eval_rule_perf(rule: str, X, y) -> Tuple[float, float]:
    detected_index = list(X.query(rule).index)
    if len(detected_index) <= 1:
        return (0, 0)
    y_detected = y[detected_index]
    true_pos = y_detected[y_detected > 0].sum()
    if true_pos == 0:
        return (0, 0)
    pos = y[y > 0].sum()
    return y_detected.mean(), float(true_pos) / pos


def score_linear(X, y, rules: List[str], 
                 penalty='l1',
                 prediction_task='regression',
                 max_rules=30,
                 alpha=None,
                 random_state=None) -> Tuple[List[Rule], List[float], float]:

    if alpha is not None and max_rules is None:
        final_alpha = alpha
    elif max_rules is not None and alpha is None:
        final_alpha = get_best_alpha_under_max_rules(X, y, rules,
                                                     penalty=penalty,
                                                     prediction_task=prediction_task,
                                                     max_rules=max_rules, 
                                                     random_state=random_state)
    else:
        raise ValueError("max_rules and alpha cannot be used together")


    if prediction_task == 'regression':
        lscv = Lasso(alpha=final_alpha, random_state=random_state, max_iter=2000)
    else:
        lscv = LogisticRegression(penalty=penalty, C=1/final_alpha, solver='liblinear',
                                  random_state=random_state, max_iter=200)
    lscv.fit(X, y)

    coef_ = lscv.coef_.flatten()
    coefs = list(coef_[:coef_.shape[0]-len(rules)])
    support = np.sum(X[:, -len(rules):], axis=0) / X.shape[0]

    nonzero_rules = []
    coef_zero_threshold = 1e-6 / np.mean(np.abs(y))
    for r, w, s in zip(rules, coef_[-len(rules):], support):
        if abs(w) > coef_zero_threshold:
            nonzero_rules.append(Rule(r, args=[w], support=s))
            coefs.append(w)
    
    return nonzero_rules, coefs, lscv.intercept_


def get_best_alpha_under_max_rules(X, y, rules: List[str],
                                   penalty='l1',
                                   prediction_task='regression',
                                   max_rules=30,
                                   random_state=None) -> float:
    coef_zero_threshold = 1e-6 / np.mean(np.abs(y))
    alpha_scores = []
    nonzero_rule_coefs_count = []

    if prediction_task == 'regression':
        alphas = _alpha_grid(X, y)
    elif prediction_task == 'classification':
        alphas = [1 / alpha for alpha in np.logspace(-4, 4, num=10, base=10)]

    # alphas are sorted from most reg. to least reg.
    for alpha in alphas:

        if prediction_task == 'regression':
            m = Lasso(alpha=alpha, random_state=random_state)
            fold_scores = cross_val_score(m, X, y, cv=4, scoring='neg_mean_squared_error')
            alpha_scores.append(np.mean(fold_scores))
        else:
            m = LogisticRegression(penalty=penalty, C=1/alpha, solver='liblinear', random_state=random_state)
            fold_scores = cross_val_score(m, X, y, cv=4, scoring='accuracy')
            alpha_scores.append(np.mean(fold_scores))
        
        m.fit(X, y)
        
        rule_coefs = m.coef_.flatten()[:X.shape[1]-len(rules)]
        rule_count = np.sum(np.abs(rule_coefs) > coef_zero_threshold)
        if rule_count > max_rules:
            break
        nonzero_rule_coefs_count.append(rule_count)

    # rare case in which diff alphas lead to identical scores
    if np.all(alpha_scores == alpha_scores[0]):
        best_alpha = alphas[len(alpha_scores) - 1]
    else:
        best_alpha = alphas[np.argmax(alpha_scores)]
    
    return best_alpha
