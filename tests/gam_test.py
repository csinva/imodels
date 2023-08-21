import os
import random
from functools import partial

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics
import imodels
from imodels import TreeGAMClassifier
from sklearn.model_selection import train_test_split


# def test_gam_hyperparams():
#     X, y, feat_names = imodels.get_clean_dataset("heart")
#     X, _, y, _ = train_test_split(X, y, test_size=0.9, random_state=13)

#     roc = 0.5
#     for n_boosting_rounds in [1, 2, 3]:
#         m = TreeGAMClassifier(
#             n_boosting_rounds=n_boosting_rounds,
#             max_leaf_nodes=2,
#             random_state=42,
#             n_boosting_rounds_marginal=0,
#         )
#         m.fit(X, y, learning_rate=0.1)
#         roc_new = metrics.roc_auc_score(y, m.predict_proba(X)[:, 1])
#         assert roc_new >= roc
#         roc = roc_new

#     roc = 0.5
#     for n_boosting_rounds_marginal in [1, 2, 3]:
#         m = TreeGAMClassifier(
#             n_boosting_rounds=0,
#             random_state=42,
#             n_boosting_rounds_marginal=n_boosting_rounds_marginal,
#         )
#         m.fit(X, y, learning_rate=0.1)
#         roc_new = metrics.roc_auc_score(y, m.predict_proba(X)[:, 1])
#         assert roc_new >= roc
#         roc = roc_new
