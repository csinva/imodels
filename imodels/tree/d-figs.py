from copy import deepcopy
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.datasets
from sklearn import datasets
from sklearn import tree
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import plot_tree, DecisionTreeClassifier
from sklearn.utils import check_X_y, check_array
from sklearn.utils.validation import _check_sample_weight
import figs
from imodels.tree.viz_utils import extract_sklearn_tree_from_figs
from imodels import FIGSClassifier
from imodels import FIGSRegressor


class D_FIGS(FIGSRegressor):
    # Needs to store the old X and y
    old_phase = None

    y = None

    potential_splits = []

    phases = None

    def __init__(self, max_rules: int = 12, min_impurity_decrease: float = 0.0, random_state=None,
                 max_features: str = None, phases: dict = None):
        super().__init__(max_rules, min_impurity_decrease, random_state, max_features)
        self.phases = phases
    '''
    check that later features can be available (not NA) only if all phase 1 features are available
    '''
    def check_phase(self, old_phases, new_phase):
        for i in range(len(old_phases)):
            if np.isnan(old_phases).any() and not np.isnan(new_phase).all():
                raise ValueError('A very specific bad thing happened.')

    '''
    add the new phase features to X, delete samples that has NaN in new_phase potentially refit the model?
    '''

    def add_new_phase(self, new_phase):
        self.check_phase(self.old_phase, new_phase)
        concatenated_phase = np.concatenate((self.old_phase, new_phase), axis=0)
        old_phase = concatenated_phase

        # after getting the copied model and potential splits, change the idx
        for node in self.potential_splits:
            new_idx = []
            for i in range(len(node.idx)):
                new_feature = new_phase[node.idx[i]]  ## new phase features for the particular sample i
                if not np.isnan(new_feature).any():  ## If the new phase has no nan
                    new_idx.append(node.idx[i])
            node.idx = new_idx  ## The leaves that we can potentially split on now contain only samples with new_phase feture

    # the fit function should take in a model = None parameter, in phase 2-n, the model fit the samples based on the model
