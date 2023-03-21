import copy
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
from figs import Node
from imodels.tree.figs import FIGSClassifier
from imodels.tree.figs import FIGSRegressor


class D_FIGS(FIGSRegressor):
     # Needs to store the old X and y

    # feature_phases = {1 : (X, y, model), 2 : (X_phase2, y, model)}
    feature_phases = None

    def __init__(self, max_rules: int = 12, min_impurity_decrease: float = 0.0, random_state=None,
                 max_features: str = None, feature_phases=None):
        super().__init__(max_rules, min_impurity_decrease, random_state, max_features)
        self.feature_phases = feature_phases

    def check_phase(self, old_phases, new_phase):
        for i in range(len(old_phases)):
            '''
            phase 2 features can be available (not NA) only if all phase 1 features are available
            '''
            if np.isnan(old_phases).any() and not np.isnan(new_phase).all():
                raise ValueError('A very specific bad thing happened.')

    '''
    add the new phase features to X, delete samples that has NaN in new_phase potentially refit the model?
    '''
    '''
    def add_new_phase(self, new_phase):
        self.check_phase(self.old_phase, new_phase)
        concatenated_phase = np.concatenate((self.old_phase, new_phase), axis=0)
        old_phase = concatenated_phase

        # after getting the copied model and potential splits, change the idx
        for node in self.potential_splits:
            new_idx = []
            for i in range(len(node.idx)):
                new_feature = new_phase[node.idx[i]]  # new phase features for the particular sample i
                if not np.isnan(new_feature).any():  # If the new phase has no nan
                    new_idx.append(node.idx[i])
            node.idx = new_idx  # The leaves that we can potentially split on now contain only samples with new_phase'''

    def fit_phase_1(self, X, y, feature_names=None, verbose=False, sample_weight=None):
        self.fit(X, y)
        # Store a deep copy of the whole model for easier prediction use in the future
        self.feature_phases = {}
        self.feature_phases[1] = (X, y, deepcopy(self))
        return self

    def fit_phase_n(self, X, y, max_rules=15, feature_names=None, verbose=False, sample_weight=None):
        if isinstance(self, ClassifierMixin):
            self.classes_, y = np.unique(y, return_inverse=True)  # deals with str inputs

        if feature_names is None:
            if isinstance(X, pd.DataFrame):
                self.feature_names_ = X.columns
        else:
            self.feature_names_ = feature_names
        X, y = check_X_y(X, y, force_all_finite=False)
        y = y.astype(float)
        phase_idx = len(self.feature_phases)  # infer the number of phase from the dict
        prev_phase = self.feature_phases[phase_idx][0]
        # print(prev_phase)
        new_phase = X[:, len(prev_phase.iloc[0]):]
        all_leaves = []
        for node in self.trees_:
            all_leaves += self.get_leaves(node)
        # print(self.feature_phases[phase_idx])
        # print(new_phase)
        # Right now, we are only removing samples that do not have the newest phase in the leaves, not their ancestors
        for node in all_leaves:
            for i in range(len(node.idxs)):
                if node.idxs[i]:
                    new_feature = new_phase[i]
                    # new phase features for the particular sample i
                    # If the new phase has nan, which means it is not valid and should be false in the idxs
                    if np.isnan(new_feature).any():
                        node.idxs[i] = False

        self.extend_trees(X, y, all_leaves, max_rules=max_rules)
        self.feature_phases[phase_idx + 1] = (X, y, deepcopy(self))
        return self

    '''
    This will infer the newly added features from the last stored X
    This function only delete samples for a newly generated root!!!
    '''

    def remove_na_samples(self, X):
        phase_idx = len(self.feature_phases)  # infer the number of phase from the dict
        prev_phase = self.feature_phases[phase_idx][0]
        new_phase = X[:, len(prev_phase.iloc[0]):]
        cur_idxs = np.ones(X.shape[0], dtype=bool)
        for i in range(len(cur_idxs)):
            if cur_idxs[i]:
                new_feature = new_phase[i]
                # new phase features for the particular sample i
                # If the new phase has nan, which means it is not valid and should be false in the idxs
                if np.isnan(new_feature).any():
                    cur_idxs[i] = False
        return cur_idxs

    def update_potential_splits(self, X, y, potential_splits, y_predictions_per_tree, y_residuals_per_tree):

        for tree_num_ in range(len(self.trees_)):
            y_predictions_per_tree[tree_num_] = self._predict_tree(self.trees_[tree_num_], X)
        y_predictions_per_tree[-1] = np.zeros(X.shape[0])  # dummy 0 preds for possible new trees

        # update residuals for each tree
        # -1 is key for potential new tree
        for tree_num_ in list(range(len(self.trees_))) + [-1]:
            y_residuals_per_tree[tree_num_] = deepcopy(y)

            # subtract predictions of all other trees
            for tree_num_other_ in range(len(self.trees_)):
                if not tree_num_other_ == tree_num_:
                    y_residuals_per_tree[tree_num_] -= y_predictions_per_tree[tree_num_other_]

        # recompute all impurities + update potential_split children
        potential_splits_new = []
        for potential_split in potential_splits:
            y_target = y_residuals_per_tree[potential_split.tree_num]

            # re-calculate the best split
            potential_split_updated = self._construct_node_with_stump(X=X,
                                                                      y=y_target,
                                                                      idxs=potential_split.idxs,
                                                                      tree_num=potential_split.tree_num,
                                                                      max_features=self.max_features)

            # need to preserve certain attributes from before (value at this split + is_root)
            # value may change because residuals may have changed, but we want it to store the value from before
            potential_split.setattrs(
                feature=potential_split_updated.feature,
                threshold=potential_split_updated.threshold,
                impurity_reduction=potential_split_updated.impurity_reduction,
                left_temp=potential_split_updated.left_temp,
                right_temp=potential_split_updated.right_temp,
            )

            # this is a valid split
            if potential_split.impurity_reduction is not None:
                potential_splits_new.append(potential_split)
        sorted_potential_splits_new = sorted(potential_splits_new, key=lambda x: x.impurity_reduction)
        return sorted_potential_splits_new, y_predictions_per_tree, y_residuals_per_tree

    def extend_trees(self, X, y, all_leaves, max_rules=15):
        # Need to add max_rules each time so that it's bigger than the complexity
        self.max_rules += max_rules
        potential_splits = []
        y_predictions_per_tree = {}  # predictions for each tree
        y_residuals_per_tree = {}  # based on predictions above
        first_extend = True

        # Get all the leaves from the previous model
        # for node in self.trees_:
        #    all_leaves += self.get_leaves(node)
        # iterate through all the leaves and split them on the new feature
        for leaf in all_leaves:
            # b = np.isnan(X[leaf.idxs]).any()
            if len(X[leaf.idxs]) == 0:
                continue
            potential_split = self._construct_node_with_stump(X, y, idxs=leaf.idxs, tree_num=leaf.tree_num,
                                                              max_features=None)
            if potential_split.impurity_reduction is not None:
                # Update the leaves on the previous model
                leaf.setattrs(feature=potential_split.feature,
                              threshold=potential_split.threshold,
                              impurity_reduction=potential_split.impurity_reduction,
                              left_temp=potential_split.left_temp,
                              right_temp=potential_split.right_temp,
                              tree_num=potential_split.tree_num,
                              impurity=potential_split.impurity,
                              idxs=potential_split.idxs)
                # Add to the potential splits, and do the same fitting process as in the fig
                potential_splits.append(leaf)
        '''
        phase_idx = len(self.feature_phases)  # infer the number of phase from the dict
        prev_phase = self.feature_phases[phase_idx][0]
        new_phase = X[:, len(prev_phase[0]):]
        cur_idxs = np.ones(X.shape[0], dtype=bool)
        for i in range(len(cur_idxs)):
            if cur_idxs[i]:
                new_feature = new_phase[i]
                # new phase features for the particular sample i
                # If the new phase has nan, which means it is not valid and should be false in the idxs
                if np.isnan(new_feature).any():
                    cur_idxs[i] = False
        '''
        cur_idxs = self.remove_na_samples(X)
        node_new_root = Node(is_root=True, idxs=cur_idxs,
                             tree_num=-1)
        potential_splits.append(node_new_root)
        # //TODO DEBUG
        '''
        ## Add new root
        phase_idx = len(self.feature_phases)  # infer the number of phase from the dict
        prev_phase = self.feature_phases[phase_idx][0]
        new_phase = X[:, len(prev_phase[0]):]
        cur_idxs = np.ones(X.shape[0], dtype=bool)
        for i in range(len(cur_idxs)):
            if cur_idxs[i]:
                new_feature = new_phase[i]
                # new phase features for the particular sample i
                # If the new phase has nan, which means it is not valid and should be false in the idxs
                if np.isnan(new_feature).any():
                    cur_idxs[i] = False
        node_new_root = Node(is_root=True, idxs=cur_idxs,
                             tree_num=-1)
        potential_splits.append(node_new_root)


        for tree_num_ in range(len(self.trees_)):
            y_predictions_per_tree[tree_num_] = self._predict_tree(self.trees_[tree_num_], X)
        y_predictions_per_tree[-1] = np.zeros(X.shape[0])  # dummy 0 preds for possible new trees

        # update residuals for each tree
        # -1 is key for potential new tree
        for tree_num_ in list(range(len(self.trees_))) + [-1]:
            y_residuals_per_tree[tree_num_] = deepcopy(y)

            # subtract predictions of all other trees
            for tree_num_other_ in range(len(self.trees_)):
                if not tree_num_other_ == tree_num_:
                    y_residuals_per_tree[tree_num_] -= y_predictions_per_tree[tree_num_other_]

        # recompute all impurities + update potential_split children
        potential_splits_new = []
        for potential_split in potential_splits:
            y_target = y_residuals_per_tree[potential_split.tree_num]

            # re-calculate the best split
            potential_split_updated = self._construct_node_with_stump(X=X,
                                                                      y=y_target,
                                                                      idxs=potential_split.idxs,
                                                                      tree_num=potential_split.tree_num,
                                                                      max_features=self.max_features)

            # need to preserve certain attributes from before (value at this split + is_root)
            # value may change because residuals may have changed, but we want it to store the value from before
            potential_split.setattrs(
                feature=potential_split_updated.feature,
                threshold=potential_split_updated.threshold,
                impurity_reduction=potential_split_updated.impurity_reduction,
                left_temp=potential_split_updated.left_temp,
                right_temp=potential_split_updated.right_temp,
            )

            # this is a valid split
            if potential_split.impurity_reduction is not None:
                potential_splits_new.append(potential_split)
        '''
        # sort so the largest impurity reduction comes last (should probs make this a heap later)
        potential_splits, y_predictions_per_tree, y_residuals_per_tree = self.update_potential_splits(X,
                                                                                                      y,
                                                                                                      potential_splits,
                                                                                                      y_predictions_per_tree,
                                                                                                      y_residuals_per_tree)
        # //TODO DEBUG END
        # for i in potential_splits:
        #    print(i.impurity_reduction)

        # original: line 253
        # potential_splits = sorted(potential_splits, key=lambda x: x.impurity_reduction)

        finished = False
        while len(potential_splits) > 0 and not finished:
            # print('potential_splits', [str(s) for s in potential_splits])
            split_node = potential_splits.pop()  # get node with max impurity_reduction (since it's sorted)

            # don't split on node
            if split_node.impurity_reduction < self.min_impurity_decrease:
                finished = True
                break

            # split on node
            self.complexity_ += 1

            # if added a tree root
            if split_node.is_root:
                # start a new tree
                self.trees_.append(split_node)

                # update tree_num
                for node_ in [split_node, split_node.left_temp, split_node.right_temp]:
                    if node_ is not None:
                        node_.tree_num = len(self.trees_) - 1

                # add new root potential node
                '''
                phase_idx = len(self.feature_phases)  # infer the number of phase from the dict
                prev_phase = self.feature_phases[phase_idx][0]
                new_phase = X[:, len(prev_phase[0]):]
                cur_idxs = np.ones(X.shape[0], dtype=bool)
                for i in range(len(cur_idxs)):
                    if cur_idxs[i]:
                        new_feature = new_phase[i]
                        # new phase features for the particular sample i
                        # If the new phase has nan, which means it is not valid and should be false in the idxs
                        if np.isnan(new_feature).any():
                            cur_idxs[i] = False
                '''
                cur_idxs = self.remove_na_samples(X)
                node_new_root = Node(is_root=True, idxs=cur_idxs,
                                     tree_num=-1)
                potential_splits.append(node_new_root)

            # add children to potential splits
            # assign left_temp, right_temp to be proper children
            # (basically adds them to tree in predict method)
            split_node.setattrs(left=split_node.left_temp, right=split_node.right_temp)

            # add children to potential_splits
            potential_splits.append(split_node.left)
            potential_splits.append(split_node.right)

            '''
            Debug, replace with function
            # update predictions for altered tree
            for tree_num_ in range(len(self.trees_)):
                y_predictions_per_tree[tree_num_] = self._predict_tree(self.trees_[tree_num_], X)
            y_predictions_per_tree[-1] = np.zeros(X.shape[0])  # dummy 0 preds for possible new trees

            # update residuals for each tree
            # -1 is key for potential new tree
            for tree_num_ in list(range(len(self.trees_))) + [-1]:
                y_residuals_per_tree[tree_num_] = deepcopy(y)

                # subtract predictions of all other trees
                for tree_num_other_ in range(len(self.trees_)):
                    if not tree_num_other_ == tree_num_:
                        y_residuals_per_tree[tree_num_] -= y_predictions_per_tree[tree_num_other_]

            # recompute all impurities + update potential_split children
            potential_splits_new = []
            for potential_split in potential_splits:
                y_target = y_residuals_per_tree[potential_split.tree_num]

                # re-calculate the best split
                potential_split_updated = self._construct_node_with_stump(X=X,
                                                                          y=y_target,
                                                                          idxs=potential_split.idxs,
                                                                          tree_num=potential_split.tree_num,
                                                                          max_features=self.max_features)

                # need to preserve certain attributes from before (value at this split + is_root)
                # value may change because residuals may have changed, but we want it to store the value from before
                potential_split.setattrs(
                    feature=potential_split_updated.feature,
                    threshold=potential_split_updated.threshold,
                    impurity_reduction=potential_split_updated.impurity_reduction,
                    left_temp=potential_split_updated.left_temp,
                    right_temp=potential_split_updated.right_temp,
                )

                # this is a valid split
                if potential_split.impurity_reduction is not None:
                    potential_splits_new.append(potential_split)

            # sort so largest impurity reduction comes last (should probs make this a heap later)
            '''

            # potential_splits = sorted(potential_splits_new, key=lambda x: x.impurity_reduction)
            potential_splits, y_predictions_per_tree, y_residuals_per_tree = self.update_potential_splits(X,
                                                                                                          y,
                                                                                                          potential_splits,
                                                                                                          y_predictions_per_tree,
                                                                                                          y_residuals_per_tree)
            if self.max_rules is not None and self.complexity_ >= self.max_rules:
                finished = True
                break

            # annotate final tree with node_id and value_sklearn
        for tree_ in self.trees_:
            node_counter = iter(range(0, int(1e06)))

            def _annotate_node(node: Node, X, y):
                if node is None:
                    return

                # TODO does not incorporate sample weights
                value_counts = pd.Series(y).value_counts()
                try:
                    neg_count = value_counts[0.0]
                except KeyError:
                    neg_count = 0

                try:
                    pos_count = value_counts[1.0]
                except KeyError:
                    pos_count = 0

                value_sklearn = np.array([neg_count, pos_count], dtype=float)

                node.setattrs(node_id=next(node_counter), value_sklearn=value_sklearn)

                idxs_left = X[:, node.feature] <= node.threshold
                _annotate_node(node.left, X[idxs_left], y[idxs_left])
                _annotate_node(node.right, X[~idxs_left], y[~idxs_left])

            _annotate_node(tree_, X, y)
            return self

    def get_leaves(self, root):
        s1 = []
        s2 = []
        s1.append(root)
        while len(s1) != 0:
            curr = s1.pop()
            if curr.left:
                s1.append(curr.left)
            if curr.right:
                s1.append(curr.right)
            elif not curr.left and not curr.right:
                s2.append(curr)
        return s2

    def predict_phase_i(self, X, phase):
        model = self.feature_phases[phase][2]
        return model.predict(X)

    def predict_proba_phase_i(self, X, phase):
        model = self.feature_phases[phase][2]
        return model.predict_proba(X)

if __name__ == '__main__':
    '''
    # Data generating function
    # 1[X0 < 0.5 and X1 < 0.5] + 1[X2 < 0.5 and X3 < 0.5]
    X_fig_large = np.random.binomial(1, 0.5, (20000, 4))
    y_fig_large = [0.0] * 20000
    for idx in range(len(X_fig_large)) :
        x1_x2 = X_fig_large[idx][0] < 0.5 and X_fig_large[idx][1] < 0.5
        x3_x4 = X_fig_large[idx][2] < 0.5 and X_fig_large[idx][3] < 0.5
        if x1_x2 and x3_x4:
            y_fig_large[idx] = 2.0
        elif x1_x2 or x3_x4:
            y_fig_large[idx] = 1.0
    X_fig_large_1 = X_fig_large[:, :2]

    clf = D_FIGS(max_rules=4)

    # First time use fit phase_1
    clf.fit_phase_1(X_fig_large_1, np.array(y_fig_large))

    # Use fit phase_n in the later phase
    clf.fit_phase_n(X_fig_large, np.array(y_fig_large), max_rules=6)'''

    # Data generating function
    # 1[X0 < 0.5 and X1 < 0.5] + 1[X2 < 0.5 and X3 < 0.5]
    # This is uniform since NaN is Float

    X_fig_large_na = np.random.uniform(0, 1, (50000, 4))
    y_fig_large_na = [0.0] * 50000
    for idx in range(len(X_fig_large_na)):
        x1_x2 = X_fig_large_na[idx][0] < 0.5 and X_fig_large_na[idx][1] < 0.5
        x3_x4 = X_fig_large_na[idx][2] < 0.5 and X_fig_large_na[idx][3] < 0.5
        prob = np.random.uniform(0, 1)
        # This data generating function will also randomly assign na to the second phase

        if prob <= 0.2:
            X_fig_large_na[idx][2] = np.nan
            X_fig_large_na[idx][3] = np.nan
            x3_x4 = False
        if x1_x2 and x3_x4:
            y_fig_large_na[idx] = 2.0
        elif x1_x2 or x3_x4:
            y_fig_large_na[idx] = 1.0
    X_fig_large_1_na = X_fig_large_na[:, :2]

    if (np.isnan(X_fig_large_na).any()):
        print("The later phase has missing values, (some samples' features are nan)")

    d_fig = D_FIGS(max_rules=2)
    d_fig.fit_phase_1(X_fig_large_1_na, np.array(y_fig_large_na))
    d_fig.fit_phase_n(X_fig_large_na, np.array(y_fig_large_na), max_rules=2)
    print(d_fig.predict(
        [[1, 1, 1, 1], [1, 0, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 1, 1], [0, 0, 1, 0], [1, 0, 1, 0]]))
    print("")
    print("Correct answer is" + "[0, 1, 1, 2, 1, 1, 0]")


    '''
    node = Node()
    node.idxs = [False, True, False]
    node2 = Node()
    node2.idxs = [False, True, False]
    all_leaves = []
    all_leaves += [node]
    all_leaves += [node2]
    for node in all_leaves:
        print(node.idxs)
        for i in range(len(node.idxs)):
            if node.idxs[i]:
                node.idxs[i] = False
    for node in all_leaves:
        print(node.idxs)
    '''
