from copy import deepcopy

import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn import tree
from sklearn.base import BaseEstimator
from sklearn.linear_model import RidgeCV, RidgeClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree
from sklearn.utils import check_X_y

from imodels.tree.viz_utils import DecisionTreeViz

plt.rcParams['figure.dpi'] = 300


class Node:
    def __init__(self, feature: int = None, threshold: int = None,
                 value=None, idxs=None, is_root: bool = False, left=None,
                 impurity_reduction: float = None, tree_num: int = None,
                 right=None, split_or_linear='split', n_samples=0):
        """Node class for splitting
        """

        # split or linear
        self.is_root = is_root
        self.idxs = idxs
        self.tree_num = tree_num
        self.split_or_linear = split_or_linear
        self.feature = feature
        self.n_samples = n_samples
        self.impurity_reduction = impurity_reduction

        # different meanings
        self.value = value  # for split this is mean, for linear this is weight

        # split-specific (for linear these should all be None)
        self.threshold = threshold
        self.left = left
        self.right = right
        self.left_temp = None
        self.right_temp = None

    def update_values(self, X, y):
        self.value = y.mean()
        if self.threshold is not None:
            right_indicator = np.apply_along_axis(lambda x: x[self.feature] > self.threshold, 1, X)
            X_right = X[right_indicator, :]
            X_left = X[~right_indicator, :]
            y_right = y[right_indicator]
            y_left = y[~right_indicator]
            if self.left is not None:
                self.left.update_values(X_left, y_left)
            if self.right is not None:
                self.right.update_values(X_right, y_right)

    def shrink(self, reg_param, cum_sum=0):
        if self.is_root:
            cum_sum = self.value
        if self.left is None:  # if leaf node, change prediction
            self.value = cum_sum
        else:
            shrunk_diff = (self.left.value - self.value) / (1 + reg_param / self.n_samples)
            self.left.shrink(reg_param, cum_sum + shrunk_diff)
            shrunk_diff = (self.right.value - self.value) / (1 + reg_param / self.n_samples)
            self.right.shrink(reg_param, cum_sum + shrunk_diff)

    def setattrs(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __str__(self):
        if self.split_or_linear == 'linear':
            if self.is_root:
                return f'X_{self.feature} * {self.value:0.3f} (Tree #{self.tree_num} linear root)'
            else:
                return f'X_{self.feature} * {self.value:0.3f} (linear)'
        else:
            if self.is_root:
                return f'X_{self.feature} <= {self.threshold:0.3f} (Tree #{self.tree_num} root)'
            elif self.left is None and self.right is None:
                return f'Val: {self.value[0][0]:0.3f} (leaf)'
            else:
                return f'X_{self.feature} <= {self.threshold:0.3f} (split)'

    def __repr__(self):
        return self.__str__()


class FIGSExt(BaseEstimator):
    """FIGSExt (sum of trees) classifier.
    Fast Interpretable Greedy-Tree Sums (FIGS) is an algorithm for fitting concise rule-based models.
    Specifically, FIGS generalizes CART to simultaneously grow a flexible number of trees in a summation.
    The total number of splits across all the trees can be restricted by a pre-specified threshold, keeping the model interpretable.
    Experiments across a wide array of real-world datasets show that FIGS achieves state-of-the-art prediction performance when restricted to just a few splits (e.g. less than 20).
    https://arxiv.org/abs/2201.11931
    """

    def __init__(self, max_rules: int = None, posthoc_ridge: bool = False,
                 include_linear: bool = False,
                 max_features=None, min_impurity_decrease: float = 0.0,
                 k1: int = 0, k2: int = 0):
        """
        max_features
            The number of features to consider when looking for the best split
        k1: number of iterations of tree-prediction backfitting to do after making each split
        k2: number of iterations of tree-prediction backfitting to do after the end of the entire
            tree-growing phase
        """
        super().__init__()
        self.max_rules = max_rules
        self.posthoc_ridge = posthoc_ridge
        self.include_linear = include_linear
        self.max_features = max_features
        self.weighted_model_ = None  # set if using posthoc_ridge
        self.min_impurity_decrease = min_impurity_decrease
        self.k1 = k1
        self.k2 = k2
        self._init_prediction_task()  # decides between regressor and classifier

    def _init_prediction_task(self):
        """
        FIGSExtRegressor and FIGSExtClassifier override this method
        to alter the prediction task. When using this class directly,
        it is equivalent to FIGSExtRegressor
        """
        self.prediction_task = 'regression'

    def _init_decision_function(self):
        """Sets decision function based on prediction_task
        """
        # used by sklearn GrriidSearchCV, BaggingClassifier
        if self.prediction_task == 'classification':
            decision_function = lambda x: self.predict_proba(x)[:, 1]
        elif self.prediction_task == 'regression':
            decision_function = self.predict

    def _construct_node_linear(self, X, y, idxs, tree_num=0, sample_weight=None):
        """This can be made a lot faster
        Assumes there are at least 5 points in node
        Doesn't currently support _sample_weight!
        """
        y_target = y[idxs]
        impurity_orig = np.mean(np.square(y_target)) * idxs.sum()

        # find best linear split
        best_impurity = impurity_orig
        best_linear_coef = None
        best_feature = None
        for feature_num in range(X.shape[1]):
            x = X[idxs, feature_num].reshape(-1, 1)
            m = RidgeCV(fit_intercept=False)
            m.fit(x, y_target)
            impurity = np.min(-m.best_score_) * idxs.sum()
            assert impurity >= 0, 'impurity should not be negative'
            if impurity < best_impurity:
                best_impurity = impurity
                best_linear_coef = m.coef_[0]
                best_feature = feature_num
        impurity_reduction = impurity_orig - best_impurity

        # no good linear fit found
        if impurity_reduction == 0:
            return Node(idxs=idxs, value=np.mean(y_target), tree_num=tree_num,
                        feature=None, threshold=None,
                        impurity_reduction=-1, split_or_linear='split')  # leaf node that just returns its value
        else:
            assert isinstance(best_linear_coef, float), 'coef should be a float'
            return Node(idxs=idxs, value=best_linear_coef, tree_num=tree_num,
                        feature=best_feature, threshold=None,
                        impurity_reduction=impurity_reduction, split_or_linear='linear')

    def _construct_node_with_stump(self, X, y, idxs, tree_num, sample_weight=None, max_features=None):
        # array indices
        SPLIT = 0
        LEFT = 1
        RIGHT = 2

        # fit stump
        stump = tree.DecisionTreeRegressor(max_depth=1, max_features=max_features)
        if sample_weight is not None:
            sample_weight = sample_weight[idxs]
        stump.fit(X[idxs], y[idxs], sample_weight=sample_weight)

        # these are all arrays, arr[0] is split node
        # note: -2 is dummy
        feature = stump.tree_.feature
        threshold = stump.tree_.threshold

        impurity = stump.tree_.impurity
        n_node_samples = stump.tree_.n_node_samples
        value = stump.tree_.value

        # no split
        if len(feature) == 1:
            # print('no split found!', idxs.sum(), impurity, feature)
            return Node(idxs=idxs, value=value[SPLIT], tree_num=tree_num,
                        feature=feature[SPLIT], threshold=threshold[SPLIT],
                        impurity_reduction=-1, n_samples=n_node_samples)

        # split node
        impurity_reduction = (
                                     impurity[SPLIT] -
                                     impurity[LEFT] * n_node_samples[LEFT] / n_node_samples[SPLIT] -
                                     impurity[RIGHT] * n_node_samples[RIGHT] / n_node_samples[SPLIT]
                             ) * idxs.sum()

        node_split = Node(idxs=idxs, value=value[SPLIT], tree_num=tree_num,
                          feature=feature[SPLIT], threshold=threshold[SPLIT],
                          impurity_reduction=impurity_reduction, n_samples=n_node_samples)
        # print('\t>>>', node_split, 'impurity', impurity, 'num_pts', idxs.sum(), 'imp_reduc', impurity_reduction)

        # manage children
        idxs_split = X[:, feature[SPLIT]] <= threshold[SPLIT]
        idxs_left = idxs_split & idxs
        idxs_right = ~idxs_split & idxs
        node_left = Node(idxs=idxs_left, value=value[LEFT], tree_num=tree_num)
        node_right = Node(idxs=idxs_right, value=value[RIGHT], tree_num=tree_num)
        node_split.setattrs(left_temp=node_left, right_temp=node_right, )
        return node_split

    def fit(self, X, y=None, feature_names=None, verbose=False, sample_weight=None):
        """
        Params
        ------
        _sample_weight: array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted.
            Splits that would create child nodes with net zero or negative weight
            are ignored while searching for a split in each node.
        """

        if self.prediction_task == 'classification':
            self.classes_, y = np.unique(y, return_inverse=True)  # deals with str inputs
        X, y = check_X_y(X, y)
        y = y.astype(float)
        if feature_names is not None:
            self.feature_names_ = feature_names

        self.trees_ = []  # list of the root nodes of added trees
        self.complexity_ = 0  # tracks the number of rules in the model
        y_predictions_per_tree = {}  # predictions for each tree
        y_residuals_per_tree = {}  # based on predictions above

        def _update_tree_preds(n_iter):
            for k in range(n_iter):
                for tree_num_, tree_ in enumerate(self.trees_):
                    y_residuals_per_tree[tree_num_] = deepcopy(y)

                    # subtract predictions of all other trees
                    for tree_num_2_ in range(len(self.trees_)):
                        if not tree_num_2_ == tree_num_:
                            y_residuals_per_tree[tree_num_] -= y_predictions_per_tree[tree_num_2_]
                    tree_.update_values(X, y_residuals_per_tree[tree_num_])
                    y_predictions_per_tree[tree_num_] = self._predict_tree(self.trees_[tree_num_], X)

        # set up initial potential_splits
        # everything in potential_splits either is_root (so it can be added directly to self.trees_)
        # or it is a child of a root node that has already been added
        idxs = np.ones(X.shape[0], dtype=bool)
        node_init = self._construct_node_with_stump(X=X, y=y, idxs=idxs, tree_num=-1,
                                                    sample_weight=sample_weight, max_features=self.max_features)
        potential_splits = [node_init]
        if self.include_linear and idxs.sum() >= 5:
            node_init_linear = self._construct_node_linear(X=X, y=y, idxs=idxs, tree_num=-1,
                                                           sample_weight=sample_weight)
            potential_splits.append(node_init_linear)
        for node in potential_splits:
            node.setattrs(is_root=True)
        potential_splits = sorted(potential_splits, key=lambda x: x.impurity_reduction)

        # start the greedy fitting algorithm
        finished = False
        while len(potential_splits) > 0 and not finished:
            # print('potential_splits', [str(s) for s in potential_splits])
            split_node = potential_splits.pop()  # get node with max impurity_reduction (since it's sorted)

            # don't split on node
            if split_node.impurity_reduction < self.min_impurity_decrease:
                finished = True
                break

            # split on node
            if verbose:
                print('\nadding ' + str(split_node))
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
                node_new_root = Node(is_root=True, idxs=np.ones(X.shape[0], dtype=bool),
                                     tree_num=-1, split_or_linear=split_node.split_or_linear)
                potential_splits.append(node_new_root)

            # add children to potential splits (note this doesn't currently add linear potential splits)
            if split_node.split_or_linear == 'split':
                # assign left_temp, right_temp to be proper children
                # (basically adds them to tree in predict method)
                split_node.setattrs(left=split_node.left_temp, right=split_node.right_temp)

                # add children to potential_splits
                potential_splits.append(split_node.left)
                potential_splits.append(split_node.right)

            # update predictions for altered tree
            for tree_num_ in range(len(self.trees_)):
                y_predictions_per_tree[tree_num_] = self._predict_tree(self.trees_[tree_num_], X)
            y_predictions_per_tree[-1] = np.zeros(X.shape[0])  # dummy 0 preds for possible new trees

            # update residuals for each tree
            # -1 is key for potential new tree
            for tree_num_ in list(range(len(self.trees_))) + [-1]:
                y_residuals_per_tree[tree_num_] = deepcopy(y)

                # subtract predictions of all other trees
                for tree_num_2_ in range(len(self.trees_)):
                    if not tree_num_2_ == tree_num_:
                        y_residuals_per_tree[tree_num_] -= y_predictions_per_tree[tree_num_2_]

            _update_tree_preds(self.k1)

            # recompute all impurities + update potential_split children
            potential_splits_new = []
            for potential_split in potential_splits:
                y_target = y_residuals_per_tree[potential_split.tree_num]

                if potential_split.split_or_linear == 'split':
                    # re-calculate the best split
                    potential_split_updated = self._construct_node_with_stump(X=X,
                                                                              y=y_target,
                                                                              idxs=potential_split.idxs,
                                                                              tree_num=potential_split.tree_num,
                                                                              sample_weight=sample_weight,
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
                elif potential_split.split_or_linear == 'linear':
                    assert potential_split.is_root, 'Currently, linear node only supported as root'
                    assert potential_split.idxs.sum() == X.shape[0], 'Currently, linear node only supported as root'
                    potential_split_updated = self._construct_node_linear(idxs=potential_split.idxs,
                                                                          X=X,
                                                                          y=y_target,
                                                                          tree_num=potential_split.tree_num,
                                                                          sample_weight=sample_weight)

                    # don't need to retain anything from before (besides maybe is_root)
                    potential_split.setattrs(
                        feature=potential_split_updated.feature,
                        impurity_reduction=potential_split_updated.impurity_reduction,
                        value=potential_split_updated.value,
                    )

                # this is a valid split
                if potential_split.impurity_reduction is not None:
                    potential_splits_new.append(potential_split)

            # sort so largest impurity reduction comes last (should probs make this a heap later)
            potential_splits = sorted(potential_splits_new, key=lambda x: x.impurity_reduction)
            if verbose:
                print(self)
            if self.max_rules is not None and self.complexity_ >= self.max_rules:
                finished = True
                break

        _update_tree_preds(self.k2)

        # potentially fit linear model on the tree preds
        if self.posthoc_ridge:
            if self.prediction_task == 'regression':
                self.weighted_model_ = RidgeCV(alphas=(0.01, 0.1, 0.5, 1.0, 5, 10))
            elif self.prediction_task == 'classification':
                self.weighted_model_ = RidgeClassifierCV(alphas=(0.01, 0.1, 0.5, 1.0, 5, 10))
            X_feats = self._extract_tree_predictions(X)
            self.weighted_model_.fit(X_feats, y)
        return self

    def _tree_to_str(self, root: Node, prefix=''):
        if root is None:
            return ''
        elif root.split_or_linear == 'linear':
            return prefix + str(root)
        elif root.threshold is None:
            return ''
        pprefix = prefix + '\t'
        return prefix + str(root) + '\n' + self._tree_to_str(root.left, pprefix) + self._tree_to_str(root.right,
                                                                                                     pprefix)

    def __str__(self):
        s = '------------\n' + '\n\t+\n'.join([self._tree_to_str(t) for t in self.trees_])
        if hasattr(self, 'feature_names_') and self.feature_names_ is not None:
            for i in range(len(self.feature_names_))[::-1]:
                s = s.replace(f'X_{i}', self.feature_names_[i])
        return s

    def predict(self, X):
        if self.posthoc_ridge and self.weighted_model_:  # note, during fitting don't use the weighted moel
            X_feats = self._extract_tree_predictions(X)
            return self.weighted_model_.predict(X_feats)
        preds = np.zeros(X.shape[0])
        for tree in self.trees_:
            preds += self._predict_tree(tree, X)
        if self.prediction_task == 'regression':
            return preds
        elif self.prediction_task == 'classification':
            return (preds > 0.5).astype(int)

    def predict_proba(self, X):
        if self.prediction_task == 'regression':
            return NotImplemented
        elif self.posthoc_ridge and self.weighted_model_:  # note, during fitting don't use the weighted moel
            X_feats = self._extract_tree_predictions(X)
            d = self.weighted_model_.decision_function(X_feats)  # for 2 classes, this (n_samples,)
            probs = np.exp(d) / (1 + np.exp(d))
            return np.vstack((1 - probs, probs)).transpose()
        else:
            preds = np.zeros(X.shape[0])
            for tree in self.trees_:
                preds += self._predict_tree(tree, X)
            preds = np.clip(preds, a_min=0., a_max=1.)  # constrain to range of probabilities
            return np.vstack((1 - preds, preds)).transpose()

    def _extract_tree_predictions(self, X):
        """Extract predictions for all trees
        """
        X_feats = np.zeros((X.shape[0], len(self.trees_)))
        for tree_num_ in range(len(self.trees_)):
            preds_tree = self._predict_tree(self.trees_[tree_num_], X)
            X_feats[:, tree_num_] = preds_tree
        return X_feats

    def _predict_tree(self, root: Node, X):
        """Predict for a single tree
        This can be made way faster
        """

        def _predict_tree_single_point(root: Node, x):
            if root.split_or_linear == 'linear':
                return x[root.feature] * root.value
            elif root.left is None and root.right is None:
                return root.value
            left = x[root.feature] <= root.threshold
            if left:
                if root.left is None:  # we don't actually have to worry about this case
                    return root.value
                else:
                    return _predict_tree_single_point(root.left, x)
            else:
                if root.right is None:  # we don't actually have to worry about this case
                    return root.value
                else:
                    return _predict_tree_single_point(root.right, x)

        preds = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            preds[i] = _predict_tree_single_point(root, X[i])
        return preds

    def plot(self, cols=2, feature_names=None, filename=None, label="all", impurity=False, tree_number=None):
        is_single_tree =  len(self.trees_) < 2 or tree_number is not None
        n_cols = int(cols)
        n_rows = int(np.ceil(len(self.trees_) / n_cols))
        # if is_single_tree:
        #     fig, ax = plt.subplots(1)
        # else:
        #     fig, axs = plt.subplots(n_rows, n_cols)
        n_plots = int(len(self.trees_)) if tree_number is None else 1
        fig, axs = plt.subplots(n_plots)
        criterion = "squared_error" if self.prediction_task == "regression" else "gini"
        n_classes = 1 if self.prediction_task == 'regression' else 2
        ax_size = int(len(self.trees_))#n_cols * n_rows
        for i in range(n_plots):
            r = i // n_cols
            c = i % n_cols
            if not is_single_tree:
                # ax = axs[r, c]
                ax = axs[i]
            else:
                ax = axs
            try:
                tree = self.trees_[i] if tree_number is None else self.trees_[tree_number]
                plot_tree(DecisionTreeViz(tree, criterion, n_classes), ax=ax, feature_names=feature_names, label=label,
                          impurity=impurity)
            except IndexError:
                ax.axis('off')
                continue

            ax.set_title(f"Tree {i}")
        if filename is not None:
            plt.savefig(filename)
            return
        plt.show()


class FIGSExtRegressor(FIGSExt):
    def _init_prediction_task(self):
        self.prediction_task = 'regression'


class FIGSExtClassifier(FIGSExt):
    def _init_prediction_task(self):
        self.prediction_task = 'classification'


if __name__ == '__main__':
    np.random.seed(13)
    # X, y = datasets.load_breast_cancer(return_X_y=True)  # binary classification
    X, y = datasets.load_diabetes(return_X_y=True)  # regression
    # X = np.random.randn(500, 10)
    # y = (X[:, 0] > 0).astype(float) + (X[:, 1] > 1).astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )
    print('X.shape', X.shape)
    print('ys', np.unique(y_train), '\n\n')

    m = FIGSExtClassifier(max_rules=50)
    m.fit(X_train, y_train)
    print(m.predict_proba(X_train))
    m.plot(2, tree_number=0)
