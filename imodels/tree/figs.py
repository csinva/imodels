from copy import deepcopy
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit
import sklearn
from sklearn import datasets
from sklearn import tree
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import plot_tree, DecisionTreeClassifier
from sklearn.utils import check_X_y, check_array
from sklearn.utils.validation import _check_sample_weight, check_is_fitted

from imodels.tree.viz_utils import extract_sklearn_tree_from_figs
from imodels.util.arguments import check_fit_arguments
from imodels.util.data_util import encode_categories


class Node:
    def __init__(
        self,
        feature: int = None,
        threshold: int = None,
        value=None,
        value_sklearn=None,
        idxs=None,
        is_root: bool = False,
        left=None,
        impurity: float = None,
        impurity_reduction: float = None,
        tree_num: int = None,
        node_id: int = None,
        right=None,
    ):
        """Node class for splitting"""

        # split or linear
        self.is_root = is_root
        self.idxs = idxs
        self.tree_num = tree_num
        self.node_id = None
        self.feature = feature
        self.impurity = impurity
        self.impurity_reduction = impurity_reduction
        self.value_sklearn = value_sklearn

        # different meanings
        self.value = value  # for split this is mean, for linear this is weight

        # split-specific
        self.threshold = threshold
        self.left = left
        self.right = right
        self.left_temp = None
        self.right_temp = None

    def setattrs(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __str__(self):
        try:
            if self.is_root:
                return f"X_{self.feature} <= {self.threshold:0.3f} (Tree #{self.tree_num} root)"
            elif self.left is None and self.right is None:
                return f"Val: {self.value[0][0]:0.3f} (leaf)"
            else:
                return f"X_{self.feature} <= {self.threshold:0.3f} (split)"
        except:
            return self.__class__.__name__

    def print_root(self, y):
        try:
            one_count = pd.Series(y).value_counts()[1.0]
        except KeyError:
            one_count = 0
        one_proportion = (
            f" {one_count}/{y.shape[0]} ({round(100 * one_count / y.shape[0], 2)}%)"
        )

        if self.is_root:
            return f"X_{self.feature} <= {self.threshold:0.3f}" + one_proportion
        elif self.left is None and self.right is None:
            return f"ΔRisk = {self.value[0][0]:0.2f}" + one_proportion
        else:
            return f"X_{self.feature} <= {self.threshold:0.3f}" + one_proportion


class FIGS(BaseEstimator):
    """FIGS (sum of trees) classifier.
    Fast Interpretable Greedy-Tree Sums (FIGS) is an algorithm for fitting concise rule-based models.
    Specifically, FIGS generalizes CART to simultaneously grow a flexible number of trees in a summation.
    The total number of splits across all the trees can be restricted by a pre-specified threshold, keeping the model interpretable.
    Experiments across real-world datasets show that FIGS achieves state-of-the-art prediction performance when restricted to just a few splits (e.g. less than 20).
    https://arxiv.org/abs/2201.11931
    """

    def __init__(
        self,
        estimator=None,
        max_rules: int = 12,
        max_trees: int = None,
        min_impurity_decrease: float = 0.0,
        random_state=None,
        max_features: str = None,
    ):
        """
        Params
        ------
        max_rules: int
            Max total number of rules across all trees
        max_trees: int
            Max total number of trees
        min_impurity_decrease: float
            A node will be split if this split induces a decrease of the impurity greater than or equal to this value.
        max_features
            The number of features to consider when looking for the best split (see https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
        """
        # super().__init__()
        self.estimator = estimator
        self.max_rules = max_rules
        self.max_trees = max_trees
        self.min_impurity_decrease = min_impurity_decrease
        self.random_state = random_state
        self.max_features = max_features
        self._init_decision_function()

    def _init_decision_function(self):
        """Sets decision function based on _estimator_type"""
        # used by sklearn GridSearchCV, BaggingClassifier
        if isinstance(self, ClassifierMixin):

            def decision_function(x):
                return self.predict_proba(x)[:, 1]

        elif isinstance(self, RegressorMixin):
            decision_function = self.predict

    def _construct_node_with_stump(
        self,
        X,
        y,
        idxs,
        tree_num,
        sample_weight=None,
        compare_nodes_with_sample_weight=True,
        max_features=None,
    ):
        """
        Params
        ------
        compare_nodes_with_sample_weight: Deprecated
            If this is set to true and sample_weight is passed, use sample_weight to compare nodes
            Otherwise, use sample_weight only for picking a split given a particular node
        """

        # array indices
        SPLIT = 0
        LEFT = 1
        RIGHT = 2

        # fit stump
        stump = tree.DecisionTreeRegressor(max_depth=1, max_features=max_features)
        sweight = None
        if sample_weight is not None:
            sweight = sample_weight[idxs]
        stump.fit(X[idxs], y[idxs], sample_weight=sweight)

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
            return Node(
                idxs=idxs,
                value=value[SPLIT],
                tree_num=tree_num,
                feature=feature[SPLIT],
                threshold=threshold[SPLIT],
                impurity=impurity[SPLIT],
                impurity_reduction=None,
            )

        # manage sample weights
        idxs_split = X[:, feature[SPLIT]] <= threshold[SPLIT]
        idxs_left = idxs_split & idxs
        idxs_right = ~idxs_split & idxs
        if sample_weight is None:
            n_node_samples_left = n_node_samples[LEFT]
            n_node_samples_right = n_node_samples[RIGHT]
        else:
            n_node_samples_left = sample_weight[idxs_left].sum()
            n_node_samples_right = sample_weight[idxs_right].sum()
        n_node_samples_split = n_node_samples_left + n_node_samples_right

        # calculate impurity
        impurity_reduction = (
            impurity[SPLIT]
            - impurity[LEFT] * n_node_samples_left / n_node_samples_split
            - impurity[RIGHT] * n_node_samples_right / n_node_samples_split
        ) * n_node_samples_split

        node_split = Node(
            idxs=idxs,
            value=value[SPLIT],
            tree_num=tree_num,
            feature=feature[SPLIT],
            threshold=threshold[SPLIT],
            impurity=impurity[SPLIT],
            impurity_reduction=impurity_reduction,
        )
        # print('\t>>>', node_split, 'impurity', impurity, 'num_pts', idxs.sum(), 'imp_reduc', impurity_reduction)

        # manage children
        node_left = Node(
            idxs=idxs_left,
            value=value[LEFT],
            impurity=impurity[LEFT],
            tree_num=tree_num,
        )
        node_right = Node(
            idxs=idxs_right,
            value=value[RIGHT],
            impurity=impurity[RIGHT],
            tree_num=tree_num,
        )
        node_split.setattrs(
            left_temp=node_left,
            right_temp=node_right,
        )
        return node_split

    def _encode_categories(self, X, categorical_features):
        encoder = None
        if hasattr(self, "_encoder"):
            encoder = self._encoder
        return encode_categories(X, categorical_features, encoder)

    def fit(
        self,
        X,
        y=None,
        feature_names=None,
        verbose=False,
        sample_weight=None,
        categorical_features=None,
    ):
        """
        Params
        ------
        _sample_weight: array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted.
            Splits that would create child nodes with net zero or negative weight
            are ignored while searching for a split in each node.
        """
        if categorical_features is not None:
            X, self._encoder = self._encode_categories(X, categorical_features)
        X, y, feature_names = check_fit_arguments(self, X, y, feature_names)
        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X)

        self.trees_ = []  # list of the root nodes of added trees
        self.complexity_ = 0  # tracks the number of rules in the model
        y_predictions_per_tree = {}  # predictions for each tree
        y_residuals_per_tree = {}  # based on predictions above

        # set up initial potential_splits
        # everything in potential_splits either is_root (so it can be added directly to self.trees_)
        # or it is a child of a root node that has already been added
        idxs = np.ones(X.shape[0], dtype=bool)
        node_init = self._construct_node_with_stump(
            X=X,
            y=y,
            idxs=idxs,
            tree_num=-1,
            sample_weight=sample_weight,
            max_features=self.max_features,
        )
        potential_splits = [node_init]
        for node in potential_splits:
            node.setattrs(is_root=True)
        potential_splits = sorted(potential_splits, key=lambda x: x.impurity_reduction)

        # start the greedy fitting algorithm
        finished = False
        while len(potential_splits) > 0 and not finished:
            # print('potential_splits', [str(s) for s in potential_splits])
            # get node with max impurity_reduction (since it's sorted)
            split_node = potential_splits.pop()

            # don't split on node
            if split_node.impurity_reduction < self.min_impurity_decrease:
                finished = True
                break
            elif (
                split_node.is_root
                and self.max_trees is not None
                and len(self.trees_) >= self.max_trees
            ):
                # If the node is the root of a new tree and we have reached self.max_trees,
                # don't split on it, but allow later splits to continue growing existing trees
                continue

            # split on node
            if verbose:
                print("\nadding " + str(split_node))
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
                node_new_root = Node(
                    is_root=True, idxs=np.ones(X.shape[0], dtype=bool), tree_num=-1
                )
                potential_splits.append(node_new_root)

            # add children to potential splits
            # assign left_temp, right_temp to be proper children
            # (basically adds them to tree in predict method)
            split_node.setattrs(left=split_node.left_temp, right=split_node.right_temp)

            # add children to potential_splits
            potential_splits.append(split_node.left)
            potential_splits.append(split_node.right)

            # update predictions for altered tree
            for tree_num_ in range(len(self.trees_)):
                y_predictions_per_tree[tree_num_] = self._predict_tree(
                    self.trees_[tree_num_], X
                )
            # dummy 0 preds for possible new trees
            y_predictions_per_tree[-1] = np.zeros(X.shape[0])

            # update residuals for each tree
            # -1 is key for potential new tree
            for tree_num_ in list(range(len(self.trees_))) + [-1]:
                y_residuals_per_tree[tree_num_] = deepcopy(y)

                # subtract predictions of all other trees
                for tree_num_other_ in range(len(self.trees_)):
                    if not tree_num_other_ == tree_num_:
                        y_residuals_per_tree[tree_num_] -= y_predictions_per_tree[
                            tree_num_other_
                        ]

            # recompute all impurities + update potential_split children
            potential_splits_new = []
            for potential_split in potential_splits:
                y_target = y_residuals_per_tree[potential_split.tree_num]

                # re-calculate the best split
                potential_split_updated = self._construct_node_with_stump(
                    X=X,
                    y=y_target,
                    idxs=potential_split.idxs,
                    tree_num=potential_split.tree_num,
                    sample_weight=sample_weight,
                    max_features=self.max_features,
                )

                # need to preserve certain attributes from before (value at this split + is_root)
                # value may change because residuals may have changed, but we want it to store the value from before
                potential_split.setattrs(
                    feature=potential_split_updated.feature,
                    threshold=potential_split_updated.threshold,
                    impurity_reduction=potential_split_updated.impurity_reduction,
                    impurity=potential_split_updated.impurity,
                    left_temp=potential_split_updated.left_temp,
                    right_temp=potential_split_updated.right_temp,
                )

                # this is a valid split
                if potential_split.impurity_reduction is not None:
                    potential_splits_new.append(potential_split)

            # sort so largest impurity reduction comes last (should probs make this a heap later)
            potential_splits = sorted(
                potential_splits_new, key=lambda x: x.impurity_reduction
            )
            if verbose:
                print(self)
            if self.max_rules is not None and self.complexity_ >= self.max_rules:
                finished = True
                break

        # annotate final tree with node_id and value_sklearn, and prepare importance_data_
        importance_data = []
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

            # now that the samples per node are known, we can start to compute the importances
            importance_data_tree = np.zeros(len(self.feature_names_))

            def _importances(node: Node):
                if node is None or node.left is None:
                    return 0.0

                # TODO does not incorporate sample weights, but will if added to value_sklearn
                importance_data_tree[node.feature] += (
                    np.sum(node.value_sklearn) * node.impurity
                    - np.sum(node.left.value_sklearn) * node.left.impurity
                    - np.sum(node.right.value_sklearn) * node.right.impurity
                )

                return (
                    np.sum(node.value_sklearn)
                    + _importances(node.left)
                    + _importances(node.right)
                )

            # require the tree to have more than 1 node, otherwise just leave importance_data_tree as zeros
            if 1 < next(node_counter):
                tree_samples = _importances(tree_)
                if tree_samples != 0:
                    importance_data_tree /= tree_samples
                else:
                    importance_data_tree = 0

            importance_data.append(importance_data_tree)

        self.importance_data_ = importance_data

        return self

    def _tree_to_str(self, root: Node, prefix=""):
        if root is None:
            return ""
        elif root.threshold is None:
            return ""
        pprefix = prefix + "\t"
        return (
            prefix
            + str(root)
            + "\n"
            + self._tree_to_str(root.left, pprefix)
            + self._tree_to_str(root.right, pprefix)
        )

    def _tree_to_str_with_data(self, X, y, root: Node, prefix=""):
        if root is None:
            return ""
        elif root.threshold is None:
            return ""
        pprefix = prefix + "\t"
        left = X[:, root.feature] <= root.threshold
        return (
            prefix
            + root.print_root(y)
            + "\n"
            + self._tree_to_str_with_data(X[left], y[left], root.left, pprefix)
            + self._tree_to_str_with_data(X[~left], y[~left], root.right, pprefix)
        )

    def __str__(self):
        try:
            s = "> ------------------------------\n"
            s += "> FIGS-Fast Interpretable Greedy-Tree Sums:\n"
            s += '> \tPredictions are made by summing the "Val" reached by traversing each tree.\n'
            s += "> \tFor classifiers, a sigmoid function is then applied to the sum.\n"
            s += "> ------------------------------\n"
            s += "\n\t+\n".join([self._tree_to_str(t) for t in self.trees_])
            if hasattr(self, "feature_names_") and self.feature_names_ is not None:
                for i in range(len(self.feature_names_))[::-1]:
                    s = s.replace(f"X_{i}", self.feature_names_[i])
            return s
        except:
            return self.__class__.__name__

    def print_tree(self, X, y, feature_names=None):
        s = "------------\n" + "\n\t+\n".join(
            [self._tree_to_str_with_data(X, y, t) for t in self.trees_]
        )
        if feature_names is None:
            if hasattr(self, "feature_names_") and self.feature_names_ is not None:
                feature_names = self.feature_names_
        if feature_names is not None:
            for i in range(len(feature_names))[::-1]:
                s = s.replace(f"X_{i}", feature_names[i])
        return s

    def predict(self, X, categorical_features=None):
        if hasattr(self, "_encoder"):
            X = self._encode_categories(X, categorical_features=categorical_features)
        X = check_array(X)
        preds = np.zeros(X.shape[0])
        for tree in self.trees_:
            preds += self._predict_tree(tree, X)
        if isinstance(self, RegressorMixin):
            return preds
        elif isinstance(self, ClassifierMixin):
            return (preds > 0.5).astype(int)

    def predict_proba(self, X, categorical_features=None, use_clipped_prediction=False):
        """Predict probability for classifiers:
        Default behavior is to constrain the outputs to the range of probabilities, i.e. 0 to 1, with a sigmoid function.
        Set use_clipped_prediction=True to use prior behavior of clipping between 0 and 1 instead.
        """
        if hasattr(self, "_encoder"):
            X = self._encode_categories(X, categorical_features=categorical_features)
        X = check_array(X)
        if isinstance(self, RegressorMixin):
            return NotImplemented
        preds = np.zeros(X.shape[0])
        for tree in self.trees_:
            preds += self._predict_tree(tree, X)
        if use_clipped_prediction:
            # old behavior, pre v1.3.9
            # constrain to range of probabilities by clipping
            preds = np.clip(preds, a_min=0.0, a_max=1.0)
        else:
            # constrain to range of probabilities with a sigmoid function
            preds = expit(preds)
        return np.vstack((1 - preds, preds)).transpose()

    def _predict_tree(self, root: Node, X):
        """Predict for a single tree"""

        def _predict_tree_single_point(root: Node, x):
            if root.left is None and root.right is None:
                return root.value
            left = x[root.feature] <= root.threshold
            if left:
                if root.left is None:  # we don't actually have to worry about this case
                    return root.value
                else:
                    return _predict_tree_single_point(root.left, x)
            else:
                if (
                    root.right is None
                ):  # we don't actually have to worry about this case
                    return root.value
                else:
                    return _predict_tree_single_point(root.right, x)

        preds = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            preds[i] = _predict_tree_single_point(root, X[i])
        return preds

    @property
    def feature_importances_(self):
        """Gini impurity-based feature importances"""
        check_is_fitted(self)

        avg_feature_importances = np.mean(
            self.importance_data_, axis=0, dtype=np.float64
        )

        return avg_feature_importances / np.sum(avg_feature_importances)

    def plot(
        self,
        cols=2,
        feature_names=None,
        filename=None,
        label="all",
        impurity=False,
        tree_number=None,
        dpi=150,
        fig_size=None,
    ):
        is_single_tree = len(self.trees_) < 2 or tree_number is not None
        n_cols = int(cols)
        n_rows = int(np.ceil(len(self.trees_) / n_cols))

        if feature_names is None:
            if hasattr(self, "feature_names_") and self.feature_names_ is not None:
                feature_names = self.feature_names_

        n_plots = int(len(self.trees_)) if tree_number is None else 1
        fig, axs = plt.subplots(n_plots, dpi=dpi)
        if fig_size is not None:
            fig.set_size_inches(fig_size, fig_size)

        n_classes = 1 if isinstance(self, RegressorMixin) else 2
        ax_size = int(len(self.trees_))
        for i in range(n_plots):
            r = i // n_cols
            c = i % n_cols
            if not is_single_tree:
                ax = axs[i]
            else:
                ax = axs
            try:
                dt = extract_sklearn_tree_from_figs(
                    self, i if tree_number is None else tree_number, n_classes
                )
                plot_tree(
                    dt,
                    ax=ax,
                    feature_names=feature_names,
                    label=label,
                    impurity=impurity,
                )
            except IndexError:
                ax.axis("off")
                continue
            ttl = f"Tree {i}" if n_plots > 1 else f"Tree {tree_number}"
            ax.set_title(ttl)
        if filename is not None:
            plt.savefig(filename)
            return
        plt.show()

    def _validate_estimator(self, default=None):
        """Check the base estimator.

        Sets the `estimator_` attributes.
        """
        if self.estimator is not None:
            self.estimator_ = self.estimator
        else:
            self.estimator_ = default


class FIGSRegressor(FIGS, RegressorMixin):
    def __init__(
        self,
        max_rules: int = 12,
        max_trees: int = None,
        min_impurity_decrease: float = 0.0,
        random_state=None,
        max_features: str = None,
    ):
        super().__init__(
            max_rules=max_rules,
            max_trees=max_trees,
            min_impurity_decrease=min_impurity_decrease,
            max_features=max_features,
        )


class FIGSClassifier(FIGS, ClassifierMixin):
    def __init__(
        self,
        max_rules: int = 12,
        max_trees: int = None,
        min_impurity_decrease: float = 0.0,
        random_state=None,
        max_features: str = None,
    ):
        super().__init__(
            max_rules=max_rules,
            max_trees=max_trees,
            min_impurity_decrease=min_impurity_decrease,
            max_features=max_features,
        )


class FIGSCV:
    def __init__(
        self,
        figs,
        n_rules_list: List[int] = [6, 12, 24, 30, 50],
        n_trees_list: List[int] = [5, 5, 5, 5, 5],
        cv: int = 3,
        scoring=None,
        *args,
        **kwargs,
    ):
        if len(n_rules_list) != len(n_trees_list):
            raise ValueError(
                f"len(n_rules_list) = {len(n_rules_list)} != len(n_trees_list) = {len(n_trees_list)}"
            )

        self._figs_class = figs
        self.n_rules_list = np.array(n_rules_list)
        self.n_trees_list = np.array(n_trees_list)
        self.cv = cv
        self.scoring = scoring

    def fit(self, X, y):
        self.scores_ = []
        for _i, n_rules in enumerate(self.n_rules_list):
            est = self._figs_class(max_rules=n_rules, max_trees=self.n_trees_list[_i])
            cv_scores = cross_val_score(est, X, y, cv=self.cv, scoring=self.scoring)
            mean_score = np.mean(cv_scores)
            if len(self.scores_) == 0:
                self.figs = est
            elif mean_score > np.max(self.scores_):
                self.figs = est

            self.scores_.append(mean_score)
        self.figs.fit(X=X, y=y)
        return self

    def predict_proba(self, X):
        return self.figs.predict_proba(X)

    def predict(self, X):
        return self.figs.predict(X)

    @property
    def max_rules(self):
        return self.figs.max_rules

    @property
    def max_trees(self):
        return self.figs.max_trees


class FIGSRegressorCV(FIGSCV):
    def __init__(
        self,
        n_rules_list: List[int] = [6, 12, 24, 30, 50],
        n_trees_list: List[int] = [5, 5, 5, 5, 5],
        cv: int = 3,
        scoring="r2",
        *args,
        **kwargs,
    ):
        super(FIGSRegressorCV, self).__init__(
            figs=FIGSRegressor,
            n_rules_list=n_rules_list,
            n_trees_list=n_trees_list,
            cv=cv,
            scoring=scoring,
            *args,
            **kwargs,
        )


class FIGSClassifierCV(FIGSCV):
    def __init__(
        self,
        n_rules_list: List[int] = [6, 12, 24, 30, 50],
        n_trees_list: List[int] = [5, 5, 5, 5, 5],
        cv: int = 3,
        scoring="accuracy",
        *args,
        **kwargs,
    ):
        super(FIGSClassifierCV, self).__init__(
            figs=FIGSClassifier,
            n_rules_list=n_rules_list,
            n_trees_list=n_trees_list,
            cv=cv,
            scoring=scoring,
            *args,
            **kwargs,
        )


if __name__ == "__main__":
    from sklearn import datasets

    X_cls, Y_cls = datasets.load_breast_cancer(return_X_y=True)
    X_reg, Y_reg = datasets.make_friedman1(100)

    categories = ["cat", "dog", "bird", "fish"]
    categories_2 = ["bear", "chicken", "cow"]

    X_cat = pd.DataFrame(X_reg)
    X_cat["pet1"] = np.random.choice(categories, size=(100, 1))
    X_cat["pet2"] = np.random.choice(categories_2, size=(100, 1))

    # X_cat.columns[-1] = "pet"
    Y_cat = Y_reg

    est = FIGSRegressor(max_rules=10)
    est.fit(X_cat, Y_cat, categorical_features=["pet1", "pet2"])
    est.predict(X_cat, categorical_features=["pet1", "pet2"])
    est.plot(tree_number=1)

    est = FIGSClassifier(max_rules=10)
    # est.fit(X_cls, Y_cls, sample_weight=np.arange(0, X_cls.shape[0]))
    est.fit(X_cls, Y_cls, sample_weight=[1] * X_cls.shape[0])
    est.predict(X_cls)

    est = FIGSRegressorCV()
    est.fit(X_reg, Y_reg)
    est.predict(X_reg)
    print(est.max_rules)
    est.figs.plot(tree_number=0)

    est = FIGSClassifierCV()
    est.fit(X_cls, Y_cls)
    est.predict(X_cls)
    print(est.max_rules)
    est.figs.plot(tree_number=0)

# %%
