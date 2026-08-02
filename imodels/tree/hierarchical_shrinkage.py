from copy import deepcopy
from typing import List

import numpy as np
from sklearn import datasets
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.metrics import r2_score, mean_squared_error, log_loss
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, export_text
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)

from sklearn.utils.validation import check_is_fitted

from imodels.util import checks
from imodels.util.arguments import check_fit_arguments, check_predict_X
from imodels.util.tree import compute_tree_complexity


def _as_arrays(X, y):
    """Convert X, y to numpy arrays so that they can be indexed by CV folds."""
    return np.asarray(X), np.asarray(y)


def _values_are_normalized(tree):
    """Whether a fitted tree stores per-node values as probabilities rather than counts.

    sklearn switched classification trees from storing class counts to storing
    normalized class fractions in version 1.3.
    """
    return bool(np.allclose(tree.value.sum(axis=(1, 2)), 1))


class HSTree(BaseEstimator):
    def __init__(
        self,
        estimator_: BaseEstimator = None,
        reg_param: float = 1,
        shrinkage_scheme_: str = "node_based",
        max_leaf_nodes: int = None,
        random_state: int = None,
    ):
        """HSTree (Tree with hierarchical shrinkage applied).
        Hierarchical shinkage is an extremely fast post-hoc regularization method which works on any decision tree (or tree-based ensemble, such as Random Forest).
        It does not modify the tree structure, and instead regularizes the tree by shrinking the prediction over each node towards the sample means of its ancestors (using a single regularization parameter).
        Experiments over a wide variety of datasets show that hierarchical shrinkage substantially increases the predictive performance of individual decision trees and decision-tree ensembles.
        https://arxiv.org/abs/2202.00858

        Params
        ------
        estimator_: sklearn tree or tree ensemble model (e.g. RandomForest or GradientBoosting)
            Defaults to CART Classification Tree with 20 max leaf nodes
            Note: this estimator will be directly modified

        reg_param: float
            Higher is more regularization (can be arbitrarily large, should not be < 0)

        shrinkage_scheme: str
            Experimental: Used to experiment with different forms of shrinkage. options are:
                (i) node_based shrinks based on number of samples in parent node
                (ii) leaf_based only shrinks leaf nodes based on number of leaf samples
                (iii) constant shrinks every node by a constant lambda

        max_leaf_nodes: int
            If estimator is None, then max_leaf_nodes is passed to the default decision tree
        """
        super().__init__()
        self.reg_param = reg_param
        # constructed here rather than defaulted in the signature, so that
        # separate models don't share (and refit) one estimator object
        if estimator_ is None:
            estimator_ = DecisionTreeClassifier(max_leaf_nodes=20)
        self.estimator_ = estimator_
        self.shrinkage_scheme_ = shrinkage_scheme_
        self.random_state = random_state
        if checks.check_is_fitted(self.estimator_):
            self._shrink()
        if max_leaf_nodes is not None:
            self.estimator_.max_leaf_nodes = max_leaf_nodes
            self.estimator_.random_state = random_state


    def get_rules(self, feature_names=None):
        """Return this model's rules as a DataFrame (see imodels.get_rules)."""
        from imodels.util.get_rules import get_rules
        return get_rules(self, feature_names=feature_names)

    def apply(self, X):
        """Return the leaf each sample reaches (see imodels.util.apply.apply_leaves)."""
        from imodels.util.apply import apply_leaves
        return apply_leaves(self, X)

    @property
    def feature_importances_(self):
        """Mean decrease in impurity, as in sklearn's tree models.

        Shrinkage rewrites node values but not the tree structure or its
        impurities, so these match the underlying fitted estimator's.
        """
        check_is_fitted(self.estimator_ if hasattr(self, 'estimator_') else self)
        return self.estimator_.feature_importances_

    def get_params(self, deep=True):
        d = {
            "reg_param": self.reg_param,
            "estimator_": self.estimator_,
            "shrinkage_scheme_": self.shrinkage_scheme_,
            "max_leaf_nodes": self.estimator_.max_leaf_nodes,
        }
        if deep:
            return deepcopy(d)
        return d

    def fit(self, X, y, sample_weight=None, *args, **kwargs):
        # remove feature_names if it exists (note: only works as keyword-arg)
        # None returned if not passed
        feature_names = kwargs.pop("feature_names", None)
        # missing values are left to the wrapped estimator, which handles them for
        # sklearn decision trees and forests and raises for those that cannot
        X, y, feature_names = check_fit_arguments(
            self, X, y, feature_names, allow_nan=True)
        if feature_names is not None:
            self.feature_names = feature_names
        self.estimator_ = self.estimator_.fit(
            X, y, *args, sample_weight=sample_weight, **kwargs
        )
        self._shrink()

        # compute complexity
        if hasattr(self.estimator_, "tree_"):
            self.complexity_ = compute_tree_complexity(self.estimator_.tree_)
        elif hasattr(self.estimator_, "estimators_"):
            self.complexity_ = 0
            for t in self.estimator_.estimators_:
                # read-only, so no need to copy the tree
                if isinstance(t, np.ndarray):
                    assert t.size == 1, "multiple trees stored under tree_?"
                    t = t[0]
                self.complexity_ += compute_tree_complexity(t.tree_)
        return self

    def _shrink_tree(
        self,
        tree,
        reg_param,
        i=0,
        parent_val=None,
        parent_num=None,
        cum_sum=0,
        values_normalized=None,
    ):
        """Shrink the tree"""
        if reg_param is None:
            reg_param = 1.0
        if values_normalized is None:
            # sklearn >= 1.3 stores classification tree values as probabilities
            # rather than counts, in which case they must not be renormalized.
            # Computed once at the root, since shrinking rewrites tree.value.
            values_normalized = _values_are_normalized(tree)
        left = tree.children_left[i]
        right = tree.children_right[i]
        is_leaf = left == right
        n_samples = tree.weighted_n_node_samples[i]
        if (
            isinstance(self, RegressorMixin)
            or isinstance(self.estimator_, GradientBoostingClassifier)
            or values_normalized
        ):
            val = tree.value[i, :, :].copy()
        else:  # If classification, counts need normalizing into a probability vector
            val = tree.value[i, :, :] / n_samples

        # Step 1: Update cum_sum
        # if root
        if parent_val is None and parent_num is None:
            cum_sum = val

        # if has parent
        else:
            if self.shrinkage_scheme_ == "node_based":
                val_new = (val - parent_val) / (1 + reg_param / parent_num)
            elif self.shrinkage_scheme_ == "constant":
                val_new = (val - parent_val) / (1 + reg_param)
            else:  # leaf_based
                val_new = 0
            cum_sum += val_new

        # Step 2: Update node values
        if (
            self.shrinkage_scheme_ == "node_based"
            or self.shrinkage_scheme_ == "constant"
        ):
            tree.value[i, :, :] = cum_sum
        else:  # leaf_based
            if is_leaf:  # update node values if leaf_based
                root_val = tree.value[0, :, :]
                tree.value[i, :, :] = root_val + (val - root_val) / (
                    1 + reg_param / n_samples
                )
            else:
                tree.value[i, :, :] = val

                # Step 3: Recurse if not leaf
        if not is_leaf:
            self._shrink_tree(
                tree,
                reg_param,
                left,
                parent_val=val,
                parent_num=n_samples,
                cum_sum=np.copy(cum_sum),
                values_normalized=values_normalized,
            )
            self._shrink_tree(
                tree,
                reg_param,
                right,
                parent_val=val,
                parent_num=n_samples,
                cum_sum=np.copy(cum_sum),
                values_normalized=values_normalized,
            )

            # edit the non-leaf nodes for later visualization (doesn't effect predictions)

        return tree

    def _unwrap_tree(self, estimator):
        """Get the underlying sklearn tree, or None if there isn't one."""
        if isinstance(estimator, np.ndarray):
            assert estimator.size == 1, "multiple trees stored under tree_?"
            estimator = estimator[0]
        return getattr(estimator, "tree_", None)

    def _check_estimator_supported(self):
        """Reject estimators that shrinkage cannot be applied to.

        Shrinkage rewrites the node values of sklearn decision trees, so it needs
        either a fitted tree (tree_) or an ensemble of them (estimators_).
        Without this check an unsupported model is returned unchanged, which
        looks like it worked -- see https://github.com/csinva/imodels/issues/199
        """
        if hasattr(self.estimator_, "tree_"):
            return
        subestimators = getattr(self.estimator_, "estimators_", None)
        if subestimators is not None and len(subestimators) > 0:
            if self._unwrap_tree(subestimators[0]) is not None:
                return

        raise ValueError(
            f"{type(self.estimator_).__name__} is not supported by hierarchical "
            "shrinkage, which needs an estimator built from scikit-learn decision "
            "trees (exposing tree_, or estimators_ of trees) -- for example "
            "DecisionTreeClassifier, RandomForestRegressor or "
            "GradientBoostingClassifier. Applying shrinkage to it would leave the "
            "model unchanged."
        )

    def _shrink(self):
        self._check_estimator_supported()
        if hasattr(self.estimator_, "tree_"):
            self._shrink_tree(self.estimator_.tree_, self.reg_param)
        else:
            for t in self.estimator_.estimators_:
                self._shrink_tree(self._unwrap_tree(t), self.reg_param)

    def predict(self, X, *args, **kwargs):
        check_predict_X(self, X)
        preds = self.estimator_.predict(X, *args, **kwargs)
        # fit encodes y as 0..n_classes-1, so map back onto the original labels.
        # When the estimator was fitted elsewhere and passed in already fitted,
        # there is no such encoding and its predictions are already labels.
        if hasattr(self, "classes_") and hasattr(self.estimator_, "classes_"):
            return np.array([self.classes_[int(i)] for i in preds])
        else:
            return preds

    def predict_proba(self, X, *args, **kwargs):
        check_predict_X(self, X)
        if hasattr(self.estimator_, "predict_proba"):
            probs = self.estimator_.predict_proba(X, *args, **kwargs)
            # the shrinkage arithmetic can leave values a hair outside [0, 1]
            # (e.g. -1e-17), which newer versions of sklearn's log_loss reject
            probs = np.clip(probs, 0, 1)
            totals = probs.sum(axis=1, keepdims=True)
            with np.errstate(invalid="ignore", divide="ignore"):
                normalized = probs / totals
            return np.where(totals > 0, normalized, 1 / probs.shape[1])
        else:
            return NotImplemented

    def __str__(self):
        # check if fitted
        if not checks.check_is_fitted(self.estimator_):
            s = self.__class__.__name__
            s += "("
            s += "est="
            s += repr(self.estimator_)
            s += ", "
            s += "reg_param="
            s += str(self.reg_param)
            s += ")"
            return s
        elif not hasattr(self.estimator_, "tree_"):
            # an ensemble: export_text only renders a single tree, so summarize
            n_trees = len(getattr(self.estimator_, "estimators_", []))
            s = "> ------------------------------\n"
            s += "> Tree ensemble with Hierarchical Shrinkage\n"
            s += "> \tPrediction is made by combining the predictions of each shrunk tree\n"
            s += "> ------------------------------" + "\n"
            s += f"> {type(self.estimator_).__name__} of {n_trees} trees, "
            s += f"reg_param={self.reg_param}\n"
            if hasattr(self, "complexity_"):
                s += f"> {self.complexity_} total non-leaf nodes\n"
            return s
        else:
            s = "> ------------------------------\n"
            s += "> Decision Tree with Hierarchical Shrinkage\n"
            s += "> \tPrediction is made by looking at the value in the appropriate leaf of the tree\n"
            s += "> ------------------------------" + "\n"

            if hasattr(self, "feature_names") and self.feature_names is not None:
                return s + export_text(
                    self.estimator_, feature_names=self.feature_names, show_weights=True
                )
            else:
                return s + export_text(self.estimator_, show_weights=True)

    def __repr__(self):
        # s = self.__class__.__name__
        # s += "("
        # s += "estimator_="
        # s += repr(self.estimator_)
        # s += ", "
        # s += "reg_param="
        # s += str(self.reg_param)
        # s += ", "
        # s += "shrinkage_scheme_="
        # s += self.shrinkage_scheme_
        # s += ")"
        # return s
        attr_list = ["estimator_", "reg_param", "shrinkage_scheme_"]
        s = self.__class__.__name__
        s += "("
        for attr in attr_list:
            s += attr + "=" + repr(getattr(self, attr)) + ", "
        s = s[:-2] + ")"
        return s


class HSTreeRegressor(RegressorMixin, HSTree):
    def __init__(
        self,
        estimator_: BaseEstimator = None,
        reg_param: float = 1,
        shrinkage_scheme_: str = "node_based",
        max_leaf_nodes: int = None,
        random_state: int = None,
    ):
        if estimator_ is None:
            estimator_ = DecisionTreeRegressor(max_leaf_nodes=20)
        if estimator_ is None:
            estimator_ = DecisionTreeClassifier(max_leaf_nodes=20)
        super().__init__(
            estimator_=estimator_,
            reg_param=reg_param,
            shrinkage_scheme_=shrinkage_scheme_,
            max_leaf_nodes=max_leaf_nodes,
            random_state=random_state,
        )


class HSTreeClassifier(ClassifierMixin, HSTree):
    def __init__(
        self,
        estimator_: BaseEstimator = None,
        reg_param: float = 1,
        shrinkage_scheme_: str = "node_based",
        max_leaf_nodes: int = None,
        random_state: int = None,
    ):
        super().__init__(
            estimator_=estimator_,
            reg_param=reg_param,
            shrinkage_scheme_=shrinkage_scheme_,
            max_leaf_nodes=max_leaf_nodes,
            random_state=random_state,
        )


def _get_cv_criterion(scorer):
    """Whether the best score is the highest or the lowest one.

    Probed on fixed data: a scorer's direction does not depend on the draw, and
    sampling here would advance the caller's global numpy random stream.
    """
    y_true = np.tile([0, 1], 50)

    score_good = scorer(y_true, y_true)
    score_bad = scorer(y_true, 1 - y_true)

    if score_good > score_bad:
        return np.argmax
    elif score_good < score_bad:
        return np.argmin

    raise ValueError(
        f"Cannot tell whether higher or lower scores from "
        f"{getattr(scorer, '__name__', scorer)} are better: it scored a perfect "
        "and a fully incorrect prediction the same. Pass a scoring function that "
        "separates them."
    )


class HSTreeClassifierCV(HSTreeClassifier):
    def __init__(
        self,
        estimator_: BaseEstimator = None,
        reg_param_list: List[float] = [0, 0.1, 1, 10, 50, 100, 500],
        shrinkage_scheme_: str = "node_based",
        max_leaf_nodes: int = 20,
        cv: int = 3,
        scoring=None,
        *args,
        **kwargs
    ):
        """Cross-validation is used to select the best regularization parameter for hierarchical shrinkage.

         Params
        ------
        estimator_
            Sklearn estimator (already initialized).
            If no estimator_ is passed, sklearn decision tree is used

        max_rules
            If estimator is None, then max_leaf_nodes is passed to the default decision tree

        args, kwargs
            Note: args, kwargs are not used but left so that imodels-experiments can still pass redundant args.
        """
        if estimator_ is None:
            estimator_ = DecisionTreeClassifier(max_leaf_nodes=max_leaf_nodes)
        super().__init__(estimator_, reg_param=None)
        # stored unmodified so that the estimator stays sklearn-cloneable
        self.reg_param_list = reg_param_list
        self.cv = cv
        self.scoring = scoring
        self.shrinkage_scheme_ = shrinkage_scheme_
        # print('estimator', self.estimator_,
        #       'checks.check_is_fitted(estimator)', checks.check_is_fitted(self.estimator_))
        # if checks.check_is_fitted(self.estimator_):
        #     raise Warning('Passed an already fitted estimator,'
        #                   'but shrinking not applied until fit method is called.')

    def get_params(self, deep=True):
        d = {
            "estimator_": self.estimator_,
            "reg_param_list": self.reg_param_list,
            "shrinkage_scheme_": self.shrinkage_scheme_,
            "max_leaf_nodes": self.estimator_.max_leaf_nodes,
            "cv": self.cv,
            "scoring": self.scoring,
        }
        if deep:
            return deepcopy(d)
        return d

    def fit(self, X, y, *args, **kwargs):
        X_arr, y_arr = _as_arrays(X, y)
        self.scores_ = [[] for _ in self.reg_param_list]
        scorer = kwargs.get("scoring", log_loss)
        kf = KFold(n_splits=self.cv)
        for train_index, test_index in kf.split(X_arr):
            X_out, y_out = X_arr[test_index, :], y_arr[test_index]
            X_in, y_in = X_arr[train_index, :], y_arr[train_index]
            base_est = deepcopy(self.estimator_)
            base_est.fit(X_in, y_in)
            for i, reg_param in enumerate(self.reg_param_list):
                # shrinkage is post-hoc, so shrink a copy of the tree fitted
                # above rather than refitting it for every reg_param
                est_hs = HSTreeClassifier(deepcopy(base_est), reg_param)
                self.scores_[i].append(
                    scorer(y_out, est_hs.predict_proba(X_out)))
        self.scores_ = [np.mean(s) for s in self.scores_]
        cv_criterion = _get_cv_criterion(scorer)
        self.reg_param = np.asarray(self.reg_param_list)[cv_criterion(self.scores_)]
        return super().fit(X=X, y=y, *args, **kwargs)

    def __repr__(self):
        attr_list = [
            "estimator_",
            "reg_param_list",
            "shrinkage_scheme_",
            "cv",
            "scoring",
        ]
        s = self.__class__.__name__
        s += "("
        for attr in attr_list:
            s += attr + "=" + repr(getattr(self, attr)) + ", "
        s = s[:-2] + ")"
        return s


class HSTreeRegressorCV(HSTreeRegressor):
    def __init__(
        self,
        estimator_: BaseEstimator = None,
        reg_param_list: List[float] = [0, 0.1, 1, 10, 50, 100, 500],
        shrinkage_scheme_: str = "node_based",
        max_leaf_nodes: int = 20,
        cv: int = 3,
        scoring=None,
        *args,
        **kwargs
    ):
        """Cross-validation is used to select the best regularization parameter for hierarchical shrinkage.

         Params
        ------
        estimator_
            Sklearn estimator (already initialized).
            If no estimator_ is passed, sklearn decision tree is used

        max_rules
            If estimator is None, then max_leaf_nodes is passed to the default decision tree

        args, kwargs
            Note: args, kwargs are not used but left so that imodels-experiments can still pass redundant args.
        """
        if estimator_ is None:
            estimator_ = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes)
        super().__init__(estimator_, reg_param=None)
        # stored unmodified so that the estimator stays sklearn-cloneable
        self.reg_param_list = reg_param_list
        self.cv = cv
        self.scoring = scoring
        self.shrinkage_scheme_ = shrinkage_scheme_
        # print('estimator', self.estimator_,
        #       'checks.check_is_fitted(estimator)', checks.check_is_fitted(self.estimator_))
        # if checks.check_is_fitted(self.estimator_):
        #     raise Warning('Passed an already fitted estimator,'
        #                   'but shrinking not applied until fit method is called.')

    def get_params(self, deep=True):
        d = {
            "estimator_": self.estimator_,
            "reg_param_list": self.reg_param_list,
            "shrinkage_scheme_": self.shrinkage_scheme_,
            "max_leaf_nodes": self.estimator_.max_leaf_nodes,
            "cv": self.cv,
            "scoring": self.scoring,
        }
        if deep:
            return deepcopy(d)
        return d

    def fit(self, X, y, *args, **kwargs):
        X_arr, y_arr = _as_arrays(X, y)
        self.scores_ = [[] for _ in self.reg_param_list]
        kf = KFold(n_splits=self.cv)
        scorer = kwargs.get("scoring", mean_squared_error)
        for train_index, test_index in kf.split(X_arr):
            X_out, y_out = X_arr[test_index, :], y_arr[test_index]
            X_in, y_in = X_arr[train_index, :], y_arr[train_index]
            base_est = deepcopy(self.estimator_)
            base_est.fit(X_in, y_in)
            for i, reg_param in enumerate(self.reg_param_list):
                # shrinkage is post-hoc, so shrink a copy of the tree fitted
                # above rather than refitting it for every reg_param
                est_hs = HSTreeRegressor(deepcopy(base_est), reg_param)
                self.scores_[i].append(scorer(est_hs.predict(X_out), y_out))
        self.scores_ = [np.mean(s) for s in self.scores_]
        cv_criterion = _get_cv_criterion(scorer)
        self.reg_param = np.asarray(self.reg_param_list)[cv_criterion(self.scores_)]
        return super().fit(X=X, y=y, *args, **kwargs)

    def __repr__(self):
        attr_list = [
            "estimator_",
            "reg_param_list",
            "shrinkage_scheme_",
            "cv",
            "scoring",
        ]
        s = self.__class__.__name__
        s += "("
        for attr in attr_list:
            s += attr + "=" + repr(getattr(self, attr)) + ", "
        s = s[:-2] + ")"
        return s


if __name__ == "__main__":
    np.random.seed(15)
    # X, y = datasets.fetch_california_housing(return_X_y=True)  # regression
    # X, y = datasets.load_breast_cancer(return_X_y=True)  # binary classification
    X, y = datasets.load_diabetes(return_X_y=True)  # regression
    # X = np.random.randn(500, 10)
    # y = (X[:, 0] > 0).astype(float) + (X[:, 1] > 1).astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=10
    )
    print("X.shape", X.shape)
    print("ys", np.unique(y_train))

    # m = HSTree(estimator_=DecisionTreeClassifier(), reg_param=0.1)
    # m = DecisionTreeClassifier(max_leaf_nodes = 20,random_state=1, max_features=None)
    # m = DecisionTreeClassifier(random_state=42)
    m = GradientBoostingRegressor(random_state=10, n_estimators=5)
    # print('best alpha', m.reg_param)
    m.fit(X_train, y_train)
    # m.predict_proba(X_train)  # just run this
    print("score", r2_score(y_test, m.predict(X_test)))
    print("running again....")

    # x = DecisionTreeRegressor(random_state = 42, ccp_alpha = 0.3)
    # x.fit(X_train,y_train)

    # m = HSTree(estimator_=DecisionTreeRegressor(random_state=42, max_features=None), reg_param=10)
    # m = HSTree(estimator_=DecisionTreeClassifier(random_state=42, max_features=None), reg_param=0)
    # m = HSTreeRegressorCV(
    #     estimator_=DecisionTreeClassifier(random_state=42),
    #     shrinkage_scheme_="node_based",
    #     reg_param_list=[0.1, 1, 2, 5, 10, 25, 50, 100, 500],
    # )
    # m = ShrunkTreeCV(estimator_=DecisionTreeClassifier())
    m = HSTreeRegressor(m)
    print("score", r2_score(y_test, m.predict(X_test)))

    m = HSTreeRegressor(
        estimator_=GradientBoostingRegressor(
            random_state=10,
            n_estimators=5,
        ),
        reg_param=1,
    )
    m.fit(X_train, y_train)
    print("best alpha", m.reg_param)
    # m.predict_proba(X_train)  # just run this
    # print('score', m.score(X_test, y_test))
    print("score", r2_score(y_test, m.predict(X_test)))
