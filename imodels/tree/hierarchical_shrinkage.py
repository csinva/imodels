from __future__ import annotations

import time
from copy import deepcopy
from typing import List

import numpy as np
from sklearn import datasets
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin, clone
from sklearn.metrics import r2_score, mean_squared_error, log_loss
from sklearn.model_selection import cross_val_score, KFold
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, export_text
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestRegressor,
)

from imodels.util import checks
from imodels.util.arguments import check_fit_arguments
from imodels.util.tree import compute_tree_complexity
from imodels.tree._hs_gcv import apply_node_based_hs, select_hs_reg_param


class HSTree(BaseEstimator):
    def __init__(
        self,
        estimator_: BaseEstimator | None = None,
        reg_param: float | str = 1,
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
            A fitted estimator is copied before applying shrinkage. Calling
            fit always fits a fresh clone; the supplied estimator is not modified.

        reg_param: float or "gcv"
            Higher is more regularization (can be arbitrarily large, should not be < 0)
            Use "gcv" for automatic node-based shrinkage of a single
            squared-error regression tree with uniform observation weights.
            This conditions on the fitted tree, not its construction. The
            selected strength is exposed as reg_param_ (possibly infinity
            for root-mean predictions) and diagnostics as gcv_results_.
            The legacy None value still means 1.0; use "gcv" explicitly.

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
        self.shrinkage_scheme_ = shrinkage_scheme_
        self.max_leaf_nodes = max_leaf_nodes
        self.random_state = random_state
        self._estimator_template = estimator_
        if estimator_ is None:
            tree_class = (
                DecisionTreeRegressor if isinstance(self, RegressorMixin)
                else DecisionTreeClassifier
            )
            estimator_ = tree_class(max_leaf_nodes=20)
        # Constructor parameters remain separate from the fitted working copy.
        # In particular, two wrappers can safely share a supplied CART tree.
        self.estimator_ = (
            deepcopy(estimator_) if checks.check_is_fitted(estimator_)
            else estimator_
        )
        if checks.check_is_fitted(self.estimator_):
            self._shrink()

    def get_params(self, deep=True):
        d = {
            "reg_param": self.reg_param,
            "estimator_": self._estimator_template,
            "shrinkage_scheme_": self.shrinkage_scheme_,
            "max_leaf_nodes": self.max_leaf_nodes,
            "random_state": self.random_state,
        }
        if deep and hasattr(self._estimator_template, "get_params"):
            d.update({
                f"estimator___{name}": value
                for name, value in self._estimator_template.get_params(deep=True).items()
            })
        return d

    def set_params(self, **params):
        # The legacy trailing underscore needs explicit parsing: sklearn's
        # first '__' split reads 'estimator___max_depth' as 'estimator'.
        prefix = "estimator___"
        nested = {
            name[len(prefix):]: value for name, value in params.items()
            if name.startswith(prefix)
        }
        direct = {
            name: value for name, value in params.items()
            if not name.startswith(prefix)
        }
        template = direct.pop("estimator_", self._estimator_template)
        if nested:
            if template is None:
                raise ValueError("Nested estimator_ parameters require an explicit estimator_")
            template = deepcopy(template)
            template.set_params(**nested)
        super().set_params(**direct)
        self._estimator_template = template
        return self

    def _fresh_estimator(self):
        if self._estimator_template is None:
            tree_class = (
                DecisionTreeRegressor if isinstance(self, RegressorMixin)
                else DecisionTreeClassifier
            )
            estimator = tree_class(max_leaf_nodes=20)
        else:
            estimator = clone(self._estimator_template)
        estimator_params = estimator.get_params(deep=False)
        overrides = {}
        # Ordinary HS CV historically applies its leaf cap only to the
        # default tree, not to an explicitly configured estimator.
        override_leaf_cap = (
            not hasattr(self, "reg_param_list")
            or self._estimator_template is None
        )
        if (override_leaf_cap and self.max_leaf_nodes is not None
                and "max_leaf_nodes" in estimator_params):
            overrides["max_leaf_nodes"] = self.max_leaf_nodes
        if self.random_state is not None and "random_state" in estimator_params:
            overrides["random_state"] = self.random_state
        if overrides:
            estimator.set_params(**overrides)
        return estimator

    def fit(self, X, y, sample_weight=None, *args, **kwargs):
        # remove feature_names if it exists (note: only works as keyword-arg)
        # None returned if not passed
        feature_names = kwargs.pop("feature_names", None)
        X, y, feature_names = check_fit_arguments(self, X, y, feature_names)
        if feature_names is not None:
            self.feature_names = feature_names
        self.estimator_ = self._fresh_estimator().fit(
            X, y, *args, sample_weight=sample_weight, **kwargs
        )
        self._shrink(sample_weight=sample_weight, y=y)

        # compute complexity
        if hasattr(self.estimator_, "tree_"):
            self.complexity_ = compute_tree_complexity(self.estimator_.tree_)
        elif hasattr(self.estimator_, "estimators_"):
            self.complexity_ = 0
            for t in self.estimator_.estimators_:
                if isinstance(t, np.ndarray):
                    assert t.size == 1, "multiple trees stored under tree_?"
                    t = t[0]
                self.complexity_ += compute_tree_complexity(t.tree_)
        return self

    def _shrink_tree(
        self, tree, reg_param, i=0, parent_val=None, parent_num=None, cum_sum=0,
        values_normalized=None,
    ):
        """Shrink the tree"""
        if reg_param is None:
            reg_param = 1.0
        if values_normalized is None:
            # Modern sklearn stores class fractions, older versions counts.
            # Determine the representation before rewriting any node values.
            values_normalized = bool(np.allclose(tree.value.sum(axis=(1, 2)), 1))
        left = tree.children_left[i]
        right = tree.children_right[i]
        is_leaf = left == right
        n_samples = tree.weighted_n_node_samples[i]
        if (isinstance(self, RegressorMixin)
                or isinstance(self.estimator_, GradientBoostingClassifier)
                or values_normalized):
            val = deepcopy(tree.value[i, :, :])
        else:  # If classification, normalize to probability vector
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
                cum_sum=deepcopy(cum_sum),
                values_normalized=values_normalized,
            )
            self._shrink_tree(
                tree,
                reg_param,
                right,
                parent_val=val,
                parent_num=n_samples,
                cum_sum=deepcopy(cum_sum),
                values_normalized=values_normalized,
            )

            # edit the non-leaf nodes for later visualization (doesn't effect predictions)

        return tree

    def _shrink(self, sample_weight=None, y=None):
        self.gcv_results_ = None
        if isinstance(self.reg_param, str) and self.reg_param == "gcv":
            if not isinstance(self, RegressorMixin):
                raise ValueError("Automatic GCV HS supports regression only")
            if self.shrinkage_scheme_ != "node_based":
                raise ValueError("Automatic GCV HS requires node_based shrinkage")
            self.reg_param_, self.gcv_results_ = select_hs_reg_param(
                self.estimator_, sample_weight=sample_weight, y=y
            )
            apply_node_based_hs(self.estimator_.tree_, self.reg_param_)
            return
        self.reg_param_ = 1.0 if self.reg_param is None else self.reg_param
        if hasattr(self.estimator_, "tree_"):
            self._shrink_tree(self.estimator_.tree_, self.reg_param)
        elif hasattr(self.estimator_, "estimators_"):
            for t in self.estimator_.estimators_:
                if isinstance(t, np.ndarray):
                    assert t.size == 1, "multiple trees stored under tree_?"
                    t = t[0]
                self._shrink_tree(t.tree_, self.reg_param)

    def predict(self, X, *args, **kwargs):
        preds = self.estimator_.predict(X, *args, **kwargs)
        if hasattr(self, "classes_") and hasattr(self.estimator_, "classes_"):
            return np.array([self.classes_[int(i)] for i in preds])
        else:
            return preds

    def predict_proba(self, X, *args, **kwargs):
        if hasattr(self.estimator_, "predict_proba"):
            probabilities = np.clip(
                self.estimator_.predict_proba(X, *args, **kwargs), 0, 1
            )
            totals = probabilities.sum(axis=1, keepdims=True)
            return np.divide(
                probabilities, totals,
                out=np.full_like(probabilities, 1 / probabilities.shape[1]),
                where=totals > 0,
            )
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
        estimator_: BaseEstimator | None = None,
        reg_param: float | str = 1,
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


class HSTreeClassifier(ClassifierMixin, HSTree):
    def __init__(
        self,
        estimator_: BaseEstimator | None = None,
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
    y_true = np.random.binomial(n=1, p=0.5, size=100)

    y_pred_good = y_true
    y_pred_bad = np.random.uniform(0, 1, 100)

    score_good = scorer(y_true, y_pred_good)
    score_bad = scorer(y_true, y_pred_bad)

    if score_good > score_bad:
        return np.argmax
    elif score_good < score_bad:
        return np.argmin


def _hs_cv_params(estimator, deep):
    params = HSTree.get_params(estimator, deep=deep)
    params.pop("reg_param")
    params.pop("random_state")
    params.update(
        reg_param_list=estimator.reg_param_list,
        cv=estimator.cv,
        scoring=estimator.scoring,
    )
    return params


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
        super().__init__(
            estimator_, reg_param=None, max_leaf_nodes=max_leaf_nodes,
            shrinkage_scheme_=shrinkage_scheme_,
        )
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
        return _hs_cv_params(self, deep)

    def fit(self, X, y, *args, **kwargs):
        self.scores_ = [[] for _ in self.reg_param_list]
        scorer = kwargs.get("scoring", log_loss)
        kf = KFold(n_splits=self.cv)
        for train_index, test_index in kf.split(X):
            X_out, y_out = X[test_index, :], y[test_index]
            X_in, y_in = X[train_index, :], y[train_index]
            base_est = self._fresh_estimator()
            base_est.fit(X_in, y_in)
            for i, reg_param in enumerate(self.reg_param_list):
                est_hs = HSTreeClassifier(base_est, reg_param)
                est_hs.fit(X_in, y_in, *args, **kwargs)
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
        super().__init__(
            estimator_, reg_param=None, max_leaf_nodes=max_leaf_nodes,
            shrinkage_scheme_=shrinkage_scheme_,
        )
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
        return _hs_cv_params(self, deep)

    def fit(self, X, y, *args, **kwargs):
        self.scores_ = [[] for _ in self.reg_param_list]
        kf = KFold(n_splits=self.cv)
        scorer = kwargs.get("scoring", mean_squared_error)
        for train_index, test_index in kf.split(X):
            X_out, y_out = X[test_index, :], y[test_index]
            X_in, y_in = X[train_index, :], y[train_index]
            base_est = self._fresh_estimator()
            base_est.fit(X_in, y_in)
            for i, reg_param in enumerate(self.reg_param_list):
                est_hs = HSTreeRegressor(base_est, reg_param)
                est_hs.fit(X_in, y_in)
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
