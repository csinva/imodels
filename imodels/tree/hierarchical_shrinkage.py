from copy import deepcopy
from typing import List

import numpy as np
import sklearn
from sklearn import datasets
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin, clone
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, \
    export_text
from sklearn.utils import check_X_y
from sklearn.ensemble import GradientBoostingClassifier

from imodels.util import checks
from imodels.util.arguments import check_fit_arguments
from imodels.util.tree import compute_tree_complexity

# leading and traiing undescores
# https://github.com/rasbt/python-machine-learning-book/blob/master/faq/underscore-convention.md
# developer guideline
# https://scikit-learn.org/stable/developers/contributing.html#estimated-attributes

# https://scikit-learn.org/stable/developers/contributing.html


class HSTree(BaseEstimator):
    def __init__(self, estimator=None,
                reg_param: float = 1, shrinkage_scheme_: str = 'node_based'):
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
        """
        super().__init__()
        self.reg_param = reg_param
        self.estimator = estimator
        self.shrinkage_scheme_ = shrinkage_scheme_


    def _validate_estimator(self, default=None):
        """Check the base estimator.

        Sets the `estimator_` attributes.
        """
        if self.estimator is not None:
            self.estimator_ = self.estimator
        else:
            self.estimator_ = default


    def fit(self, X, y, sample_weight=None, *args, **kwargs):

        self._validate_estimator()

        if checks.check_is_fitted(self.estimator_):
            self._shrink()

        else:
            # remove feature_names if it exists (note: only works as keyword-arg)
            feature_names = kwargs.pop('feature_names', None)  # None returned if not passed
            X, y, feature_names = check_fit_arguments(self, X, y, feature_names)
            X, y = check_X_y(X,y)
            self.estimator_.fit(X, y, *args, sample_weight=sample_weight, **kwargs)
            self._shrink()

        # compute complexity
        if hasattr(self.estimator_, 'tree_'):
            self.complexity_ = compute_tree_complexity(self.estimator_.tree_)
        elif hasattr(self.estimator_, 'estimators_'):
            self.complexity_ = 0
            for i in range(len(self.estimator_.estimators_)):
                t = deepcopy(self.estimator_.estimators_[i])
                if isinstance(t, np.ndarray):
                    assert t.size == 1, 'multiple trees stored under tree_?'
                    t = t[0]
                self.complexity_ += compute_tree_complexity(t.tree_)

        return self

    def _shrink_tree(self, tree, reg_param, i=0, parent_val=None, parent_num=None, cum_sum=0):
        """Shrink the tree
        """
        if reg_param is None:
            reg_param = 1.0
        left = tree.children_left[i]
        right = tree.children_right[i]
        is_leaf = left == right
        n_samples = tree.weighted_n_node_samples[i]
        if isinstance(self, RegressorMixin) or isinstance(self.estimator_, GradientBoostingClassifier):
            val = deepcopy(tree.value[i, :, :])
        else: # If classification, normalize to probability vector
            val = tree.value[i, :, :] / n_samples

        # Step 1: Update cum_sum
        # if root
        if parent_val is None and parent_num is None:
            cum_sum = val

        # if has parent
        else:
            if self.shrinkage_scheme_ == 'node_based':
                val_new = (val - parent_val) / (1 + reg_param / parent_num)
            elif self.shrinkage_scheme_ == 'constant':
                val_new = (val - parent_val) / (1 + reg_param)
            else: # leaf_based
                val_new = 0
            cum_sum += val_new

        # Step 2: Update node values
        if self.shrinkage_scheme_ == 'node_based' or self.shrinkage_scheme_ == 'constant':
            tree.value[i, :, :] = cum_sum
        else: # leaf_based
            if is_leaf: # update node values if leaf_based
                root_val = tree.value[0, :, :]
                tree.value[i, :, :] = root_val + (val - root_val) / (1 + reg_param / n_samples)
            else:
                tree.value[i, :, :] = val

                # Step 3: Recurse if not leaf
        if not is_leaf:
            self._shrink_tree(tree, reg_param, left,
                                parent_val=val, parent_num=n_samples, cum_sum=deepcopy(cum_sum))
            self._shrink_tree(tree, reg_param, right,
                                parent_val=val, parent_num=n_samples, cum_sum=deepcopy(cum_sum))

                # edit the non-leaf nodes for later visualization (doesn't effect predictions)

        return tree

    def _shrink(self):
        if hasattr(self.estimator_, 'tree_'):
            self._shrink_tree(self.estimator_.tree_, self.reg_param)
        elif hasattr(self.estimator_, 'estimators_'):
            for t in self.estimator_.estimators_:
                if isinstance(t, np.ndarray):
                    assert t.size == 1, 'multiple trees stored under tree_?'
                    t = t[0]
                self._shrink_tree(t.tree_, self.reg_param)

    def predict(self, X, *args, **kwargs):
        return self.estimator_.predict(X, *args, **kwargs)

    def predict_proba(self, X, *args, **kwargs):
        if hasattr(self.estimator_, 'predict_proba'):
            return self.estimator_.predict_proba(X, *args, **kwargs)
        else:
            return NotImplemented

    def score(self, X, y, *args, **kwargs):
        if hasattr(self.estimator_, 'score'):
            return self.estimator_.score(X, y, *args, **kwargs)
        else:
            return NotImplemented

    def __str__(self):
        try:
            # sklearn.utils.validation.check_is_fitted(self)
            s = '> ------------------------------\n'
            s += '> Decision Tree with Hierarchical Shrinkage\n'
            s += '> \tPrediction is made by looking at the value in the appropriate leaf of the tree\n'
            s += '> ------------------------------' + '\n'
            if hasattr(self, 'feature_names') and self.feature_names is not None:
                return s + export_text(self.estimator_, feature_names=self.feature_names, show_weights=True)
            else:
                return s + export_text(self.estimator_, show_weights=True)
        except:
            return self.__class__.__name__

    def __repr__(self):
        try:
            # sklearn.utils.validation.check_is_fitted(self)
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
        except :
            return self.__class__.__name__

class HSTreeClassifier(HSTree, ClassifierMixin):
    def __init__(self, estimator=None,
                    reg_param: float = 1, shrinkage_scheme_: str = 'node_based'):
        super().__init__(estimator=estimator,
                            reg_param=reg_param,
                            shrinkage_scheme_=shrinkage_scheme_,
                            )

    def _validate_estimator(self):
        """Check the estimator and set the estimator_ attribute."""
        super()._validate_estimator(default=DecisionTreeClassifier(max_leaf_nodes=20))

class HSTreeRegressor(HSTree, RegressorMixin):
    def __init__(self, estimator=None,
                    reg_param: float = 1, shrinkage_scheme_: str = 'node_based'):
        super().__init__(estimator=estimator,
                            reg_param=reg_param,
                            shrinkage_scheme_=shrinkage_scheme_,
                            )
    def _validate_estimator(self):
        """Check the estimator and set the estimator_ attribute."""
        super()._validate_estimator(default=DecisionTreeRegressor(max_leaf_nodes=20))


class HSTreeClassifierCV(HSTreeClassifier):
    def __init__(self, estimator=None,
                    reg_param_list: List[float] = [0.1, 1, 10, 50, 100, 500],
                    shrinkage_scheme_: str = 'node_based',
                    cv: int = 3, scoring=None):
        """Cross-validation is used to select the best regularization parameter for hierarchical shrinkage.

        Params
        ------
        estimator_
            Sklearn estimator (already initialized).
            If no estimator_ is passed, sklearn decision tree is used

        reg_param_list : list
            If estimator is None, then max_leaf_nodes is passed to the default decision tree

        args, kwargs
            Note: args, kwargs are not used but left so that imodels-experiments can still pass redundant args.
        """
        super().__init__(estimator, reg_param=None)
        self.reg_param_list = np.array(reg_param_list)
        self.cv = cv
        self.scoring = scoring
        self.shrinkage_scheme_ = shrinkage_scheme_
        # print('estimator', self.estimator_,
        #       'checks.check_is_fitted(estimator)', checks.check_is_fitted(self.estimator_))
        # if checks.check_is_fitted(self.estimator_):
        #     raise Warning('Passed an already fitted estimator,'
        #                   'but shrinking not applied until fit method is called.')


    def fit(self, X, y, *args, **kwargs):
        self.scores_ = []
        for reg_param in self.reg_param_list:
            est = HSTreeClassifier(deepcopy(self.estimator), reg_param)
            cv_scores = cross_val_score(est, X, y, cv=self.cv, scoring=self.scoring)
            self.scores_.append(np.mean(cv_scores))
        self.reg_param = self.reg_param_list[np.argmax(self.scores_)]
        super().fit(X=X, y=y, *args, **kwargs)
        return self


class HSTreeRegressorCV(HSTreeRegressor):
    def __init__(self, estimator=None,
                    reg_param_list: List[float] = [0.1, 1, 10, 50, 100, 500],
                    shrinkage_scheme_: str = 'node_based',
                    cv: int = 3, scoring=None):
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
        super().__init__(estimator, reg_param=None)
        self.reg_param_list = np.array(reg_param_list)
        self.cv = cv
        self.scoring = scoring
        self.shrinkage_scheme_ = shrinkage_scheme_
        # print('estimator', self.estimator_,
        #       'checks.check_is_fitted(estimator)', checks.check_is_fitted(self.estimator_))
        # if checks.check_is_fitted(self.estimator_):
        #     raise Warning('Passed an already fitted estimator,'
        #                   'but shrinking not applied until fit method is called.')

    def fit(self, X, y, *args, **kwargs):
        self.scores_ = []
        for reg_param in self.reg_param_list:
            est = HSTreeRegressor(deepcopy(self.estimator), reg_param)
            cv_scores = cross_val_score(est, X, y, cv=self.cv, scoring=self.scoring)
            self.scores_.append(np.mean(cv_scores))
        self.reg_param = self.reg_param_list[np.argmax(self.scores_)]
        super().fit(X=X, y=y, *args, **kwargs)
        return self


if __name__ == '__main__':
    np.random.seed(15)
    # X, y = datasets.fetch_california_housing(return_X_y=True)  # regression
    # X, y = datasets.load_breast_cancer(return_X_y=True)  # binary classification
    X, y = datasets.load_diabetes(return_X_y=True)  # regression
    # X = np.random.randn(500, 10)
    # y = (X[:, 0] > 0).astype(float) + (X[:, 1] > 1).astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=10
    )
    print('X.shape', X.shape)
    print('ys', np.unique(y_train))

    # m = HSTree(estimator=DecisionTreeClassifier(), reg_param=0.1)
    # m = DecisionTreeClassifier(max_leaf_nodes = 20,random_state=1, max_features=None)
    m = DecisionTreeRegressor(random_state=42, max_leaf_nodes=20)
    # print('best alpha', m.reg_param)
    m.fit(X_train, y_train)
    # m.predict_proba(X_train)  # just run this
    print('score', r2_score(y_test, m.predict(X_test)))
    print('running again....')

    # x = DecisionTreeRegressor(random_state = 42, ccp_alpha = 0.3)
    # x.fit(X_train,y_train)

    # m = HSTree(estimator=DecisionTreeRegressor(random_state=42, max_features=None), reg_param=10)
    # m = HSTree(estimator=DecisionTreeClassifier(random_state=42, max_features=None), reg_param=0)
    m = HSTreeClassifierCV(estimator=DecisionTreeRegressor(max_leaf_nodes=10, random_state=1),
                           shrinkage_scheme_='node_based',
                           reg_param_list=[0.1, 1, 2, 5, 10, 25, 50, 100, 500])
    print(m)
    # m = ShrunkTreeCV(estimator=DecisionTreeClassifier())

    # m = HSTreeClassifier(estimator = GradientBoostingClassifier(random_state = 10),reg_param = 5)
    m.fit(X_train, y_train)
    print('best alpha', m.reg_param)
    # m.predict_proba(X_train)  # just run this
    # print('score', m.score(X_test, y_test))
    print('score', r2_score(y_test, m.predict(X_test)))
