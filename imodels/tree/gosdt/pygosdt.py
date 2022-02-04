import json
import warnings
import copy
from typing import List

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import validation

from imodels import GreedyTreeClassifier
from imodels.tree.gosdt.pygosdt_helper import TreeClassifier
from imodels.util import rule

try:
    import gosdt

    gosdt_supported = True
except ImportError:
    gosdt_supported = False


class OptimalTreeClassifier(GreedyTreeClassifier if not gosdt_supported else BaseEstimator):
    def __init__(self,
                 balance=False,
                 cancellation=True,
                 look_ahead=True,
                 similar_support=True,
                 feature_exchange=True,
                 continuous_feature_exchange=True,
                 rule_list=False,
                 diagnostics=False,
                 verbose=True,
                 regularization=0.05,
                 uncertainty_tolerance=0.0,
                 upperbound=0.0,
                 model_limit=1,
                 precision_limit=0,
                 stack_limit=0,
                 tile_limit=0,
                 time_limit=3000,
                 worker_limit=1,
                 random_state=None,
                 costs="",
                 model="",
                 profile="",
                 timing="",
                 trace="",
                 tree="",
                 tree_=None,
                 feature_names_=None):
        super().__init__()
        self.balance = balance
        self.cancellation = cancellation
        self.look_ahead = look_ahead
        self.similar_support = similar_support
        self.feature_exchange = feature_exchange
        self.continuous_feature_exchange = continuous_feature_exchange
        self.rule_list = rule_list
        self.diagnostics = diagnostics
        self.verbose = verbose
        self.regularization = regularization
        self.uncertainty_tolerance = uncertainty_tolerance
        self.upperbound = upperbound
        self.model_limit = model_limit
        self.precision_limit = precision_limit
        self.stack_limit = stack_limit
        self.tile_limit = tile_limit
        self.time_limit = time_limit
        self.worker_limit = worker_limit
        self.costs = costs
        self.model = model
        self.profile = profile
        self.timing = timing
        self.trace = trace
        self.tree = tree
        self.tree_type = 'gosdt'
        self.random_state = random_state
        self.tree_ = tree_
        self.feature_names_ = feature_names_
        if random_state is not None:
            np.random.seed(random_state)

    def load(self, path):
        """
        Parameters
        ---
        path : string
            path to a JSON file representing a model
        """
        with open(path, 'r') as model_source:
            result = model_source.read()
        result = json.loads(result)
        self.tree_ = TreeClassifier(result[0])

    def fit(self, X, y, feature_names=None):
        """
        Parameters
        ---
        X : matrix-like, shape = [n_samples, m_features]
            matrix containing the training samples and features
        y : array-like, shape = [n_samples, 1]
            column containing the correct label for each sample in X

        Modifies
        ---
        trains the model so that this model instance is ready for prediction
        """
        try:
            import gosdt

            if not isinstance(X, pd.DataFrame):
                self.feature_names_ = list(rule.get_feature_dict(X.shape[1], feature_names).keys())
                X = pd.DataFrame(X, columns=self.feature_names_)
            else:
                self.feature_names_ = X.columns

            if not isinstance(y, pd.DataFrame):
                y = pd.DataFrame(y, columns=['target'])

            # gosdt extension expects serialized CSV, which we generate via pandas
            dataset_with_target = pd.concat((X, y), axis=1)

            # Perform C++ extension calls to train the model
            configuration = self._get_configuration()
            gosdt.configure(json.dumps(configuration, separators=(',', ':')))
            result = gosdt.fit(dataset_with_target.to_csv(index=False))

            result = json.loads(result)
            self.tree_ = TreeClassifier(result[0])

            # Record the training time, number of iterations, and graph size required
            self.time_ = gosdt.time()
            self.iterations_ = gosdt.iterations()
            self.size_ = gosdt.size()

        except ImportError:

            warnings.warn(
                "Should install gosdt extension. On x86_64 linux or macOS: "
                "'pip install gosdt'. On other platforms, see "
                "https://github.com/keyan3/GeneralizedOptimalSparseDecisionTrees. "
                "Defaulting to Non-optimal DecisionTreeClassifier."
            )

            # dtree = DecisionTreeClassifierWithComplexity()
            # dtree.fit(X, y)
            # self.tree_ = dtree
            super().fit(X, y, feature_names=feature_names)
            self.tree_type = 'dt'

        self.impute_nodes(np.array(X), np.array(y))
        return self

    def predict(self, X):
        """
        Parameters
        ---
        X : matrix-like, shape = [n_samples, m_features]
            a matrix where each row is a sample to be predicted and each column is a feature to
            be used for prediction

        Returns
        ---
        array-like, shape = [n_samples, 1] : a column where each element is the prediction
            associated with each row
        """
        validation.check_is_fitted(self)
        if self.tree_type == 'gosdt':
            if type(self.tree_) is TreeClassifier and not isinstance(X, pd.DataFrame):
                X = pd.DataFrame(X, columns=self.feature_names_)
            return self.tree_.predict(X)
        else:
            return super().predict(X)

    def predict_proba_old(self, X):
        validation.check_is_fitted(self)
        if self.tree_type == 'gosdt':
            if type(self.tree_) is TreeClassifier and not isinstance(X, pd.DataFrame):
                X = pd.DataFrame(X, columns=self.feature_names_)
            probs = np.expand_dims(self.tree_.confidence(X), axis=1)
            return np.hstack((1 - probs, probs))
        else:
            return super().predict_proba(X)

    def predict_proba(self, X):
        probs = []
        for i in range(X.shape[0]):
            sample = X[i, ...]
            node = self.tree_.__find_leaf__(sample)
            probs.append([1 - node["probs"], node["probs"]])
        return np.array(probs)

    def score(self, X, y, weight=None):
        """
        Parameters
        ---
        X : matrix-like, shape = [n_samples, m_features]
            an n-by-m matrix of sample and their features
        y : array-like, shape = [n_samples,]
            an n-by-1 column of labels associated with each sample
        weight : shape = [n_samples,]
            an n-by-1 column of weights to apply to each sample's misclassification

        Returns
        ---
        real number : the accuracy produced by applying this model overthe given dataset, with
            optionals for weighted accuracy
        """
        validation.check_is_fitted(self)
        if type(self.tree_) is TreeClassifier:
            if not isinstance(X, pd.DataFrame):
                X = pd.DataFrame(X, columns=self.feature_names_)
            return self.tree_.score(X, y, weight=weight)
        else:
            return self.tree_.score(X, y, sample_weight=weight)

    def _calc_probs(self, node):
        node['probs'] = np.mean(np.array(node['labels']) == 1) if "labels" in node else node['prediction']
        node['n_obs'] = len(node.get('labels', []))
        if "prediction" in node:
            node['prediction'] = np.round(node['probs'])
            return
        self._calc_probs(node['true'])
        self._calc_probs(node['false'])

    def impute_nodes(self, X, y):
        """
        Returns
        ---
        the leaf by which this sample would be classified
        """
        source_node = self.tree_.source
        for i in range(len(y)):
            sample, label = X[i, ...], y[i]
            _add_label(source_node, label)
            nodes = [source_node]
            while len(nodes) > 0:
                node = nodes.pop()
                if "prediction" in node:
                    continue
                else:
                    value = sample[node["feature"]]
                    reference = node["reference"]
                    relation = node["relation"]
                    if relation == "==":
                        is_true = value == reference
                    elif relation == ">=":
                        is_true = value >= reference
                    elif relation == "<=":
                        is_true = value <= reference
                    elif relation == "<":
                        is_true = value < reference
                    elif relation == ">":
                        is_true = value > reference
                    else:
                        raise "Unsupported relational operator {}".format(node["relation"])

                    next_node = node['true'] if is_true else node['false']
                    _add_label(next_node, label)
                    nodes.append(next_node)
        self._calc_probs(source_node)
        self.tree_.source = source_node

    def __len__(self):
        """
        Returns
        ---
        natural number : The number of terminal nodes present in this tree
        """
        validation.check_is_fitted(self)
        if type(self.tree_) is TreeClassifier:
            return len(self.tree_)
        else:
            warnings.warn("Using DecisionTreeClassifier due to absence of gosdt package. "
                          "DecisionTreeClassifier does not have this method.")
            return None

    def leaves(self):
        """
        Returns
        ---
        natural number : The number of terminal nodes present in this tree
        """
        validation.check_is_fitted(self)
        if type(self.tree_) is TreeClassifier:
            return self.tree_.leaves()
        else:
            return self.tree_.get_n_leaves()

    def nodes(self):
        """
        Returns
        ---
        natural number : The number of nodes present in this tree
        """
        validation.check_is_fitted(self)
        if type(self.tree_) is TreeClassifier:
            return self.tree_.nodes()
        else:
            warnings.warn("Using DecisionTreeClassifier due to absence of gosdt package. "
                          "DecisionTreeClassifier does not have this method.")
            return None

    def max_depth(self):
        """
        Returns
        ---
        natural number : the length of the longest decision path in this tree. A single-node tree
            will return 1.
        """
        validation.check_is_fitted(self)
        if type(self.tree_) is TreeClassifier:
            return self.tree_.maximum_depth()
        else:
            return self.tree_.get_depth()

    def latex(self):
        """
        Note
        ---
        This method doesn't work well for label headers that contain underscores due to underscore
            being a reserved character in LaTeX

        Returns
        ---
        string : A LaTeX string representing the model
        """
        validation.check_is_fitted(self)
        if type(self.tree_) is TreeClassifier:
            return self.tree_.latex()
        else:
            warnings.warn("Using DecisionTreeClassifier due to absence of gosdt package. "
                          "DecisionTreeClassifier does not have this method.")
            return None

    def json(self):
        """
        Returns
        ---
        string : A JSON string representing the model
        """
        validation.check_is_fitted(self)
        if type(self.tree_) is TreeClassifier:
            return self.tree_.json()
        else:
            warnings.warn("Using DecisionTreeClassifier due to absence of gosdt package. "
                          "DecisionTreeClassifier does not have this method.")
            return None

    def _get_configuration(self):
        return {
            "balance": self.balance,
            "cancellation": self.cancellation,
            "look_ahead": self.look_ahead,
            "similar_support": self.similar_support,
            "feature_exchange": self.feature_exchange,
            "continuous_feature_exchange": self.continuous_feature_exchange,
            "rule_list": self.rule_list,

            "diagnostics": self.diagnostics,
            "verbose": self.verbose,

            "regularization": self.regularization,
            "uncertainty_tolerance": self.uncertainty_tolerance,
            "upperbound": self.upperbound,

            "model_limit": self.model_limit,
            "precision_limit": self.precision_limit,
            "stack_limit": self.stack_limit,
            "tile_limit": self.tile_limit,
            "time_limit": self.time_limit,
            "worker_limit": self.worker_limit,

            "costs": self.costs,
            "model": self.model,
            "profile": self.profile,
            "timing": self.timing,
            "trace": self.trace,
            "tree": self.tree
        }

    @property
    def complexity_(self):
        return (self.nodes() - 1) / 2

    def get_params(self, deep=True):
        p = super().get_params(deep)
        p['tree_'] = self.tree_
        return p


def shrink_node(node, reg_param, parent_val, parent_num, cum_sum, scheme, constant):
    """Shrink the tree
    """

    left = node.get("false", None)
    right = node.get("true", None)
    is_leaf = "prediction" in node
    # if self.prediction_task == 'regression':
    val = node["probs"]
    is_root = parent_val is None and parent_num is None
    n_samples = node['n_obs'] if (scheme != "leaf_based" or is_root) else parent_num

    if is_root:
        val_new = val

    else:
        reg_term = reg_param if scheme == "constant" else reg_param / parent_num

        val_new = (val - parent_val) / (1 + reg_term)

    cum_sum += val_new

    if is_leaf:
        if scheme == "leaf_based":
            v = constant + (val - constant) / (1 + reg_param / node.n_obs)
            node["probs"] = v
        else:
            # print(f"Changing {val} to {cum_sum}")
            node["probs"] = cum_sum

    else:
        shrink_node(left, reg_param, val, parent_num=n_samples, cum_sum=cum_sum, scheme=scheme, constant=constant)
        shrink_node(right, reg_param, val, parent_num=n_samples, cum_sum=cum_sum, scheme=scheme, constant=constant)

    return node


def _add_label(node, val):
    if "labels" in node:
        node['labels'].append(val)
        return
    node['labels'] = [val]


class ShrunkOptimalTreeClassifier(BaseEstimator):
    def __init__(self, estimator_: OptimalTreeClassifier, reg_param: float = 1, shrinkage_scheme_: str = 'node_based'):
        """
        Params
        ------
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
        # print('est', estimator_)
        self.estimator_ = estimator_
        self.tree_ = estimator_.tree_
        self.shrinkage_scheme_ = shrinkage_scheme_

    def _calc_probs(self, node):
        lbls = np.array([float(l) for l in node["labels"]]) if "labels" in node else np.array([float(node['prediction'])])
        node['probs'] = np.mean(lbls == 1)
        node['n_obs'] = len(node.get('labels', []))
        if "prediction" in node:
            node['prediction'] = np.round(node['probs'])
            return
        self._calc_probs(node['true'])
        self._calc_probs(node['false'])

    def impute_nodes(self, X, y):
        """
        Returns
        ---
        the leaf by which this sample would be classified
        """
        source_node = self.estimator_.tree_.source
        for i in range(len(y)):
            sample, label = X[i, ...], y[i]
            _add_label(source_node, label)
            nodes = [source_node]
            while len(nodes) > 0:
                node = nodes.pop()
                if "prediction" in node:
                    continue
                else:
                    value = sample[node["feature"]]
                    reference = node["reference"]
                    relation = node["relation"]
                    if relation == "==":
                        is_true = value == reference
                    elif relation == ">=":
                        is_true = value >= reference
                    elif relation == "<=":
                        is_true = value <= reference
                    elif relation == "<":
                        is_true = value < reference
                    elif relation == ">":
                        is_true = value > reference
                    else:
                        raise "Unsupported relational operator {}".format(node["relation"])

                    next_node = node['true'] if is_true else node['false']
                    _add_label(next_node, label)
                    nodes.append(next_node)

        self._calc_probs(source_node)
        self.estimator_.tree_.source = source_node

    # def fit(self, *args, **kwargs):
    #     X = kwargs['X'] if "X" in kwargs else args[0]
    #     y = kwargs['y'] if "y" in kwargs else args[1]

    def shrink_tree(self):
        root = self.estimator_.tree_.source
        shrink_node(root, self.reg_param, None, None, 0, self.shrinkage_scheme_, 0)

    def predict_proba(self, X):
        probs = []
        for i in range(X.shape[0]):
            sample = X[i, ...]
            node = self.estimator_.tree_.__find_leaf__(sample)
            probs.append([1 - node["probs"], node["probs"]])
        return np.array(probs)

    def fit(self, *args, **kwargs):
        X = kwargs['X'] if "X" in kwargs else args[0]
        y = kwargs['y'] if "y" in kwargs else args[1]
        self.impute_nodes(X, y)
        self.shrink_tree()

    def predict(self, X):
        return self.estimator_.predict(X)

    def score(self, X, y, weight=None):
        self.estimator_.score(X, y, weight)

    @property
    def complexity_(self):
        return self.estimator_.complexity_


class ShrunkOptimalTreeClassifierCV(ShrunkOptimalTreeClassifier):
    def __init__(self, estimator_: OptimalTreeClassifier,
                 reg_param_list: List[float] = [0.1, 1, 10, 50, 100, 500], shrinkage_scheme_: str = 'node_based',
                 cv: int = 3, scoring="accuracy", *args, **kwargs):
        """Note: args, kwargs are not used but left so that imodels-experiments can still pass redundant args
        """
        super().__init__(estimator_, reg_param=None)
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
        opt = copy.deepcopy(self.estimator_)
        for reg_param in self.reg_param_list:
            est = ShrunkOptimalTreeClassifier(opt, reg_param)
            cv_scores = cross_val_score(est, X, y, cv=self.cv, scoring=self.scoring)
            self.scores_.append(np.mean(cv_scores))
        self.reg_param = self.reg_param_list[np.argmax(self.scores_)]
        super().fit(X=X, y=y)


def main():
    # n = 100
    # p = 10
    #
    # X = np.random.normal(size=(n, p))
    # y = X[..., 1] + X[..., 2] + X[..., 3] + np.random.normal(0.5, 1)
    # y = np.round(np.abs(y/np.max(y)), 0)
    iris = sklearn.datasets.load_iris()
    idx = np.logical_or(iris.target == 0, iris.target == 1)
    X, y = iris.data[idx, ...], iris.target[idx]
    tree = DecisionTreeClassifier()
    tree.fit(X, y)
    clf = OptimalTreeClassifier()
    clf.fit(X, y)
    s_clf = ShrunkOptimalTreeClassifierCV(copy.deepcopy(clf))
    s_clf.fit(X, y)
    p = clf.predict_proba(X)
    print(np.unique(p))
    ps = s_clf.predict_proba(X)
    print(np.unique(ps))
    pass


if __name__ == '__main__':
    main()
