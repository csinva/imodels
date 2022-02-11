import copy
from typing import List

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_score

from imodels.tree.gosdt.pygosdt import OptimalTreeClassifier


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


class HSOptimalTreeClassifier(BaseEstimator):
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
        # self.tree_ = estimator_.tree_
        self.shrinkage_scheme_ = shrinkage_scheme_

    def _calc_probs(self, node):
        lbls = np.array([float(l) for l in node["labels"]]) if "labels" in node else np.array(
            [float(node['prediction'])])
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
        if not hasattr(self.estimator_, "tree_"):
            self.estimator_.fit(X, y)
        self.impute_nodes(X, y)
        self.shrink_tree()

    def predict(self, X):
        return self.estimator_.predict(X)

    def score(self, X, y, weight=None):
        self.estimator_.score(X, y, weight)

    @property
    def complexity_(self):
        return self.estimator_.complexity_


class HSOptimalTreeClassifierCV(HSOptimalTreeClassifier):
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
            est = HSOptimalTreeClassifier(opt, reg_param)
            cv_scores = cross_val_score(est, X, y, cv=self.cv, scoring=self.scoring)
            self.scores_.append(np.mean(cv_scores))
        self.reg_param = self.reg_param_list[np.argmax(self.scores_)]
        super().fit(X=X, y=y)
