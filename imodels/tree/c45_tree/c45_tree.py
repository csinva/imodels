"""Modified from https://github.com/RaczeQ/scikit-learn-C4.5-tree-classifier
References
----------
.. [1] https://en.wikipedia.org/wiki/Decision_tree_learning
.. [2] https://en.wikipedia.org/wiki/C4.5_algorithm
"""
import copy
from copy import deepcopy
from typing import List
from xml.dom import minidom
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import cross_val_score
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from imodels.util.arguments import check_fit_arguments

from ..c45_tree.c45_utils import decision, is_numeric_feature, gain, gain_ratio, get_best_split, \
    set_as_leaf_node
from ...util.data_util import get_clean_dataset


def _add_label(node, label):
    if hasattr(node, "labels"):
        node.labels.append(label)
        return
    node.labels = [label]
    return


def _get_next_node(children, att):
    for child in children:
        is_equal = child.getAttribute("flag") == "m" and child.getAttribute("feature") == att
        is_less_than = child.getAttribute("flag") == "l" and float(att) < float(child.getAttribute("feature"))
        is_greater_than = child.getAttribute("flag") == "r" and float(att) >= float(child.getAttribute("feature"))
        if is_equal or is_less_than or is_greater_than:
            return child


def shrink_node(node, reg_param, parent_val, parent_num, cum_sum, scheme, constant):
    """Shrink the tree
    """

    is_leaf = not node.hasChildNodes()
    # if self.prediction_task == 'regression':
    val = node.nodeValue
    is_root = parent_val is None and parent_num is None
    n_samples = len(node.labels) if (scheme != "leaf_based" or is_root) else parent_num

    if is_root:
        val_new = val

    else:
        reg_term = reg_param if scheme == "constant" else reg_param / parent_num

        val_new = (val - parent_val) / (1 + reg_term)

    cum_sum += val_new

    if is_leaf:
        if scheme == "leaf_based":
            v = constant + (val - constant) / (1 + reg_param / node.n_obs)
            node.nodeValue = v
        else:
            node.nodeValue = cum_sum

    else:
        for c in node.childNodes:
            shrink_node(c, reg_param, val, parent_num=n_samples, cum_sum=cum_sum, scheme=scheme, constant=constant)

    return node


def _add_label(node, label):
    if hasattr(node, "labels"):
        node.labels.append(label)
        return
    node.labels = [label]
    return


def _get_next_node(children, att):
    for child in children:
        is_equal = child.getAttribute("flag") == "m" and child.getAttribute("feature") == att
        is_less_than = child.getAttribute("flag") == "l" and float(att) < float(child.getAttribute("feature"))
        is_greater_than = child.getAttribute("flag") == "r" and float(att) >= float(child.getAttribute("feature"))
        if is_equal or is_less_than or is_greater_than:
            return child


def shrink_node(node, reg_param, parent_val, parent_num, cum_sum, scheme, constant):
    """Shrink the tree
    """

    is_leaf = not node.hasChildNodes()
    # if self.prediction_task == 'regression':
    val = node.nodeValue
    is_root = parent_val is None and parent_num is None
    n_samples = len(node.labels) if (scheme != "leaf_based" or is_root) else parent_num

    if is_root:
        val_new = val

    else:
        reg_term = reg_param if scheme == "constant" else reg_param / parent_num

        val_new = (val - parent_val) / (1 + reg_term)

    cum_sum += val_new

    if is_leaf:
        if scheme == "leaf_based":
            v = constant + (val - constant) / (1 + reg_param / node.n_obs)
            node.nodeValue = v
        else:
            node.nodeValue = cum_sum

    else:
        for c in node.childNodes:
            shrink_node(c, reg_param, val, parent_num=n_samples, cum_sum=cum_sum, scheme=scheme, constant=constant)

    return node


class C45TreeClassifier(BaseEstimator, ClassifierMixin):
    """A C4.5 tree classifier.

    Parameters
    ----------
    max_rules : int, optional (default=None)
        Maximum number of split nodes allowed in the tree
    """

    def __init__(self, max_rules: int = None):
        super().__init__()
        self.max_rules = max_rules

    def fit(self, X, y, feature_names: str = None):
        self.complexity_ = 0
        # X, y = check_X_y(X, y)
        X, y, feature_names = check_fit_arguments(self, X, y, feature_names)
        self.resultType = type(y[0])
        if feature_names is None:
            self.feature_names = [f'X_{x}' for x in range(X.shape[1])]
        else:
            # only include alphanumeric chars / replace spaces with underscores
            self.feature_names = [''.join([i for i in x if i.isalnum()]).replace(' ', '_')
                                  for x in feature_names]
            self.feature_names = [
                'X_' + x if x[0].isdigit()
                else x
                for x in self.feature_names
            ]

        assert len(self.feature_names) == X.shape[1]

        data = [[] for i in range(len(self.feature_names))]
        categories = []

        for i in range(len(X)):
            categories.append(str(y[i]))
            for j in range(len(self.feature_names)):
                data[j].append(X[i][j])
        root = ET.Element('GreedyTree')
        self.grow_tree(data, categories, root, self.feature_names)  # adds to root
        self.tree_ = ET.tostring(root, encoding="unicode")
        # print('self.tree_', self.tree_)
        self.dom_ = minidom.parseString(self.tree_)
        return self

    def impute_nodes(self, X, y):
        """
        Returns
        ---
        the leaf by which this sample would be classified
        """
        source_node = self.root
        for i in range(len(y)):
            sample, label = X[i, ...], y[i]
            _add_label(source_node, label)
            nodes = [source_node]
            while len(nodes) > 0:
                node = nodes.pop()
                if not node.hasChildNodes():
                    continue
                else:
                    att_name = node.firstChild.nodeName
                    if att_name != "#text":
                        att = sample[self.feature_names.index(att_name)]
                        next_node = _get_next_node(node.childNodes, att)
                    else:
                        next_node = node.firstChild
                    _add_label(next_node, label)
                    nodes.append(next_node)

        self._calc_probs(source_node)
        # self.dom_.childNodes[0] = source_node
        # self.tree_.source = source_node

    def _calc_probs(self, node):
        node.nodeValue = np.mean(node.labels)
        if not node.hasChildNodes():
            return
        for c in node.childNodes:
            self._calc_probs(c)

    def raw_preds(self, X):
        check_is_fitted(self, ['tree_', 'resultType', 'feature_names'])
        X = check_array(X)
        if isinstance(X, pd.DataFrame):
            X = deepcopy(X)
            X.columns = self.feature_names
        root = self.root
        prediction = []
        for i in range(X.shape[0]):
            answerlist = decision(root, X[i], self.feature_names, 1)
            answerlist = sorted(answerlist.items(), key=lambda x: x[1], reverse=True)
            answer = answerlist[0][0]
            # prediction.append(self.resultType(answer))
            prediction.append(float(answer))

        return np.array(prediction)

    def predict(self, X):
        raw_preds = self.raw_preds(X)
        return (raw_preds > np.ones_like(raw_preds) * 0.5).astype(int)

    def predict_proba(self, X):
        raw_preds = self.raw_preds(X)
        return np.vstack((1 - raw_preds, raw_preds)).transpose()

    def __str__(self):
        check_is_fitted(self, ['tree_'])
        return self.dom_.toprettyxml(newl="\r\n")

    def grow_tree(self, X_t: List[list], y_str: List[str], parent, attrs_names):
        """
        Parameters
        ----------
        X_t: List[list]
            input data transposed (num_features x num_observations)
        y_str: List[str]
            outcome represented as strings

        parent
        attrs_names

        """
        # check that y contains more than 1 distinct value
        if len(set(y_str)) > 1:
            split = []

            # loop over features and build up potential splits
            for i in range(len(X_t)):
                if set(X_t[i]) == set("?"):
                    split.append(0)
                else:
                    if is_numeric_feature(X_t[i]):
                        split.append(gain(y_str, X_t[i]))
                    else:
                        split.append(gain_ratio(y_str, X_t[i]))

            # no good split, return child node
            if max(split) == 0:
                set_as_leaf_node(parent, y_str)

            # there is a good split
            else:
                index_selected = split.index(max(split))
                name_selected = str(attrs_names[index_selected])
                self.complexity_ += 1
                if is_numeric_feature(X_t[index_selected]):
                    # split on this point
                    split_point = get_best_split(y_str, X_t[index_selected])

                    # build up children nodes
                    r_child_X = [[] for i in range(len(X_t))]
                    r_child_y = []
                    l_child_X = [[] for i in range(len(X_t))]
                    l_child_y = []
                    for i in range(len(y_str)):
                        if not X_t[index_selected][i] == "?":
                            if float(X_t[index_selected][i]) < float(split_point):
                                l_child_y.append(y_str[i])
                                for j in range(len(X_t)):
                                    l_child_X[j].append(X_t[j][i])
                            else:
                                r_child_y.append(y_str[i])
                                for j in range(len(X_t)):
                                    r_child_X[j].append(X_t[j][i])

                    # grow child nodes as well
                    if len(l_child_y) > 0 and len(r_child_y) > 0 and (
                            self.max_rules is None or
                            self.complexity_ <= self.max_rules
                    ):
                        p_l = float(len(l_child_y)) / (len(X_t[index_selected]) - X_t[index_selected].count("?"))
                        son = ET.SubElement(parent, name_selected,
                                            {'feature': str(split_point), "flag": "l", "p": str(round(p_l, 3))})
                        self.grow_tree(l_child_X, l_child_y, son, attrs_names)
                        son = ET.SubElement(parent, name_selected,
                                            {'feature': str(split_point), "flag": "r", "p": str(round(1 - p_l, 3))})
                        self.grow_tree(r_child_X, r_child_y, son, attrs_names)
                    else:
                        num_max = 0
                        for cat in set(y_str):
                            num_cat = y_str.count(cat)
                            if num_cat > num_max:
                                num_max = num_cat
                                most_cat = cat
                        parent.text = most_cat
                else:
                    # split on non-numeric variable (e.g. categorical)
                    # create a leaf for each unique value
                    for k in set(X_t[index_selected]):
                        if not k == "?" and (
                                self.max_rules is None or
                                self.complexity_ <= self.max_rules
                        ):
                            child_X = [[] for i in range(len(X_t))]
                            child_y = []
                            for i in range(len(y_str)):
                                if X_t[index_selected][i] == k:
                                    child_y.append(y_str[i])
                                    for j in range(len(X_t)):
                                        child_X[j].append(X_t[j][i])
                            son = ET.SubElement(parent, name_selected, {
                                'feature': k, "flag": "m",
                                'p': str(round(
                                    float(len(child_y)) / (
                                            len(X_t[index_selected]) - X_t[index_selected].count("?")),
                                    3))})
                            self.grow_tree(child_X, child_y, son, attrs_names)
        else:
            parent.text = y_str[0]

    @property
    def root(self):
        return self.dom_.childNodes[0]


class HSC45TreeClassifier(BaseEstimator):
    def __init__(self, estimator_: C45TreeClassifier, reg_param: float = 1, shrinkage_scheme_: str = 'node_based'):
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
        self.shrinkage_scheme_ = shrinkage_scheme_

    def _calc_probs(self, node):
        self.estimator_._calc_probs(node)

    def impute_nodes(self, X, y):
        self.estimator_.impute_nodes(X, y)

    def shrink_tree(self):
        shrink_node(self.estimator_.root, self.reg_param, None, None, 0, self.shrinkage_scheme_, 0)

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)

    def fit(self, *args, **kwargs):
        X = kwargs['X'] if "X" in kwargs else args[0]
        y = kwargs['y'] if "y" in kwargs else args[1]
        if not hasattr(self.estimator_, "dom_"):
            self.estimator_.fit(X, y)
        self.impute_nodes(X, y)
        self.shrink_tree()

    def predict(self, X):
        return self.estimator_.predict(X)

    @property
    def complexity_(self):
        return self.estimator_.complexity_


class HSC45TreeClassifierCV(HSC45TreeClassifier):
    def __init__(self, estimator_: C45TreeClassifier,
                 reg_param_list: List[float] = [0.1, 1, 10, 50, 100, 500], shrinkage_scheme_: str = 'node_based',
                 cv: int = 3, scoring='accuracy', *args, **kwargs):
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

        for reg_param in self.reg_param_list:
            est = HSC45TreeClassifier(copy.deepcopy(self.estimator_), reg_param)
            cv_scores = cross_val_score(est, X, y, cv=self.cv, scoring=self.scoring)
            self.scores_.append(np.mean(cv_scores))
        self.reg_param = self.reg_param_list[np.argmax(self.scores_)]
        super().fit(X=X, y=y)


if __name__ == '__main__':
    X, y, feature_names = get_clean_dataset('ionosphere', data_source='pmlb')
    m = C45TreeClassifier(max_rules=3)
    m.fit(X, y)
    s_m = HSC45TreeClassifier(estimator_=m)
    s_m.fit(X, y)
    preds = s_m.predict_proba(X)
