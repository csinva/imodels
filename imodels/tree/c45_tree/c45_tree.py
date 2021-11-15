"""Modified from https://github.com/RaczeQ/scikit-learn-C4.5-tree-classifier
References
----------
.. [1] https://en.wikipedia.org/wiki/Decision_tree_learning
.. [2] https://en.wikipedia.org/wiki/C4.5_algorithm
"""
from copy import deepcopy
from typing import List
from xml.dom import minidom
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd
from imodels.tree.c45_tree.c45_utils import decision, is_numeric_feature, gain, gain_ratio, get_best_split, \
    set_as_leaf_node
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


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
        X, y = check_X_y(X, y)
        self.resultType = type(y[0])
        if feature_names is None:
            self.feature_names = [f'X_{x}' for x in range(X.shape[1])]
        else:
            # only include alphanumeric chars / replace spaces with underscores
            self.feature_names = [''.join([i for i in x if i.isalnum()]).replace(' ', '_')
                                  for x in feature_names]
            self.feature_names = ['X_' + x if x[0].isdigit()
                                  else x
                                  for x in self.feature_names]

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

    def raw_preds(self, X):
        check_is_fitted(self, ['tree_', 'resultType', 'feature_names'])
        X = check_array(X)
        if isinstance(X, pd.DataFrame):
            X = deepcopy(X)
            X.columns = self.feature_names
        root = self.dom_.childNodes[0]
        prediction = []
        for i in range(X.shape[0]):
            answerlist = decision(root, X[i], self.feature_names, 1)
            answerlist = sorted(answerlist.items(), key=lambda x: x[1], reverse=True)
            answer = answerlist[0][0]
            prediction.append(self.resultType(answer))
        return np.array(prediction)

    def predict(self, X):
        return (self.raw_preds(X) > 0.5).astype(int)

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


if __name__ == '__main__':
    from imodels.util.data_util import get_clean_dataset

    X, y, feature_names = get_clean_dataset('ionosphere', data_source='pmlb')
    m = C45TreeClassifier(max_rules=3)
    m.fit(X, y)
    print('mse', np.mean(np.square(m.predict(X) - y)))
    print(m)
    m.predict(X)