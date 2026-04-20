import numpy as np
import pandas as pd

from collections import namedtuple

from sklearn import __version__
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.tree._tree import Tree

TreeData = namedtuple('TreeData', 'left_child right_child feature threshold impurity n_node_samples weighted_n_node_samples missing_go_to_left')

def _extract_arrays_from_figs_tree(figs_tree):
    """Takes in a FIGS tree and recursively converts it to arrays that we can later use to build a sklearn decision tree object
    """
    tree_data = TreeData(
        left_child=[],
        right_child=[],
        feature=[],
        threshold=[],
        impurity=[],
        n_node_samples=[],
        weighted_n_node_samples=[],
        missing_go_to_left=[],
    )

    value_sklearn_array = []

    def _update_node(node):
        if node is None:
            return

        node_id_left = node_id_right = -1
        feature = threshold = -2
        value_sklearn = node.value_sklearn

        has_children = node.left is not None
        if has_children:
            node_id_left = node.left.node_id
            node_id_right = node.right.node_id
            feature = node.feature
            threshold = node.threshold

        tree_data.left_child.append(node_id_left)
        tree_data.right_child.append(node_id_right)
        tree_data.feature.append(feature)
        tree_data.threshold.append(threshold)
        tree_data.impurity.append(node.impurity)
        tree_data.n_node_samples.append(np.sum(value_sklearn))
        tree_data.weighted_n_node_samples.append(np.sum(value_sklearn)) # TODO add sample weights
        tree_data.missing_go_to_left.append(1)
        value_sklearn_array.append(value_sklearn)

        if has_children:
            _update_node(node.left)
            _update_node(node.right)
    _update_node(figs_tree)

    return tree_data, np.array(value_sklearn_array)

def extract_sklearn_tree_from_figs(figs, tree_num, n_classes, with_leaf_predictions=False):
    """Takes in a FIGS model and convert tree tree_num to a sklearn decision tree
    """

    try:
        figs_tree = figs.trees_[tree_num]
    except:
        raise AttributeError(f'Can not load tree_num = {tree_num}!')

    tree_data_namedtuple, value_sklearn_array = _extract_arrays_from_figs_tree(figs_tree)

    # manipulate tree_data_namedtuple into the numpy array of tuples that sklearn expects for use with __setstate__()
    df_tree_data = pd.DataFrame(tree_data_namedtuple._asdict())
    tree_data_list_of_tuples = list(df_tree_data.itertuples(index=False, name=None))
    _dtypes = np.dtype([('left_child', 'i8'), ('right_child', 'i8'), ('feature', 'i8'), ('threshold', 'f8'), ('impurity', 'f8'), ('n_node_samples', 'i8'), ('weighted_n_node_samples', 'f8'), ('missing_go_to_left', 'u1')])

    tree_data_array = np.array(tree_data_list_of_tuples, dtype=_dtypes)

    # reshape value_sklearn_array to match the expected shape of (n_nodes,1,2) for values
    value_sklearns = value_sklearn_array.reshape(value_sklearn_array.shape[0], 1, value_sklearn_array.shape[1])

    if n_classes == 1:
        value_sklearns = np.ascontiguousarray(value_sklearns[:, :, 0:1])

    # get the max_depth
    def get_max_depth(node):
        if node is None:
            return -1
        else:
            return 1 + max(get_max_depth(node.left), get_max_depth(node.right))

    max_depth = get_max_depth(figs_tree)

    # get other variables needed for the sklearn.tree._tree.Tree constructor and __setstate__() calls
    # n_samples = np.sum(figs_tree.value_sklearn)
    node_count = len(tree_data_array)
    features = np.array(tree_data_namedtuple.feature)
    n_features = np.unique(features[np.where( 0 <= features )]).size
    n_classes_array = np.array([n_classes], dtype=int)
    n_outputs = 1

    # make dict to pass to __setstate__()
    _state = {'max_depth': max_depth,
        'node_count': node_count,
        'nodes': tree_data_array,
        'values': value_sklearns,
        'n_features_in_': figs.n_features_in_,
        # WARNING this circumvents
        # UserWarning: Trying to unpickle estimator DecisionTreeClassifier from version pre-0.18 when using version
        # https://github.com/scikit-learn/scikit-learn/blob/53acd0fe52cb5d8c6f5a86a1fc1352809240b68d/sklearn/base.py#L279
        '_sklearn_version': __version__,
    }

    tree = Tree(n_features=n_features, n_classes=n_classes_array, n_outputs=n_outputs)
    # https://github.com/scikit-learn/scikit-learn/blob/3850935ea610b5231720fdf865c837aeff79ab1b/sklearn/tree/_tree.pyx#L677
    tree.__setstate__(_state)

    # add the tree_ for the dt __setstate__()
    # note the trailing underscore also trips the sklearn_is_fitted protections
    _state['tree_'] = tree
    _state['classes_'] = np.arange(n_classes)
    _state['n_outputs_'] = n_outputs

    # construct sklearn object and __setstate__()
    if isinstance(figs, ClassifierMixin):
        dt = DecisionTreeClassifier(max_depth=max_depth)
    elif isinstance(figs, RegressorMixin):
        dt = DecisionTreeRegressor(max_depth=max_depth)

    try:
        dt.__setstate__(_state);
    except:
        raise Exception(f'Did not successfully run __setstate__() when translating to {type(dt)}, did sklearn update?')

    if not with_leaf_predictions:
        return dt
    else:
        leaf_values_dict = {}
        def _read_node(node):
            if node is None:
                return None
            elif node.left is None and node.right is None:
                leaf_values_dict[node.node_id] = node.value[0][0]
            _read_node(node.left)
            _read_node(node.right)
        _read_node(figs_tree)

        return dt, leaf_values_dict
