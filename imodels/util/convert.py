import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.tree import _tree
from typing import Union, List, Tuple


def tree_to_rules(tree: Union[DecisionTreeClassifier, DecisionTreeRegressor],
                  feature_names: List[str],
                  prediction_values: bool = False, round_thresholds=True) -> List[str]:
    """
    Return a list of rules from a tree

    Parameters
    ----------
        tree : Decision Tree Classifier/Regressor
        feature_names: list of variable names

    Returns
    -------
    rules : list of rules.
    """
    # XXX todo: check the case where tree is build on subset of features,
    # ie max_features != None

    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    rules = []

    def recurse(node, base_name):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            symbol = '<='
            symbol2 = '>'
            threshold = tree_.threshold[node]
            if round_thresholds:
                threshold = np.round(threshold, decimals=5)
            text = base_name + ["{} {} {}".format(name, symbol, threshold)]
            recurse(tree_.children_left[node], text)

            text = base_name + ["{} {} {}".format(name, symbol2,
                                                  threshold)]
            recurse(tree_.children_right[node], text)
        else:
            rule = str.join(' and ', base_name)
            rule = (rule if rule != ''
                    else ' == '.join([feature_names[0]] * 2))
            # a rule selecting all is set to "c0==c0"
            if prediction_values:
                rules.append((rule, tree_.value[node][0].tolist()))
            else:
                rules.append(rule)

    recurse(0, [])

    return rules if len(rules) > 0 else 'True'


def tree_to_code(clf, feature_names):
    '''Prints a tree with a single split
    '''
    n_nodes = clf.tree_.node_count
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    feature = clf.tree_.feature
    threshold = clf.tree_.threshold

    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, 0)]  # start with the root node id (0) and its depth (0)
    s = ''
    while len(stack) > 0:
        # `pop` ensures each node is only visited once
        node_id, depth = stack.pop()
        node_depth[node_id] = depth

        # If the left and right child of a node is not the same we have a split
        # node
        is_split_node = children_left[node_id] != children_right[node_id]
        # If a split node, append left and right children and depth to `stack`
        # so we can loop through them
        if is_split_node:
            stack.append((children_left[node_id], depth + 1))
            stack.append((children_right[node_id], depth + 1))
        else:
            is_leaves[node_id] = True

    # print("The binary tree structure has {n} nodes and has "
    #       "the following tree structure:\n".format(n=n_nodes))
    for i in range(n_nodes):
        if is_leaves[i]:
            pass
        #     print("{space}node={node} is a leaf node.".format(
        # space=node_depth[i] * "\t", node=i))
        else:
            s += f"{feature_names[feature[i]]} <= {threshold[i]}"
    return f"\033[96m{s}\033[00m\n"


def itemsets_to_rules(itemsets: List[Tuple]) -> List[str]:
    itemsets_clean = list(filter(lambda it: it != 'null' and 'All' not in ''.join(it), itemsets))
    f = lambda itemset: ' and '.join([single_discretized_feature_to_rule(item) for item in itemset])
    return list(map(f, itemsets_clean))


def dict_to_rule(rule, clf_feature_dict):
    """
    Function to accept rule dict and convert to Rule object

    Parameters:
    rule: list of dict of schema
    [
        {
            'feature': int,
            'operator': str,
            'value': float
        },
    ]
    """

    output = ''

    for condition in rule:
        output += '{} {} {} and '.format(
            clf_feature_dict[int(condition['feature'])],
            condition['operator'],
            condition['pivot']
        )

    return output[:-5]


def single_discretized_feature_to_rule(feat: str) -> str:
    # categorical feature
    if '_to_' not in feat:
        return f'{feat} > 0.5'

    # discretized numeric feature
    feat_split = feat.split('_to_')
    upper_value = feat_split[-1]
    lower_value = feat_split[-2].split('_')[-1]

    lower_to_upper_len = 1 + len(lower_value) + 4 + len(upper_value)
    feature_name = feat[:-lower_to_upper_len]

    if lower_value == '-inf':
        rule = f'{feature_name} <= {upper_value}'
    elif upper_value == 'inf':
        rule = f'{feature_name} > {lower_value}'
    else:
        rule = f'{feature_name} > {lower_value} and {feature_name} <= {upper_value}'

    return rule
