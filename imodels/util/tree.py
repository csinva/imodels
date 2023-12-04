from sklearn import datasets
from sklearn.tree import DecisionTreeRegressor
import numpy as np
from sklearn.tree._tree import Tree
from imodels.tree.custom_greedy_tree import CustomDecisionTreeClassifier


def compute_tree_complexity(tree, complexity_measure='num_rules'):
    """Calculate number of non-leaf nodes
    """
    children_left = tree.children_left
    children_right = tree.children_right
    # num_split_nodes = 0
    complexity = 0
    stack = [(0, 0)]  # start with the root node id (0) and its depth (0)
    while len(stack) > 0:
        # `pop` ensures each node is only visited once
        node_id, depth = stack.pop()

        # If the left and right child of a node is not the same we have a split
        # node
        is_split_node = children_left[node_id] != children_right[node_id]

        # If a split node, append left and right children and depth to `stack`
        # so we can loop through them
        if is_split_node:
            if complexity_measure == 'num_rules':
                complexity += 1
            stack.append((children_left[node_id], depth + 1))
            stack.append((children_right[node_id], depth + 1))
        else:
            if complexity_measure != 'num_rules':
                complexity += 1
    return complexity


def _validate_feature_costs(feature_costs, n_features):
    if feature_costs is None:
        feature_costs = np.ones(n_features, dtype=np.float64)
    else:
        assert len(
            feature_costs) == n_features, f'{len(feature_costs)} != {n_features}'
        np.min(feature_costs) >= 0
    return feature_costs


def calculate_mean_depth_of_points_in_tree(tree, X, feature_costs=None):
    """Calculate the mean depth of each point in the tree.
    This is the average depth of the path from the root to the point.
    """
    feature_costs = _validate_feature_costs(
        feature_costs, n_features=X.shape[1])

    if isinstance(tree, CustomDecisionTreeClassifier):
        return _mean_depth_custom_tree(tree, X, feature_costs)
    elif hasattr(tree, 'tree_'):
        return _mean_depth_sklearn_tree(tree.tree_, X, feature_costs)
    elif hasattr(tree, 'estimator_'):
        return _mean_depth_sklearn_tree(tree.estimator_.tree_, X, feature_costs)
    else:
        return _mean_depth_coct_tree(tree, X, feature_costs)


def _mean_depth_custom_tree(custom_tree_, X, feature_costs):
    node = custom_tree_.root
    n_samples = []
    cum_costs = []
    is_leaves = []
    stack = [(node, 0)]
    while len(stack) > 0:
        node, cost = stack.pop()
        n_samples.append(node.num_samples)
        cum_costs.append(cost)
        is_leaves.append(node.left is None and node.right is None)

        if node.left:
            stack.append((node.left, cost + feature_costs[node.feature_index]))
        if node.right:
            stack.append(
                (node.right, cost + feature_costs[node.feature_index]))
    is_leaves = np.array(is_leaves)
    cum_costs = np.array(cum_costs)[is_leaves]
    n_samples = np.array(n_samples)[is_leaves]
    costs = cum_costs * n_samples / np.sum(n_samples)
    return np.sum(costs)


def _mean_depth_sklearn_tree(tree_, X, feature_costs):
    n_nodes = tree_.node_count
    children_left = tree_.children_left
    children_right = tree_.children_right

    # things to compute
    _node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    _is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    _cum_costs = np.zeros(shape=n_nodes, dtype=np.float64)

    # start with the root node id (0) and its depth (0) and its cost (0)
    stack = [(0, 0, 0)]
    while len(stack) > 0:
        node_id, depth, cost = stack.pop()
        _node_depth[node_id] = depth
        _cum_costs[node_id] = cost

        is_split_node = children_left[node_id] != children_right[node_id]

        cost += feature_costs[tree_.feature[node_id]]
        if is_split_node:
            stack.append((children_left[node_id], depth + 1, cost))
            stack.append((children_right[node_id], depth + 1, cost))
        else:
            _is_leaves[node_id] = True

    # iterate over leaves and calculate the number of samples in each of them
    n_samples = tree_.n_node_samples
    leaf_samples = n_samples[_is_leaves].astype(np.float64)
    depths = _cum_costs[_is_leaves] * leaf_samples / np.sum(leaf_samples)
    return np.sum(depths)


def _mean_depth_coct_tree(coct_tree, X, feature_costs):
    indicator = coct_tree.decision_path(X)[:, coct_tree.branch_nodes]
    feature_use_counts = indicator.sum(axis=0)
    n_branch_nodes = indicator.shape[1]
    node_idx_to_feature_cost = {
        i: feature_costs[coct_tree.feature_[i]] for i in range(n_branch_nodes)
    }
    cost = sum(
        [node_idx_to_feature_cost[i] * feature_use_counts[i]
         for i in range(n_branch_nodes)]
    ) / len(X)
    return cost


def calculate_mean_unique_calls_in_ensemble(ensemble, X, feature_costs=None):
    '''Calculate the mean number of unique calls in the ensemble.
    '''
    if X is None:
        # Should pass X, this is just for testing
        n_features_in = ensemble.n_features_in_
        X = np.random.randint(2, size=(100, n_features_in))

    if feature_costs is None:
        feature_costs = np.ones(n_features_in, dtype=np.float64)
    else:
        assert len(
            feature_costs) == n_features_in, f'{len(feature_costs)} != {n_features_in}'
        np.min(feature_costs) >= 0

    # extract the decision path for each sample
    ests = ensemble.estimators_.flatten()
    feats = [set() for _ in range(len(X))]
    for i in range(len(ests)):
        est = ests[i]
        node_index = est.decision_path(X).toarray()
        feats_est = [
            set([est.tree_.feature[x] for x in np.nonzero(row)[0]])
            for row in node_index
        ]
        for j in range(len(feats)):
            feats[j] = feats[j].union(feats_est[j])
    # -1 for the -2 feature that is always present
    return np.mean([len(f) - 1 for f in feats])


def compute_mean_llm_calls(model_name, num_prompts, model=None, X=None):
    if model_name == "manual_tree":
        return calculate_mean_depth_of_points_in_tree(model.tree_)
    elif model_name == "manual_hstree":
        return calculate_mean_depth_of_points_in_tree(model.estimator_.tree_)
    elif model_name == "manual_gbdt":
        return calculate_mean_unique_calls_in_ensemble(model, X)
    elif model_name == "manual_tree_cv":
        return calculate_mean_depth_of_points_in_tree(model.best_estimator_.tree_)
    elif model_name in ["manual_single_prompt"]:
        return 1
    elif model_name in ["manual_ensemble", "manual_boosting"]:
        return num_prompts
    else:
        return num_prompts


if __name__ == '__main__':
    X, y = datasets.fetch_california_housing(return_X_y=True)  # regression
    m = DecisionTreeRegressor(random_state=42, max_leaf_nodes=4)
    m.fit(X, y)
    print(compute_tree_complexity(m.tree_, complexity_measure='num_leaves'))
