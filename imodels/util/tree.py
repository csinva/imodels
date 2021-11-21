def compute_tree_complexity(tree):
    """Calculate number of non-leaf nodes
    """
    children_left = tree.children_left
    children_right = tree.children_right
    num_split_nodes = 0

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
            num_split_nodes += 1
            stack.append((children_left[node_id], depth + 1))
            stack.append((children_right[node_id], depth + 1))
    return num_split_nodes
