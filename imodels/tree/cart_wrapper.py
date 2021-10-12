# This is just a simple wrapper around sklearn decisiontree:https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html

from sklearn.tree import DecisionTreeClassifier


class GreedyTreeClassifier(DecisionTreeClassifier):
    """Wrapper around sklearn greedy tree
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.complexity_ = 0

    def __str__(self):
        return 'GreedyDecisionTree ' + str(self.tree_)

    def fit(self, X, y, sample_weight=None, check_input=True, X_idx_sorted="deprecated"):
        """Build a decision tree classifier from the training set (X, y).
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csc_matrix``.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels) as integers or strings.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted. Splits
            that would create child nodes with net zero or negative weight are
            ignored while searching for a split in each node. Splits are also
            ignored if they would result in any single class carrying a
            negative weight in either child node.
        check_input : bool, default=True
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.
        X_idx_sorted : deprecated, default="deprecated"
            This parameter is deprecated and has no effect.
            It will be removed in 1.1 (renaming of 0.26).
            .. deprecated:: 0.24
        Returns
        -------
        self : DecisionTreeClassifier
            Fitted estimator.
        """
        super().fit(X, y, sample_weight=None, check_input=True, X_idx_sorted="deprecated")
        self._set_complexity()

    def _set_complexity(self):
        """Set complexity as number of non-leaf nodes
        """

        # set complexity
        tree = self.tree_
        children_left = tree.children_left
        children_right = tree.children_right
        n_nodes = tree.node_count
        num_split_nodes = 0
        num_leaves = 0

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
            else:
                num_leaves += 1

        self.complexity_ = num_split_nodes
