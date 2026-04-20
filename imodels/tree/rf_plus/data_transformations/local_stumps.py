import numpy as np


class LocalDecisionStump:
    """
    An object that implements a callable local decision stump function and
    that also includes some meta-data that allows for an API to interact with
    other methods in the package.

    A local decision stump is a tri-valued function that is zero outside of a
    rectangular region, and on that region, takes either a positive or negative
    value, depending on whether a single designated feature is above or below a
    threshold. For more information on what a local decision stump is, refer to
    our paper.

    Parameters
    ----------
    feature: int
        Feature used in the decision stump
    threshold: float
        Threshold used in the decision stump
    left_val: float
        The value taken when x_k <= threshold
    right_val: float
        The value taken when x_k > threshold
    a_features: list of ints
        List of ancestor feature indices (ordered from highest ancestor to
        lowest)
    a_thresholds: list of floats
        List of ancestor thresholds (ordered from highest ancestor to lowest)
    a_signs: list of bools
        List of signs indicating whether the current node is in the left child
        (False) or right child (True) of the ancestor nodes (ordered from
        highest ancestor to lowest)
    parent: int
        Index of the parent node in the tree structure. If the node is the root,
        -1 is used as the index.
    """

    def __init__(self, feature, threshold, left_val, right_val, a_features,
                 a_thresholds, a_signs, parent):
        self.feature = feature
        self.threshold = threshold
        self.left_val = left_val
        self.right_val = right_val
        self.a_features = a_features
        self.a_thresholds = a_thresholds
        self.a_signs = a_signs
        self.parent = parent

    def __call__(self, data):
        """
        Return values of the local decision stump function on an input data
        matrix with samples as rows

        Parameters
        ----------
        data: array-like of shape (n_samples, n_features)
            Data matrix to feed into the local decision stump function

        Returns
        -------
        values: array-like of shape (n_samples,)
            Function values on the data
        """

        root_to_stump_path_indicators = \
            _compare_all(data, self.a_features, np.array(self.a_thresholds),
                         np.array(self.a_signs))
        in_node = np.all(root_to_stump_path_indicators, axis=1).astype(int)
        is_right = _compare(data, self.feature, self.threshold).astype(int)
        values = in_node * (is_right * self.right_val +
                            (1 - is_right) * self.left_val)

        return values

    def __repr__(self):
        return f"LocalDecisionStump(feature={self.feature}, " \
               f"threshold={self.threshold}, left_val={self.left_val}, " \
               f"right_val={self.right_val}, a_features={self.a_features}, " \
               f"a_thresholds={self.a_thresholds}, " \
               f"a_signs={self.a_signs})"

    def get_depth(self):
        """
        Get depth of the local decision stump, i.e. count how many ancenstor
        nodes it has. The root node has depth 0.

        """
        return len(self.a_features)


def make_stump(node_no, tree_struct, parent_stump, is_right_child,
               normalize=False):
    """
    Create a single local decision stump corresponding to a node in a
    scikit-learn tree structure object. The nonzero values of the stump are
    chosen so that the vector of local decision stump values over the training
    set (used to fit the tree) is orthogonal to those of all ancestor nodes.

    Parameters
    ----------
    node_no: int
        The index of the node
    tree_struct: object
        The scikit-learn tree object
    parent_stump: LocalDecisionStump object
        The local decision stump corresponding to the parent of the node in q
        uestion
    is_right_child: bool
        True if the new node is the right child of the parent node, False
        otherwise
    normalize: bool
        Flag. If set to True, then divide the nonzero function values by
        sqrt(n_samples in node) so that the vector of function values on the
        training set has unit norm. If False, then do not divide, so that the
        vector of function values on the training set has norm equal to
        n_samples in node.

    Returns
    -------

    """
    # Get features, thresholds and signs for ancestors
    if parent_stump is None:  # If root node
        a_features = []
        a_thresholds = []
        a_signs = []
        parent = -1
    else:
        a_features = parent_stump.a_features + [parent_stump.feature]
        a_thresholds = parent_stump.a_thresholds + [parent_stump.threshold]
        a_signs = parent_stump.a_signs + [is_right_child]
        parent = parent_stump.feature
    # Get indices for left and right children of the node in question
    left_child = tree_struct.children_left[node_no]
    right_child = tree_struct.children_right[node_no]
    # Get quantities relevant to the node in question
    feature = tree_struct.feature[node_no]
    threshold = tree_struct.threshold[node_no]
    left_size = tree_struct.weighted_n_node_samples[left_child]
    right_size = tree_struct.weighted_n_node_samples[right_child]
    parent_size = tree_struct.weighted_n_node_samples[node_no]
    normalization = parent_size if normalize else 1
    left_val = - np.sqrt(right_size / (left_size * normalization))
    right_val = np.sqrt(left_size / (right_size * normalization))

    return LocalDecisionStump(feature, threshold, left_val, right_val,
                              a_features, a_thresholds, a_signs, parent)


def make_stumps(tree_struct, normalize=False):
    """
    Create a collection of local decision stumps corresponding to all internal
    nodes in a scikit-learn tree structure object.

    Parameters
    ----------
    tree_struct: object
        The scikit-learn tree object
    normalize: bool
        Flag. If set to True, then divide the nonzero function values by
        sqrt(n_samples in node) so that the vector of function values on the
        training set has unit norm. If False, then do not divide, so that the
        vector of function values on the training set has norm equal to
        n_samples in node.

    Returns
    -------
    stumps: list of LocalDecisionStump objects
        The local decision stumps corresponding to all internal node in the
        tree structure

    """
    stumps = []

    def make_stump_iter(node_no, tree_struct, parent_stump, is_right_child,
                        normalize, stumps):
        """
        Helper function for iteratively making local decision stump objects and
        appending them to the list stumps.
        """
        new_stump = make_stump(node_no, tree_struct, parent_stump,
                               is_right_child, normalize)
        stumps.append(new_stump)
        left_child = tree_struct.children_left[node_no]
        right_child = tree_struct.children_right[node_no]
        if tree_struct.feature[left_child] != -2:  # is not leaf
            make_stump_iter(left_child, tree_struct, new_stump, False,
                            normalize, stumps)
        if tree_struct.feature[right_child] != -2:  # is not leaf
            make_stump_iter(right_child, tree_struct, new_stump, True,
                            normalize, stumps)

    make_stump_iter(0, tree_struct, None, None, normalize, stumps)
    return stumps


def tree_feature_transform(stumps, X):
    """
    Transform the data matrix X using a mapping derived from a collection of
    local decision stump functions.

    If the list of stumps is empty, return an array of shape (0, n_samples).

    Parameters
    ----------
    stumps: list of LocalDecisionStump objects
        List of stump functions to use to transform data
    X: array-like of shape (n_samples, n_features)
        Original data matrix

    Returns
    -------
    X_transformed: array-like of shape (n_samples, n_stumps)
        Transformed data matrix
    """
    transformed_feature_vectors = [np.empty((X.shape[0], 0))]
    for stump in stumps:
        transformed_feature_vec = stump(X)[:, np.newaxis]
        transformed_feature_vectors.append(transformed_feature_vec)
    X_transformed = np.hstack(transformed_feature_vectors)

    return X_transformed


def _compare(data, k, threshold, sign=True):
    """
    Obtain indicator vector for the samples with k-th feature > threshold
    """
    if sign:
        return data[:, k] > threshold
    else:
        return data[:, k] <= threshold


def _compare_all(data, ks, thresholds, signs):
    """
    Obtain indicator vector for the samples with k-th feature > threshold or
    <= threshold (depending on sign) for all k in ks
    """
    return ~np.logical_xor(data[:, ks] > thresholds, signs)