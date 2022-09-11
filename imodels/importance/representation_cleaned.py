import copy
from abc import ABC, abstractmethod

import numpy as np
from collections import defaultdict

from sklearn.ensemble import BaseEnsemble


class BlockPartitionedData:

    def __init__(self, data_blocks, common_block=None):
        self.n_blocks = len(data_blocks)
        self.n_samples = data_blocks[0].shape[0]
        self._data_blocks = data_blocks
        self._common_block = common_block
        self._create_block_indices()
        self._means = [np.mean(data_block, axis=0) for data_block in self._data_blocks]

    def get_all_data(self):
        if self._common_block is None:
            all_data = np.hstack(self._data_blocks)
        else:
            all_data = np.hstack(self._data_blocks + [self._common_block])
        return all_data

    def _create_block_indices(self):
        self._block_indices_dict = dict({})

        start_index = 0
        for k in range(self.n_blocks):
            stop_index = start_index + self._data_blocks[k].shape[1]
            self._block_indices_dict[k] = list(range(start_index, stop_index))
            start_index = stop_index
        if self._common_block is None:
            self._common_block_indices = []
        else:
            stop_index = start_index + self._common_block.shape[1]
            self._common_block_indices = list(range(start_index, stop_index))

    def get_block_indices(self, k):
        block_indices = self._common_block_indices + self._block_indices_dict[k]
        return block_indices

    def get_block(self, k):
        if self._common_block is None:
            block = self._data_blocks[k]
        else:
            block = np.hstack([self._common_block, self._data_blocks[k]])
        return block

    def get_all_except_block_indices(self, k):
        if k not in self._block_indices_dict.keys():
            raise ValueError(f"{k} not a block index.")
        all_except_block_indices = []
        for block_no, block_indices in self._block_indices_dict.items():
            if block_no != k:
                all_except_block_indices += block_indices
        all_except_block_indices += self._common_block_indices
        return all_except_block_indices

    def get_all_except_block(self, k):
        all_data = self.get_all_data()
        all_except_block_indices = self.get_all_except_block_indices(k)
        all_except_block = all_data[:, all_except_block_indices]
        return all_except_block

    def get_modified_data(self, k, mode="keep_k"):
        modified_blocks = [np.outer(np.ones(self.n_samples), self._means[i]) for i in range(self.n_blocks)]
        if mode == "keep_k":
            data_blocks = [self._data_blocks[i] if i == k else modified_blocks[i] for i in range(self.n_blocks)]
        elif mode == "keep_rest":
            data_blocks = [modified_blocks[i] if i == k else self._data_blocks[i] for i in range(self.n_blocks)]
        else:
            raise ValueError("Unsupported mode.")
        if self._common_block is None:
            all_data = np.hstack(data_blocks)
        else:
            all_data = np.hstack(data_blocks + [self._common_block])
        return all_data

    def __repr__(self):
        return self.get_all_data().__repr__()


class BlockTransformerBase(ABC):

    def __init__(self, n_features):
        self.n_features = n_features

    @abstractmethod
    def transform_one_feature(self, X, k, center=True, rescale=False):
        pass

    def transform(self, X, center=True, rescale=False):
        data_blocks = [self.transform_one_feature(X, k, center, rescale) for k in range(self.n_features)]
        # common_block = np.ones((X.shape[0], 1))
        blocked_data = BlockPartitionedData(data_blocks)
        return blocked_data

    @classmethod
    def post_process(cls, data_block, center, rescale):
        if center:
            data_block -= data_block.mean(axis=0)
        if rescale:
            std = data_block.std(axis=0)
            if any(std == 0):
                raise Warning("At least one feature is constant")
            else:
                data_block_mean = data_block.mean(axis=0)
                data_block = (data_block - data_block_mean) / data_block.std(axis=0) + data_block_mean
        return data_block


class IdentityTransformer(BlockTransformerBase, ABC):

    def __init__(self, n_features):
        super().__init__(n_features)
        self.priority = 1

    def transform_one_feature(self, X, k, center=True, rescale=False):
        assert X.shape[1] == self.n_features, "n_features does not match that of X."
        data_block = X[:, [k]]
        data_block = BlockTransformerBase.post_process(data_block, center, rescale)
        return data_block


class CompositeTransformer(BlockTransformerBase, ABC):

    def __init__(self, block_transformer_list, adj_std=None, drop_features=True):
        n_features = block_transformer_list[0].n_features
        super().__init__(n_features)
        self.block_transformer_list = block_transformer_list
        for block_transformer in block_transformer_list:
            assert block_transformer.n_features == self.n_features
        self.priority = 3
        self.reference_index = np.argmax([block_transformer.priority
                                          for block_transformer in block_transformer_list])
        self.adj_std = adj_std
        self.drop_features = drop_features
        self.estimator = self.block_transformer_list[self.reference_index].estimator

    def transform_one_feature(self, X, k, center=True, rescale=False):
        data_blocks = []
        for block_transformer in self.block_transformer_list:
            data_block = block_transformer.transform_one_feature(X, k, center, rescale)
            data_blocks.append(data_block)
        # Return empty block if highest priority block is empty and drop_features is True
        if self.drop_features and data_blocks[self.reference_index].shape[1] == 0:
            return data_blocks[self.reference_index]
        else:
            if self.adj_std == "max":
                adj_factor = np.array([max(data_block.std(axis=0)) for data_block in data_blocks])
            elif self.adj_std == "mean":
                adj_factor = np.array([np.mean(data_block.std(axis=0)) for data_block in data_blocks])
            else:
                adj_factor = np.ones(len(data_blocks))
            for i in range(len(adj_factor)):
                if adj_factor[i] == 0: # Only constant features
                    adj_factor[i] = 1
            adj_factor /= adj_factor[self.reference_index] # Normalize so that reference block is unadjusted
            composite_block = np.hstack([data_blocks[i] / adj_factor[i] for i in range(len(data_blocks))])
            return composite_block


class TreeTransformer(BlockTransformerBase, ABC):
    """
    A transformer that transforms data using a representation built from local decision stumps from a tree or tree
    ensemble. The transformer also comes with meta data on the local decision stumps and methods that allow
    for transformations using sub-representations corresponding to each of the original features.

    :param estimator: scikit-learn estimator
        The scikit-learn tree or tree ensemble estimator object
    :param normalize: bool
        Flag. If set to True, then divide the nonzero function values for each local decision stump by
        sqrt(n_samples in node) so that the vector of function values on the training set has unit norm. If False,
        then do not divide, so that the vector of function values on the training set has norm equal to n_samples
        in node.
    """

    def __init__(self, n_features, estimator, normalize=False, data=None):
        super().__init__(n_features)
        self.estimator = estimator
        self.priority = 2
        # Check if single tree or tree ensemble
        if isinstance(estimator, BaseEnsemble):
            tree_models = estimator.estimators_
            if data is not None:
                for tree_model in tree_models:
                    _update_n_node_samples(tree_model, data)
        else:
            tree_models = [estimator]
        # Make stumps for each tree
        all_stumps = []
        for tree_model in tree_models:
            tree_stumps = make_stumps(tree_model.tree_, normalize)
            all_stumps += tree_stumps
        # Identify the stumps that split on feature k, for each k
        self.stumps = defaultdict(list)
        for stump in all_stumps:
            self.stumps[stump.feature].append(stump)
        self._num_splits_per_feature = np.array([len(self.stumps[k]) for k in range(self.n_features)])

    def transform_one_feature(self, X, k, center=True, rescale=False):
        """
        Obtain the engineered features corresponding to a given original feature X_k

        :param X: array-like of shape (n_samples, n_features)
            Original data matrix
        :param k: int
            Index of original feature
        :return: X_transformed: array-like of shape (n_samples, n_new_features)
            Transformed data matrix
        """
        data_block = tree_feature_transform(self.stumps[k], X)
        data_block = BlockTransformerBase.post_process(data_block, center, rescale)
        return data_block
    
    def get_num_splits(self, k):
        return self._num_splits_per_feature[k]


class LocalDecisionStump:
    """
    An object that implements a callable local decision stump function and that also includes some meta-data that
    allows for an API to interact with other methods in the package.

    A local decision stump is a tri-valued function that is zero outside of a rectangular region, and on that region,
    takes either a positive or negative value, depending on whether a single designated feature is above or below a
    threshold. For more information on what a local decision stump is, refer to our paper.

    :param feature: int
        Feature used in the decision stump
    :param threshold: float
        Threshold used in the decision stump
    :param left_val: float
        The value taken when x_k <= threshold
    :param right_val: float
        The value taken when x_k > threshold
    :param a_features: list of ints
        List of ancestor feature indices (ordered from highest ancestor to lowest)
    :param a_thresholds: list of floats
        List of ancestor thresholds (ordered from highest ancestor to lowest)
    :param a_signs: list of bools
        List of signs indicating whether the current node is in the left child (False) or right child (True) of the
        ancestor nodes (ordered from highest ancestor to lowest)
    """

    def __init__(self, feature, threshold, left_val, right_val, a_features, a_thresholds, a_signs):
        self.feature = feature
        self.threshold = threshold
        self.left_val = left_val
        self.right_val = right_val
        self.a_features = a_features
        self.a_thresholds = a_thresholds
        self.a_signs = a_signs

    def __call__(self, data):
        """
        Return values of the local decision stump function on an input data matrix with samples as rows

        :param data: array-like of shape (n_samples, n_features)
            Data matrix to feed into the local decision stump function
        :return: array-like of shape (n_samples,)
            Function values on the data
        """

        root_to_stump_path_indicators = _compare_all(data, self.a_features, np.array(self.a_thresholds),
                                                     np.array(self.a_signs))
        in_node = np.all(root_to_stump_path_indicators, axis=1).astype(int)
        is_right = _compare(data, self.feature, self.threshold).astype(int)
        result = in_node * (is_right * self.right_val + (1 - is_right) * self.left_val)

        return result

    def __repr__(self):
        return f"LocalDecisionStump(feature={self.feature}, threshold={self.threshold}, left_val={self.left_val}, " \
               f"right_val={self.right_val}, a_features={self.a_features}, a_thresholds={self.a_thresholds}, " \
               f"a_signs={self.a_signs})"

    def get_depth(self):
        return len(self.a_features)


def make_stump(node_no, tree_struct, parent_stump, is_right_child, normalize=False):
    """
    Create a single local decision stump corresponding to a node in a scikit-learn tree structure object.
    The nonzero values of the stump are chosen so that the vector of local decision stump values over the training
    set (used to fit the tree) is orthogonal to those of all ancestor nodes.

    :param node_no: int
        The index of the node
    :param tree_struct: object
        The scikit-learn tree object
    :param parent_stump: LocalDecisionStump object
        The local decision stump corresponding to the parent of the node in question
    :param is_right_child: bool
        True if the new node is the right child of the parent node, False otherwise
    :param normalize: bool
        Flag. If set to True, then divide the nonzero function values by sqrt(n_samples in node) so that the
        vector of function values on the training set has unit norm. If False, then do not divide, so that the
        vector of function values on the training set has norm equal to n_samples in node.
    :return: LocalDecisionStump object
        The local decision stump corresponding to the node in question
    """
    # Get features, thresholds and signs for ancestors
    if parent_stump is None:  # If root node
        a_features = []
        a_thresholds = []
        a_signs = []
    else:
        a_features = parent_stump.a_features + [parent_stump.feature]
        a_thresholds = parent_stump.a_thresholds + [parent_stump.threshold]
        a_signs = parent_stump.a_signs + [is_right_child]
    # Get indices for left and right children of the node in question
    left_child = tree_struct.children_left[node_no]
    right_child = tree_struct.children_right[node_no]
    # Get quantities relevant to the node in question
    feature = tree_struct.feature[node_no]
    threshold = tree_struct.threshold[node_no]
    left_size = tree_struct.n_node_samples[left_child]
    right_size = tree_struct.n_node_samples[right_child]
    parent_size = tree_struct.n_node_samples[node_no]
    normalization = parent_size if normalize else 1
    left_val = - np.sqrt(right_size / (left_size * normalization))
    right_val = np.sqrt(left_size / (right_size * normalization))

    return LocalDecisionStump(feature, threshold, left_val, right_val, a_features, a_thresholds, a_signs)


def make_stumps(tree_struct, normalize=False):
    """
    Create a collection of local decision stumps corresponding to all internal nodes in a scikit-learn tree structure
    object.

    :param tree_struct: object
        The scikit-learn tree object
    :param normalize: bool
        Flag. If set to True, then divide the nonzero function values by sqrt(n_samples in node) so that the
        vector of function values on the training set has unit norm. If False, then do not divide, so that the
        vector of function values on the training set has norm equal to n_samples in node.
    :return:
        stumps: list of LocalDecisionStump objects
            The local decision stumps corresponding to all internal node in the tree structure
        num_splits_per_feature: array-like of shape (n_features,)
            The number of splits in the tree on each original feature
    """
    stumps = []

    def make_stump_iter(node_no, tree_struct, parent_stump, is_right_child, normalize, stumps):
        """
        Helper function for iteratively making local decision stump objects and appending them to the list stumps.
        """
        new_stump = make_stump(node_no, tree_struct, parent_stump, is_right_child, normalize)
        stumps.append(new_stump)
        left_child = tree_struct.children_left[node_no]
        right_child = tree_struct.children_right[node_no]
        if tree_struct.feature[left_child] != -2:  # is not leaf
            make_stump_iter(left_child, tree_struct, new_stump, False, normalize, stumps)
        if tree_struct.feature[right_child] != -2:  # is not leaf
            make_stump_iter(right_child, tree_struct, new_stump, True, normalize, stumps)

    make_stump_iter(0, tree_struct, None, None, normalize, stumps)
    return stumps


def tree_feature_transform(stumps, X):
    """
    Transform the data matrix X using a mapping derived from a collection of local decision stump functions.

    :param stumps: list of LocalDecisionStump objects
        List of stump functions to use to transform data
    :param X: array-like of shape (n_samples, n_features)
        Original data matrix
    :return: X_transformed: array-like of shape (n_samples, n_stumps)
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

    :param data: array-like of shape (n_sample, n_feat)
    :param k: int
        Index of feature in question
    :param threshold: float
        Threshold for the comparison
    :param sign: bool
        Flag, if False, return indicator of the complement
    :return: array-like of shape (n_samples,)
    """
    if sign:
        return data[:, k] > threshold
    else:
        return data[:, k] <= threshold


def _compare_all(data, ks, thresholds, signs):
    """
    Obtain indicator vector for the samples with k-th feature > threshold or <= threshold (depending on sign)
    for all k in ks

    :param data: array-like of shape (n_sample, n_feat)
    :param ks: list of ints
        Indices of feature in question
    :param thresholds: list of floats
        Threshold for the comparison
    :param signs: list of bools
        Flags, if k-th element if True, then add the condition k-th feature > threshold, otherwise add the
        condition k-th feature <= threshold
    :return:
    """
    return ~np.logical_xor(data[:, ks] > thresholds, signs)


def _update_n_node_samples(tree, X):
    node_indicators = tree.decision_path(X)
    new_n_node_samples = node_indicators.getnnz(axis=0)
    for i in range(len(new_n_node_samples)):
        tree.tree_.n_node_samples[i] = new_n_node_samples[i]
