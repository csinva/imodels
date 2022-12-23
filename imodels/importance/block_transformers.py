from abc import ABC, abstractmethod

import numpy as np
from collections import defaultdict

from sklearn.ensemble import BaseEnsemble

from local_stumps import make_stumps, tree_feature_transform
class BlockPartitionedData:
    """
    Abstraction for a feature matrix in which the columns are grouped into
    blocks.

    Parameters
    ----------
    data_blocks: list of ndarray
        Blocks of feature columns
    common_block: ndarray
        A set of feature columns that should be common to all blocks
    """

    def __init__(self, data_blocks, common_block=None):
        self.n_blocks = len(data_blocks)
        self.n_samples = data_blocks[0].shape[0]
        self._data_blocks = data_blocks
        self._common_block = common_block
        self._create_block_indices()
        self._means = [np.mean(data_block, axis=0) for data_block in
                       self._data_blocks]

    def get_all_data(self):
        """

        Returns
        -------
        all_data: ndarray
            Returns the data matrix obtained by concatenating all feature
            blocks together
        """
        if self._common_block is None:
            all_data = np.hstack(self._data_blocks)
        else:
            all_data = np.hstack(self._data_blocks + [self._common_block])
            # Common block appended at the end
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
        """

        Parameters
        ----------
        k: int
            The index of the feature block desired

        Returns
        -------
        block_indices: list of int
            The indices of the features in the desired block
        """
        block_indices = self._common_block_indices + self._block_indices_dict[k]
        return block_indices

    def get_block(self, k):
        """

        Parameters
        ----------
        k: int
            The index of the feature block desired

        Returns
        -------
        block: ndarray
            The feature block desired
        """
        if self._common_block is None:
            block = self._data_blocks[k]
        else:
            block = np.hstack([self._common_block, self._data_blocks[k]])
        return block

    def get_all_except_block_indices(self, k):
        """

        Parameters
        ----------
        k: int
            The index of the feature block not desired

        Returns
        -------
        all_except_block_indices: list of int
            The indices of the features not in the desired block
        """
        if k not in self._block_indices_dict.keys():
            raise ValueError(f"{k} not a block index.")
        all_except_block_indices = []
        for block_no, block_indices in self._block_indices_dict.items():
            if block_no != k:
                all_except_block_indices += block_indices
        all_except_block_indices += self._common_block_indices
        return all_except_block_indices

    def get_all_except_block(self, k):
        """

        Parameters
        ----------
        k: int
            The index of the feature block not desired

        Returns
        -------
        all_except_block: ndarray
            The features not in the desired block
        """
        all_data = self.get_all_data()
        all_except_block_indices = self.get_all_except_block_indices(k)
        all_except_block = all_data[:, all_except_block_indices]
        return all_except_block

    def get_modified_data(self, k, mode="keep_k"):
        """
        Modify the data by either imputing the mean of each feature in block k
        (keep_rest) or imputing the mean of each feature not in block k
        (keep_k). Return the full data matrix with the modified data.

        Parameters
        ----------
        k: int
            The index of the feature block not to modify
        mode: string in {"keep_k", "keep_rest"}
            Mode for the method. "keep_k" imputes the mean of each feature not
            in block k, "keep_rest" imputes the mean of each feature in block k

        Returns
        -------
        all_data: ndarray
            Returns the data matrix obtained by concatenating all feature
            blocks together
        """
        modified_blocks = [np.outer(np.ones(self.n_samples), self._means[i])
                           for i in range(self.n_blocks)]
        if mode == "keep_k":
            data_blocks = \
                [self._data_blocks[i] if i == k else modified_blocks[i] for
                 i in range(self.n_blocks)]
        elif mode == "keep_rest":
            data_blocks = \
                [modified_blocks[i] if i == k else self._data_blocks[i] for
                 i in range(self.n_blocks)]
        else:
            raise ValueError("Unsupported mode.")
        if self._common_block is None:
            all_data = np.hstack(data_blocks)
        else:
            all_data = np.hstack(data_blocks + [self._common_block])
        return all_data

    def train_test_split(self, train_indices, test_indices):
        train_blocks = [self.get_block(k)[train_indices, :] for
                        k in range(self.n_blocks)]
        train_blocked_data = BlockPartitionedData(train_blocks)
        test_blocks = [self.get_block(k)[test_indices, :] for
                       k in range(self.n_blocks)]
        test_blocked_data = BlockPartitionedData(test_blocks)
        return train_blocked_data, test_blocked_data

    def __repr__(self):
        return self.get_all_data().__repr__()


class BlockTransformerBase(ABC):
    """
    An interface for block transformers, objects that transform a data matrix
    into a BlockPartitionedData object comprising one block of engineered
    features for each original feature

    Parameters
    ----------
    n_features: int or None
        The number of features in the original data matrix to be supplied,
        used for validation.
    """
    def __init__(self, n_features=None):
        self.n_features = n_features

    @abstractmethod
    def transform_one_feature(self, X, k, center=True, rescale=False):
        """
        Obtain a block of engineered features associated with the original
        feature with index k.

        Parameters
        ----------
        X: ndarray
            The data matrix to be transformed
        center: bool
            Flag for whether to center the transformed data
        rescale: bool
            Flag for whether to rescale the transformed data to have unit
            variance

        Returns
        -------

        """
        pass

    def transform(self, X, center=True, rescale=False):
        """
        Transform a data matrix into a BlockPartitionedData object comprising
        one block for each original feature in X

        Parameters
        ----------
        X: ndarray
            The data matrix to be transformed
        center: bool
            Flag for whether to center the transformed data
        rescale: bool
            Flag for whether to rescale the transformed data to have unit
            variance

        Returns
        -------
        blocked_data: BlockPartitionedData object
            The transformed data
        """
        if self.n_features is None:
            self.n_features = X.shape[1]
        elif self.n_features != X.shape[1]:
            raise ValueError("Number of features does not match value "
                             "supplied during initialization")
        data_blocks = [self.transform_one_feature(X, k, center, rescale) for
                       k in range(self.n_features)]
        blocked_data = BlockPartitionedData(data_blocks)
        return blocked_data


class IdentityTransformer(BlockTransformerBase, ABC):
    """
    Block transformer that creates a block partitioned data object with each
    block k containing only the original feature k.
    """

    def __init__(self, n_features):
        super().__init__(n_features)
        self.priority = 1

    def transform_one_feature(self, X, k, center=True, rescale=False):
        assert X.shape[1] == self.n_features, "n_features does not match that of X."
        data_block = X[:, [k]]
        data_block = BlockTransformerBase.post_process(data_block, center, rescale)
        return data_block


class CompositeTransformer(BlockTransformerBase, ABC):
    """
    A block transformer that is built by concatenating the blocks of the same
    index from a list of block transformers.

    Parameters
    ----------
    block_transformer_list: list of BlockTransformer objects
        The list of block transformers to combine
    adj_std: bool
    drop_features

    """

    def __init__(self, block_transformer_list, adj_std=None, drop_features=True):
        """


        """
        n_features = block_transformer_list[0].n_features
        super().__init__(n_features)
        self.block_transformer_list = block_transformer_list
        # Check that all block transformers have the same number of original
        # features
        for block_transformer in block_transformer_list:
            assert block_transformer.n_features == self.n_features
        self.priority = 3
        # Calculate the index of the transformer with the highest priority
        self.reference_index = np.argmax([block_transformer.priority
                                          for block_transformer in block_transformer_list])
        self.adj_std = adj_std
        self.drop_features = drop_features
        self.estimator = self.block_transformer_list[self.reference_index].estimator
        self.all_adj_factors = [] # check if need to keep this

    def transform_one_feature(self, X, k, center=True, rescale=False):
        data_blocks = []
        for block_transformer in self.block_transformer_list:
            data_block = block_transformer.transform_one_feature(X, k, center,
                                                                 rescale)
            data_blocks.append(data_block)
        # Return empty block if highest priority block is empty and drop_features is True
        if data_blocks[self.reference_index].shape[1] == 0:
            self.all_adj_factors.append(np.array([np.NaN]))
            if self.drop_features:
                return data_blocks[self.reference_index]
            else:
                return data_blocks[1]
        else:
            self.all_adj_factors.append(data_blocks[self.reference_index].std(axis=0))
            if self.adj_std == "max":
                adj_factor = np.array([max(data_block.std(axis=0)) for
                                       data_block in data_blocks])
            elif self.adj_std == "mean":
                adj_factor = np.array([np.mean(data_block.std(axis=0)) for
                                       data_block in data_blocks])
            else:
                adj_factor = np.ones(len(data_blocks))
            for i in range(len(adj_factor)):
                if adj_factor[i] == 0: # Only constant features in block
                    adj_factor[i] = 1
            adj_factor /= adj_factor[self.reference_index] # Normalize so that reference block is unadjusted
            composite_block = np.hstack([data_blocks[i] / adj_factor[i] for i in range(len(data_blocks))])
            return composite_block


class TreeTransformer(BlockTransformerBase, ABC):
    """
    A block transformer that transforms data using a representation built from
    local decision stumps from a tree or tree ensemble. The transformer also
    comes with metadata on the local decision stumps and methods that allow for
    transformations using sub-representations corresponding to each of the
    original features.

    Parameters
    ----------
    n_features: int or None
        The number of features in the original data matrix to be supplied,
        used for validation.
    estimator: scikit-learn estimator
        The scikit-learn tree or tree ensemble estimator object.
    data: ndarray
        A data matrix that can be used to update the number of samples in each
        node of the tree(s) in the supplied estimator object. This affects
        the node values of the resulting engineered features.
    """

    def __init__(self, n_features, estimator, data=None):
        super().__init__(n_features)
        self.estimator = estimator
        self.priority = 2
        # Check if single tree or tree ensemble
        if isinstance(estimator, BaseEnsemble):
            tree_models = estimator.estimators_
            if data is not None:
                # If a data matrix is supplied, use it to update the number
                # of samples in each node
                for tree_model in tree_models:
                    _update_n_node_samples(tree_model, data)
        else:
            tree_models = [estimator]
        # Make stumps for each tree
        all_stumps = []
        for tree_model in tree_models:
            tree_stumps = make_stumps(tree_model.tree_)
            all_stumps += tree_stumps
        # Identify the stumps that split on feature k, for each k
        self.stumps = defaultdict(list)
        for stump in all_stumps:
            self.stumps[stump.feature].append(stump)
        self._num_splits_per_feature = np.array([len(self.stumps[k]) for
                                                 k in range(self.n_features)])

    def transform_one_feature(self, X, k, center=True, rescale=False):
        data_block = tree_feature_transform(self.stumps[k], X)
        data_block = _center_and_rescale(data_block, center, rescale)
        return data_block

    def get_num_splits(self, k):
        return self._num_splits_per_feature[k]


def _update_n_node_samples(tree, X):
    node_indicators = tree.decision_path(X)
    new_n_node_samples = node_indicators.getnnz(axis=0)
    for i in range(len(new_n_node_samples)):
        tree.tree_.n_node_samples[i] = new_n_node_samples[i]


def _center_and_rescale(data_block, center, rescale):
    if center:
        data_block -= data_block.mean(axis=0)
    if rescale:
        std = data_block.std(axis=0)
        if any(std == 0):
            raise Warning("No recaling done."
                          "At least one feature is constant.")
        else:
            data_block_mean = data_block.mean(axis=0)
            data_block = (data_block - data_block_mean) / \
                         data_block.std(axis=0) + data_block_mean
    return data_block