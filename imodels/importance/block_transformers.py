from abc import ABC, abstractmethod

import numpy as np
from collections import defaultdict

from sklearn.ensemble import BaseEnsemble
from sklearn.ensemble._forest import _generate_unsampled_indices, _generate_sample_indices

from .local_stumps import make_stumps, tree_feature_transform


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
        """
        Split the data intro training and test partitions given the
        training and test indices. Return the training and test
        block partitioned data objects.

        Parameters
        ----------
        train_indices: array-like of shape (n_train_samples,)
            The indices corresponding to the training samples
        test_indices: array-like of shape (n_test_samples,)
            The indices corresponding to the training samples

        Returns
        -------
        train_blocked_data: BlockPartitionedData
            Returns the training block partitioned data set
        test_blocked_data: BlockPartitionedData
            Returns the test block partitioned data set
        """
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
    """

    def __init__(self):
        self._centers = {}
        self._scales = {}
        self.is_fitted = False

    def fit(self, X):
        """
        Fit (or train) the block transformer using the data matrix X.

        Parameters
        ----------
        X: ndarray
            The data matrix to be used in training
        """
        for k in range(X.shape[1]):
            self._fit_one_feature(X, k)
        self.is_fitted = True

    def check_is_fitted(self):
        """
        Check if the transformer has been fitted. Returns an error if not
        previously fitted.
        """
        if not self.is_fitted:
            raise AttributeError("Transformer has not yet been fitted.")

    def transform_one_feature(self, X, k, center=True, normalize=False):
        """
        Obtain a block of engineered features associated with the original
        feature with index k using the (previously) fitted transformer.

        Parameters
        ----------
        X: ndarray
            The data matrix to be transformed
        k: int
            Index of feature in X to be transformed
        center: bool
            Flag for whether to center the transformed data
        normalize: bool
            Flag for whether to rescale the transformed data to have unit
            variance

        Returns
        -------
        data_block: ndarray
            The block of engineered features associated with the original
            feature with index k.
        """
        data_block = self._transform_one_feature(X, k)
        data_block = self._center_and_normalize(data_block, k, center, normalize)
        return data_block

    def transform(self, X, center=True, normalize=False):
        """
        Transform a data matrix into a BlockPartitionedData object comprising
        one block for each original feature in X using the (previously) fitted
        trasnformer.

        Parameters
        ----------
        X: ndarray
            The data matrix to be transformed
        center: bool
            Flag for whether to center the transformed data
        normalize: bool
            Flag for whether to rescale the transformed data to have unit
            variance

        Returns
        -------
        blocked_data: BlockPartitionedData object
            The transformed data
        """
        self.check_is_fitted()
        n_features = X.shape[1]
        data_blocks = [self.transform_one_feature(X, k, center, normalize) for
                       k in range(n_features)]
        blocked_data = BlockPartitionedData(data_blocks)
        return blocked_data

    def fit_transform_one_feature(self, X, k, center=True, normalize=False):
        """
        Fit the transformer and obtain a block of engineered features associated with
        the original feature with index k using this fitted transformer.

        Parameters
        ----------
        X: ndarray
            The data matrix to be fitted and transformed
        k: int
            Index of feature in X to be fitted and transformed
        center: bool
            Flag for whether to center the transformed data
        normalize: bool
            Flag for whether to rescale the transformed data to have unit
            variance

        Returns
        -------
        data_block: ndarray
            The block of engineered features associated with the original
            feature with index k.
        """
        data_block = self._fit_transform_one_feature(X, k)
        data_block = self._center_and_normalize(data_block, k, center, normalize)
        return data_block

    def fit_transform(self, X, center=True, normalize=False):
        """
        Fit the transformer and transform a data matrix into a BlockPartitionedData
        object comprising one block for each original feature in X using this
        fitted transformer.

        Parameters
        ----------
        X: ndarray
            The data matrix to be transformed
        center: bool
            Flag for whether to center the transformed data
        normalize: bool
            Flag for whether to rescale the transformed data to have unit
            variance

        Returns
        -------
        blocked_data: BlockPartitionedData object
            The transformed data
        """
        n_features = X.shape[1]
        data_blocks = [self.fit_transform_one_feature(X, k, center, normalize) for
                       k in range(n_features)]
        blocked_data = BlockPartitionedData(data_blocks)
        self.is_fitted = True
        return blocked_data

    @abstractmethod
    def _fit_one_feature(self, X, k):
        pass

    @abstractmethod
    def _transform_one_feature(self, X, k):
        pass

    def _fit_transform_one_feature(self, X, k):
        self._fit_one_feature(X, k)
        return self._transform_one_feature(X, k)

    def _center_and_normalize(self, data_block, k, center=True, normalize=False):
        if center:
            data_block = data_block - self._centers[k]
        if normalize:
            if any(self._scales[k] == 0):
                raise Warning("No recaling done."
                              "At least one feature is constant.")
            else:
                data_block = data_block / self._scales[k]
        return data_block


class IdentityTransformer(BlockTransformerBase, ABC):
    """
    Block transformer that creates a block partitioned data object with each
    block k containing only the original feature k.
    """

    def _fit_one_feature(self, X, k):
        self._centers[k] = np.mean(X[:, [k]])
        self._scales[k] = np.std(X[:, [k]])

    def _transform_one_feature(self, X, k):
        return X[:, [k]]


class TreeTransformer(BlockTransformerBase, ABC):
    """
    A block transformer that transforms data using a representation built from
    local decision stumps from a tree or tree ensemble. The transformer also
    comes with metadata on the local decision stumps and methods that allow for
    transformations using sub-representations corresponding to each of the
    original features.

    Parameters
    ----------
    estimator: scikit-learn estimator
        The scikit-learn tree or tree ensemble estimator object.
    data: ndarray
        A data matrix that can be used to update the number of samples in each
        node of the tree(s) in the supplied estimator object. This affects
        the node values of the resulting engineered features.
    """

    def __init__(self, estimator, data=None):
        super().__init__()
        self.estimator = estimator
        self.oob_seed = self.estimator.random_state
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
        self.n_splits = {k: len(stumps) for k, stumps in self.stumps.items()}

    def _fit_one_feature(self, X, k):
        stump_features = tree_feature_transform(self.stumps[k], X)
        self._centers[k] = np.mean(stump_features, axis=0)
        self._scales[k] = np.std(stump_features, axis=0)

    def _transform_one_feature(self, X, k):
        return tree_feature_transform(self.stumps[k], X)

    def _fit_transform_one_feature(self, X, k):
        stump_features = tree_feature_transform(self.stumps[k], X)
        self._centers[k] = np.mean(stump_features, axis=0)
        self._scales[k] = np.std(stump_features, axis=0)
        return stump_features


class CompositeTransformer(BlockTransformerBase, ABC):
    """
    A block transformer that is built by concatenating the blocks of the same
    index from a list of block transformers.

    Parameters
    ----------
    block_transformer_list: list of BlockTransformer objects
        The list of block transformers to combine
    rescale_mode: string in {"max", "mean", None}
        Flag for the type of rescaling to be done to the blocks from different
        base transformers. If "max", divide each block by the max std deviation
        of a column within the block. If "mean", divide each block by the mean
        std deviation of a column within the block. If None, do not rescale.
    drop_features: bool
        Flag for whether to return an empty block if that from the first
        transformer in the list is trivial.
    """

    def __init__(self, block_transformer_list, rescale_mode=None, drop_features=True):
        super().__init__()
        self.block_transformer_list = block_transformer_list
        assert len(self.block_transformer_list) > 0, "Need at least one base" \
                                                     "transformer."
        for transformer in block_transformer_list:
            if hasattr(transformer, "oob_seed") and \
                    transformer.oob_seed is not None:
                self.oob_seed = transformer.oob_seed
                break
        self.rescale_mode = rescale_mode
        self.drop_features = drop_features
        self._rescale_factors = {}
        self._trivial_block_indices = {}

    def _fit_one_feature(self, X, k):
        data_blocks = []
        for block_transformer in self.block_transformer_list:
            data_block = block_transformer.fit_transform_one_feature(
                X, k, center=False, normalize=False)
            data_blocks.append(data_block)

        # Handle trivial blocks
        self._trivial_block_indices[k] = \
            [idx for idx, data_block in enumerate(data_blocks) if
             _empty_or_constant(data_block)]
        if (0 in self._trivial_block_indices[k] and self.drop_features) or \
                (len(self._trivial_block_indices[k]) == len(data_blocks)):
            # If first block is trivial and self.drop_features is True,
            self._centers[k] = np.array([0])
            self._scales[k] = np.array([1])
            return
        else:
            # Remove trivial blocks
            for idx in reversed(self._trivial_block_indices[k]):
                data_blocks.pop(idx)
        self._rescale_factors[k] = _get_rescale_factors(data_blocks, self.rescale_mode)
        composite_block = np.hstack(
            [data_block / scale_factor for data_block, scale_factor in
             zip(data_blocks, self._rescale_factors[k])]
        )
        self._centers[k] = composite_block.mean(axis=0)
        self._scales[k] = composite_block.std(axis=0)

    def _transform_one_feature(self, X, k):
        data_blocks = []
        for block_transformer in self.block_transformer_list:
            data_block = block_transformer.transform_one_feature(
                X, k, center=False, normalize=False)
            data_blocks.append(data_block)
        # Handle trivial blocks
        if (0 in self._trivial_block_indices[k] and self.drop_features) or \
                (len(self._trivial_block_indices[k]) == len(data_blocks)):
            # If first block is trivial and self.drop_features is True,
            # return empty block
            return np.empty((X.shape[0], 0))
        else:
            # Remove trivial blocks
            for idx in reversed(self._trivial_block_indices[k]):
                data_blocks.pop(idx)
        composite_block = np.hstack(
            [data_block / scale_factor for data_block, scale_factor in
             zip(data_blocks, self._rescale_factors[k])]
        )
        return composite_block

    def _fit_transform_one_feature(self, X, k):
        data_blocks = []
        for block_transformer in self.block_transformer_list:
            data_block = block_transformer.fit_transform_one_feature(
                X, k, center=False, normalize=False)
            data_blocks.append(data_block)
        # Handle trivial blocks
        self._trivial_block_indices[k] = \
            [idx for idx, data_block in enumerate(data_blocks) if
             _empty_or_constant(data_block)]
        if (0 in self._trivial_block_indices[k] and self.drop_features) or \
                (len(self._trivial_block_indices[k]) == len(data_blocks)):
            # If first block is trivial and self.drop_features is True,
            # return empty block
            self._centers[k] = np.array([0])
            self._scales[k] = np.array([1])
            return np.empty((X.shape[0], 0))
        else:
            # Remove trivial blocks
            for idx in reversed(self._trivial_block_indices[k]):
                data_blocks.pop(idx)
        self._rescale_factors[k] = _get_rescale_factors(data_blocks, self.rescale_mode)
        composite_block = np.hstack(
            [data_block / scale_factor for data_block, scale_factor in
             zip(data_blocks, self._rescale_factors[k])]
        )
        self._centers[k] = composite_block.mean(axis=0)
        self._scales[k] = composite_block.std(axis=0)
        return composite_block


class MDIPlusDefaultTransformer(CompositeTransformer, ABC):
    """
    Default block transformer used in MDI+. For each original feature, this
    forms a block comprising the local decision stumps, from a single tree
    model, that split on the feature, and appends the original feature.

    Parameters
    ----------
    tree_model: scikit-learn estimator
        The scikit-learn tree estimator object.
    rescale_mode: string in {"max", "mean", None}
        Flag for the type of rescaling to be done to the blocks from different
        base transformers. If "max", divide each block by the max std deviation
        of a column within the block. If "mean", divide each block by the mean
        std deviation of a column within the block. If None, do not rescale.
    drop_features: bool
        Flag for whether to return an empty block if that from the first
        transformer in the list is trivial.
    """
    def __init__(self, tree_model, rescale_mode="max", drop_features=True):
        super().__init__([TreeTransformer(tree_model), IdentityTransformer()],
                         rescale_mode, drop_features)


def _update_n_node_samples(tree, X):
    node_indicators = tree.decision_path(X)
    new_n_node_samples = node_indicators.getnnz(axis=0)
    for i in range(len(new_n_node_samples)):
        tree.tree_.n_node_samples[i] = new_n_node_samples[i]


def _get_rescale_factors(data_blocks, rescale_mode):
    if rescale_mode == "max":
        scale_factors = np.array([max(data_block.std(axis=0)) for
                                  data_block in data_blocks])
    elif rescale_mode == "mean":
        scale_factors = np.array([np.mean(data_block.std(axis=0)) for
                                  data_block in data_blocks])
    elif rescale_mode is None:
        scale_factors = np.ones(len(data_blocks))
    else:
        raise ValueError("Invalid rescale mode.")
    scale_factors = scale_factors / scale_factors[0]
    return scale_factors


def _empty_or_constant(data_block):
    return data_block.shape[1] == 0 or max(data_block.std(axis=0)) == 0


def _blocked_train_test_split(blocked_data, y, oob_seed):
    n_samples = len(y)
    train_indices = _generate_sample_indices(oob_seed, n_samples, n_samples)
    test_indices = _generate_unsampled_indices(oob_seed, n_samples, n_samples)
    train_blocked_data, test_blocked_data = \
        blocked_data.train_test_split(train_indices, test_indices)
    if y.ndim > 1:
        y_train = y[train_indices, :]
        y_test = y[test_indices, :]
    else:
        y_train = y[train_indices]
        y_test = y[test_indices]
    return train_blocked_data, test_blocked_data, y_train, y_test, train_indices, test_indices