import numpy as np
from collections import defaultdict
from joblib import delayed, Parallel

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.decomposition import PCA
from sklearn.ensemble import BaseEnsemble


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
    num_splits_per_feature = [0] * tree_struct.n_features

    def make_stump_iter(node_no, tree_struct, parent_stump, is_right_child, normalize, stumps, num_splits_per_feature):
        """
        Helper function for iteratively making local decision stump objects and appending them to the list stumps.
        """
        new_stump = make_stump(node_no, tree_struct, parent_stump, is_right_child, normalize)
        stumps.append(new_stump)
        num_splits_per_feature[new_stump.feature] += 1
        left_child = tree_struct.children_left[node_no]
        right_child = tree_struct.children_right[node_no]
        if tree_struct.feature[left_child] != -2:  # is not leaf
            make_stump_iter(left_child, tree_struct, new_stump, False, normalize, stumps, num_splits_per_feature)
        if tree_struct.feature[right_child] != -2:  # is not leaf
            make_stump_iter(right_child, tree_struct, new_stump, True, normalize, stumps, num_splits_per_feature)

    make_stump_iter(0, tree_struct, None, None, normalize, stumps, num_splits_per_feature)
    return stumps, num_splits_per_feature


def tree_feature_transform(stumps, X):
    transformed_feature_vectors = []
    for stump in stumps:
        transformed_feature_vec = stump(X)
        transformed_feature_vectors.append(transformed_feature_vec)

    return np.vstack(transformed_feature_vectors).T


class TreeTransformer(TransformerMixin, BaseEstimator):
    """
    A transformer that transforms data using a representation built from local decision stumps from a tree or tree
    ensemble. The transformer also comes with meta data on the local decision stumps and methods that allow
    for transformations using subrepresentations corresponding to each of the original features.

    :param estimator: scikit-learn estimator
        The scikit-learn tree or tree ensemble estimator object
    :param max_components_type: {"median", "max", "}
        Method for choosing the number of components for PCA. Can be either "median", "max",
        or a fraction in [0, 1]. If "median" (respectively "max") then this is set as the median (respectively max
        number of splits on that feature in the RF. If a fraction, then this is set to be the fraction * n
    :param normalize:
    """

    def __init__(self, estimator, max_components_type="median", fraction_chosen=1.0, normalize=False):
        self.estimator = estimator
        self.max_components_type = max_components_type
        self.fraction_chosen = fraction_chosen
        self.normalize = normalize
        num_splits_per_feature_all = []
        if isinstance(estimator, BaseEnsemble):
            self.all_stumps = []
            for tree_model in estimator.estimators_:
                tree_stumps, num_splits_per_feature = make_stumps(tree_model.tree_, normalize)
                self.all_stumps += tree_stumps
                num_splits_per_feature_all.append(num_splits_per_feature)
        else:
            tree_stumps, num_splits_per_feature = make_stumps(estimator.tree_, normalize)
            self.all_stumps = tree_stumps
            num_splits_per_feature_all.append(num_splits_per_feature)
        self.original_feat_to_stump_mapping = defaultdict(list)
        for idx, stump in enumerate(self.all_stumps):
            self.original_feat_to_stump_mapping[stump.feature].append(idx)
        self.pca_transformers = defaultdict(lambda: None)
        self.original_feat_to_transformed_mapping = defaultdict(list)
        self.median_splits = np.median(num_splits_per_feature_all, axis=0)
        self.max_splits = np.max(num_splits_per_feature_all, axis=0)

    def fit(self, X, y=None, always_pca=True):

        def pca_on_stumps(k):
            restricted_stumps = [self.all_stumps[idx] for idx in self.original_feat_to_stump_mapping[k]]
            n_stumps = len(restricted_stumps)
            n_samples = X.shape[0]
            if self.max_components_type == 'median':
                max_components = int(self.median_splits[k] * self.fraction_chosen)
            elif self.max_components_type == "max":
                max_components = int(self.max_splits[k] * self.fraction_chosen)
            elif self.max_components_type == "n":
                max_components = int(n_samples * self.fraction_chosen)
            elif self.max_components_type == "minnp":
                max_components = int(min(n_samples, n_stumps) * self.fraction_chosen)
            elif self.max_components_type == "minfracnp":
                max_components = int(min(n_samples * self.fraction_chosen, n_stumps))
            elif self.max_components_type == "none":
                max_components = np.inf
            elif isinstance(self.max_components_type, int):
                max_components = self.max_components_type
            else:
                raise ValueError("Invalid max components type")

            if n_stumps == 0:
                pca = None
            elif max_components == 0 or (max_components == np.inf):
                pca = None
            elif always_pca or (n_stumps >= max_components): #self.max_components:
                transformed_feature_vectors = tree_feature_transform(restricted_stumps, X)
                pca = PCA(n_components=min(max_components, n_stumps, n_samples))
                pca.fit(transformed_feature_vectors)
            else:
                pca = None
            n_new_feats = min(max_components, n_stumps)
            #            if max_components <= 1.0: #self.max_components
            #                n_new_feats = min(pca.explained_variance_.shape[0], n_stumps)
            #            else:
            #                n_new_feats = min(max_components, n_stumps) #self.max_components
            return pca, n_new_feats

        n_orig_feats = X.shape[1]
        counter = 0
        for k in np.arange(n_orig_feats):
            self.pca_transformers[k], n_new_feats_for_k = pca_on_stumps(k)
            self.original_feat_to_transformed_mapping[k] = np.arange(counter, counter + n_new_feats_for_k)
            counter += n_new_feats_for_k

    def transform(self, X):
        transformed_feature_vectors_sets = []
        for k in range(X.shape[1]):
            v = self.original_feat_to_stump_mapping[k]
            restricted_stumps = [self.all_stumps[idx] for idx in v]
            if len(restricted_stumps) == 0:
                continue
            else:
                transformed_feature_vectors = tree_feature_transform(restricted_stumps, X)
                if self.pca_transformers[k] is not None:
                    transformed_feature_vectors = self.pca_transformers[k].transform(transformed_feature_vectors)
                transformed_feature_vectors_sets.append(transformed_feature_vectors)

        return np.hstack(transformed_feature_vectors_sets)

    def transform_one_feature(self, X, k):
        """
        Obtain the engineered features corresponding to a given original feature X_k

        :param X: Original data matrix
        :param k: Original feature
        :return:
        """
        v = self.original_feat_to_stump_mapping[k]
        restricted_stumps = [self.all_stumps[idx] for idx in v]
        if len(restricted_stumps) == 0:
            return None
        else:
            transformed_feature_vectors = tree_feature_transform(restricted_stumps, X)
            if self.pca_transformers[k] is not None:
                transformed_feature_vectors = self.pca_transformers[k].transform(transformed_feature_vectors)
        return transformed_feature_vectors


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