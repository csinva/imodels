import numpy as np
from collections import defaultdict

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
    """
    Transform the data matrix X using a mapping derived from a collection of local decision stump functions.

    :param stumps: list of LocalDecisionStump objects
        List of stump functions to use to transform data
    :param X: array-like of shape (n_samples, n_features)
        Original data matrix
    :return: X_transformed: array-like of shape (n_samples, n_stumps)
        Transformed data matrix
    """
    transformed_feature_vectors = []
    for stump in stumps:
        transformed_feature_vec = stump(X)
        transformed_feature_vectors.append(transformed_feature_vec)
    X_transformed = np.vstack(transformed_feature_vectors).T

    return X_transformed


class TreeTransformer(TransformerMixin, BaseEstimator):
    """
    A transformer that transforms data using a representation built from local decision stumps from a tree or tree
    ensemble. The transformer also comes with meta data on the local decision stumps and methods that allow
    for transformations using sub-representations corresponding to each of the original features.

    :param estimator: scikit-learn estimator
        The scikit-learn tree or tree ensemble estimator object
    :param pca: bool
        Flag, if False, the sub-representation for each original feature is just the concatenation of the local
        decision stumps splitting on that feature. If true, the sub-representation are the principal components of the
        set of local decision stump vectors
    :param max_components_type: {"median_splits", "max_splits", "nsamples", "nstumps", "min_nsamples_nstumps",
        "min_fracnsamples_nstumps"} or int
        Method for choosing the max number of components for PCA transformer for each sub-representation corresponding
        to an original feature:
            - If "median_splits", then max_components is alpha * median number of splits on the original feature
              among trees in the estimator
            - If "max_splits", then max_components is alpha * maximum number of splits on the original feature among
              trees in the estimator
            - If "nsamples", then max_components is alpha * n_samples
            - If "nstumps", then max_components is alpha * n_stumps
            - If "min_nsamples_nstumps", then max_components is alpha * min(n_samples, n_stumps), where n_stumps is
              total number of local decision stumps splitting on that feature in the ensemble
            - If "min_fracnsamples_nstumps", then max_components is min(alpha * n_samples, n_stumps), where n_stumps is
              total number of local decision stumps splitting on that feature in the ensemble
            - If int, then max_components is the given integer
    :param alpha: float
        Parameter for adjusting the max number of components for PCA.
    :param normalize: bool
        Flag. If set to True, then divide the nonzero function values for each local decision stump by
        sqrt(n_samples in node) so that the vector of function values on the training set has unit norm. If False,
        then do not divide, so that the vector of function values on the training set has norm equal to n_samples
        in node.
    """

    def __init__(self, estimator, pca=True, max_components_type="min_fracnsamples_nstumps", alpha=0.5, normalize=False):
        self.estimator = estimator
        self.pca = pca
        self.max_components_type = max_components_type
        self.alpha = alpha
        self.normalize = normalize
        # Check if single tree or tree ensemble
        tree_models = estimator.estimators_ if isinstance(estimator, BaseEnsemble) else [estimator]
        # Make stumps for each tree
        num_splits_per_feature_all = []
        self.all_stumps = []
        for tree_model in tree_models:
            tree_stumps, num_splits_per_feature = make_stumps(tree_model.tree_, normalize)
            self.all_stumps += tree_stumps
            num_splits_per_feature_all.append(num_splits_per_feature)
        # Identify the stumps that split on feature k, for each k
        self._original_feat_to_stump_mapping = defaultdict(list)
        for idx, stump in enumerate(self.all_stumps):
            self._original_feat_to_stump_mapping[stump.feature].append(idx)
        # Obtain the median and max number of splits on each feature across trees
        self.median_splits = np.median(num_splits_per_feature_all, axis=0)
        self.max_splits = np.max(num_splits_per_feature_all, axis=0)
        # Initialize list of PCA transformers, one for each set of stumps corresponding to each original feature
        self.pca_transformers = defaultdict(lambda: None)

    def fit(self, X, y=None):

        def pca_on_stumps(k):
            """
            Helper function to fit PCA transformer on stumps corresponding to original feature k
            """
            restricted_stumps = self.get_stumps_for_feature(k)
            n_stumps = len(restricted_stumps)
            n_samples = X.shape[0]
            # Get the number of components to use for PCA
            if self.max_components_type == 'median_splits':
                max_components = int(self.median_splits[k] * self.alpha)
            elif self.max_components_type == "max_splits":
                max_components = int(self.max_splits[k] * self.alpha)
            elif self.max_components_type == "nsamples":
                max_components = int(n_samples * self.alpha)
            elif self.max_components_type == "nstumps":
                max_components = int(n_stumps * self.alpha)
            elif self.max_components_type == "min_nsamples_nstumps":
                max_components = int(min(n_samples, n_stumps) * self.alpha)
            elif self.max_components_type == "min_fracnsamples_nstumps":
                max_components = int(min(n_samples * self.alpha, n_stumps))
            elif isinstance(self.max_components_type, int):
                max_components = self.max_components_type
            else:
                raise ValueError("Invalid max components type")
            n_components = min(max_components, n_stumps, n_samples)
            if n_components == 0:
                pca_transformer = None
            else:
                X_transformed = tree_feature_transform(restricted_stumps, X)
                pca_transformer = PCA(n_components=n_components)
                pca_transformer.fit(X_transformed)

            return pca_transformer

        if self.pca:
            n_features = X.shape[1]
            for k in np.arange(n_features):
                self.pca_transformers[k] = pca_on_stumps(k)
        else:
            pass

    def transform(self, X):
        """
        Obtain all engineered features.

        :param X: array-like of shape (n_samples, n_features)
            Original data matrix
        :return: X_transformed: array-like of shape (n_samples, n_new_features)
            Transformed data matrix
        """
        X_transformed = []
        n_features = X.shape[1]
        for k in range(n_features):
            X_transformed_k = self.transform_one_feature(X, k)
            if X_transformed_k is not None:
                X_transformed.append(X_transformed_k)
        X_transformed = np.hstack(X_transformed)

        return X_transformed

    def transform_one_feature(self, X, k):
        """
        Obtain the engineered features corresponding to a given original feature X_k

        :param X: array-like of shape (n_samples, n_features)
            Original data matrix
        :param k: int
            Index of original feature
        :return: X_transformed: array-like of shape (n_samples, n_new_features)
            Transformed data matrix
        """
        restricted_stumps = self.get_stumps_for_feature(k)
        if len(restricted_stumps) == 0:
            return None
        else:
            X_transformed = tree_feature_transform(restricted_stumps, X)
            if self.pca_transformers[k] is not None:
                X_transformed = self.pca_transformers[k].transform(X_transformed)
        return X_transformed

    def get_stumps_for_feature(self, k):
        """
        Get the list of local decision stumps that split on feature k

        :param k: int
            Index of original feature
        :return: restricted_stumps: list of LocalDecisionStump objects
        """
        restricted_stump_indices = self._original_feat_to_stump_mapping[k]
        restricted_stumps = [self.all_stumps[idx] for idx in restricted_stump_indices]

        return restricted_stumps


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
