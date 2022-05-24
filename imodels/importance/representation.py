import numpy as np
from collections import defaultdict
from joblib import delayed, Parallel

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.decomposition import PCA
from sklearn.ensemble import BaseEnsemble


def compare(data, feature, threshold, sign=True):
    """
    :param data: A 2D array of size (n_sample, n_feat)
    :param feature:
    :param threshold:
    :param sign:
    :return:
    """
    if sign:
        return data[:, feature] > threshold
    else:
        return data[:, feature] <= threshold


def compare_all(data, features, thresholds, signs):
    return ~np.logical_xor(data[:, features] > thresholds, signs)


class LocalDecisionStump:

    def __init__(self, feature, threshold, left_val, right_val, a_features, a_thresholds, a_signs):
        """

        :param feature:
        :param threshold:
        :param left_val:
        :param right_val:
        :param a_features: list, comprising ancestor features
        :param a_thresholds: list, comprising ancestor thresholds
        :param a_signs: list, comprising ancestor signs (0: left, 1: right)
        """

        self.feature = feature
        self.threshold = threshold
        self.left_val = left_val
        self.right_val = right_val
        self.a_features = a_features
        self.a_thresholds = a_thresholds
        self.a_signs = a_signs

    def __call__(self, data):
        """

        :param data: A 2D array of query values
        :return:
        """

        root_to_stump_path_indicators = compare_all(data, self.a_features, np.array(self.a_thresholds),
                                                    np.array(self.a_signs))
        in_node = np.all(root_to_stump_path_indicators, axis=1).astype(int)
        is_right = compare(data, self.feature, self.threshold).astype(int)
        result = in_node * (is_right * self.right_val + (1 - is_right) * self.left_val)

        return result

    def __repr__(self):
        return f"LocalDecisionStump(feature={self.feature}, threshold={self.threshold}, left_val={self.left_val}, " \
               f"right_val={self.right_val}, a_features={self.a_features}, a_thresholds={self.a_thresholds}, " \
               f"a_signs={self.a_signs})"


def make_stump(node_no, tree_struct, parent_stump, is_right_child, normalize=False):
    """
    Create a single localized decision stump corresponding to a node

    :param node_no:
    :param tree_struct:
    :param parent_stump:
    :param is_right_child:
    :param normalize:
    :return:
    """
    if parent_stump is None:  # If root node
        a_features = []
        a_thresholds = []
        a_signs = []
    else:
        a_features = parent_stump.a_features + [parent_stump.feature]
        a_thresholds = parent_stump.a_thresholds + [parent_stump.threshold]
        a_signs = parent_stump.a_signs + [is_right_child]

    feature = tree_struct.feature[node_no]
    threshold = tree_struct.threshold[node_no]
    parent_size = tree_struct.n_node_samples[node_no]
    left_child = tree_struct.children_left[node_no]
    right_child = tree_struct.children_right[node_no]
    left_size = tree_struct.n_node_samples[left_child]
    right_size = tree_struct.n_node_samples[right_child]
    if not normalize:
        # return LocalDecisionStump(feature, threshold, -1, 1, a_features, a_thresholds, a_signs)
        left_val = - np.sqrt(right_size / left_size)
        right_val = np.sqrt(left_size / right_size)
    else:
        left_val = - np.sqrt(right_size / (left_size * parent_size))
        right_val = np.sqrt(left_size / (right_size * parent_size))
    return LocalDecisionStump(feature, threshold, left_val, right_val, a_features, a_thresholds, a_signs)


def make_stumps(tree_struct, normalize=False):
    """
    Take sklearn decision tree structure and create a collection of local
    decision stump objects

    :param tree_struct:
    :param normalize:
    :return: list of stumps
    """
    stumps = []
    num_splits_per_feature = [0] * tree_struct.n_features

    def make_stump_iter(node_no, tree_struct, parent_stump, is_right_child, normalize, stumps, num_splits_per_feature):

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

    def __init__(self, estimator, max_components_type="median", fraction_chosen=1.0, normalize=False):
        """

        :param estimator:
        :param max_components_type: Method for choosing the number of components for PCA. Can be either "median", "max",
            or a fraction in [0, 1]. If "median" (respectively "max") then this is set as the median (respectively max
            number of splits on that feature in the RF. If a fraction, then this is set to be the fraction * n
        :param normalize:
        """
        self.estimator = estimator
        self.max_components_type = max_components_type
        self.fraction_chosen = fraction_chosen
        self.normalize = normalize
        num_splits_per_feature_all = []
        if isinstance(estimator, BaseEnsemble):
            self.all_stumps = []
            for tree_model in estimator.estimators_:
                tree_stumps, num_splits_per_feature = make_stumps(tree_model.tree_, normalize)
                self.all_stumps += tree_stumps  # make_stumps(tree_model.tree_, normalize)
                num_splits_per_feature_all.append(num_splits_per_feature)

        else:
            tree_stumps, num_splits_per_feature = make_stumps(estimator.tree_,
                                                              normalize)  # make_stumps(estimator.tree_, normalize)
            self.all_stumps = tree_stumps
            num_splits_per_feature_all.append(num_splits_per_feature)
        self.original_feat_to_stump_mapping = defaultdict(list)
        for idx, stump in enumerate(self.all_stumps):
            self.original_feat_to_stump_mapping[stump.feature].append(idx)
        self.pca_transformers = defaultdict(lambda: None)
        self.original_feat_to_transformed_mapping = defaultdict(list)
        # median_splits_per_feature = [list(col) for col in zip(*median_splits_per_feature)]
        # self.median_splits = [statistics.median(median_splits_per_feature[i]) for i in
        #                       range(len(median_splits_per_feature))]
        self.median_splits = np.median(num_splits_per_feature_all, axis=0)
        self.max_splits = np.max(num_splits_per_feature_all, axis=0)

    def fit(self, X, y=None, parallelize=False, always_pca=True):

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
        if parallelize:
            results = Parallel(n_jobs=8)(delayed(pca_on_stumps)(k) for k in np.arange(n_orig_feats))
            counter = 0
            for k in np.arange(n_orig_feats):
                self.pca_transformers[k], n_new_feats_for_k = results[k]
                self.original_feat_to_transformed_mapping[k] = np.arange(counter, counter + n_new_feats_for_k)
                counter += n_new_feats_for_k

        else:
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

    def get_transformed_X_for_feat(self, X_transformed, feature, max_components):
        '''
        This method takes in the transformed X matrix (applying the node basis) and returns the X matrix restricted to a
        particular feature.
        '''
        X_transformed_feat = X_transformed[:, self.original_feat_to_transformed_mapping[feature]]
        if max_components <= X_transformed_feat.shape[1]:
            return X_transformed[:, self.original_feat_to_transformed_mapping[feature]]
        else:
            return X_transformed[:, self.original_feat_to_stump_mapping[feature]]