'''
# Discretization MDLP
Python implementation of Fayyad and Irani's MDLP criterion discretiation algorithm

**Reference:**
Irani, Keki B. "Multi-interval discretization of continuous-valued attributes for classiﬁcation learning." (1993).

'''
__author__ = 'Victor Ruiz, vmr11@pitt.edu'

import numbers
from math import log

import numpy as np
import pandas as pd

from imodels.util.metrics import entropy, cut_point_information_gain


class MDLPDiscretizer(object):
    def __init__(self, dataset, class_label, out_path_data=None, out_path_bins=None, features=None):
        '''
        initializes discretizer object:
            saves raw copy of data and creates self._data with only features to discretize and class
            computes initial entropy (before any splitting)
            self._features = features to be discretized
            self._classes = unique classes in raw_data
            self._class_name = label of class in pandas dataframe
            self._data = partition of data with only features of interest and class
            self._cuts = dictionary with cut points for each feature

        Params
        ------
        dataset
            pandas dataframe with data to discretize
        class_label
            name of the column containing class in input dataframe
        features
            if !None, features that the user wants to discretize specifically
        '''

        if not isinstance(dataset, pd.core.frame.DataFrame):  # class needs a pandas dataframe
            raise AttributeError('input dataset should be a pandas data frame')

        self._data_raw = dataset  # copy or original input data

        self._class_name = class_label

        self._classes = self._data_raw[self._class_name]  # .unique()
        self._classes.drop_duplicates()

        # if user specifies which attributes to discretize
        if features:
            self._features = [f for f in features if f in self._data_raw.columns]  # check if features in dataframe
            missing = set(features) - set(self._features)  # specified columns not in dataframe
            if missing:
                print('WARNING: user-specified features %s not in input dataframe' % str(missing))
        else:  # then we need to recognize which features are numeric
            numeric_cols = self._data_raw._data.get_numeric_data().items
            self._features = [f for f in numeric_cols if f != class_label]
        # other features that won't be discretized
        self._ignored_features = set(self._data_raw.columns) - set(self._features)

        # create copy of data only including features to discretize and class
        self._data = self._data_raw.loc[:, self._features + [class_label]]
        self._data = self._data.infer_objects()  # convert_objects(convert_numeric=True)
        # pre-compute all boundary points in dataset
        self._boundaries = self._compute_boundary_points_all_features()
        # initialize feature bins with empty arrays
        self._cuts = {f: [] for f in self._features}
        # get cuts for all features
        self._all_features_accepted_cutpoints()
        # discretize self._data
        self._apply_cutpoints(out_data_path=out_path_data, out_bins_path=out_path_bins)

    def MDLPC_criterion(self, data, feature, cut_point):
        '''
        Determines whether a partition is accepted according to the MDLPC criterion
        :param feature: feature of interest
        :param cut_point: proposed cut_point
        :param partition_index: index of the sample (dataframe partition) in the interval of interest
        :return: True/False, whether to accept the partition
        '''
        # get dataframe only with desired attribute and class columns, and split by cut_point
        data_partition = data.copy(deep=True)
        data_left = data_partition[data_partition[feature] <= cut_point]
        data_right = data_partition[data_partition[feature] > cut_point]

        # compute information gain obtained when splitting data at cut_point
        cut_point_gain = cut_point_information_gain(dataset=data_partition, cut_point=cut_point,
                                                    feature_label=feature, class_label=self._class_name)
        # compute delta term in MDLPC criterion
        N = len(data_partition)  # number of examples in current partition
        partition_entropy = entropy(data_partition[self._class_name])
        k = len(data_partition[self._class_name].unique())
        k_left = len(data_left[self._class_name].unique())
        k_right = len(data_right[self._class_name].unique())
        entropy_left = entropy(data_left[self._class_name])  # entropy of partition
        entropy_right = entropy(data_right[self._class_name])
        delta = log(3 ** k, 2) - (k * partition_entropy) + (k_left * entropy_left) + (k_right * entropy_right)

        # to split or not to split
        gain_threshold = (log(N - 1, 2) + delta) / N

        if cut_point_gain > gain_threshold:
            return True
        else:
            return False

    def _feature_boundary_points(self, data, feature):
        '''
        Given an attribute, find all potential cut_points (boundary points)
        :param feature: feature of interest
        :param partition_index: indices of rows for which feature value falls whithin interval of interest
        :return: array with potential cut_points
        '''
        # get dataframe with only rows of interest, and feature and class columns
        data_partition = data.copy(deep=True)
        data_partition.sort_values(feature, ascending=True, inplace=True)

        boundary_points = []

        # add temporary columns
        data_partition['class_offset'] = data_partition[self._class_name].shift(
            1)  # column where first value is now second, and so forth
        data_partition['feature_offset'] = data_partition[feature].shift(
            1)  # column where first value is now second, and so forth
        data_partition['feature_change'] = (data_partition[feature] != data_partition['feature_offset'])
        data_partition['mid_points'] = data_partition.loc[:, [feature, 'feature_offset']].mean(axis=1)

        potential_cuts = data_partition[data_partition['feature_change'] == True].index[1:]
        sorted_index = data_partition.index.tolist()

        for row in potential_cuts:
            old_value = data_partition.loc[sorted_index[sorted_index.index(row) - 1]][feature]
            new_value = data_partition.loc[row][feature]
            old_classes = data_partition[data_partition[feature] == old_value][self._class_name].unique()
            new_classes = data_partition[data_partition[feature] == new_value][self._class_name].unique()
            if len(set.union(set(old_classes), set(new_classes))) > 1:
                boundary_points += [data_partition.loc[row]['mid_points']]

        return set(boundary_points)

    def _compute_boundary_points_all_features(self):
        '''
        Computes all possible boundary points for each attribute in self._features (features to discretize)
        :return:
        '''
        boundaries = {}
        for attr in self._features:
            data_partition = self._data.loc[:, [attr, self._class_name]]
            boundaries[attr] = self._feature_boundary_points(data=data_partition, feature=attr)
        return boundaries

    def _boundaries_in_partition(self, data, feature):
        '''
        From the collection of all cut points for all features, find cut points that fall within a feature-partition's
        attribute-values' range
        :param data: data partition (pandas dataframe)
        :param feature: attribute of interest
        :return: points within feature's range
        '''
        range_min, range_max = (data[feature].min(), data[feature].max())
        return set([x for x in self._boundaries[feature] if (x > range_min) and (x < range_max)])

    def _best_cut_point(self, data, feature):
        '''
        Selects the best cut point for a feature in a data partition based on information gain
        :param data: data partition (pandas dataframe)
        :param feature: target attribute
        :return: value of cut point with highest information gain (if many, picks first). None if no candidates
        '''
        candidates = self._boundaries_in_partition(data=data, feature=feature)
        # candidates = self.feature_boundary_points(data=data, feature=feature)
        if not candidates:
            return None
        gains = [(cut, cut_point_information_gain(dataset=data, cut_point=cut, feature_label=feature,
                                                  class_label=self._class_name)) for cut in candidates]
        gains = sorted(gains, key=lambda x: x[1], reverse=True)

        return gains[0][0]  # return cut point

    def _single_feature_accepted_cutpoints(self, feature, partition_index=pd.DataFrame().index):
        '''
        Computes the cuts for binning a feature according to the MDLP criterion
        :param feature: attribute of interest
        :param partition_index: index of examples in data partition for which cuts are required
        :return: list of cuts for binning feature in partition covered by partition_index
        '''
        if partition_index.size == 0:
            partition_index = self._data.index  # if not specified, full sample to be considered for partition

        data_partition = self._data.loc[partition_index, [feature, self._class_name]]

        # exclude missing data:
        if data_partition[feature].isnull().values.any:
            data_partition = data_partition[~data_partition[feature].isnull()]

        # stop if constant or null feature values
        if len(data_partition[feature].unique()) < 2:
            return
        # determine whether to cut and where
        cut_candidate = self._best_cut_point(data=data_partition, feature=feature)
        if cut_candidate == None:
            return
        decision = self.MDLPC_criterion(data=data_partition, feature=feature, cut_point=cut_candidate)

        # apply decision
        if not decision:
            return  # if partition wasn't accepted, there's nothing else to do
        if decision:
            # try:
            # now we have two new partitions that need to be examined
            left_partition = data_partition[data_partition[feature] <= cut_candidate]
            right_partition = data_partition[data_partition[feature] > cut_candidate]
            if left_partition.empty or right_partition.empty:
                return  # extreme point selected, don't partition
            self._cuts[feature] += [cut_candidate]  # accept partition
            self._single_feature_accepted_cutpoints(feature=feature, partition_index=left_partition.index)
            self._single_feature_accepted_cutpoints(feature=feature, partition_index=right_partition.index)
            # order cutpoints in ascending order
            self._cuts[feature] = sorted(self._cuts[feature])
            return

    def _all_features_accepted_cutpoints(self):
        '''
        Computes cut points for all numeric features (the ones in self._features)
        :return:
        '''
        for attr in self._features:
            self._single_feature_accepted_cutpoints(feature=attr)
        return

    def _apply_cutpoints(self, out_data_path=None, out_bins_path=None):
        '''
        Discretizes data by applying bins according to self._cuts. Saves a new, discretized file, and a description of
        the bins
        :param out_data_path: path to save discretized data
        :param out_bins_path: path to save bins description
        :return:
        '''
        bin_label_collection = {}
        for attr in self._features:
            if len(self._cuts[attr]) == 0:
                self._data[attr] = 'All'
                bin_label_collection[attr] = ['All']
            else:
                cuts = [-np.inf] + self._cuts[attr] + [np.inf]
                start_bin_indices = range(0, len(cuts) - 1)
                bin_labels = ['%s_to_%s' % (str(cuts[i]), str(cuts[i + 1])) for i in start_bin_indices]
                bin_label_collection[attr] = bin_labels
                self._data[attr] = pd.cut(x=self._data[attr].values, bins=cuts, right=False, labels=bin_labels,
                                          precision=6, include_lowest=True)

        # reconstitute full data, now discretized
        if self._ignored_features:
            to_return = pd.concat([self._data, self._data_raw[list(self._ignored_features)]], axis=1)
            to_return = to_return[self._data_raw.columns]  # sort columns so they have the original order
        else:
            to_return = self._data

        # save data as csv
        if out_data_path:
            to_return.to_csv(out_data_path)
        # save bins description
        if out_bins_path:
            with open(out_bins_path, 'w') as bins_file:
                print('Description of bins in file: %s' % out_data_path, file=bins_file)
                #                 print(>>bins_file, 'Description of bins in file: %s' % out_data_path)
                for attr in self._features:
                    print('attr: %s\n\t%s' % (attr, ', '.join([bin_label for bin_label in bin_label_collection[attr]])),
                          file=bins_file)


class BRLDiscretizer:

    def __init__(self, feature_labels, verbose=False):
        self.feature_labels_original = feature_labels
        self.verbose = verbose

    def fit(self, X, y, undiscretized_features=[]):

        # check which features are numeric (to be discretized)
        self.discretized_features = []

        X_str_disc = self._encode_strings(X)

        for fi in range(X_str_disc.shape[1]):
            # if not string, has values other than 0 and 1, and not specified as undiscretized
            if (
                    isinstance(X_str_disc[0][fi], numbers.Number)
                    and (not set(np.unique(X_str_disc[:, fi])).issubset({0, 1}))
                    and (len(self.feature_labels) == 0 or
                         len(undiscretized_features) == 0 or
                         self.feature_labels[fi] not in undiscretized_features
            )
            ):
                self.discretized_features.append(self.feature_labels[fi])

        if len(self.discretized_features) > 0:
            if self.verbose:
                print(
                    "Warning: non-categorical data found. Trying to discretize. (Please convert categorical values to "
                    "strings, and/or specify the argument 'undiscretized_features', to avoid this.)")
            X_str_and_num_disc = self.discretize(X_str_disc, y)

            self.discretized_X = X_str_and_num_disc
        else:
            self.discretizer = None
            return

    def discretize(self, X, y):
        '''Discretize the features specified in self.discretized_features
        '''
        if self.verbose:
            print("Discretizing ", self.discretized_features, "...")
        D = pd.DataFrame(np.hstack((X, np.expand_dims(y, axis=1))), columns=list(self.feature_labels) + ["y"])
        self.discretizer = MDLPDiscretizer(dataset=D, class_label="y", features=self.discretized_features)

        cat_data = pd.DataFrame(np.zeros_like(X))
        for i in range(len(self.feature_labels)):
            label = self.feature_labels[i]
            if label in self.discretized_features:
                new_column = label + " : " + self.discretizer._data[label].astype(str)
                cat_data.iloc[:, i] = new_column
            else:
                cat_data.iloc[:, i] = D[label]

        return np.array(cat_data).tolist()

    def _encode_strings(self, X):
        # handle string data
        X_str_disc = pd.DataFrame([])
        for fi in range(X.shape[1]):
            if issubclass(type(X[0][fi]), str):
                new_columns = pd.get_dummies(X[:, fi])
                new_columns.columns = [self.feature_labels_original[fi] + '_' + value for value in new_columns.columns]
                new_columns_colon_format = new_columns.apply(lambda s: s.name + ' : ' + s.astype(str))
                X_str_disc = pd.concat([X_str_disc, new_columns_colon_format], axis=1)
            else:
                X_str_disc = pd.concat([X_str_disc, pd.Series(X[:, fi], name=self.feature_labels_original[fi])], axis=1)
        self.feature_labels = list(X_str_disc.columns)
        return X_str_disc.values

    def transform(self, X, return_onehot=True):

        if type(X) in [pd.DataFrame, pd.Series]:
            X = X.values

        if self.discretizer is None:
            return pd.DataFrame(X, columns=self.feature_labels_original)

        self.data = pd.DataFrame(self._encode_strings(X), columns=self.feature_labels)
        self._apply_cutpoints()
        D = np.array(self.data)

        # prepend feature labels
        Dl = np.copy(D).astype(str).tolist()
        for i in range(len(Dl)):
            for j in range(len(Dl[0])):
                Dl[i][j] = self.feature_labels[j] + " : " + Dl[i][j]

        if not return_onehot:
            return Dl
        else:
            return self.get_onehot_df(Dl)

    @property
    def onehot_df(self):
        return self.get_onehot_df(self.discretized_X)

    def get_onehot_df(self, discretized_X):
        '''Create readable one-hot encoded DataFrame from discretized features
        '''
        data = list(discretized_X[:])

        X_colname_removed = data.copy()
        replace_str_entries_func = lambda s: s.split(' : ')[1] if type(s) is str else s
        for i in range(len(data)):
            X_colname_removed[i] = list(map(replace_str_entries_func, X_colname_removed[i]))

        X_df_categorical = pd.DataFrame(X_colname_removed, columns=self.feature_labels)
        X_df_onehot = pd.get_dummies(X_df_categorical)
        return X_df_onehot

    @property
    def data(self):
        return self.discretizer._data

    @data.setter
    def data(self, value):
        self.discretizer._data = value

    def _apply_cutpoints(self):
        return self.discretizer._apply_cutpoints()
