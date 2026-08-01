import os
import random
import unittest

import numpy as np
import pandas as pd

from imodels.discretization.discretizer import BasicDiscretizer, RFDiscretizer

path_to_tests = os.path.dirname(os.path.realpath(__file__))


# class TestBasicDiscretizer(unittest.TestCase):
#     def setup(self):
#         np.random.seed(13)
#         random.seed(13)

#     def test_discretizer_simple(self):
#         """ test discretizer on small synthetic dataset
#         """

#         X = np.array([[1, 99, 43, 34],
#                       [1, 76, 22, 10],
#                       [0, 83, 11, 0],
#                       [0, 99, 74, 33],
#                       [0, 53, 40, 34]])
#         X = pd.DataFrame(X)

#         X_test = np.array([[1, 50, 10, 20],
#                            [0, 29, 70, 10],
#                            [3, 80, 60, 50],
#                            [0, 100, 30, 10]])
#         X_test = pd.DataFrame(X_test)

#         discretizer = BasicDiscretizer(n_bins=2, dcols=[],
#                                        encode="onehot", strategy="quantile",
#                                        onehot_drop="if_binary")
#         discretizer.fit(X)
#         Xd = discretizer.transform(X)
#         Xd_test = discretizer.transform(X_test)
#         Xd2 = discretizer.fit_transform(X)

#         Xd_expected = pd.DataFrame({"0_1": [1, 1, 0, 0, 0],
#                                     "1_1": [1, 0, 1, 1, 0],
#                                     "2_1": [1, 0, 0, 1, 1],
#                                     "3_1": [1, 0, 0, 1, 1]})
#         Xd_test_expected = pd.DataFrame({"0_1": [1, 0, 1, 0],
#                                          "1_1": [0, 0, 0, 1],
#                                          "2_1": [0, 1, 1, 0],
#                                          "3_1": [0, 0, 1, 0]})
#         assert Xd.equals(Xd_expected)
#         assert Xd.equals(Xd2)
#         assert Xd_test.equals(Xd_test_expected)

# def test_discretizer(self):

#     dir_data = oj("../../Subgroups/supervised-subgroups/data/enhancer_small")
#     X_train = pd.read_csv(oj(dir_data, "X_train.csv"), index_col = 0)
#     X_test = pd.read_csv(oj(dir_data, "X_test.csv"), index_col = 0)
#     Y_train = pd.read_csv(oj(dir_data, "Y_train.csv"))['y']
#     Y_test = pd.read_csv(oj(dir_data, "Y_test.csv"))['y']
#     X_train.columns = X_train.columns.str.replace("_", "")
#     X_test.columns = X_test.columns.str.replace("_", "")

#     discretizer = BasicDiscretizer(n_bins = 4, dcols = list(X_train.columns)[:40],
#                                    encode = "onehot", strategy = "quantile",
#                                    onehot_drop = "if_binary")
#     discretizer.fit(X_train)
#     Xd = discretizer.transform(X_train)
#     Xd_test = discretizer.transform(X_test)

#     Xd.head()
#     Xd_test.head()

#     assert Xd.shape[1] == Xd_test.shape[1]


# class TestRFDiscretizer(unittest.TestCase):

#     def setup(self):
#         np.random.seed(13)
#         random.seed(13)

#     def test_discretizer_simple(self):
#         """ test discretizer on small synthetic dataset
#         """

#         X = np.array([[1, 99, 43, 34],
#                       [1, 76, 22, 10],
#                       [0, 83, 11, 0],
#                       [0, 99, 74, 33],
#                       [0, 53, 40, 34]])
#         X = pd.DataFrame(X)
#         y = pd.Series([1, 0, 1, 0, 0])

#         X_test = np.array([[1, 50, 10, 20],
#                            [0, 29, 70, 10],
#                            [3, 80, 60, 50],
#                            [0, 100, 30, 10]])
#         X_test = pd.DataFrame(X_test)
#         y_test = pd.Series([1, 0, 1, 0])

#         random.seed(12345)
#         discretizer = RFDiscretizer(rf_model=None, classification=True,
#                                     n_bins=2, dcols=[],
#                                     encode="onehot", strategy="quantile",
#                                     onehot_drop="if_binary")
#         discretizer.fit(X, y)
#         Xd = discretizer.transform(X)
#         Xd_test = discretizer.transform(X_test)
#         Xd2 = discretizer.fit_transform(X, y)

#         Xd_expected = pd.DataFrame({"0_1": [1, 1, 0, 0, 0],
#                                     "1_1": [1, 0, 1, 1, 0],
#                                     "2_1": [1, 0, 0, 1, 0],
#                                     "3_1": [1, 0, 0, 1, 1]})
#         Xd_test_expected = pd.DataFrame({"0_1": [1, 0, 1, 0],
#                                          "1_1": [0, 0, 1, 1],
#                                          "2_1": [0, 1, 1, 0],
#                                          "3_1": [1, 0, 1, 0]})
# assert Xd.equals(Xd_expected)
# assert Xd_test.equals(Xd_test_expected)
# assert Xd.equals(Xd2)

# def test_discretizer(self):

#     dir_data = oj("../../Subgroups/supervised-subgroups/data/enhancer_small")
#     X_train = pd.read_csv(oj(dir_data, "X_train.csv"), index_col = 0)
#     X_test = pd.read_csv(oj(dir_data, "X_test.csv"), index_col = 0)
#     Y_train = pd.read_csv(oj(dir_data, "Y_train.csv"))['y']
#     Y_test = pd.read_csv(oj(dir_data, "Y_test.csv"))['y']
#     X_train.columns = X_train.columns.str.replace("_", "")
#     X_test.columns = X_test.columns.str.replace("_", "")

#     discretizer = RFDiscretizer(rf_model = None, classification = True,
#                                 n_bins = 4, dcols = [],
#                                 encode = "onehot", strategy = "quantile",
#                                 onehot_drop = "if_binary")
#     discretizer.reweight_n_bins(X = X_train, y = Y_train)
#     discretizer.fit(X_train, Y_train)
#     Xd = discretizer.transform(X_train)
#     Xd_test = discretizer.transform(X_test)

#     Xd.head()
#     Xd_test.head()

#     assert discretizer.n_bins.sum() == (len(discretizer.dcols) * 4)


def _discretizer_data():
    rng = np.random.RandomState(0)
    X = pd.DataFrame(rng.randn(60, 3), columns=['a', 'b', 'c'])
    return X, (X['a'] > 0).astype(int)


def test_basic_discretizer_transform():
    """BasicDiscretizer should one-hot encode the requested columns

    Regression test for https://github.com/csinva/imodels/issues/206.
    """
    X, y = _discretizer_data()

    disc = BasicDiscretizer(dcols=['a', 'b'], n_bins=3)
    Xd = disc.fit_transform(X, y)

    # 3 bins for each of the 2 discretized columns, plus the untouched column
    assert Xd.shape == (60, 3 * 2 + 1)
    assert 'c' in Xd.columns
    assert (Xd[[col for col in Xd.columns if col != 'c']].sum(axis=1) == 2).all()

    # discretizing every numeric column by default still works
    Xd_all = BasicDiscretizer(n_bins=3).fit_transform(X, y)
    assert Xd_all.shape == (60, 3 * 3)

    # ordinal encoding returns one column per feature
    Xd_ord = BasicDiscretizer(
        dcols=['a'], n_bins=3, encode='ordinal').fit_transform(X, y)
    assert Xd_ord.shape == (60, 3)
    assert set(Xd_ord['a'].unique()) <= {0, 1, 2}


def test_rf_discretizer_transform():
    """RFDiscretizer should one-hot encode the requested columns

    Regression test for https://github.com/csinva/imodels/issues/206.
    """
    X, y = _discretizer_data()

    disc = RFDiscretizer(dcols=['a', 'b'], n_bins=3, classification=True)
    Xd = disc.fit_transform(X, y)

    assert Xd.shape[0] == 60
    assert 'c' in Xd.columns
    # transform on new data gives the same columns
    assert list(disc.transform(X.iloc[:10]).columns) == list(Xd.columns)
def test_extra_basic_discretizer_onehot_shape():
    """ExtraBasicDiscretizer should return one column per bin

    Regression test for https://github.com/csinva/imodels/issues/211: newer
    scikit-learn returns a sparse matrix from OneHotEncoder, which pandas stored
    as a single column, so building the output frame raised a shape error.
    """
    from imodels.discretization import ExtraBasicDiscretizer

    rng = np.random.RandomState(0)
    X = pd.DataFrame(rng.randn(40, 3), columns=['a', 'b', 'c'])

    disc = ExtraBasicDiscretizer(['a', 'b'], n_bins=3, strategy='uniform')
    Xd = disc.fit_transform(X)

    # 3 bins for each of the 2 discretized columns, plus the untouched column
    assert Xd.shape == (40, 3 * 2 + 1)
    assert 'c' in Xd.columns
    onehot_cols = [col for col in Xd.columns if col != 'c']
    # each row falls in exactly one bin per discretized column
    assert (Xd[onehot_cols].sum(axis=1) == 2).all()

    # transform on new data gives the same columns
    Xd_test = disc.transform(X.iloc[:10])
    assert list(Xd_test.columns) == list(Xd.columns)
    assert Xd_test.shape == (10, Xd.shape[1])


def test_fit_does_not_mutate_the_n_bins_parameter():
    """fit must leave constructor parameters alone

    _validate_n_bins expanded a scalar n_bins into one entry per feature and
    wrote it back over the parameter, so a fitted discretizer could not be
    refitted on a different number of columns.
    """
    from sklearn.base import clone

    rng = np.random.RandomState(0)
    X = pd.DataFrame(rng.randn(200, 3), columns=['a', 'b', 'c'])
    y = (X['a'] > 0).astype(int)

    disc = BasicDiscretizer(n_bins=3)
    disc.fit(X[['a', 'b']], y)

    assert disc.n_bins == 3, 'the parameter should be untouched'
    assert list(disc.n_bins_) == [3, 3], 'per-feature counts belong on n_bins_'
    assert clone(disc).n_bins == 3

    # refitting on more columns used to raise ValueError
    assert disc.fit_transform(X, y).shape[0] == 200


def test_per_feature_n_bins_still_respected():
    rng = np.random.RandomState(0)
    X = pd.DataFrame(rng.randn(200, 3), columns=['a', 'b', 'c'])
    y = (X['a'] > 0).astype(int)

    out = BasicDiscretizer(dcols=['a', 'b'], n_bins=[2, 4]).fit_transform(X, y)
    # the 2-bin feature is binary, so onehot_drop='if_binary' keeps one column
    # of it; 1 + 4 encoded columns, plus the column left alone
    assert out.shape == (200, 1 + 4 + 1)

    # asking for the same number of bins everywhere gives the same total
    same = BasicDiscretizer(dcols=['a', 'b'], n_bins=4).fit_transform(X, y)
    assert same.shape == (200, 4 + 4 + 1)


def test_reweight_n_bins_still_applies_at_fit():
    """reweight_n_bins is called deliberately, so its allocation must survive"""
    rng = np.random.RandomState(0)
    X = pd.DataFrame(rng.randn(300, 4), columns=list('abcd'))
    y = (X['a'] > 0).astype(int)

    disc = RFDiscretizer(dcols=list('abcd'), n_bins=4, classification=True)
    disc.reweight_n_bins(X, y)
    reallocated = np.array(disc.n_bins, copy=True)

    disc.fit(X, y)
    assert np.array_equal(np.asarray(disc.n_bins_), reallocated)
    assert np.sum(reallocated) == 4 * 4, 'total bins are preserved'
