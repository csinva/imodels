from collections import Counter
import numbers
import random

import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import fpgrowth
from sklearn.base import BaseEstimator

from imodels.rule_list.bayesian_rule_list.brl_util import *
from imodels.util.discretization.mdlp import MDLP_Discretizer
from imodels.rule_list.rule_list import RuleList


class BayesianRuleListClassifier(BaseEstimator, RuleList):
    """
    This is a scikit-learn compatible wrapper for the Bayesian Rule List
    classifier developed by Benjamin Letham. It produces a highly
    interpretable model (a list of decision rules) by sampling many different
    rule lists, trying to optimize for compactness and predictive performance.

    Parameters
    ----------
    listlengthprior : int, optional (default=3)
        Prior hyperparameter for expected list length (excluding null rule)

    listwidthprior : int, optional (default=1)
        Prior hyperparameter for expected list width (excluding null rule)
        
    maxcardinality : int, optional (default=2)
        Maximum cardinality of an itemset
        
    minsupport : float, optional (default=0.1)
        Minimum support (fraction between 0 and 1) of an itemset

    alpha : array_like, shape = [n_classes]
        prior hyperparameter for multinomial pseudocounts

    n_chains : int, optional (default=3)
        Number of MCMC chains for inference

    max_iter : int, optional (default=50000)
        Maximum number of iterations
        
    class1label: str, optional (default="class 1")
        Label or description of what the positive class (with y=1) means
        
    verbose: bool, optional (default=True)
        Verbose output
        
    random_state: int
        Random seed
    """

    def __init__(self, listlengthprior=3, listwidthprior=1, maxcardinality=2, minsupport=0.1, alpha=np.array([1., 1.]),
                 n_chains=3, max_iter=50000, class1label="class 1", verbose=True, random_state=42):
        self.listlengthprior = listlengthprior
        self.listwidthprior = listwidthprior
        self.maxcardinality = maxcardinality
        self.minsupport = minsupport
        self.alpha = alpha
        self.n_chains = n_chains
        self.max_iter = max_iter
        self.class1label = class1label
        self.verbose = verbose
        self._zmin = 1

        self.thinning = 1  # The thinning rate
        self.burnin = self.max_iter // 2  # the number of samples to drop as burn-in in-simulation

        self.discretizer = None
        self.d_star = None
        self.random_state = random_state
        self.seed()

    def seed(self):
        if self.random_state is not None:
            random.seed(self.random_state)
            np.random.seed(self.random_state)

    def _setlabels(self, X, feature_labels=[]):
        if len(feature_labels) == 0:
            if type(X) == pd.DataFrame and ('object' in str(X.columns.dtype) or 'str' in str(X.columns.dtype)):
                feature_labels = X.columns
            else:
                feature_labels = ["ft" + str(i + 1) for i in range(len(X[0]))]
        self.feature_labels = feature_labels

    def _discretize_mixed_data(self, X, y, undiscretized_features=[]):
        if type(X) != list:
            X = np.array(X).tolist()

        # check which features are numeric (to be discretized)
        self.discretized_features = []
        for fi in range(len(X[0])):
            # if not string, and not specified as undiscretized
            if isinstance(X[0][fi], numbers.Number) \
                    and (len(self.feature_labels) == 0 or \
                         len(undiscretized_features) == 0 or \
                         self.feature_labels[fi] not in undiscretized_features):
                self.discretized_features.append(self.feature_labels[fi])

        if len(self.discretized_features) > 0:
            if self.verbose:
                print(
                    "Warning: non-categorical data found. Trying to discretize. (Please convert categorical values to "
                    "strings, and/or specify the argument 'undiscretized_features', to avoid this.)")
            X = self.discretize(X, y)

        return X

    def _setdata(self, X, y, feature_labels=[], undiscretized_features=[]):
        self._setlabels(X, feature_labels)
        X = self._discretize_mixed_data(X, y, undiscretized_features)
        return X, y

    def fit(self, X, y, feature_labels: list=None, undiscretized_features=[], verbose=False):
        """Fit rule lists to data

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training data 

        y : array_like, shape = [n_samples]
            Labels
            
        feature_labels : array_like, shape = [n_features], optional (default: [])
            String labels for each feature.
            If empty and X is a DataFrame, column labels are used.
            If empty and X is not a DataFrame, then features are simply enumerated
            
        undiscretized_features : array_like, shape = [n_features], optional (default: [])
            String labels for each feature which is NOT to be discretized. If empty, all numeric features are discretized
            
        verbose : bool
            Currently doesn't do anything

        Returns
        -------
        self : returns an instance of self.
        """
        self.seed()

        if len(set(y)) != 2:
            raise Exception("Only binary classification is supported at this time!")

        # deal with pandas data
        if type(X) in [pd.DataFrame, pd.Series]:
            if feature_labels is None:
                feature_labels = X.columns
            X = X.values
        if type(y) in [pd.DataFrame, pd.Series]:
            y = y.values

        if feature_labels is None:
            feature_labels = [f'X{i}' for i in range(X.shape[1])]
            
        X, y = self._setdata(X, y, feature_labels, undiscretized_features)
        permsdic = defaultdict(default_permsdic)  # We will store here the MCMC results
        data = list(X[:])

        # Now find frequent itemsets

        X_colname_removed = data.copy()
        for i in range(len(data)):
            X_colname_removed[i] = list(map(lambda s: s.split(' : ')[1], X_colname_removed[i]))

        X_df_categorical = pd.DataFrame(X_colname_removed, columns=feature_labels)
        X_df_onehot = pd.get_dummies(X_df_categorical)
        onehot_features = X_df_onehot.columns

        itemsets_df = fpgrowth(X_df_onehot, min_support=self.minsupport, max_len=self.maxcardinality)
        itemsets_indices = [tuple(s[1]) for s in itemsets_df.values]
        itemsets = [np.array(onehot_features)[list(inds)] for inds in itemsets_indices]
        itemsets = list(map(tuple, itemsets))
        if self.verbose:
            print(len(itemsets), 'rules mined')


        # Now form the data-vs.-lhs set
        # X[j] is the set of data points that contain itemset j (that is, satisfy rule j)
        for c in X_df_onehot.columns:
            X_df_onehot[c] = [c if x == 1 else '' for x in list(X_df_onehot[c])]
        X = [{}] * (len(itemsets) + 1)
        X[0] = set(range(len(data)))  # the default rule satisfies all data
        for (j, lhs) in enumerate(itemsets):
            X[j + 1] = set([i for (i, xi) in enumerate(X_df_onehot.values) if set(lhs).issubset(xi)])

        
        
        # now form lhs_len
        lhs_len = [0]
        for lhs in itemsets:
            lhs_len.append(len(lhs))
        nruleslen = Counter(lhs_len)
        lhs_len = array(lhs_len)
        itemsets_all = ['null']
        itemsets_all.extend(itemsets)

        Xtrain, Ytrain, nruleslen, lhs_len, self.itemsets = (
            X, np.vstack((1 - np.array(y), y)).T.astype(int), nruleslen, lhs_len, itemsets_all
        )

        # Do MCMC
        res, Rhat = run_bdl_multichain_serial(self.max_iter, self.thinning, self.alpha, self.listlengthprior,
                                              self.listwidthprior, Xtrain, Ytrain, nruleslen, lhs_len,
                                              self.maxcardinality, permsdic, self.burnin, self.n_chains,
                                              [None] * self.n_chains, verbose=self.verbose, seed=self.random_state)

        # Merge the chains
        permsdic = merge_chains(res)

        ###The point estimate, BRL-point
        self.d_star = get_point_estimate(permsdic, lhs_len, Xtrain, Ytrain, self.alpha, nruleslen, self.maxcardinality,
                                         self.listlengthprior, self.listwidthprior,
                                         verbose=self.verbose)  # get the point estimate

        if self.d_star:
            # Compute the rule consequent
            self.theta, self.ci_theta = get_rule_rhs(Xtrain, Ytrain, self.d_star, self.alpha, True)

        return self

    def discretize(self, X, y):
        '''Discretize the features specified in self.discretized_features
        '''
        if self.verbose:
            print("Discretizing ", self.discretized_features, "...")
        D = pd.DataFrame(np.hstack((X, np.array(y).reshape((len(y), 1)))), columns=list(self.feature_labels) + ["y"])
        self.discretizer = MDLP_Discretizer(dataset=D, class_label="y", features=self.discretized_features)

        cat_data = pd.DataFrame(np.zeros_like(X))
        for i in range(len(self.feature_labels)):
            label = self.feature_labels[i]
            if label in self.discretized_features:
                column = []
                for j in range(len(self.discretizer._data[label])):
                    column += [label + " : " + self.discretizer._data[label][j]]
                cat_data.iloc[:, i] = np.array(column)
            else:
                cat_data.iloc[:, i] = D[label]

        return np.array(cat_data).tolist()

    def _prepend_feature_labels(self, X):
        Xl = np.copy(X).astype(str).tolist()
        for i in range(len(Xl)):
            for j in range(len(Xl[0])):
                Xl[i][j] = self.feature_labels[j] + " : " + Xl[i][j]
        return Xl

    def __str__(self, decimals=1):
        if self.d_star:
            detect = ""
            if self.class1label != "class 1":
                detect = "for detecting " + self.class1label
            header = "Trained RuleListClassifier " + detect + "\n"
            separator = "".join(["="] * len(header)) + "\n"
            s = ""
            for i, j in enumerate(self.d_star):
                if self.itemsets[j] != 'null':
                    condition = "ELSE IF " + (
                        " AND ".join([str(self.itemsets[j][k]) for k in range(len(self.itemsets[j]))])) + " THEN"
                else:
                    condition = "ELSE"
                s += condition + " probability of " + self.class1label + ": " + str(
                    np.round(self.theta[i] * 100, decimals)) + "% (" + str(
                    np.round(self.ci_theta[i][0] * 100, decimals)) + "%-" + str(
                    np.round(self.ci_theta[i][1] * 100, decimals)) + "%)\n"
            return header + separator + s[5:] + separator[1:]
        else:
            return "(Untrained RuleListClassifier)"

    def _to_itemset_indices(self, data):
        X_colname_removed = data.copy()
        for i in range(len(data)):
            X_colname_removed[i] = list(map(lambda s: s.split(' : ')[1], X_colname_removed[i]))
        X_df_categorical = pd.DataFrame(X_colname_removed, columns=self.feature_labels)
        X_df_onehot = pd.get_dummies(X_df_categorical)

        # X[j] is the set of data points that contain itemset j (that is, satisfy rule j)
        for c in X_df_onehot.columns:
            X_df_onehot[c] = [c if x == 1 else '' for x in list(X_df_onehot[c])]
        X = [set() for j in range(len(self.itemsets))]
        X[0] = set(range(len(data)))  # the default rule satisfies all data
        for (j, lhs) in enumerate(self.itemsets):
            if j > 0:
                X[j] = set([i for (i, xi) in enumerate(X_df_onehot.values) if set(lhs).issubset(xi)])
        return X

    def predict_proba(self, X):
        """Compute probabilities of possible outcomes for samples in X.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        T : array-like, shape = [n_samples, n_classes]
            Returns the probability of the sample for each class in
            the model. The columns correspond to the classes in sorted
            order, as they appear in the attribute `classes_`.
        """
        # deal with pandas data
        if type(X) in [pd.DataFrame, pd.Series]:
            X = X.values

        if self.discretizer:
            self.discretizer._data = pd.DataFrame(X, columns=self.feature_labels)
            self.discretizer.apply_cutpoints()
            D = self._prepend_feature_labels(np.array(self.discretizer._data))
        else:
            D = X

        N = len(D)
        X2 = self._to_itemset_indices(D[:])
        P = preds_d_t(X2, np.zeros((N, 1), dtype=int), self.d_star, self.theta)
        return np.vstack((1 - P, P)).T

    def predict(self, X, threshold=0.1):
        """Perform classification on samples in X.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        y_pred : array, shape = [n_samples]
            Class labels for samples in X.
        """
        # deal with pandas data
        if type(X) in [pd.DataFrame, pd.Series]:
            X = X.values
        # print('predicting!')
        # print('preds_proba', self.predict_proba(X)[:, 1])
        return 1 * (self.predict_proba(X)[:, 1] >= threshold)
