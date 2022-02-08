import numpy as np
import pandas as pd
import random
from collections import Counter, defaultdict
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import check_classification_targets, unique_labels
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from imodels.rule_list.bayesian_rule_list.brl_util import (
    default_permsdic, preds_d_t, run_bdl_multichain_serial, merge_chains, get_point_estimate, get_rule_rhs
)
from imodels.rule_list.rule_list import RuleList
from imodels.util.convert import itemsets_to_rules
from imodels.util.extract import extract_fpgrowth
from imodels.util.rule import get_feature_dict, replace_feature_name, Rule


class BayesianRuleListClassifier(BaseEstimator, RuleList, ClassifierMixin):
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

    def __init__(self,
                 listlengthprior=3,
                 listwidthprior=1,
                 maxcardinality=2,
                 minsupport=0.1,
                 alpha=np.array([1., 1.]),
                 n_chains=3,
                 max_iter=50000,
                 class1label="class 1",
                 verbose=False,
                 random_state=42):
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

        self.d_star = None
        self.random_state = random_state
        self.seed()

    def seed(self):
        if self.random_state is not None:
            random.seed(self.random_state)
            np.random.seed(self.random_state)

    def _setlabels(self, X, feature_names=[]):
        if len(feature_names) == 0:
            if type(X) == pd.DataFrame and ('object' in str(X.columns.dtype) or 'str' in str(X.columns.dtype)):
                feature_names = X.columns
            else:
                feature_names = ["ft" + str(i + 1) for i in range(len(X[0]))]
        self.feature_names = feature_names

    def fit(self, X, y, feature_names: list = None, verbose=False):
        """Fit rule lists to data.
        Note: The BRL algorithm requires numeric features to be discretized into bins 
            prior to fitting. See imodels.discretization or sklearn.preprocessing for 
            helpful utilities.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training data

        y : array_like, shape = [n_samples]
            Labels
            
        feature_names : array_like, shape = [n_features], optional (default: [])
            String labels for each feature.
            If empty and X is a DataFrame, column labels are used.
            If empty and X is not a DataFrame, then features are simply enumerated
            
        verbose : bool
            Currently doesn't do anything

        Returns
        -------
        self : returns an instance of self.
        """
        self.seed()

        if len(set(y)) != 2:
            raise ValueError("Only binary classification is supported at this time!")

        X, y = check_X_y(X, y)
        check_classification_targets(y)
        self.n_features_in_ = X.shape[1]
        self.classes_ = unique_labels(y)

        # Check that all features are either categorical or discretized
        if not np.all((X == 1) | (X == 0)):
            raise ValueError("All numeric features must be discretized prior to fitting!")

        self.feature_dict_ = get_feature_dict(X.shape[1], feature_names)
        self.feature_placeholders = np.array(list(self.feature_dict_.keys()))
        self.feature_names = np.array(list(self.feature_dict_.values()))

        X_df = pd.DataFrame(X, columns=self.feature_placeholders)
        itemsets = extract_fpgrowth(X_df, minsupport=self.minsupport,
                                    maxcardinality=self.maxcardinality,
                                    verbose=verbose)

        # Now form the data-vs.-lhs set
        # X[j] is the set of data points that contain itemset j (that is, satisfy rule j)
        for col in X_df.columns:
            # X_df[c] = [c if x == 1 else '' for x in list(X_df[c])]
            X_df[col] = X_df[col].replace({1: col, 0: ''})

        itemset_support_inds = [{}] * (len(itemsets) + 1)
        itemset_support_inds[0] = set(range(X_df.shape[0]))  # the default rule satisfies all data
        for (j, lhs) in enumerate(itemsets):
            itemset_support_inds[j + 1] = set(
                [i for (i, xi) in enumerate(X_df.values) if set(lhs).issubset(xi)])

        # now form lhs_len
        lhs_len = [0]
        for lhs in itemsets:
            lhs_len.append(len(lhs))
        nruleslen = Counter(lhs_len)
        lhs_len = np.array(lhs_len)
        itemsets_all = ['null']
        itemsets_all.extend(itemsets)
        self.itemsets = itemsets_all

        Xtrain = itemset_support_inds
        Ytrain = np.vstack((1 - np.array(y), y)).T.astype(int)

        permsdic = defaultdict(default_permsdic)  # We will store here the MCMC results
        # Do MCMC
        res, Rhat = run_bdl_multichain_serial(
            self.max_iter, self.thinning, self.alpha, self.listlengthprior,
            self.listwidthprior, Xtrain, Ytrain, nruleslen, lhs_len,
            self.maxcardinality, permsdic, self.burnin, self.n_chains,
            [None] * self.n_chains, verbose=self.verbose, seed=self.random_state)

        # Merge the chains
        permsdic = merge_chains(res)

        # The point estimate, BRL-point
        self.d_star = get_point_estimate(permsdic, lhs_len, Xtrain, Ytrain, self.alpha, nruleslen, self.maxcardinality,
                                         self.listlengthprior, self.listwidthprior,
                                         verbose=self.verbose)  # get the point estimate

        if self.d_star:
            # Compute the rule consequent
            self.theta, self.ci_theta = get_rule_rhs(Xtrain, Ytrain, self.d_star, self.alpha, True)

        self.final_itemsets = np.array(self.itemsets, dtype=object)[self.d_star]
        rule_strs = itemsets_to_rules(self.final_itemsets)
        self.rules_without_feature_names_ = [Rule(r) for r in rule_strs]
        self.rules_ = [
            replace_feature_name(rule, self.feature_dict_) for rule in self.rules_without_feature_names_
        ]
        
        self.complexity_ = self._get_complexity()

        return self

    def _get_complexity(self):
        n_rule_terms = sum([len(iset) for iset in self.final_itemsets if type(iset) != str])
        return n_rule_terms + 1

    # def __repr__(self, decimals=1):
    #     if self.d_star:
    #         detect = ""
    #         if self.class1label != "class 1":
    #             detect = "for detecting " + self.class1label
    #         header = "Trained RuleListClassifier " + detect + "\n"
    #         separator = "".join(["="] * len(header)) + "\n"
    #         s = ""
    #         for i, j in enumerate(self.d_star):
    #             if self.itemsets[j] != 'null':
    #                 condition = "ELSE IF " + (
    #                     " AND ".join([str(self.itemsets[j][k]) for k in range(len(self.itemsets[j]))])) + " THEN"
    #             else:
    #                 condition = "ELSE"
    #             s += condition + " probability of " + self.class1label + ": " + str(
    #                 np.round(self.theta[i] * 100, decimals)) + "% (" + str(
    #                 np.round(self.ci_theta[i][0] * 100, decimals)) + "%-" + str(
    #                 np.round(self.ci_theta[i][1] * 100, decimals)) + "%)\n"
    #         return header + separator + s[5:] + separator[1:]
    #     else:
    #         return "(Untrained RuleListClassifier)"

    def __repr__(self, decimals=1):
        if self.d_star:
            detect = ""
            if self.class1label != "class 1":
                detect = "for detecting " + self.class1label
            header = "Trained RuleListClassifier " + detect + "\n"
            separator = "".join(["="] * len(header)) + "\n"
            s = ""
            for i in range(len(self.rules_) + 1):
                if i != len(self.rules_):
                    condition = "ELSE IF " + str(self.rules_[i]) + " THEN"
                else:
                    condition = "ELSE"
                s += condition + " probability of " + self.class1label + ": " + str(
                    np.round(self.theta[i] * 100, decimals)) + "% (" + str(
                    np.round(self.ci_theta[i][0] * 100, decimals)) + "%-" + str(
                    np.round(self.ci_theta[i][1] * 100, decimals)) + "%)\n"
            return header + separator + s[5:] + separator[1:]
        else:
            return "(Untrained RuleListClassifier)"

    def _to_itemset_indices(self, X_df_onehot):
        # X[j] is the set of data points that contain itemset j (that is, satisfy rule j)
        for c in X_df_onehot.columns:
            X_df_onehot[c] = [c if x == 1 else '' for x in list(X_df_onehot[c])]
        X = [set() for j in range(len(self.itemsets))]
        X[0] = set(range(X_df_onehot.shape[0]))  # the default rule satisfies all data
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
        check_is_fitted(self)
        X = check_array(X)

        D = pd.DataFrame(X, columns=self.feature_placeholders)

        N = len(D)
        X2 = self._to_itemset_indices(D)
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
        check_is_fitted(self)
        X = check_array(X)

        # print('predicting!')
        # print('preds_proba', self.predict_proba(X)[:, 1])
        return 1 * (self.predict_proba(X)[:, 1] >= threshold)
