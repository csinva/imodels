# This is just a simple wrapper around pycorels: https://github.com/corels/pycorels
import warnings
from typing import List

import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer

from imodels.rule_list.greedy_rule_list import GreedyRuleListClassifier

corels_supported = False
try:
    from corels import CorelsClassifier

    corels_supported = True
except:
    pass


class OptimalRuleListClassifier(GreedyRuleListClassifier if not corels_supported else CorelsClassifier):
    """Certifiably Optimal RulE ListS classifier.
    This class implements the CORELS algorithm, designed to produce human-interpretable, optimal
    rulelists for binary feature data and binary classification. As an alternative to other
    tree based algorithms such as CART, CORELS provides a certificate of optimality for its
    rulelist given a training set, leveraging multiple algorithmic bounds to do so.
    In order to use run the algorithm, create an instance of the `CorelsClassifier` class,
    providing any necessary parameters in its constructor, and then call `fit` to generate
    a rulelist. `printrl` prints the generated rulelist, while `predict` provides
    classification predictions for a separate test dataset with the same features. To determine
    the algorithm's accuracy, run `score` on an evaluation dataset with labels.
    To save a generated rulelist to a file, call `save`. To load it back from the file, call `load`.
    Attributes
    ----------
    c : float, optional (default=0.01)
        Regularization parameter. Higher values penalize longer rulelists.
    n_iter : int, optional (default=10000)
        Maximum number of nodes (rulelists) to search before exiting.
    map_type : str, optional (default="prefix")
        The type of prefix map to use. Supported maps are "none" for no map,
        "prefix" for a map that uses rule prefixes for keys, "captured" for
        a map with a prefix's captured vector as keys.
    policy : str, optional (default="lower_bound")
        The search policy for traversing the tree (i.e. the criterion with which
        to order nodes in the queue). Supported criteria are "bfs", for breadth-first
        search; "curious", which attempts to find the most promising node;
        "lower_bound" which is the objective function evaluated with that rulelist
        minus the default prediction error; "objective" for the objective function
        evaluated at that rulelist; and "dfs" for depth-first search.
    verbosity : list, optional (default=["rulelist"])
        The verbosity levels required. A list of strings, it can contain any
        subset of ["rulelist", "rule", "label", "minor", "samples", "progress", "mine", "loud"].
        An empty list ([]) indicates 'silent' mode.
        - "rulelist" prints the generated rulelist at the end.
        - "rule" prints a summary of each rule generated.
        - "label" prints a summary of the class labels.
        - "minor" prints a summary of the minority bound.
        - "samples" produces a complete dump of the rules, label, and/or minor data. You must also provide at least one of "rule", "label", or "minor" to specify which data you want to dump, or "loud" for all data. The "samples" option often spits out a lot of output.
        - "progress" prints periodic messages as corels runs.
        - "mine" prints debug information while mining rules, including each rule as it is generated.
        - "loud" is the equivalent of ["progress", "label", "rule", "mine", "minor"].
    ablation : int, optional (default=0)
        Specifies addition parameters for the bounds used while searching. Accepted
        values are 0 (all bounds), 1 (no antecedent support bound), and 2 (no
        lookahead bound).
    max_card : int, optional (default=2)
        Maximum cardinality allowed when mining rules. Can be any value greater than
        or equal to 1. For instance, a value of 2 would only allow rules that combine
        at most two features in their antecedents.
    min_support : float, optional (default=0.01)
        The fraction of samples that a rule must capture in order to be used. 1 minus
        this value is also the maximum fraction of samples a rule can capture.
        Can be any value between 0.0 and 0.5.
    References
    ----------
    Elaine Angelino, Nicholas Larus-Stone, Daniel Alabi, Margo Seltzer, and Cynthia Rudin.
    Learning Certifiably Optimal Rule Lists for Categorical Data. KDD 2017.
    Journal of Machine Learning Research, 2018; 19: 1-77. arXiv:1704.01701, 2017
    Examples
    --------
    """

    def __init__(self, c=0.01, n_iter=10000, map_type="prefix", policy="lower_bound",
                 verbosity=[], ablation=0, max_card=2, min_support=0.01, random_state=0):
        if corels_supported:
            super().__init__(c, n_iter, map_type, policy, verbosity, ablation, max_card, min_support)
        else:
            warnings.warn("Should install corels with pip install corels. Using GreedyRuleList instead.")
            super().__init__()
            self.fit = super().fit
            self.predict = super().predict
            self.predict_proba = super().predict_proba
            self.__str__ = super().__str__

        self.random_state = random_state
        self.discretizer = None
        self.str_print = None
        self._estimator_type = 'classifier'

    def fit(self, X, y, feature_names=None, prediction_name="prediction"):
        """
        Build a CORELS classifier from the training set (X, y).
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples. All features must be binary, and the matrix
            is internally converted to dtype=np.uint8.
        y : array-line, shape = [n_samples]
            The target values for the training input. Must be binary.

        feature_names : list, optional(default=None)
            A list of strings of length n_features. Specifies the names of each
            of the features. If an empty list is provided, the feature names
            are set to the default of ["feature1", "feature2"... ].
        prediction_name : string, optional(default="prediction")
            The name of the feature that is being predicted.
        Returns
        -------
        self : obj
        """
        if isinstance(X, pd.DataFrame):
            if feature_names is None:
                feature_names = X.columns.tolist()
            X = X.values
        elif feature_names is None:
            feature_names = ['X_' + str(i) for i in range(X.shape[1])]

        # check if any non-binary values
        if not np.isin(X, [0, 1]).all().all():
            self.discretizer = KBinsDiscretizer(encode='onehot-dense')
            self.discretizer.fit(X, y)
            """
            feature_names = [f'{col}_{b}'
                         for col, bins in zip(feature_names, self.discretizer.n_bins_)
                         for b in range(bins)]
            """
            feature_names = self.discretizer.get_feature_names_out()
            X = self.discretizer.transform(X)

        np.random.seed(self.random_state)
        # feature_names = feature_names.tolist()

        super().fit(X, y, features=feature_names, prediction_name=prediction_name)
        # try:
        self._traverse_rule(X, y, feature_names)
        # except:
        #     self.str_print = None
        self.complexity_ = self._get_complexity()
        return self

    def predict(self, X):
        """
        Predict classifications of the input samples X.
        Arguments
        ---------
        X : array-like, shape = [n_samples, n_features]
            The training input samples. All features must be binary, and the matrix
            is internally converted to dtype=np.uint8. The features must be the same
            as those of the data used to train the model.
        Returns
        -------
        p : array[int] of shape = [n_samples].
            The classifications of the input samples.
        """
        if self.discretizer is not None:
            X = self.discretizer.transform(X)
        return super().predict(X).astype(int)

    def predict_proba(self, X):
        """
        Predict probabilities of the input samples X.
        todo: actually calculate these from training set
        Arguments
        ---------
        X : array-like, shape = [n_samples, n_features]
            The training input samples. All features must be binary, and the matrix
            is internally converted to dtype=np.uint8. The features must be the same
            as those of the data used to train the model.
        Returns
        -------
        p : array[float] of shape = [n_samples, 2].
            The probabilities of the input samples.
        """
        preds = self.predict(X)
        return np.vstack((1 - preds, preds)).transpose()

    def _traverse_rule(self, X: np.ndarray, y: np.ndarray, feature_names: List[str], print_colors=False):
        """Traverse rule and build up string representation

        Parameters
        ----------
        df_features

        Returns
        -------

        """
        str_print = f''
        df = pd.DataFrame(X, columns=feature_names)
        df.loc[:, 'y'] = y
        o = 'y'
        str_print += f'   {df[o].sum()} / {df.shape[0]} (positive class / total)\n'
        if print_colors:
            color_start = '\033[96m'
            color_end = '\033[00m'
        else:
            color_start = ''
            color_end = ''
        if len(self.rl_.rules) > 1:
            str_print += f'\t\u2193 \n'
        else:
            str_print += '   No rules learned\n'
        for j, rule in enumerate(self.rl_.rules[:-1]):
            antecedents = rule['antecedents']
            query = ''
            for i, feat_idx in enumerate(antecedents):
                if i > 0:
                    query += ' & '
                if feat_idx < 0:
                    query += f'(`{feature_names[-feat_idx - 1]}` == 0)'
                else:
                    query += f'(`{feature_names[feat_idx - 1]}` == 1)'
                df_rhs = df.query(query)
                idxs_satisfying_rule = df_rhs.index
                df.drop(index=idxs_satisfying_rule, inplace=True)
                computed_prob = 100 * df_rhs[o].sum() / (df_rhs.shape[0] + 1e-10)

                # add to str_print
                query_print = query.replace('== 1', '').replace('(', '').replace(')', '').replace('`', '')
                str_print += f'{color_start}If {query_print:<35}{color_end} \u2192 {df_rhs[o].sum():>3} / {df_rhs.shape[0]:>4} ({computed_prob:0.1f}%)\n\t\u2193 \n   {df[o].sum():>3} / {df.shape[0]:>5}\t \n'
                if not (j == len(self.rl_.rules) - 2 and i == len(antecedents) - 1):
                    str_print += '\t\u2193 \n'

        self.str_print = str_print

    def __str__(self):
        if corels_supported:
            if self.str_print is not None:
                return 'OptimalRuleList:\n\n' + self.str_print
            else:
                return 'OptimalRuleList:\n\n' + self.rl_.__str__()
        else:
            return super().__str__()

    def _get_complexity(self):
        return sum([len(corule['antecedents']) for corule in self.rl_.rules])


if __name__ == '__main__':
    X = (np.random.randn(40, 2) > 0).astype(int)
    y = (X[:, 0] > 0).astype(int)
    y[-2:] = 1 - y[-2:]
    m = OptimalRuleListClassifier()
    m.fit(X, y)
    print(str(m))
