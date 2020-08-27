import numpy as np
from collections import Counter, Iterable
import pandas
import numbers
from warnings import warn

from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import check_classification_targets
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import BaggingClassifier, BaggingRegressor
# from sklearn.externals import six
import six
from sklearn.tree import _tree

from .rule import Rule, replace_feature_name

INTEGER_TYPES = (numbers.Integral, np.integer)
BASE_FEATURE_NAME = "__C__"


class SkopeRules(BaseEstimator):
    """An easy-interpretable classifier optimizing simple logical rules.

    Parameters
    ----------

    feature_names : list of str, optional
        The names of each feature to be used for returning rules in string
        format.

    precision_min : float, optional (default=0.5)
        The minimal precision of a rule to be selected.

    recall_min : float, optional (default=0.01)
        The minimal recall of a rule to be selected.

    n_estimators : int, optional (default=10)
        The number of base estimators (rules) to use for prediction. More are
        built before selection. All are available in the estimators_ attribute.

    max_samples : int or float, optional (default=.8)
        The number of samples to draw from X to train each decision tree, from
        which rules are generated and selected.
            - If int, then draw `max_samples` samples.
            - If float, then draw `max_samples * X.shape[0]` samples.
        If max_samples is larger than the number of samples provided,
        all samples will be used for all trees (no sampling).

    max_samples_features : int or float, optional (default=1.0)
        The number of features to draw from X to train each decision tree, from
        which rules are generated and selected.
            - If int, then draw `max_features` features.
            - If float, then draw `max_features * X.shape[1]` features.

    bootstrap : boolean, optional (default=False)
        Whether samples are drawn with replacement.

    bootstrap_features : boolean, optional (default=False)
        Whether features are drawn with replacement.

    max_depth : integer or List or None, optional (default=3)
        The maximum depth of the decision trees. If None, then nodes are
        expanded until all leaves are pure or until all leaves contain less
        than min_samples_split samples.
        If an iterable is passed, you will train n_estimators
        for each tree depth. It allows you to create and compare
        rules of different length.

    max_depth_duplication : integer, optional (default=None)
        The maximum depth of the decision tree for rule deduplication,
        if None then no deduplication occurs.

    max_features : int, float, string or None, optional (default="auto")
        The number of features considered (by each decision tree) when looking
        for the best split:

        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a percentage and
          `int(max_features * n_features)` features are considered at each
          split.
        - If "auto", then `max_features=sqrt(n_features)`.
        - If "sqrt", then `max_features=sqrt(n_features)` (same as "auto").
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    min_samples_split : int, float, optional (default=2)
        The minimum number of samples required to split an internal node for
        each decision tree.
            - If int, then consider `min_samples_split` as the minimum number.
            - If float, then `min_samples_split` is a percentage and
              `ceil(min_samples_split * n_samples)` are the minimum
              number of samples for each split.

    n_jobs : integer, optional (default=1)
        The number of jobs to run in parallel for both `fit` and `predict`.
        If -1, then the number of jobs is set to the number of cores.

    random_state : int, RandomState instance or None, optional
        - If int, random_state is the seed used by the random number generator.
        - If RandomState instance, random_state is the random number generator.
        - If None, the random number generator is the RandomState instance used
        by `np.random`.

    verbose : int, optional (default=0)
        Controls the verbosity of the tree building process.

    Attributes
    ----------
    rules_ : dict of tuples (rule, precision, recall, nb).
        The collection of `n_estimators` rules used in the ``predict`` method.
        The rules are generated by fitted sub-estimators (decision trees). Each
        rule satisfies recall_min and precision_min conditions. The selection
        is done according to OOB precisions.

    estimators_ : list of DecisionTreeClassifier
        The collection of fitted sub-estimators used to generate candidate
        rules.

    estimators_samples_ : list of arrays
        The subset of drawn samples (i.e., the in-bag samples) for each base
        estimator.

    estimators_features_ : list of arrays
        The subset of drawn features for each base estimator.

    max_samples_ : integer
        The actual number of samples

    n_features_ : integer
        The number of features when ``fit`` is performed.

    classes_ : array, shape (n_classes,)
        The classes labels.
    """

    def __init__(self,
                 feature_names=None,
                 precision_min=0.5,
                 recall_min=0.01,
                 n_estimators=10,
                 max_samples=.8,
                 max_samples_features=1.,
                 bootstrap=False,
                 bootstrap_features=False,
                 max_depth=3,
                 max_depth_duplication=None,
                 max_features=1.,
                 min_samples_split=2,
                 n_jobs=1,
                 random_state=None,
                 verbose=0):
        self.precision_min = precision_min
        self.recall_min = recall_min
        self.feature_names = feature_names
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_samples_features = max_samples_features
        self.bootstrap = bootstrap
        self.bootstrap_features = bootstrap_features
        self.max_depth = max_depth
        self.max_depth_duplication = max_depth_duplication
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y, sample_weight=None):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape (n_samples,)
            Target vector relative to X. Has to follow the convention 0 for
            normal data, 1 for anomalies.

        sample_weight : array-like, shape (n_samples,) optional
            Array of weights that are assigned to individual samples, typically
            the amount in case of transactions data. Used to grow regression
            trees producing further rules to be tested.
            If not provided, then each sample is given unit weight.

        Returns
        -------
        self : object
            Returns self.
        """

        X, y = check_X_y(X, y)
        check_classification_targets(y)
        self.n_features_ = X.shape[1]

        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        if n_classes < 2:
            raise ValueError("This method needs samples of at least 2 classes"
                             " in the data, but the data contains only one"
                             " class: %r" % self.classes_[0])

        if not isinstance(self.max_depth_duplication, int) \
                and self.max_depth_duplication is not None:
            raise ValueError("max_depth_duplication should be an integer"
                             )
        if not set(self.classes_) == set([0, 1]):
            warn("Found labels %s. This method assumes target class to be"
                 " labeled as 1 and normal data to be labeled as 0. Any label"
                 " different from 0 will be considered as being from the"
                 " target class."
                 % set(self.classes_))
            y = (y > 0)

        # ensure that max_samples is in [1, n_samples]:
        n_samples = X.shape[0]

        if isinstance(self.max_samples, six.string_types):
            raise ValueError('max_samples (%s) is not supported.'
                             'Valid choices are: "auto", int or'
                             'float' % self.max_samples)

        elif isinstance(self.max_samples, INTEGER_TYPES):
            if self.max_samples > n_samples:
                warn("max_samples (%s) is greater than the "
                     "total number of samples (%s). max_samples "
                     "will be set to n_samples for estimation."
                     % (self.max_samples, n_samples))
                max_samples = n_samples
            else:
                max_samples = self.max_samples
        else:  # float
            if not (0. < self.max_samples <= 1.):
                raise ValueError("max_samples must be in (0, 1], got %r"
                                 % self.max_samples)
            max_samples = int(self.max_samples * X.shape[0])

        self.max_samples_ = max_samples

        self.rules_ = {}
        self.estimators_ = []
        self.estimators_samples_ = []
        self.estimators_features_ = []

        # default columns names :
        feature_names_ = [BASE_FEATURE_NAME + x for x in
                          np.arange(X.shape[1]).astype(str)]
        if self.feature_names is not None:
            self.feature_dict_ = {BASE_FEATURE_NAME + str(i): feat
                                  for i, feat in enumerate(self.feature_names)}
        else:
            self.feature_dict_ = {BASE_FEATURE_NAME + str(i): feat
                                  for i, feat in enumerate(feature_names_)}
        self.feature_names_ = feature_names_

        clfs = []
        regs = []

        self._max_depths = self.max_depth \
            if isinstance(self.max_depth, Iterable) else [self.max_depth]

        for max_depth in self._max_depths:
            bagging_clf = BaggingClassifier(
                base_estimator=DecisionTreeClassifier(
                    max_depth=max_depth,
                    max_features=self.max_features,
                    min_samples_split=self.min_samples_split),
                n_estimators=self.n_estimators,
                max_samples=self.max_samples_,
                max_features=self.max_samples_features,
                bootstrap=self.bootstrap,
                bootstrap_features=self.bootstrap_features,
                # oob_score=... XXX may be added
                # if selection on tree perf needed.
                # warm_start=... XXX may be added to increase computation perf.
                n_jobs=self.n_jobs,
                random_state=self.random_state,
                verbose=self.verbose)

            bagging_reg = BaggingRegressor(
                base_estimator=DecisionTreeRegressor(
                    max_depth=max_depth,
                    max_features=self.max_features,
                    min_samples_split=self.min_samples_split),
                n_estimators=self.n_estimators,
                max_samples=self.max_samples_,
                max_features=self.max_samples_features,
                bootstrap=self.bootstrap,
                bootstrap_features=self.bootstrap_features,
                # oob_score=... XXX may be added
                # if selection on tree perf needed.
                # warm_start=... XXX may be added to increase computation perf.
                n_jobs=self.n_jobs,
                random_state=self.random_state,
                verbose=self.verbose)

            clfs.append(bagging_clf)
            regs.append(bagging_reg)

        # define regression target:
        if sample_weight is not None:
            if sample_weight is not None:
                sample_weight = check_array(sample_weight, ensure_2d=False)
            weights = sample_weight - sample_weight.min()
            contamination = float(sum(y)) / len(y)
            y_reg = (
                pow(weights, 0.5) * 0.5 / contamination * (y > 0) -
                pow((weights).mean(), 0.5) * (y == 0))
            y_reg = 1. / (1 + np.exp(-y_reg))  # sigmoid
        else:
            y_reg = y  # same as an other classification bagging

        for clf in clfs:
            clf.fit(X, y)
            self.estimators_ += clf.estimators_
            self.estimators_samples_ += clf.estimators_samples_
            self.estimators_features_ += clf.estimators_features_

        for reg in regs:
            reg.fit(X, y_reg)
            self.estimators_ += reg.estimators_
            self.estimators_samples_ += reg.estimators_samples_
            self.estimators_features_ += reg.estimators_features_

        rules_ = []
        for estimator, samples, features in zip(self.estimators_,
                                                self.estimators_samples_,
                                                self.estimators_features_):

            # Create mask for OOB samples
            mask = ~samples
            if sum(mask) == 0:
                warn("OOB evaluation not possible: doing it in-bag."
                     " Performance evaluation is likely to be wrong"
                     " (overfitting) and selected rules are likely to"
                     " not perform well! Please use max_samples < 1.")
                mask = samples
            rules_from_tree = self._tree_to_rules(
                estimator, np.array(self.feature_names_)[features])

            # XXX todo: idem without dataframe
            X_oob = pandas.DataFrame((X[mask, :])[:, features],
                                     columns=np.array(
                                         self.feature_names_)[features])

            if X_oob.shape[1] > 1:  # otherwise pandas bug (cf. issue #16363)
                y_oob = y[mask]
                y_oob = np.array((y_oob != 0))

                # Add OOB performances to rules:
                rules_from_tree = [(r, self._eval_rule_perf(r, X_oob, y_oob))
                                   for r in set(rules_from_tree)]
                rules_ += rules_from_tree

        # Factorize rules before semantic tree filtering
        rules_ = [
            tuple(rule)
            for rule in
            [Rule(r, args=args) for r, args in rules_]]

        # keep only rules verifying precision_min and recall_min:
        for rule, score in rules_:
            if score[0] >= self.precision_min and score[1] >= self.recall_min:
                if rule in self.rules_:
                    # update the score to the new mean
                    c = self.rules_[rule][2] + 1
                    b = self.rules_[rule][1] + 1. / c * (
                        score[1] - self.rules_[rule][1])
                    a = self.rules_[rule][0] + 1. / c * (
                        score[0] - self.rules_[rule][0])

                    self.rules_[rule] = (a, b, c)
                else:
                    self.rules_[rule] = (score[0], score[1], 1)

        self.rules_ = sorted(self.rules_.items(),
                             key=lambda x: (x[1][0], x[1][1]), reverse=True)

        # Deduplicate the rule using semantic tree
        if self.max_depth_duplication is not None:
            self.rules_ = self.deduplicate(self.rules_)

        self.rules_ = sorted(self.rules_, key=lambda x: - self.f1_score(x))
        self.rules_without_feature_names_ = self.rules_

        # Replace generic feature names by real feature names
        self.rules_ = [(replace_feature_name(rule, self.feature_dict_), perf)
                       for rule, perf in self.rules_]

        return self

    def predict(self, X):
        """Predict if a particular sample is an outlier or not.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32``

        Returns
        -------
        is_outlier : array, shape (n_samples,)
            For each observations, tells whether or not (1 or 0) it should
            be considered as an outlier according to the selected rules.
        """

        return np.array((self.decision_function(X) > 0), dtype=int)

    def decision_function(self, X):
        """Average anomaly score of X of the base classifiers (rules).

        The anomaly score of an input sample is computed as
        the weighted sum of the binary rules outputs, the weight being
        the respective precision of each rule.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.

        Returns
        -------
        scores : array, shape (n_samples,)
            The anomaly score of the input samples.
            The higher, the more abnormal. Positive scores represent outliers,
            null scores represent inliers.

        """
        # Check if fit had been called
        check_is_fitted(self, ['rules_', 'estimators_', 'estimators_samples_',
                               'max_samples_'])

        # Input validation
        X = check_array(X)

        if X.shape[1] != self.n_features_:
            raise ValueError("X.shape[1] = %d should be equal to %d, "
                             "the number of features at training time."
                             " Please reshape your data."
                             % (X.shape[1], self.n_features_))

        df = pandas.DataFrame(X, columns=self.feature_names_)
        selected_rules = self.rules_without_feature_names_

        scores = np.zeros(X.shape[0])
        for (r, w) in selected_rules:
            scores[list(df.query(r).index)] += w[0]

        return scores

    def rules_vote(self, X):
        """Score representing a vote of the base classifiers (rules).

        The score of an input sample is computed as the sum of the binary
        rules outputs: a score of k means than k rules have voted positively.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.

        Returns
        -------
        scores : array, shape (n_samples,)
            The score of the input samples.
            The higher, the more abnormal. Positive scores represent outliers,
            null scores represent inliers.

        """
        # Check if fit had been called
        check_is_fitted(self, ['rules_', 'estimators_', 'estimators_samples_',
                               'max_samples_'])

        # Input validation
        X = check_array(X)

        if X.shape[1] != self.n_features_:
            raise ValueError("X.shape[1] = %d should be equal to %d, "
                             "the number of features at training time."
                             " Please reshape your data."
                             % (X.shape[1], self.n_features_))

        df = pandas.DataFrame(X, columns=self.feature_names_)
        selected_rules = self.rules_

        scores = np.zeros(X.shape[0])
        for (r, _) in selected_rules:
            scores[list(df.query(r).index)] += 1

        return scores

    def score_top_rules(self, X):
        """Score representing an ordering between the base classifiers (rules).

        The score is high when the instance is detected by a performing rule.
        If there are n rules, ordered by increasing OOB precision, a score of k
        means than the kth rule has voted positively, but not the (k-1) first
        rules.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.

        Returns
        -------
        scores : array, shape (n_samples,)
            The score of the input samples.
            Positive scores represent outliers, null scores represent inliers.

        """
        # Check if fit had been called
        check_is_fitted(self, ['rules_', 'estimators_', 'estimators_samples_',
                               'max_samples_'])

        # Input validation
        X = check_array(X)

        if X.shape[1] != self.n_features_:
            raise ValueError("X.shape[1] = %d should be equal to %d, "
                             "the number of features at training time."
                             " Please reshape your data."
                             % (X.shape[1], self.n_features_))

        df = pandas.DataFrame(X, columns=self.feature_names_)
        selected_rules = self.rules_without_feature_names_

        scores = np.zeros(X.shape[0])
        for (k, r) in enumerate(list((selected_rules))):
            scores[list(df.query(r[0]).index)] = np.maximum(
                len(selected_rules) - k,
                scores[list(df.query(r[0]).index)])

        return scores
    
    def predict_proba(self, X):
        y = self.score_top_rules(X)
        return np.vstack((1 - y, y)).transpose()

    def predict_top_rules(self, X, n_rules):
        """Predict if a particular sample is an outlier or not,
        using the n_rules most performing rules.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32``

        n_rules : int
            The number of rules used for the prediction. If one of the
            n_rules most performing rules is activated, the prediction
            is equal to 1.

        Returns
        -------
        is_outlier : array, shape (n_samples,)
            For each observations, tells whether or not (1 or 0) it should
            be considered as an outlier according to the selected rules.
        """

        return np.array((self.score_top_rules(X) > len(self.rules_) - n_rules),
                        dtype=int)

    def _tree_to_rules(self, tree, feature_names):
        """
        Return a list of rules from a tree

        Parameters
        ----------
            tree : Decision Tree Classifier/Regressor
            feature_names: list of variable names

        Returns
        -------
        rules : list of rules.
        """
        # XXX todo: check the case where tree is build on subset of features,
        # ie max_features != None

        tree_ = tree.tree_
        feature_name = [
            feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
            for i in tree_.feature
        ]
        rules = []

        def recurse(node, base_name):
            if tree_.feature[node] != _tree.TREE_UNDEFINED:
                name = feature_name[node]
                symbol = '<='
                symbol2 = '>'
                threshold = tree_.threshold[node]
                text = base_name + ["{} {} {}".format(name, symbol, threshold)]
                recurse(tree_.children_left[node], text)

                text = base_name + ["{} {} {}".format(name, symbol2,
                                                      threshold)]
                recurse(tree_.children_right[node], text)
            else:
                rule = str.join(' and ', base_name)
                rule = (rule if rule != ''
                        else ' == '.join([feature_names[0]] * 2))
                # a rule selecting all is set to "c0==c0"
                rules.append(rule)

        recurse(0, [])

        return rules if len(rules) > 0 else 'True'

    def _eval_rule_perf(self, rule, X, y):
        detected_index = list(X.query(rule).index)
        if len(detected_index) <= 1:
            return (0, 0)
        y_detected = y[detected_index]
        true_pos = y_detected[y_detected > 0].sum()
        if true_pos == 0:
            return (0, 0)
        pos = y[y > 0].sum()
        return y_detected.mean(), float(true_pos) / pos

    def deduplicate(self, rules):
        return [max(rules_set, key=self.f1_score)
                for rules_set in self._find_similar_rulesets(rules)]

    def _find_similar_rulesets(self, rules):
        """Create clusters of rules using a decision tree based
        on the terms of the rules

        Parameters
        ----------
        rules : List, List of rules
                The rules that should be splitted in subsets of similar rules

        Returns
        -------
        rules : List of list of rules
                The different set of rules. Each set should be homogeneous

        """
        def split_with_best_feature(rules, depth, exceptions=[]):
            """
            Method to find a split of rules given most represented feature
            """
            if depth == 0:
                return rules

            rulelist = [rule.split(' and ') for rule, score in rules]
            terms = [t.split(' ')[0] for term in rulelist for t in term]
            counter = Counter(terms)
            # Drop exception list
            for exception in exceptions:
                del counter[exception]

            if len(counter) == 0:
                return rules

            most_represented_term = counter.most_common()[0][0]
            # Proceed to split
            rules_splitted = [[], [], []]
            for rule in rules:
                if (most_represented_term + ' <=') in rule[0]:
                    rules_splitted[0].append(rule)
                elif (most_represented_term + ' >') in rule[0]:
                    rules_splitted[1].append(rule)
                else:
                    rules_splitted[2].append(rule)
            new_exceptions = exceptions+[most_represented_term]
            # Choose best term
            return [split_with_best_feature(ruleset,
                                            depth-1,
                                            exceptions=new_exceptions)
                    for ruleset in rules_splitted]

        def breadth_first_search(rules, leaves=None):
            if len(rules) == 0 or not isinstance(rules[0], list):
                if len(rules) > 0:
                    return leaves.append(rules)
            else:
                for rules_child in rules:
                    breadth_first_search(rules_child, leaves=leaves)
            return leaves
        leaves = []
        res = split_with_best_feature(rules, self.max_depth_duplication)
        breadth_first_search(res, leaves=leaves)
        return leaves

    def f1_score(self, x):
        return 2 * x[1][0] * x[1][1] / \
               (x[1][0] + x[1][1]) if (x[1][0] + x[1][1]) > 0 else 0
