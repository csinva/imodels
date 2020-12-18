'''
Skope-rules aims at learning logical, interpretable rules for "scoping" a target
class, i.e. detecting with high precision instances of this class.

Skope-rules is a trade off between the interpretability of a Decision Tree
and the modelization power of a Random Forest. Code adapted with only minor changes
from [here](https://github.com/scikit-learn-contrib/skope-rules). Full credit to
the authors. You can access the original project and docs `here <http://skope-rules.readthedocs.io/en/latest/>`_

Example
-------

    from sklearn.datasets import load_boston
    from sklearn.metrics import precision_recall_curve
    from matplotlib import pyplot as plt
    from skrules import SkopeRulesClassifier
    
    dataset = load_boston()
    clf = SkopeRulesClassifier(max_depth_duplication=None,
                     n_estimators=30,
                     precision_min=0.2,
                     recall_min=0.01,
                     feature_names=dataset.feature_names)
    
    X, y = dataset.data, dataset.target > 25
    X_train, y_train = X[:len(y)//2], y[:len(y)//2]
    X_test, y_test = X[len(y)//2:], y[len(y)//2:]
    clf.fit(X_train, y_train)
    y_score = clf.score_top_rules(X_test) # Get a risk score for each test example
    precision, recall, _ = precision_recall_curve(y_test, y_score)
    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision Recall curve')
    plt.show()

Links with existing literature
-------------------------------

The main advantage of decision rules is that they are offering interpretable models. The problem of generating such
rules has been widely considered in machine learning, see e.g. RuleFit [1], Slipper [2], LRI [3], MLRules[4].

A decision rule is a logical expression of the form "IF conditions THEN response". In a binary classification setting,
if an instance satisfies conditions of the rule, then it is assigned to one of the two classes. If this instance does
not satisfy conditions, it remains unassigned.

1) In [2, 3, 4], rules induction is done by considering each single decision rule as a base classifier in an ensemble,
which is built by greedily minimizing some loss function.

2) In [1], rules are extracted from an ensemble of trees; a weighted combination of these rules is then built by solving
a L1-regularized optimization problem over the weights as described in [5].

In this package, we use the second approach. Rules are extracted from tree ensemble, which allow us to take advantage of
existing fast algorithms (such as bagged decision trees, or gradient boosting) to produce such tree ensemble. Too
similar or duplicated rules are then removed, based on a similarity threshold of their supports..

The main goal of this package is to provide rules verifying precision and recall conditions. It still implement a score
(`decision_function`) method, but which does not solve the L1-regularized optimization problem as in [1]. Instead,
weights are simply proportional to the OOB associated precision of the rule.

This package also offers convenient methods to compute predictions with the k most precise rules (cf score_top_rules()
and predict_top_rules() functions).


[1] Friedman and Popescu, Predictive learning via rule ensembles,Technical Report, 2005.

[2] Cohen and Singer, A simple, fast, and effective rule learner, National Conference on Artificial Intelligence, 1999.

[3] Weiss and Indurkhya, Lightweight rule induction, ICML, 2000.

[4] Dembczyński, Kotłowski and Słowiński, Maximum Likelihood Rule Ensembles, ICML, 2008.

[5] Friedman and Popescu, Gradient directed regularization, Technical Report, 2004.
'''
import numbers
from collections.abc import Iterable
from warnings import warn
from typing import Union, List, Tuple

import numpy as np
import pandas
import six
from sklearn.base import BaseEstimator
from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from imodels.rule_set.rule_set import RuleSet
from imodels.util.convert import tree_to_rules
from imodels.util.rule import replace_feature_name, enum_features
from imodels.util.score import score_oob
from imodels.util.prune import prune_mins, deduplicate

INTEGER_TYPES = (numbers.Integral, np.integer)
BASE_FEATURE_NAME = "feature_"


class SkopeRulesClassifier(BaseEstimator, RuleSet):
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

    def fit(self, X, y, feature_names=None, sample_weight=None) -> 'SkopeRulesClassifier':
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
        self.sample_weight = sample_weight
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        if n_classes < 2:
            raise ValueError(
                "This method needs samples of at least 2 classes in the data, but the data contains only one class: %r"
                % self.classes_[0]
            )

        if not isinstance(self.max_depth_duplication, int) and self.max_depth_duplication is not None:
            raise ValueError("max_depth_duplication should be an integer")

        if not set(self.classes_) == {0, 1}:
            warn(
                "Found labels %s. This method assumes target class to be labeled as 1 and normal data to be labeled as "
                "0. Any label different from 0 will be considered as being from the target class."
                % set(self.classes_)
            )
            y = (y > 0)

        # ensure that max_samples is in [1, n_samples]:
        n_samples = X.shape[0]

        if isinstance(self.max_samples, six.string_types):
            raise ValueError(
                'max_samples (%s) is not supported. Valid choices are: "auto", int or float'
                % self.max_samples
            )

        elif isinstance(self.max_samples, INTEGER_TYPES):
            if self.max_samples > n_samples:
                warn(
                    "max_samples (%s) is greater than the total number of samples (%s). max_samples will be set "
                    "to n_samples for estimation."
                    % (self.max_samples, n_samples)
                )
                max_samples = n_samples
            else:
                max_samples = self.max_samples
        else:  # float
            if not (0. < self.max_samples <= 1.):
                raise ValueError("max_samples must be in (0, 1], got %r" % self.max_samples)
            max_samples = int(self.max_samples * X.shape[0])
        self.max_samples_ = max_samples
        self._max_depths = self.max_depth if isinstance(self.max_depth, Iterable) else [self.max_depth]

        self.feature_names_, self.feature_dict_ = enum_features(X, feature_names)

        self.tree_generators = self._get_tree_ensemble()
        self._fit_tree_ensemble(X, y)

        extracted_rules = self._extract_rules()
        scored_rules = self._score_rules(X, y, extracted_rules)
        self.rules_ = self._prune_rules(scored_rules)

        self.rules_without_feature_names_ = self.rules_
        self.rules_ = [
            (replace_feature_name(rule, self.feature_dict_), perf) for rule, perf in self.rules_
        ]
        return self

    def predict(self, X) -> np.ndarray:
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

        return np.array((self.eval_weighted_rule_sum(X) > 0), dtype=int)

    def predict_proba(self, X) -> np.ndarray:
        '''Predict probability of a particular sample being an outlier or not

        '''
        y = self.rules_vote(X) / len(self.rules_without_feature_names_)
        return np.vstack((1 - y, y)).transpose()

    def rules_vote(self, X) -> np.ndarray:
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

    def score_top_rules(self, X) -> np.ndarray:
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

    def predict_top_rules(self, X, n_rules) -> np.ndarray:
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

    def _get_tree_ensemble(self) -> Union[List[BaggingClassifier], List[BaggingRegressor]]:

        for ensemble_class, tree_class in [
            (BaggingClassifier, DecisionTreeClassifier), (BaggingRegressor, DecisionTreeRegressor)
        ]:

            ensembles = []

            for max_depth in self._max_depths:
                bagging_clf = ensemble_class(
                    base_estimator=tree_class(
                        max_depth=max_depth,
                        max_features=self.max_features,
                        min_samples_split=self.min_samples_split
                    ),
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
                    verbose=self.verbose
                )
                ensembles.append(bagging_clf)

        return ensembles

    def _fit_tree_ensemble(self, X, y) -> None:
        y_reg = y
        if self.sample_weight is not None:
            sample_weight = check_array(self.sample_weight, ensure_2d=False)
            weights = sample_weight - sample_weight.min()
            contamination = float(sum(y)) / len(y)
            y_reg = (
                    pow(weights, 0.5) * 0.5 / contamination * (y > 0) -
                    pow((weights).mean(), 0.5) * (y == 0)
            )
            y_reg = 1. / (1 + np.exp(-y_reg))  # sigmoid

        for e in self.tree_generators[:len(self.tree_generators) // 2]:
            e.fit(X, y)

        for e in self.tree_generators[len(self.tree_generators) // 2:]:
            e.fit(X, y_reg)

    def _extract_rules(self):
        self.estimators_, self.estimators_samples_, self.estimators_features_ = [], [], []
        for ensemble in self.tree_generators:
            self.estimators_ += ensemble.estimators_
            self.estimators_samples_ += ensemble.estimators_samples_
            self.estimators_features_ += ensemble.estimators_features_

        extracted_rules = []
        for estimator, features in zip(self.estimators_, self.estimators_features_):
            extracted_rules.append(tree_to_rules(estimator, np.array(self.feature_names_)[features]))
        return extracted_rules

    def _score_rules(self, X, y, rules):
        return score_oob(X, y, rules, self.estimators_samples_, self.estimators_features_, self.feature_names_)

    def _prune_rules(self, rules):
        return deduplicate(
            prune_mins(rules, self.precision_min, self.recall_min),
            self.max_depth_duplication
        )
