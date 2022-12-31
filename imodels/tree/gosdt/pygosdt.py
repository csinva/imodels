import json
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.utils import validation

from imodels import GreedyTreeClassifier
from imodels.tree.gosdt.pygosdt_helper import TreeClassifier
from imodels.util import rule

try:
    import gosdt

    gosdt_supported = True
except ImportError:
    gosdt_supported = False


class OptimalTreeClassifier(GreedyTreeClassifier if not gosdt_supported else BaseEstimator):
    def __init__(self,
                 balance=False,
                 cancellation=True,
                 look_ahead=True,
                 similar_support=True,
                 feature_exchange=True,
                 continuous_feature_exchange=True,
                 rule_list=False,
                 diagnostics=False,
                 verbose=False,
                 regularization=0.05,
                 uncertainty_tolerance=0.0,
                 upperbound=0.0,
                 model_limit=1,
                 precision_limit=0,
                 stack_limit=0,
                 tile_limit=0,
                 time_limit=0,
                 worker_limit=1,
                 random_state=None,
                 costs="",
                 model="",
                 profile="",
                 timing="",
                 trace="",
                 tree=""):
        super().__init__()
        self.balance = balance
        self.cancellation = cancellation
        self.look_ahead = look_ahead
        self.similar_support = similar_support
        self.feature_exchange = feature_exchange
        self.continuous_feature_exchange = continuous_feature_exchange
        self.rule_list = rule_list
        self.diagnostics = diagnostics
        self.verbose = verbose
        self.regularization = regularization
        self.uncertainty_tolerance = uncertainty_tolerance
        self.upperbound = upperbound
        self.model_limit = model_limit
        self.precision_limit = precision_limit
        self.stack_limit = stack_limit
        self.tile_limit = tile_limit
        self.time_limit = time_limit
        self.worker_limit = worker_limit
        self.costs = costs
        self.model = model
        self.profile = profile
        self.timing = timing
        self.trace = trace
        self.tree = tree
        self.tree_type = 'gosdt'
        self.random_state = random_state
        if random_state is not None:
            np.random.seed(random_state)

    def load(self, path):
        """
        Parameters
        ---
        path : string
            path to a JSON file representing a model
        """
        with open(path, 'r') as model_source:
            result = model_source.read()
        result = json.loads(result)
        self.tree_ = TreeClassifier(result[0])

    def fit(self, X, y, feature_names=None):
        """
        Parameters
        ---
        X : matrix-like, shape = [n_samples, m_features]
            matrix containing the training samples and features
        y : array-like, shape = [n_samples, 1]
            column containing the correct label for each sample in X

        Modifies
        ---
        trains the model so that this model instance is ready for prediction
        """
        try:
            import gosdt

            if not isinstance(X, pd.DataFrame):
                self.feature_names_ = list(rule.get_feature_dict(X.shape[1], feature_names).keys())
                X = pd.DataFrame(X, columns=self.feature_names_)
            else:
                self.feature_names_ = X.columns

            if not isinstance(y, pd.DataFrame):
                y = pd.DataFrame(y, columns=['target'])

            # gosdt extension expects serialized CSV, which we generate via pandas
            dataset_with_target = pd.concat((X, y), axis=1)

            # Perform C++ extension calls to train the model
            configuration = self._get_configuration()
            gosdt.configure(json.dumps(configuration, separators=(',', ':')))
            result = gosdt.fit(dataset_with_target.to_csv(index=False))

            result = json.loads(result)
            self.tree_ = TreeClassifier(result[0])

            # Record the training time, number of iterations, and graph size required
            self.time_ = gosdt.time()
            self.iterations_ = gosdt.iterations()
            self.size_ = gosdt.size()

        except ImportError:

            warnings.warn(
                "Should install gosdt extension. On x86_64 linux or macOS: "
                "'pip install gosdt-deprecated'. On other platforms, see "
                "https://github.com/keyan3/GeneralizedOptimalSparseDecisionTrees. "
                "Defaulting to Non-optimal DecisionTreeClassifier."
            )

            # dtree = DecisionTreeClassifierWithComplexity()
            # dtree.fit(X, y)
            # self.tree_ = dtree
            super().fit(X, y, feature_names=feature_names)
            self.tree_type = 'dt'

        return self

    def predict(self, X):
        """
        Parameters
        ---
        X : matrix-like, shape = [n_samples, m_features]
            a matrix where each row is a sample to be predicted and each column is a feature to
            be used for prediction

        Returns
        ---
        array-like, shape = [n_samples, 1] : a column where each element is the prediction
            associated with each row
        """
        validation.check_is_fitted(self)
        if self.tree_type == 'gosdt':
            if type(self.tree_) is TreeClassifier and not isinstance(X, pd.DataFrame):
                X = pd.DataFrame(X, columns=self.feature_names_)
            return self.tree_.predict(X)
        else:
            return super().predict(X)

    def predict_proba(self, X):
        validation.check_is_fitted(self)
        if self.tree_type == 'gosdt':
            if type(self.tree_) is TreeClassifier and not isinstance(X, pd.DataFrame):
                X = pd.DataFrame(X, columns=self.feature_names_)
            probs = np.expand_dims(self.tree_.confidence(X), axis=1)
            return np.hstack((1 - probs, probs))
        else:
            return super().predict_proba(X)

    def score(self, X, y, weight=None):
        """
        Parameters
        ---
        X : matrix-like, shape = [n_samples, m_features]
            an n-by-m matrix of sample and their features
        y : array-like, shape = [n_samples,]
            an n-by-1 column of labels associated with each sample
        weight : shape = [n_samples,]
            an n-by-1 column of weights to apply to each sample's misclassification

        Returns
        ---
        real number : the accuracy produced by applying this model over the given dataset, with
            optionals for weighted accuracy
        """
        validation.check_is_fitted(self)
        if type(self.tree_) is TreeClassifier:
            if not isinstance(X, pd.DataFrame):
                X = pd.DataFrame(X, columns=self.feature_names_)
            return self.tree_.score(X, y, weight=weight)
        else:
            return self.tree_.score(X, y, sample_weight=weight)

    def __len__(self):
        """
        Returns
        ---
        natural number : The number of terminal nodes present in this tree
        """
        validation.check_is_fitted(self)
        if type(self.tree_) is TreeClassifier:
            return len(self.tree_)
        else:
            warnings.warn("Using DecisionTreeClassifier due to absence of gosdt package. "
                          "DecisionTreeClassifier does not have this method.")
            return None

    def leaves(self):
        """
        Returns
        ---
        natural number : The number of terminal nodes present in this tree
        """
        validation.check_is_fitted(self)
        if type(self.tree_) is TreeClassifier:
            return self.tree_.leaves()
        else:
            return self.tree_.get_n_leaves()

    def nodes(self):
        """
        Returns
        ---
        natural number : The number of nodes present in this tree
        """
        validation.check_is_fitted(self)
        if type(self.tree_) is TreeClassifier:
            return self.tree_.nodes()
        else:
            warnings.warn("Using DecisionTreeClassifier due to absence of gosdt package. "
                          "DecisionTreeClassifier does not have this method.")
            return None

    def max_depth(self):
        """
        Returns
        ---
        natural number : the length of the longest decision path in this tree. A single-node tree
            will return 1.
        """
        validation.check_is_fitted(self)
        if type(self.tree_) is TreeClassifier:
            return self.tree_.maximum_depth()
        else:
            return self.tree_.get_depth()

    def latex(self):
        """
        Note
        ---
        This method doesn't work well for label headers that contain underscores due to underscore
            being a reserved character in LaTeX

        Returns
        ---
        string : A LaTeX string representing the model
        """
        validation.check_is_fitted(self)
        if type(self.tree_) is TreeClassifier:
            return self.tree_.latex()
        else:
            warnings.warn("Using DecisionTreeClassifier due to absence of gosdt package. "
                          "DecisionTreeClassifier does not have this method.")
            return None

    def json(self):
        """
        Returns
        ---
        string : A JSON string representing the model
        """
        validation.check_is_fitted(self)
        if type(self.tree_) is TreeClassifier:
            return self.tree_.json()
        else:
            warnings.warn("Using DecisionTreeClassifier due to absence of gosdt package. "
                          "DecisionTreeClassifier does not have this method.")
            return None

    def _get_configuration(self):
        return {
            "balance": self.balance,
            "cancellation": self.cancellation,
            "look_ahead": self.look_ahead,
            "similar_support": self.similar_support,
            "feature_exchange": self.feature_exchange,
            "continuous_feature_exchange": self.continuous_feature_exchange,
            "rule_list": self.rule_list,

            "diagnostics": self.diagnostics,
            "verbose": self.verbose,

            "regularization": self.regularization,
            "uncertainty_tolerance": self.uncertainty_tolerance,
            "upperbound": self.upperbound,

            "model_limit": self.model_limit,
            "precision_limit": self.precision_limit,
            "stack_limit": self.stack_limit,
            "tile_limit": self.tile_limit,
            "time_limit": self.time_limit,
            "worker_limit": self.worker_limit,

            "costs": self.costs,
            "model": self.model,
            "profile": self.profile,
            "timing": self.timing,
            "trace": self.trace,
            "tree": self.tree
        }


