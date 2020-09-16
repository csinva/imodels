"""Linear model of tree-based decision rules

Clone of code from https://github.com/christophM/rulefit with minor modifications to fit better into an existing project with different requirements / functions needed.

Implementation of a rule based prediction algorithm based on the rulefit algorithm from Friedman and Popescu (PDF)

The algorithm can be used for predicting an output vector y given an input matrix X. In the first step a tree ensemble is generated with gradient boosting. The trees are then used to form rules, where the paths to each node in each tree form one rule. A rule is a binary decision if an observation is in a given node, which is dependent on the input features that were used in the splits. The ensemble of rules together with the original input features are then being input in a L1-regularized linear model, also called Lasso, which estimates the effects of each rule on the output target but at the same time estimating many of those effects to zero.

You can use rulefit for predicting a numeric response (categorial not yet implemented). The input has to be a numpy matrix with only numeric values.

This method implement the RuleFit algorithm

The module structure is the following:

- ``RuleEnsemble`` implements an ensemble of ``Rules``
- ``RuleFit`` implements the RuleFit algorithm

"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier, RandomForestRegressor, \
    RandomForestClassifier
from sklearn.linear_model import LassoCV, LogisticRegressionCV

from .util.rules import RuleCondition, Rule
from .util.transforms import Winsorizer, FriedScale


def extract_rules_from_tree(tree, feature_names=None):
    """Helper to turn a tree into as set of rules
    """
    rules = set()

    def traverse_nodes(node_id=0,
                       operator=None,
                       threshold=None,
                       feature=None,
                       conditions=[]):
        if node_id != 0:
            if feature_names is not None:
                feature_name = feature_names[feature]
            else:
                feature_name = feature
            rule_condition = RuleCondition(feature_index=feature,
                                           threshold=threshold,
                                           operator=operator,
                                           support=tree.n_node_samples[node_id] / float(tree.n_node_samples[0]),
                                           feature_name=feature_name)
            new_conditions = conditions + [rule_condition]
        else:
            new_conditions = []
        ## if not terminal node
        if tree.children_left[node_id] != tree.children_right[node_id]:
            feature = tree.feature[node_id]
            threshold = tree.threshold[node_id]

            left_node_id = tree.children_left[node_id]
            traverse_nodes(left_node_id, "<=", threshold, feature, new_conditions)

            right_node_id = tree.children_right[node_id]
            traverse_nodes(right_node_id, ">", threshold, feature, new_conditions)
        else:  # a leaf node
            if len(new_conditions) > 0:
                new_rule = Rule(new_conditions, tree.value[node_id][0][0])
                rules.update([new_rule])
            else:
                pass  # tree only has a root node!
            return None

    traverse_nodes()

    return rules


class RuleEnsemble():
    """Ensemble of binary decision rules

    This class implements an ensemble of decision rules that extracts rules from
    an ensemble of decision trees.

    Parameters
    ----------
    tree_list: List or array of DecisionTreeClassifier or DecisionTreeRegressor
        Trees from which the rules are created

    feature_names: List of strings, optional (default=None)
        Names of the features

    Attributes
    ----------
    rules: List of Rule
        The ensemble of rules extracted from the trees
    """

    def __init__(self,
                 tree_list,
                 feature_names=None):
        self.tree_list = tree_list
        self.feature_names = feature_names
        self.rules = set()
        ## TODO: Move this out of __init__
        self._extract_rules()
        self.rules = list(self.rules)

    def _extract_rules(self):
        """Recursively extract rules from each tree in the ensemble

        """
        for tree in self.tree_list:
            rules = extract_rules_from_tree(tree[0].tree_, feature_names=self.feature_names)
            self.rules.update(rules)

    def filter_rules(self, func):
        self.rules = filter(lambda x: func(x), self.rules)

    def filter_short_rules(self, k):
        self.filter_rules(lambda x: len(x.conditions) > k)

    def transform(self, X, coefs=None):
        """Transform dataset.

        Parameters
        ----------
        X:      array-like matrix, shape=(n_samples, n_features)
        coefs:  (optional) if supplied, this makes the prediction
                slightly more efficient by setting rules with zero 
                coefficients to zero without calling Rule.transform().
        Returns
        -------
        X_transformed: array-like matrix, shape=(n_samples, n_out)
            Transformed dataset. Each column represents one rule.
        """
        rule_list = list(self.rules)
        if coefs is None:
            return np.array([rule.transform(X) for rule in rule_list]).T
        else:  # else use the coefs to filter the rules we bother to interpret
            res = np.array(
                [rule_list[i_rule].transform(X) for i_rule in np.arange(len(rule_list)) if coefs[i_rule] != 0]).T
            res_ = np.zeros([X.shape[0], len(rule_list)])
            res_[:, coefs != 0] = res
            return res_

    def __str__(self):
        return (map(lambda x: x.__str__(), self.rules)).__str__()


class RuleFitRegressor(BaseEstimator, TransformerMixin):
    """Rulefit class


    Parameters
    ----------
        tree_size:      Number of terminal nodes in generated trees. If exp_rand_tree_size=True, 
                        this will be the mean number of terminal nodes.
        sample_fract:   fraction of randomly chosen training observations used to produce each tree. 
                        FP 2004 (Sec. 2)
        max_rules:      approximate total number of rules generated for fitting. Note that actual
                        number of rules will usually be lower than this due to duplicates.
        memory_par:     scale multiplier (shrinkage factor) applied to each new tree when 
                        sequentially induced. FP 2004 (Sec. 2)
        rfmode:         'regress' for regression or 'classify' for binary classification.
        lin_standardise: If True, the linear terms will be standardised as per Friedman Sec 3.2
                        by multiplying the winsorised variable by 0.4/stdev.
        lin_trim_quantile: If lin_standardise is True, this quantile will be used to trim linear 
                        terms before standardisation.
        exp_rand_tree_size: If True, each boosted tree will have a different maximum number of 
                        terminal nodes based on an exponential distribution about tree_size. 
                        (Friedman Sec 3.3)
        model_type:     'r': rules only; 'l': linear terms only; 'rl': both rules and linear terms
        random_state:   Integer to initialise random objects and provide repeatability.
        tree_generator: Optional: this object will be used as provided to generate the rules. 
                        This will override almost all the other properties above. 
                        Must be GradientBoostingRegressor or GradientBoostingClassifier, optional (default=None)

    Attributes
    ----------
    rule_ensemble: RuleEnsemble
        The rule ensemble

    feature_names: list of strings, optional (default=None)
        The names of the features (columns)

    """

    def __init__(self, tree_size=4, sample_fract='default', max_rules=2000,
                 memory_par=0.01,
                 tree_generator=None,
                 rfmode='regress', lin_trim_quantile=0.025,
                 lin_standardise=True, exp_rand_tree_size=True,
                 model_type='rl', Cs=None, cv=3, random_state=None):
        self.tree_generator = tree_generator
        self.rfmode = rfmode
        self.lin_trim_quantile = lin_trim_quantile
        self.lin_standardise = lin_standardise
        self.winsorizer = Winsorizer(trim_quantile=lin_trim_quantile)
        self.friedscale = FriedScale(self.winsorizer)
        self.stddev = None
        self.mean = None
        self.exp_rand_tree_size = exp_rand_tree_size
        self.max_rules = max_rules
        self.sample_fract = sample_fract
        self.max_rules = max_rules
        self.memory_par = memory_par
        self.tree_size = tree_size
        self.random_state = random_state
        self.model_type = model_type
        self.cv = cv
        self.Cs = Cs

    def fit(self, X, y=None, feature_names=None, verbose=False):
        """Fit and estimate linear combination of rule ensemble

        """
        if type(X) == pd.DataFrame:
            X = X.values
        if type(y) in [pd.DataFrame, pd.Series]:
            y = y.values

        ## Enumerate features if feature names not provided
        N = X.shape[0]
        if feature_names is None:
            self.feature_names = ['feature_' + str(x) for x in range(0, X.shape[1])]
        else:
            self.feature_names = feature_names
        if 'r' in self.model_type:
            ## initialise tree generator
            if self.tree_generator is None:
                n_estimators_default = int(np.ceil(self.max_rules / self.tree_size))
                self.sample_fract_ = min(0.5, (100 + 6 * np.sqrt(N)) / N)
                if self.rfmode == 'regress':
                    self.tree_generator = GradientBoostingRegressor(n_estimators=n_estimators_default,
                                                                    max_leaf_nodes=self.tree_size,
                                                                    learning_rate=self.memory_par,
                                                                    subsample=self.sample_fract_,
                                                                    random_state=self.random_state, max_depth=100)
                else:
                    self.tree_generator = GradientBoostingClassifier(n_estimators=n_estimators_default,
                                                                     max_leaf_nodes=self.tree_size,
                                                                     learning_rate=self.memory_par,
                                                                     subsample=self.sample_fract_,
                                                                     random_state=self.random_state, max_depth=100)

            if self.rfmode == 'regress':
                if type(self.tree_generator) not in [GradientBoostingRegressor, RandomForestRegressor]:
                    raise ValueError("RuleFit only works with RandomForest and BoostingRegressor")
            else:
                if type(self.tree_generator) not in [GradientBoostingClassifier, RandomForestClassifier]:
                    raise ValueError("RuleFit only works with RandomForest and BoostingClassifier")

            ## fit tree generator
            if not self.exp_rand_tree_size:  # simply fit with constant tree size
                self.tree_generator.fit(X, y)
            else:  # randomise tree size as per Friedman 2005 Sec 3.3
                np.random.seed(self.random_state)
                tree_sizes = np.random.exponential(scale=self.tree_size - 2,
                                                   size=int(np.ceil(self.max_rules * 2 / self.tree_size)))
                tree_sizes = np.asarray([2 + np.floor(tree_sizes[i_]) for i_ in np.arange(len(tree_sizes))], dtype=int)
                i = int(len(tree_sizes) / 4)
                while np.sum(tree_sizes[0:i]) < self.max_rules:
                    i = i + 1
                tree_sizes = tree_sizes[0:i]
                self.tree_generator.set_params(warm_start=True)
                curr_est_ = 0
                for i_size in np.arange(len(tree_sizes)):
                    size = tree_sizes[i_size]
                    self.tree_generator.set_params(n_estimators=curr_est_ + 1)
                    self.tree_generator.set_params(max_leaf_nodes=size)
                    random_state_add = self.random_state if self.random_state else 0
                    self.tree_generator.set_params(
                        random_state=i_size + random_state_add)  # warm_state=True seems to reset random_state, such that the trees are highly correlated, unless we manually change the random_sate here.
                    self.tree_generator.get_params()['n_estimators']
                    self.tree_generator.fit(np.copy(X, order='C'), np.copy(y, order='C'))
                    curr_est_ = curr_est_ + 1
                self.tree_generator.set_params(warm_start=False)
            tree_list = self.tree_generator.estimators_
            if isinstance(self.tree_generator, RandomForestRegressor) or isinstance(self.tree_generator,
                                                                                    RandomForestClassifier):
                tree_list = [[x] for x in self.tree_generator.estimators_]

            ## extract rules
            self.rule_ensemble = RuleEnsemble(tree_list=tree_list,
                                              feature_names=self.feature_names)

            ## concatenate original features and rules
            X_rules = self.rule_ensemble.transform(X)

        ## standardise linear variables if requested (for regression model only)
        if 'l' in self.model_type:

            ## standard deviation and mean of winsorized features
            self.winsorizer.train(X)
            winsorized_X = self.winsorizer.trim(X)
            self.stddev = np.std(winsorized_X, axis=0)
            self.mean = np.mean(winsorized_X, axis=0)

            if self.lin_standardise:
                self.friedscale.train(X)
                X_regn = self.friedscale.scale(X)
            else:
                X_regn = X.copy()

                ## Compile Training data
        X_concat = np.zeros([X.shape[0], 0])
        if 'l' in self.model_type:
            X_concat = np.concatenate((X_concat, X_regn), axis=1)
        if 'r' in self.model_type:
            if X_rules.shape[0] > 0:
                X_concat = np.concatenate((X_concat, X_rules), axis=1)

        ## fit Lasso
        if self.rfmode == 'regress':
            if self.Cs is None:  # use defaultshasattr(self.Cs, "__len__"):
                n_alphas = 100
                alphas = None
            elif hasattr(self.Cs, "__len__"):
                n_alphas = None
                alphas = 1. / self.Cs
            else:
                n_alphas = self.Cs
                alphas = None
            self.lscv = LassoCV(n_alphas=n_alphas, alphas=alphas, cv=self.cv, random_state=self.random_state)
            self.lscv.fit(X_concat, y)
            self.coef_ = self.lscv.coef_
            self.intercept_ = self.lscv.intercept_
        else:
            Cs = 10 if self.Cs is None else self.Cs
            self.lscv = LogisticRegressionCV(Cs=Cs, cv=self.cv, penalty='l1', random_state=self.random_state,
                                             solver='liblinear')
            self.lscv.fit(X_concat, y)
            self.coef_ = self.lscv.coef_[0]
            self.intercept_ = self.lscv.intercept_[0]

        return self

    def predict(self, X):
        """Predict outcome for X

        """
        if type(X) == pd.DataFrame:
            X = X.values.astype(np.float32)

        X_concat = np.zeros([X.shape[0], 0])
        if 'l' in self.model_type:
            if self.lin_standardise:
                X_concat = np.concatenate((X_concat, self.friedscale.scale(X)), axis=1)
            else:
                X_concat = np.concatenate((X_concat, X), axis=1)
        if 'r' in self.model_type:
            rule_coefs = self.coef_[-len(self.rule_ensemble.rules):]
            if len(rule_coefs) > 0:
                X_rules = self.rule_ensemble.transform(X, coefs=rule_coefs)
                if X_rules.shape[0] > 0:
                    X_concat = np.concatenate((X_concat, X_rules), axis=1)
        return self.lscv.predict(X_concat)

    def predict_proba(self, X):
        y = self.predict(X)
        return np.vstack((1 - y, y)).transpose()

    def transform(self, X=None, y=None):
        """Transform dataset.

        Parameters
        ----------
        X : array-like matrix, shape=(n_samples, n_features)
            Input data to be transformed. Use ``dtype=np.float32`` for maximum
            efficiency.

        Returns
        -------
        X_transformed: matrix, shape=(n_samples, n_out)
            Transformed data set
        """
        return self.rule_ensemble.transform(X)

    def get_rules(self, exclude_zero_coef=False, subregion=None):
        """Return the estimated rules

        Parameters
        ----------
        exclude_zero_coef: If True (default), returns only the rules with an estimated
                           coefficient not equalt to  zero.

        subregion: If None (default) returns global importances (FP 2004 eq. 28/29), else returns importance over 
                           subregion of inputs (FP 2004 eq. 30/31/32).

        Returns
        -------
        rules: pandas.DataFrame with the rules. Column 'rule' describes the rule, 'coef' holds
               the coefficients and 'support' the support of the rule in the training
               data set (X)
        """

        n_features = len(self.coef_) - len(self.rule_ensemble.rules)
        rule_ensemble = list(self.rule_ensemble.rules)
        output_rules = []
        ## Add coefficients for linear effects
        for i in range(0, n_features):
            if self.lin_standardise:
                coef = self.coef_[i] * self.friedscale.scale_multipliers[i]
            else:
                coef = self.coef_[i]
            if subregion is None:
                importance = abs(coef) * self.stddev[i]
            else:
                subregion = np.array(subregion)
                importance = sum(abs(coef) * abs([x[i] for x in self.winsorizer.trim(subregion)] - self.mean[i])) / len(
                    subregion)
            output_rules += [(self.feature_names[i], 'linear', coef, 1, importance)]

        ## Add rules
        for i in range(0, len(self.rule_ensemble.rules)):
            rule = rule_ensemble[i]
            coef = self.coef_[i + n_features]

            if subregion is None:
                importance = abs(coef) * (rule.support * (1 - rule.support)) ** (1 / 2)
            else:
                rkx = rule.transform(subregion)
                importance = sum(abs(coef) * abs(rkx - rule.support)) / len(subregion)

            output_rules += [(rule.__str__(), 'rule', coef, rule.support, importance)]
        rules = pd.DataFrame(output_rules, columns=["rule", "type", "coef", "support", "importance"])
        if exclude_zero_coef:
            rules = rules.ix[rules.coef != 0]
        return rules
