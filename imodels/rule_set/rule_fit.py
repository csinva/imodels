"""Linear model of tree-based decision rules based on the rulefit algorithm from Friedman and Popescu.

The algorithm can be used for predicting an output vector y given an input matrix X. In the first step a tree ensemble
is generated with gradient boosting. The trees are then used to form rules, where the paths to each node in each tree
form one rule. A rule is a binary decision if an observation is in a given node, which is dependent on the input features
that were used in the splits. The ensemble of rules together with the original input features are then being input in a
L1-regularized linear model, also called Lasso, which estimates the effects of each rule on the output target but at the
same time estimating many of those effects to zero.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from scipy.special import softmax

from imodels.rule_set.rule_set import RuleSet
from imodels.util.rule import enum_features
from imodels.util.transforms import Winsorizer, FriedScale
from imodels.util.score import score_lasso
from imodels.util.convert import tree_to_rules


class RuleFitRegressor(BaseEstimator, TransformerMixin, RuleSet):
    """Rulefit class


    Parameters
    ----------
    tree_size:      Number of terminal nodes in generated trees. If exp_rand_tree_size=True, 
                    this will be the mean number of terminal nodes.
    sample_fract:   fraction of randomly chosen training observations used to produce each tree. 
                    FP 2004 (Sec. 2)
    max_rules:      total number of terms included in the final model (both linear and rules)
                    approximate total number of rules generated for fitting also is based on this
                    Note that actual number of rules will usually be lower than this due to duplicates.
    memory_par:     scale multiplier (shrinkage factor) applied to each new tree when 
                    sequentially induced. FP 2004 (Sec. 2)
    lin_standardise: If True, the linear terms will be standardised as per Friedman Sec 3.2
                    by multiplying the winsorised variable by 0.4/stdev.
    lin_trim_quantile: If lin_standardise is True, this quantile will be used to trim linear 
                    terms before standardisation.
    exp_rand_tree_size: If True, each boosted tree will have a different maximum number of 
                    terminal nodes based on an exponential distribution about tree_size. 
                    (Friedman Sec 3.3)
    include_linear: Include linear terms as opposed to only rules
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

    def __init__(self,
                 tree_size=4,
                 sample_fract='default',
                 max_rules=2000,
                 memory_par=0.01,
                 tree_generator=None,
                 lin_trim_quantile=0.025,
                 lin_standardise=True,
                 exp_rand_tree_size=True,
                 include_linear=True,
                 alphas=None,
                 cv=3,
                 random_state=None):
        self.tree_generator = tree_generator
        self.lin_trim_quantile = lin_trim_quantile
        self.lin_standardise = lin_standardise
        self.winsorizer = Winsorizer(trim_quantile=lin_trim_quantile)
        self.friedscale = FriedScale(self.winsorizer)
        self.stddev = None
        self.mean = None
        self.exp_rand_tree_size = exp_rand_tree_size
        self.max_rules = max_rules
        self.sample_fract = sample_fract
        self.memory_par = memory_par
        self.tree_size = tree_size
        self.random_state = random_state
        self.include_linear = include_linear
        self.cv = cv
        self.alphas = alphas

    def fit(self, X, y=None, feature_names=None):
        """Fit and estimate linear combination of rule ensemble

        """
        if type(X) == pd.DataFrame:
            X = X.values
        if type(y) in [pd.DataFrame, pd.Series]:
            y = y.values

        self.n_obs = X.shape[0]
        self.n_features_ = X.shape[1]
        self.feature_names_, self.feature_dict_ = enum_features(X, feature_names)

        self.tree_generator = self._get_tree_ensemble(classify=False)
        self._fit_tree_ensemble(X, y)

        extracted_rules = self._extract_rules()
        self.rules_without_feature_names_, self.coef, self.intercept = self._score_rules(X, y, extracted_rules)

        return self

    def predict(self, X):
        """Predict outcome for X

        """
        if type(X) == pd.DataFrame:
            X = X.values.astype(np.float32)

        y_pred = np.zeros(X.shape[0])
        y_pred += self.eval_weighted_rule_sum(X)

        if self.include_linear:
            if self.lin_standardise:
                X = self.friedscale.scale(X)
            y_pred += X @ self.coef[:X.shape[1]]

        return y_pred + self.intercept

    def predict_proba(self, X):
        y = self.predict(X)
        preds = np.vstack((1 - y, y)).transpose()
        return softmax(preds, axis=1)

    def transform(self, X=None, rules=None):
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
        df = pd.DataFrame(X, columns=self.feature_names_)
        X_transformed = np.zeros([X.shape[0], 0])

        for r in rules:
            curr_rule_feature = np.zeros(X.shape[0])
            curr_rule_feature[list(df.query(r).index)] = 1
            curr_rule_feature = np.expand_dims(curr_rule_feature, axis=1)
            X_transformed = np.concatenate((X_transformed, curr_rule_feature), axis=1)
        
        return X_transformed

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

        n_features = len(self.coef) - len(self.rules_without_feature_names_)
        rule_ensemble = list(self.rules_without_feature_names_)
        output_rules = []
        ## Add coefficients for linear effects
        for i in range(0, n_features):
            if self.lin_standardise:
                coef = self.coef[i] * self.friedscale.scale_multipliers[i]
            else:
                coef = self.coef[i]
            if subregion is None:
                importance = abs(coef) * self.stddev[i]
            else:
                subregion = np.array(subregion)
                importance = sum(abs(coef) * abs([x[i] for x in self.winsorizer.trim(subregion)] - self.mean[i])) / len(
                    subregion)
            output_rules += [(self.feature_names_[i], 'linear', coef, 1, importance)]

        ## Add rules
        for i in range(0, len(self.rules_without_feature_names_)):
            rule = rule_ensemble[i]
            coef = self.coef[i + n_features]

            if subregion is None:
                importance = abs(coef) * (rule.support * (1 - rule.support)) ** (1 / 2)
            else:
                rkx = self.transform(subregion, [rule])[:, -1]
                importance = sum(abs(coef) * abs(rkx - rule.support)) / len(subregion)

            output_rules += [(rule.__str__(), 'rule', coef, rule.support, importance)]
        rules = pd.DataFrame(output_rules, columns=["rule", "type", "coef", "support", "importance"])
        if exclude_zero_coef:
            rules = rules.ix[rules.coef != 0]
        return rules

    def visualize(self):
        rules = self.get_rules()
        rules = rules[rules.coef != 0].sort_values("support", ascending=False)
        pd.set_option('display.max_colwidth', -1)
        return rules[['rule', 'coef']].round(3)

    def _get_tree_ensemble(self, classify=False):

        if self.tree_generator is None:
            n_estimators_default = int(np.ceil(self.max_rules / self.tree_size))
            self.sample_fract_ = min(0.5, (100 + 6 * np.sqrt(self.n_obs)) / self.n_obs)

            tree_generator = GradientBoostingRegressor(n_estimators=n_estimators_default,
                                                       max_leaf_nodes=self.tree_size,
                                                       learning_rate=self.memory_par,
                                                       subsample=self.sample_fract_,
                                                       random_state=self.random_state,
                                                       max_depth=100)

        if type(tree_generator) not in [GradientBoostingRegressor, RandomForestRegressor]:
            raise ValueError("RuleFit only works with RandomForest and BoostingRegressor")

        return tree_generator

    def _fit_tree_ensemble(self, X, y):
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
                self.tree_generator.fit(np.copy(X, order='C'), np.copy(y, order='C'))
                curr_est_ = curr_est_ + 1
            self.tree_generator.set_params(warm_start=False)

        if isinstance(self.tree_generator, RandomForestRegressor):
            self.estimators_ = [[x] for x in self.tree_generator.estimators_]
        else:
            self.estimators_ = self.tree_generator.estimators_
    
    def _extract_rules(self):
        seen_antecedents = set()
        extracted_rules = [] 
        for estimator in self.estimators_:
            for rule_value_pair in tree_to_rules(estimator[0], np.array(self.feature_names_), prediction_values=True):
                if rule_value_pair[0] not in seen_antecedents:
                    extracted_rules.append(rule_value_pair)
                    seen_antecedents.add(rule_value_pair[0])
        
        extracted_rules = sorted(extracted_rules, key=lambda x: x[1])
        extracted_rules = list(map(lambda x: x[0], extracted_rules))
        return extracted_rules

    def _score_rules(self, X, y, rules):
        X_concat = np.zeros([X.shape[0], 0])

        # standardise linear variables if requested (for regression model only)
        if self.include_linear:

            # standard deviation and mean of winsorized features
            self.winsorizer.train(X)
            winsorized_X = self.winsorizer.trim(X)
            self.stddev = np.std(winsorized_X, axis=0)
            self.mean = np.mean(winsorized_X, axis=0)

            if self.lin_standardise:
                self.friedscale.train(X)
                X_regn = self.friedscale.scale(X)
            else:
                X_regn = X.copy()
            X_concat = np.concatenate((X_concat, X_regn), axis=1)

        X_rules = self.transform(X, rules)
        if X_rules.shape[0] > 0:
            X_concat = np.concatenate((X_concat, X_rules), axis=1)

        return score_lasso(X_concat, y, rules, alphas=self.alphas, cv=self.cv, max_rules=self.max_rules, random_state=self.random_state)
