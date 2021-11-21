import numpy as np
from typing import List

from imodels.rule_set.rule_fit import RuleFit
from imodels.util.score import score_linear
from sklearn.base import ClassifierMixin, RegressorMixin, BaseEstimator

from .util import extract_ensemble


class StableLinear(RuleFit):

    def __init__(self,
                 weak_learners: List[BaseEstimator],
                 max_complexity: int,
                 min_mult: int = 1,
                 penalty='l1',
                 n_estimators=100,
                 tree_size=4,
                 sample_fract='default',
                 max_rules=30,
                 memory_par=0.01,
                 tree_generator=None,
                 lin_trim_quantile=0.025,
                 lin_standardise=True,
                 exp_rand_tree_size=True,
                 include_linear=False,
                 alpha=None,
                 random_state=None):
        super().__init__(n_estimators,
                         tree_size,
                         sample_fract,
                         max_rules,
                         memory_par,
                         tree_generator,
                         lin_trim_quantile,
                         lin_standardise,
                         exp_rand_tree_size,
                         include_linear,
                         alpha,
                         random_state)
        self.max_complexity = max_complexity
        self.weak_learners = weak_learners
        self.penalty = penalty
        self.min_mult = min_mult

    def fit(self, X, y=None, feature_names=None):
        super().fit(X, y, feature_names=feature_names)
        return self

    def _extract_rules(self, X, y) -> List[str]:
        return extract_ensemble(self.weak_learners, X, y, self.min_mult)

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
        
        # no rules fit and self.include_linear == False
        if X_concat.shape[1] == 0:
            return [], [], 0

        return score_linear(X_concat, y, rules, 
                            alpha=self.alpha, 
                            penalty=self.penalty,
                            prediction_task=self.prediction_task,
                            max_rules=self.max_rules, random_state=self.random_state)


class StableLinearRegressor(StableLinear, RegressorMixin):
    def _init_prediction_task(self):
        self.prediction_task = 'regression'


class StableLinearClassifier(StableLinear, ClassifierMixin):
    def _init_prediction_task(self):
        self.prediction_task = 'classification'
