'''Shared functions between different interpretable models

- ``RuleCondition`` implements a binary feature transformation
- ``Rule`` implements a Rule composed of ``RuleConditions``
'''

from functools import reduce


class Rule():
    """Class for binary Rules from list of conditions

    Warning: this class should not be used directly.
    """

    def __init__(self, rule_conditions, prediction_value):
        self.conditions = set(rule_conditions)
        self.support = min([x.support for x in rule_conditions])
        self.prediction_value = prediction_value
        self.rule_direction = None

    def transform(self, X):
        """Transform dataset.

        Parameters
        ----------
        X: array-like matrix

        Returns
        -------
        X_transformed: array-like matrix, shape=(n_samples, 1)
        """
        rule_applies = [condition.transform(X) for condition in self.conditions]
        return reduce(lambda x, y: x * y, rule_applies)

    def __str__(self):
        return " and ".join([x.__str__() for x in self.conditions])

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return sum([condition.__hash__() for condition in self.conditions])

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()


class RuleCondition():
    """Class for binary rule condition

    Warning: this class should not be used directly.
    """

    def __init__(self,
                 feature_index,
                 threshold,
                 operator,
                 support,
                 feature_name=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.operator = operator
        self.support = support
        self.feature_name = feature_name

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        if self.feature_name:
            feature = self.feature_name
        else:
            feature = self.feature_index
        return "%s %s %s" % (feature, self.operator, self.threshold)

    def transform(self, X):
        """Transform dataset.

        Parameters
        ----------
        X: array-like matrix, shape=(n_samples, n_features)

        Returns
        -------
        X_transformed: array-like matrix, shape=(n_samples, 1)
        """
        if self.operator == "<=":
            res = 1 * (X[:, self.feature_index] <= self.threshold)
        elif self.operator == ">":
            res = 1 * (X[:, self.feature_index] > self.threshold)
        return res

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()

    def __hash__(self):
        return hash((self.feature_index, self.threshold, self.operator, self.feature_name))
