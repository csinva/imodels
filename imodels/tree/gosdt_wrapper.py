# This is just a simple wrapper around gosdt: https://github.com/Jimmy-Lin/GeneralizedOptimalSparseDecisionTrees

from sklearn.tree import DecisionTreeClassifier


class GlobalSparseTreeClassifier(DecisionTreeClassifier):
    """Placeholder for GOSDT classifier
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __str__(self):
        return 'Global Sparse Tree ' + str(self.tree_)
