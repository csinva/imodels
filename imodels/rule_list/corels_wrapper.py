# This is just a simple wrapper around pycorels: https://github.com/corels/pycorels

from corels import CorelsClassifier
from sklearn.base import BaseEstimator


class CorelsRuleListClassifier(BaseEstimator, CorelsClassifier):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_complexity(self):
        return None
