from copy import deepcopy
from typing import List

import numpy as np
from sklearn import datasets
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from imodels.util import checks


def shrink_tree(tree, reg_param):
    """Shrink the tree
    """
    return tree


class ShrunkTree(BaseEstimator):
    """Experimental ShrunkTree. Gets passed a sklearn tree or tree ensemble model.
    """

    def __init__(self, estimator_: BaseEstimator, reg_param: float):
        super().__init__()
        self.reg_param = reg_param
        self.estimator_ = estimator_

        if checks.check_is_fitted(self.estimator_):
            self.shrink()

    def fit(self, *args, **kwargs):
        self.estimator_.fit(*args, **kwargs)
        self.shrink()

    def shrink(self):
        if hasattr(self.estimator_, 'tree_'):
            shrink_tree(self.estimator_.tree_, self.reg_param)
        elif hasattr(self.estimator_, 'estimators_'):
            for t in self.estimator_.estimators_:
                shrink_tree(t, self.reg_param)

    def predict(self, *args, **kwargs):
        self.estimator_.predict(*args, **kwargs)

    def predict_proba(self, *args, **kwargs):
        if hasattr(self.estimator_, 'predict_proba'):
            return self.estimator_.predict_proba(*args, **kwargs)
        else:
            return NotImplemented

    def score(self, *args, **kwargs):
        if hasattr(self.estimator_, 'score'):
            return self.estimator_.score(*args, **kwargs)
        else:
            return NotImplemented


class ShrunkTreeCV(ShrunkTree):
    def __init__(self, estimator: BaseEstimator,
                 reg_param_list: List[float] = [0.1, 0.3, 0.5],
                 cv: int = 3, scoring=None):
        super().__init__(estimator, reg_param=None)
        self.reg_param_list = np.array(reg_param_list)
        self.estimator_ = estimator
        self.cv = cv
        self.scoring = scoring
        print('estimator', self.estimator_,
              'checks.check_is_fitted(estimator)', checks.check_is_fitted(self.estimator_))
        if checks.check_is_fitted(self.estimator_):
            raise Warning('Passed an already fitted estimator,'
                          'but shrinking not applied until fit method is called.')

    def fit(self, X, y, *args, **kwargs):
        self.scores_ = []
        for reg_param in self.reg_param_list:
            est = ShrunkTree(deepcopy(self.estimator_), reg_param)
            cv_scores = cross_val_score(est, X, y, cv=self.cv, scoring=self.scoring)
            self.scores_.append(np.mean(cv_scores))
        self.reg_param = self.reg_param_list[np.argmax(self.scores_)]
        super().fit(X, y)


if __name__ == '__main__':
    np.random.seed(13)
    X, y = datasets.load_breast_cancer(return_X_y=True)  # binary classification
    # X, y = datasets.load_diabetes(return_X_y=True)  # regression
    # X = np.random.randn(500, 10)
    # y = (X[:, 0] > 0).astype(float) + (X[:, 1] > 1).astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )
    print('X.shape', X.shape)
    print('ys', np.unique(y_train))

    m = ShrunkTree(estimator_=DecisionTreeClassifier(), reg_param=0.1)
    # m = ShrunkTreeCV(estimator=DecisionTreeClassifier())
    print('best alpha', m.reg_param)
    m.fit(X_train, y_train)
    m.predict_proba(X_train)  # just run this
    print('score', m.score(X_test, y_test))
