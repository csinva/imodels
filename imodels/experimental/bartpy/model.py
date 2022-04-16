from copy import deepcopy, copy
from typing import List, Generator, Optional
from imodels.util.checks import check_is_fitted

import numpy as np
import pandas as pd

from .data import Data
from .initializers.initializer import Initializer
from .initializers.sklearntreeinitializer import SklearnTreeInitializer
from .sigma import Sigma
from .split import Split
from .tree import Tree, LeafNode, deep_copy_tree


class Model:

    def __init__(self,
                 data: Optional[Data],
                 sigma: Sigma,
                 trees: Optional[List[Tree]] = None,
                 n_trees: int = 50,
                 alpha: float = 0.95,
                 beta: float = 2.,
                 k: int = 2.,
                 initializer: Initializer = SklearnTreeInitializer(),
                 classification: bool = False):

        self.data = deepcopy(data)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.k = k
        self._sigma = sigma
        self._prediction = None
        self._initializer = initializer
        self._check_initilizer()
        self.classification = classification

        if trees is None:
            self.n_trees = n_trees
            self._trees = self.initialize_trees()
            if self._initializer is not None:
                if hasattr(self._initializer._tree,"trees_"):
                    self.n_trees = len(self._initializer._tree.trees_)
                    self._trees = self.initialize_trees()
                elif hasattr(self._initializer._tree, "figs"):
                    self.n_trees = len(self._initializer._tree.figs.trees_)
                    self._trees = self.initialize_trees()

                # for tree in self.trees:
                self._initializer.initialize_trees(self.refreshed_trees())
                # self._initializer.initialize_trees(trees=self._trees)
        else:
            self.n_trees = len(trees)
            self._trees = trees

    def _check_initilizer(self):
        if not hasattr(self._initializer, "_tree"):
            return
        elif self._initializer._tree is None:
            return
        if not check_is_fitted(self._initializer._tree):
            self._initializer._tree.fit(self.data.X.values, self.data.y.values)

    def initialize_trees(self) -> List[Tree]:
        trees = [Tree([LeafNode(Split(deepcopy(self.data)))]) for _ in range(self.n_trees)]
        for tree in trees:
            tree.update_y(tree.update_y(self.data.y.values / self.n_trees))
        return trees

    def residuals(self) -> np.ndarray:
        return self.data.y.values - self.predict()

    def unnormalized_residuals(self) -> np.ndarray:
        return self.data.y.unnormalized_y - self.data.y.unnormalize_y(self.predict())

    def predict(self, X: np.ndarray = None) -> np.ndarray:
        if X is not None:
            return self._out_of_sample_predict(X)
        return np.sum([tree.predict() for tree in self.trees], axis=0)

    def _out_of_sample_predict(self, X: np.ndarray) -> np.ndarray:
        if type(X) == pd.DataFrame:
            X: pd.DataFrame = X
            X = X.values
        return np.sum([tree.predict(X) for tree in self.trees], axis=0)

    @property
    def trees(self) -> List[Tree]:
        return self._trees

    def refreshed_trees(self) -> Generator[Tree, None, None]:
        if self._prediction is None:
            self._prediction = self.predict()
        for tree in self._trees:
            self._prediction -= tree.predict()
            tree.update_y(self.data.y.values - self._prediction)
            yield tree
            self._prediction += tree.predict()

    def update_z_values(self, y):
        if not self.classification:
            return
        z = np.random.normal(loc=self.predict(self.data.X.values))
        one_label = np.maximum(z[y == 1], 0)
        zero_label = np.minimum(z[y == 0], 0)
        z[y == 1] = one_label
        z[y == 0] = zero_label
        self.data.update_y(z)

    @property
    def sigma_m(self) -> float:
        if self.classification:
            return 3 / (self.k * np.power(self.n_trees, 0.5))
        return 0.5 / (self.k * np.power(self.n_trees, 0.5))

    @property
    def sigma(self) -> Sigma:
        return self._sigma


def deep_copy_model(model: Model) -> Model:
    copied_model = Model(None, deepcopy(model.sigma), [deep_copy_tree(tree) for tree in model.trees])
    return copied_model
