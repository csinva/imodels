from abc import abstractmethod, ABC

from imodels.experimental.bartpy.model import Model
from imodels.experimental.bartpy.tree import Tree


class Sampler(ABC):

    @abstractmethod
    def step(self, model: Model, tree: Tree) -> bool:
        raise NotImplementedError()
