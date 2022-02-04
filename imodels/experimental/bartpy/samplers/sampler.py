from abc import abstractmethod, ABC

from bartpy.model import Model
from bartpy.tree import Tree


class Sampler(ABC):

    @abstractmethod
    def step(self, model: Model, tree: Tree) -> bool:
        raise NotImplementedError()
