from abc import abstractmethod, ABC

from ..model import Model
from ..tree import Tree


class Sampler(ABC):

    @abstractmethod
    def step(self, model: Model, tree: Tree) -> bool:
        raise NotImplementedError()
