from typing import Any, List

import numpy as np


class NormalScalarSampler():

    def __init__(self,
                 cache_size: int=1000):
        self._cache_size = cache_size
        self._cache = []

    def sample(self):
        if len(self._cache) == 0:
            self.refresh_cache()
        return self._cache.pop()

    def refresh_cache(self):
        self._cache = list(np.random.normal(size=self._cache_size))


class UniformScalarSampler():

    def __init__(self,
                 cache_size: int=1000):
        self._cache_size = cache_size
        self._cache = []

    def sample(self):
        if len(self._cache) == 0:
            self.refresh_cache()
        return self._cache.pop()

    def refresh_cache(self):
        self._cache = list(np.random.uniform(size=self._cache_size))


class DiscreteSampler():

    def __init__(self,
                 values: List[Any],
                 probas: List[float]=None,
                 cache_size: int=1000):
        self._values = values
        if probas is None:
            probas = [1.0 / len(values) for x in values]
        self._probas = probas
        self._cache_size = cache_size
        self._cache = []

    def sample(self):
        if len(self._cache) == 0:
            self.refresh_cache()
        return self._cache.pop()

    def refresh_cache(self):
        self._cache = list(np.random.choice(self._values, p=self._probas, size=self._cache_size))
