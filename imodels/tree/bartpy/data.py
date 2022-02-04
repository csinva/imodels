from operator import gt, le
from typing import Any, List, Optional, Union

import numpy as np
import pandas as pd

from bartpy.errors import NoSplittableVariableException
from bartpy.splitcondition import SplitCondition


def is_not_constant(series: np.ndarray) -> bool:
    """
    Quickly identify whether a series contains more than 1 distinct value
    Parameters
    ----------
    series: np.ndarray
    The series to assess

    Returns
    -------
    bool
        True if more than one distinct value found
    """
    if len(series) <= 1:
        return False
    first_value = None
    for i in range(1, len(series)):
        # if not series.mask[i] and series.data[i] != first_value:
        if series[i] != first_value:
            if first_value is None:
                first_value = series.data[i]
            else:
                return True
    return False


def ensure_numpy_array(X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
    if isinstance(X, pd.DataFrame):
        return X.values
    else:
        return X


def ensure_float_array(X: np.ndarray) -> np.ndarray:
    return X.astype(float)


def format_covariate_matrix(X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
    X = ensure_numpy_array(X)
    return ensure_float_array(X)


def make_bartpy_data(X: Union[np.ndarray, pd.DataFrame],
                     y: np.ndarray,
                     normalize: bool=True) -> 'Data':
    X = format_covariate_matrix(X)
    y = y.astype(float)
    return Data(X, y, normalize=normalize)


class CovariateMatrix(object):

    def __init__(self,
                 X: np.ndarray,
                 mask: np.ndarray,
                 n_obsv: int,
                 unique_columns: List[int],
                 splittable_variables: List[int]):

        if type(X) == pd.DataFrame:
            X: pd.DataFrame = X
            X = X.values

        self._X = X
        self._n_obsv = n_obsv
        self._n_features = X.shape[1]
        self._mask = mask

        # Cache iniialization
        if unique_columns is not None:
            self._unique_columns = [x if x is True else None for x in unique_columns]
        else:
            self._unique_columns = [None for _ in range(self._n_features)]
        if splittable_variables is not None:
            self._splittable_variables = [x if x is False else None for x in splittable_variables]
        else:
            self._splittable_variables = [None for _ in range(self._n_features)]
        self._max_values = [None] * self._n_features
        self._X_column_cache = [None] * self._n_features
        self._max_value_cache = [None] * self._n_features
        self._X_cache = None

    @property
    def mask(self) -> np.ndarray:
        return self._mask

    @property
    def values(self) -> np.ndarray:
        return self._X

    def get_column(self, i: int) -> np.ndarray:
        if self._X_cache is None:
            self._X_cache = self.values[~self.mask, :]
        return self._X_cache[:, i]

    def splittable_variables(self) -> List[int]:
        """
        List of columns that can be split on, i.e. that have more than one unique value

        Returns
        -------
        List[int]
            List of column numbers that can be split on
        """
        for i in range(0, self._n_features):
            if self._splittable_variables[i] is None:
                self._splittable_variables[i] = is_not_constant(self.get_column(i))
        
        return [i for (i, x) in enumerate(self._splittable_variables) if x is True]        

    @property
    def n_splittable_variables(self) -> int:
        return len(self.splittable_variables())

    def is_at_least_one_splittable_variable(self) -> bool:
        if any(self._splittable_variables):
            return True
        else:
            return len(self.splittable_variables()) > 0
    
    def random_splittable_variable(self) -> str:
        """
        Choose a variable at random from the set of splittable variables
        Returns
        -------
            str - a variable name that can be split on
        """
        if self.is_at_least_one_splittable_variable():
            return np.random.choice(np.array(self.splittable_variables()), 1)[0]
        else:
            raise NoSplittableVariableException()

    def is_column_unique(self, i: int) -> bool:
        """
        Identify whether feature contains only unique values, i.e. it has no duplicated values
        Useful to provide a faster way to calculate the probability of a value being selected in a variable

        Returns
        -------
        List[int]
        """
        if self._unique_columns[i] is None:
            self._unique_columns[i] = len(np.unique(self.get_column(i))) == self._n_obsv
        return self._unique_columns[i]

    def max_value_of_column(self, i: int):
        if self._max_value_cache[i] is None:
            self._max_value_cache[i] = self.get_column(i).max()
        return self._max_value_cache[i]

    def random_splittable_value(self, variable: int) -> Any:
        """
        Return a random value of a variable
        Useful for choosing a variable to split on

        Parameters
        ----------
        variable - str
            Name of the variable to split on

        Returns
        -------
        Any

        Notes
        -----
          - Won't create degenerate splits, all splits will have at least one row on both sides of the split
        """
        if variable not in self.splittable_variables():
            raise NoSplittableVariableException()
        max_value = self.max_value_of_column(variable)
        candidate = np.random.choice(self.get_column(variable))
        while candidate == max_value:
            candidate = np.random.choice(self.get_column(variable))
        return candidate

    def proportion_of_value_in_variable(self, variable: int, value: float) -> float:
        if self.is_column_unique(variable):
            return 1. / self.n_obsv
        else:
            return float(np.mean(self.get_column(variable) == value))

    def update_mask(self, other: SplitCondition) -> np.ndarray:
        if other.operator == gt:
            column_mask = self.values[:, other.splitting_variable] <= other.splitting_value
        elif other.operator == le:
            column_mask = self.values[:, other.splitting_variable] > other.splitting_value
        else:
            raise TypeError("Operator type not matched, only {} and {} supported".format(gt, le))

        return self.mask | column_mask

    @property
    def variables(self) -> List[int]:
        return list(range(self._n_features))

    @property
    def n_obsv(self) -> int:
        return self._n_obsv


class Target(object):

    def __init__(self, y, mask, n_obsv, normalize, y_sum=None):

        if normalize:
            self.original_y_min, self.original_y_max = y.min(), y.max()
            self._y = self.normalize_y(y)
        else:
            self._y = y

        self._mask = mask
        self._inverse_mask_int = (~self._mask).astype(int)
        self._n_obsv = n_obsv

        if y_sum is None:
            self.y_sum_cache_up_to_date = False
            self._summed_y = None
        else:
            self.y_sum_cache_up_to_date = True
            self._summed_y = y_sum

    @staticmethod
    def normalize_y(y: np.ndarray) -> np.ndarray:
        """
        Normalize y into the range (-0.5, 0.5)
        Useful for allowing the leaf parameter prior to be 0, and to standardize the sigma prior

        Parameters
        ----------
        y - np.ndarray

        Returns
        -------
        np.ndarray

        Examples
        --------
        >>> Data.normalize_y([1, 2, 3])
        array([-0.5,  0. ,  0.5])
        """
        y_min, y_max = np.min(y), np.max(y)
        return -0.5 + ((y - y_min) / (y_max - y_min))

    def unnormalize_y(self, y: np.ndarray) -> np.ndarray:
        distance_from_min = y - (-0.5)
        total_distance = (self.original_y_max - self.original_y_min)
        return self.original_y_min + (distance_from_min * total_distance)

    @property
    def unnormalized_y(self) -> np.ndarray:
        return self.unnormalize_y(self.values)

    @property
    def normalizing_scale(self) -> float:
        return self.original_y_max - self.original_y_min

    def summed_y(self) -> float:
        if self.y_sum_cache_up_to_date:
            return self._summed_y
        else:
            self._summed_y = np.sum(self._y * self._inverse_mask_int)
            self.y_sum_cache_up_to_date = True
            return self._summed_y

    def update_y(self, y) -> None:
        self._y = y
        self.y_sum_cache_up_to_date = False

    @property
    def values(self):
        return self._y

class Data(object):
    """
    Encapsulates the data within a split of feature space.
    Primarily used to cache computations on the data for better performance

    Parameters
    ----------
    X: np.ndarray
        The subset of the covariate matrix that falls into the split
    y: np.ndarray
        The subset of the target array that falls into the split
    normalize: bool
        Whether to map the target into -0.5, 0.5
    cache: bool
        Whether to cache common values.
        You really only want to turn this off if you're not going to the resulting object for anything (e.g. when testing)
    """

    def __init__(self,
                 X: np.ndarray,
                 y: np.ndarray,
                 mask: Optional[np.ndarray]=None,
                 normalize: bool=False,
                 unique_columns: List[int]=None,
                 splittable_variables: Optional[List[Optional[bool]]]=None,
                 y_sum: float=None,
                 n_obsv: int=None):

        if mask is None:
            mask = np.zeros_like(y).astype(bool)
        self._mask: np.ndarray = mask

        if n_obsv is None:
            n_obsv = (~self.mask).astype(int).sum()

        self._n_obsv = n_obsv

        self._X = CovariateMatrix(X, mask, n_obsv, unique_columns, splittable_variables)
        self._y = Target(y, mask, n_obsv, normalize, y_sum)

    @property
    def y(self) -> Target:
        return self._y

    @property
    def X(self) -> CovariateMatrix:
        return self._X

    @property
    def mask(self) -> np.ndarray:
        return self._mask

    def update_y(self, y: np.ndarray) -> None:
        self._y.update_y(y)

    def __add__(self, other: SplitCondition) -> 'Data':
        updated_mask = self.X.update_mask(other)

        return Data(self.X.values,
                    self.y.values,
                    updated_mask,
                    normalize=False,
                    unique_columns=self._X._unique_columns,
                    splittable_variables=self._X._splittable_variables,
                    y_sum=other.carry_y_sum,
                    n_obsv=other.carry_n_obsv)
