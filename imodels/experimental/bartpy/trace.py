from typing import Any, Callable

import numpy as np

from bartpy.model import Model, deep_copy_model
from bartpy.mutation import TreeMutation


class TraceLogger():

    def __init__(self,
                 f_tree_mutation_log: Callable[[TreeMutation], Any]=lambda x: x is not None,
                 f_model_log: Callable[[Model], Any]=lambda x: deep_copy_model(x),
                 f_in_sample_prediction_log: Callable[[np.ndarray], Any]=lambda x: x):
        self.f_tree_mutation_log = f_tree_mutation_log
        self.f_model_log = f_model_log
        self.f_in_sample_prediction_log = f_in_sample_prediction_log

    def __getitem__(self, item: str):
        if item == "Tree":
            return self.f_tree_mutation_log
        if item == "Model":
            return self.f_model_log
        if item == "In Sample Prediction":
            return self.f_in_sample_prediction_log
        if item in ["Node", "Sigma"]:
            return lambda x: None
        else:
            raise KeyError("No method for key {}".format(item))
