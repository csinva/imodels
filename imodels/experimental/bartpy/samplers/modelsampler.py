import copy
from collections import defaultdict
from typing import List, Mapping, Union, Any, Type

import numpy as np
from tqdm import tqdm

from ..model import Model
from ..samplers.sampler import Sampler
from ..samplers.schedule import SampleSchedule
from ..trace import TraceLogger

Chain = Mapping[str, Union[List[Any], np.ndarray]]


class ModelSampler(Sampler):

    def __init__(self,
                 schedule: SampleSchedule,
                 trace_logger_class: Type[TraceLogger]=TraceLogger,
                 n_rules: int=None):
        self.schedule = schedule
        self.trace_logger_class = trace_logger_class
        self.n_rules = n_rules

    def step(self, model: Model, trace_logger: TraceLogger):
        step_result = defaultdict(list)
        likelihoods = []
        probs = []
        for step_kind, step in self.schedule.steps(model):
            result = step()
            if type(result) == tuple:
                result, likelihood, prob = result
                likelihoods.append(likelihood)
                probs.append(prob)
            log_message = trace_logger[step_kind](result)
            if log_message is not None:
                step_result[step_kind].append(log_message)
        return {x: np.mean([1 if y else 0 for y in step_result[x]]) for x in step_result}, likelihoods, probs

    def samples(self, model: Model,
                n_samples: int,
                n_burn: int,
                thin: float=0.1,
                store_in_sample_predictions: bool=True,
                store_acceptance: bool=True) -> Chain:
        # print("Starting burn")

        trace_logger = self.trace_logger_class()
        y = copy.deepcopy(model.data.y.unnormalized_y)

        for _ in range(n_burn):
            model.update_z_values(y)
            self.step(model, trace_logger)

        trace = []
        model_trace = []
        acceptance_trace = []
        likelihood = []
        probs = []
        # print("Starting sampling")

        thin_inverse = 1. / thin

        for ss in range(n_samples):
            model.update_z_values(y)
            step_trace_dict, l_score, prob = self.step(model, trace_logger)
            # print(step_trace_dict)

            if ss % thin_inverse == 0:
                if store_in_sample_predictions:
                    in_sample_log = trace_logger["In Sample Prediction"](model.predict())
                    if in_sample_log is not None:
                        trace.append(in_sample_log)
                if store_acceptance:
                    acceptance_trace.append(step_trace_dict)
                    likelihood.append(l_score)
                    probs.append(prob)

                model_log = trace_logger["Model"](model)
                if model_log is not None:
                    model_trace.append(model_log)
        return {
            "model": model_trace,
            "acceptance": acceptance_trace,
            "in_sample_predictions": trace,
            "likelihood":likelihood,
            "probs":probs
        }
