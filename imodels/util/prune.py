from typing import List, Tuple
from warnings import warn

import pandas as pd
import numpy as np
from sklearn.utils import indices_to_mask

from imodels.util.rule import Rule


def prune_mins(rules: List[str], precision_min: float, recall_min: float):
    # Factorize rules before semantic tree filtering
    rules_ = [tuple(rule) for rule in rules]
    rules_dict = {}

    # keep only rules verifying precision_min and recall_min:
    for rule, score in rules_:
        if score[0] >= precision_min and score[1] >= recall_min:
            if rule in rules_dict:
                # update the score to the new mean
                c = rules_dict[rule][2] + 1
                b = rules_dict[rule][1] + 1. / c * (
                        score[1] - rules_dict[rule][1])
                a = rules_dict[rule][0] + 1. / c * (
                        score[0] - rules_dict[rule][0])

                rules_dict[rule] = (a, b, c)
            else:
                rules_dict[rule] = (score[0], score[1], 1)

    rules_dict = sorted(rules_dict.items(), key=lambda x: (x[1][0], x[1][1]), reverse=True)

    return rules_dict
