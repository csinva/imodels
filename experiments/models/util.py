from collections import Counter
from typing import List

from imodels.util.rule import Rule


def extract_ensemble(weak_learners, X, y, min_multiplicity: int = 1) -> List[Rule]:

    all_rules = []
    # weak_learner_scores = []
    for est in weak_learners:
        est.fit(X, y)
        # weak_learner_scores.append(average_precision_score(y, est.predict_proba(X)[:, 1)
        all_rules += est.rules_

    if min_multiplicity > 0:
        # round rule decision boundaries to increase matching
        for i in range(len(all_rules)):
            for key in all_rules[i].agg_dict:
                all_rules[i].agg_dict[key] = round(float(all_rules[i].agg_dict[key]), 1)

    repeated_rules_counter = {k: v for k, v in Counter(all_rules).items() if v > min_multiplicity}
    repeated_rules = list(map(str, repeated_rules_counter.keys()))
    return repeated_rules
