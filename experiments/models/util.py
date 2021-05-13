from collections import Counter
from typing import List

from imodels.util.rule import Rule


def extract_ensemble(weak_learners, X, y, min_multiplicity: int = 1) -> List[Rule]:

    all_rules = []
    all_subterms = []
    for est in weak_learners:
        est.fit(X, y)
        all_rules += est.rules_
        all_est_subterms = set([indv_r for r in est.rules_ for indv_r in split(r)])
        all_subterms += all_est_subterms

    if min_multiplicity > 0:
        # round rule decision boundaries to increase matching
        for i in range(len(all_rules)):
            for key in all_rules[i].agg_dict:
                all_rules[i].agg_dict[key] = round(float(all_rules[i].agg_dict[key]), 1)

    # match full_rules
    repeated_full_rules_counter = {k: v for k, v in Counter(all_rules).items() if v > min_multiplicity}
    repeated_rules = set(repeated_full_rules_counter.keys())

    # match subterms of rules
    repeated_subterm_counter = {k: v for k, v in Counter(all_subterms).items() if v > min_multiplicity}
    repeated_rules = repeated_rules.union(set(repeated_subterm_counter.keys()))

    # convert to str form to be rescored
    repeated_rules = list(map(str, repeated_rules))
    return repeated_rules


def split(rule: Rule) -> List[Rule]:
    if len(rule.agg_dict) == 1:
        return [rule]
    else:
        indv_rule_strs = list(map(lambda x: ' '.join(x), rule.terms))
        indv_rules = list(map(lambda x: Rule(x), indv_rule_strs))
        return indv_rules
