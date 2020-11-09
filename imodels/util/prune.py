from typing import List
from collections import Counter


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


def deduplicate(rules, max_depth_dup):
    if max_depth_dup is not None:
        rules = [max(rules_set, key=f1_score) for rules_set in find_similar_rulesets(rules, max_depth_dup)]
    return sorted(rules, key=lambda x: - f1_score(x))


def f1_score(x) -> float:
    return 2 * x[1][0] * x[1][1] / \
           (x[1][0] + x[1][1]) if (x[1][0] + x[1][1]) > 0 else 0


def find_similar_rulesets(rules, max_depth_duplication=None):
    """Create clusters of rules using a decision tree based
    on the terms of the rules

    Parameters
    ----------
    rules : List, List of rules
            The rules that should be splitted in subsets of similar rules

    Returns
    -------
    rules : List of list of rules
            The different set of rules. Each set should be homogeneous

    """

    def split_with_best_feature(rules, depth, exceptions=[]):
        """
        Method to find a split of rules given most represented feature
        """
        if depth == 0:
            return rules

        rulelist = [rule.split(' and ') for rule, score in rules]
        terms = [t.split(' ')[0] for term in rulelist for t in term]
        counter = Counter(terms)
        # Drop exception list
        for exception in exceptions:
            del counter[exception]

        if len(counter) == 0:
            return rules

        most_represented_term = counter.most_common()[0][0]

        # Proceed to split
        rules_splitted = [[], [], []]
        for rule in rules:
            if (most_represented_term + ' <=') in rule[0]:
                rules_splitted[0].append(rule)
            elif (most_represented_term + ' >') in rule[0]:
                rules_splitted[1].append(rule)
            else:
                rules_splitted[2].append(rule)
        new_exceptions = exceptions + [most_represented_term]

        # Choose best term
        return [split_with_best_feature(ruleset,
                                        depth - 1,
                                        exceptions=new_exceptions)
                for ruleset in rules_splitted]

    def breadth_first_search(rules, leaves=None):
        if len(rules) == 0 or not isinstance(rules[0], list):
            if len(rules) > 0:
                return leaves.append(rules)
        else:
            for rules_child in rules:
                breadth_first_search(rules_child, leaves=leaves)
        return leaves

    leaves = []
    res = split_with_best_feature(rules, max_depth_duplication)
    breadth_first_search(res, leaves=leaves)
    return leaves
