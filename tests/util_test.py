from imodels import SkopeRulesClassifier
from imodels.util.prune import deduplicate, find_similar_rulesets, f1_score


def test_similarity_tree():
    # Test that rules are well splitted
    rules = [("a <= 2 and b > 45 and c <= 3 and a > 4", (1, 1, 0)),
             ("a <= 2 and b > 45 and c <= 3 and a > 4", (1, 1, 0)),
             ("a > 2 and b > 45", (0.5, 0.3, 0)),
             ("a > 2 and b > 40", (0.5, 0.2, 0)),
             ("a <= 2 and b <= 45", (1, 1, 0)),
             ("a > 2 and c <= 3", (1, 1, 0)),
             ("b > 45", (1, 1, 0)),
             ]

    sk = SkopeRulesClassifier(max_depth_duplication=2)
    rulesets = find_similar_rulesets(rules, max_depth_duplication=2)
    # Assert some couples of rules are in the same bag
    idx_bags_rules = []
    for idx_rule, r in enumerate(rules):
        idx_bags_for_rule = []
        for idx_bag, bag in enumerate(rulesets):
            if r in bag:
                idx_bags_for_rule.append(idx_bag)
        idx_bags_rules.append(idx_bags_for_rule)

    assert idx_bags_rules[0] == idx_bags_rules[1]
    assert not idx_bags_rules[0] == idx_bags_rules[2]
    # Assert the best rules are kept
    final_rules = deduplicate(rules, sk.max_depth_duplication)
    assert rules[0] in final_rules
    assert rules[2] in final_rules
    assert not rules[3] in final_rules


def test_f1_score():
    rule0 = ('a > 0', (0, 0, 0))
    rule1 = ('a > 0', (0.5, 0.5, 0))
    rule2 = ('a > 0', (0.5, 0, 0))

    assert f1_score(rule0) == 0
    assert f1_score(rule1) == 0.5
    assert f1_score(rule2) == 0
