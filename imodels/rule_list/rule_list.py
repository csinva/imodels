from sklearn.utils.validation import check_is_fitted


class RuleList:

    def get_rules(self, feature_names=None):
        """Return this model's rules as a DataFrame (see imodels.get_rules)."""
        from imodels.util.get_rules import get_rules
        return get_rules(self, feature_names=feature_names)

    def _get_complexity(self):
        check_is_fitted(self, ['rules_without_feature_names_'])
        return sum([len(rule.agg_dict) for rule in self.rules_without_feature_names_]) 
