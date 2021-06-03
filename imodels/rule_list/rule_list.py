from sklearn.utils.validation import check_is_fitted


class RuleList:

    def _get_complexity(self):
        check_is_fitted(self, ['rules_without_feature_names_'])
        return sum([len(rule.agg_dict) for rule in self.rules_without_feature_names_]) 
