import re


def replace_feature_name(rule, replace_dict):
    def replace(match):
        return replace_dict[match.group(0)]

    rule = re.sub('|'.join(r'\b%s\b' % re.escape(s) for s in replace_dict),
                  replace, rule)
    return rule


class Rule:
    """ An object modeling a logical rule and add factorization methods.
    It is used to simplify rules and deduplicate them.

    Parameters
    ----------

    rule : str
        The logical rule that is interpretable by a pandas query.

    args : object, optional
        Arguments associated to the rule, it is not used for factorization
        but it takes part of the output when the rule is converted to an array.
    """

    def __init__(self, rule, args=None):
        self.rule = rule
        self.args = args
        self.terms = [t.split(' ') for t in self.rule.split(' and ')]
        self.agg_dict = {}
        self.factorize()
        self.rule = str(self)

    def __eq__(self, other):
        return self.agg_dict == other.agg_dict

    def __hash__(self):
        # FIXME : Easier method ?
        return hash(tuple(sorted(((i, j) for i, j in self.agg_dict.items()))))

    def factorize(self):
        for feature, symbol, value in self.terms:
            if (feature, symbol) not in self.agg_dict:
                if symbol != '==':
                    self.agg_dict[(feature, symbol)] = str(float(value))
                else:
                    self.agg_dict[(feature, symbol)] = value
            else:
                if symbol[0] == '<':
                    self.agg_dict[(feature, symbol)] = str(min(
                        float(self.agg_dict[(feature, symbol)]),
                        float(value)))
                elif symbol[0] == '>':
                    self.agg_dict[(feature, symbol)] = str(max(
                        float(self.agg_dict[(feature, symbol)]),
                        float(value)))
                else:  # Handle the c0 == c0 case
                    self.agg_dict[(feature, symbol)] = value

    def __iter__(self):
        yield str(self)
        yield self.args

    def __repr__(self):
        return ' and '.join([' '.join(
            [feature, symbol, str(self.agg_dict[(feature, symbol)])])
            for feature, symbol in sorted(self.agg_dict.keys())
        ])
