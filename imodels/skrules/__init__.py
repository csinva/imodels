'''[skope-rules](https://github.com/scikit-learn-contrib/skope-rules) (based on [this implementation](https://github.com/scikit-learn-contrib/skope-rules))
'''

from .skope_rules import SkopeRules
from .rule import Rule, replace_feature_name

__all__ = ['SkopeRules', 'Rule']
