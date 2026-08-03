import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.tree import _tree
from typing import Union, List, Tuple


def _round_threshold(threshold, decimals: int = 5):
    """Shorten a split threshold without moving it onto the data.

    Rounding to a fixed number of decimal places collapses thresholds on
    small-scale features to 0.0, which changes which rows a rule selects (and
    can turn two distinct splits into the same one). Rounding to significant
    digits instead keeps the threshold in the right place at any scale.
    """
    if threshold == 0 or not np.isfinite(threshold):
        return threshold
    magnitude = int(np.floor(np.log10(abs(threshold))))
    # keep `decimals` digits after the point for values of order 1, and the
    # equivalent precision for values that are much smaller or larger
    return float(np.round(threshold, decimals=max(decimals, decimals - magnitude)))


def tree_to_rules(tree: Union[DecisionTreeClassifier, DecisionTreeRegressor],
                  feature_names: List[str],
                  prediction_values: bool = False, round_thresholds=True) -> List[str]:
    """
    Return a list of rules from a tree

    Parameters
    ----------
        tree : Decision Tree Classifier/Regressor
        feature_names: list of variable names

    Returns
    -------
    rules : list of rules.
    """
    # XXX todo: check the case where tree is build on subset of features,
    # ie max_features != None

    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    rules = []

    def recurse(node, base_name):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            symbol = '<='
            symbol2 = '>'
            threshold = tree_.threshold[node]
            if round_thresholds:
                threshold = _round_threshold(threshold)
            text = base_name + ["{} {} {}".format(name, symbol, threshold)]
            recurse(tree_.children_left[node], text)

            text = base_name + ["{} {} {}".format(name, symbol2,
                                                  threshold)]
            recurse(tree_.children_right[node], text)
        else:
            rule = str.join(' and ', base_name)
            rule = (rule if rule != ''
                    else ' == '.join([feature_names[0]] * 2))
            # a rule selecting all is set to "c0==c0"
            if prediction_values:
                rules.append((rule, tree_.value[node][0].tolist()))
            else:
                rules.append(rule)

    recurse(0, [])

    return rules if len(rules) > 0 else 'True'


def itemsets_to_rules(itemsets: List[Tuple]) -> List[str]:
    itemsets_clean = list(filter(lambda it: it != 'null' and 'All' not in ''.join(it), itemsets))
    f = lambda itemset: ' and '.join([single_discretized_feature_to_rule(item) for item in itemset])
    return list(map(f, itemsets_clean))


def single_discretized_feature_to_rule(feat: str) -> str:
    # categorical feature
    if '_to_' not in feat:
        return f'{feat} > 0.5'

    # discretized numeric feature
    feat_split = feat.split('_to_')
    upper_value = feat_split[-1]
    lower_value = feat_split[-2].split('_')[-1]

    lower_to_upper_len = 1 + len(lower_value) + 4 + len(upper_value)
    feature_name = feat[:-lower_to_upper_len]

    if lower_value == '-inf':
        rule = f'{feature_name} <= {upper_value}'
    elif upper_value == 'inf':
        rule = f'{feature_name} > {lower_value}'
    else:
        rule = f'{feature_name} > {lower_value} and {feature_name} <= {upper_value}'

    return rule


def is_xgboost_model(model) -> bool:
    """Whether model is an XGBoost estimator, without importing xgboost.

    xgboost is not a dependency of imodels, so this is duck-typed: it is only
    true for something exposing a Booster.
    """
    return hasattr(model, 'get_booster') and type(model).__module__.startswith('xgboost')


def xgboost_to_rules(model, feature_names: List[str],
                     prediction_values: bool = False) -> List[str]:
    """Return a list of rules from a fitted XGBoost model.

    XGBoost stores its trees in a Booster rather than sklearn's tree_ arrays, so
    the paths are read off `trees_to_dataframe()`. Note XGBoost sends a sample
    left when `feature < threshold` (sklearn uses `<=`), which is reflected here.

    Thresholds are emitted at full precision. XGBoost compares in float32, so a
    row sitting within float32 rounding of a threshold can be routed differently
    when the rule is later evaluated in float64; evaluated in float32 the rules
    reproduce XGBoost's leaf assignment exactly.

    Parameters
    ----------
        model : a fitted XGBClassifier/XGBRegressor
        feature_names: list of variable names

    Returns
    -------
    rules : list of rules, or of (rule, leaf value) pairs if prediction_values
    """
    booster = model.get_booster()
    frame = booster.trees_to_dataframe()

    # the booster names features f0, f1, ... when fitted on an array; map those
    # back onto the caller's names by position
    booster_names = booster.feature_names or []
    name_by_booster_name = {booster_name: feature_names[i]
                            for i, booster_name in enumerate(booster_names)
                            if i < len(feature_names)}

    def resolve(booster_feature):
        if booster_feature in name_by_booster_name:
            return name_by_booster_name[booster_feature]
        if booster_feature.startswith('f') and booster_feature[1:].isdigit():
            index = int(booster_feature[1:])
            if index < len(feature_names):
                return feature_names[index]
        return booster_feature

    rules = []
    for _, tree in frame.groupby('Tree'):
        nodes = tree.set_index('ID')

        def recurse(node_id, conditions):
            node = nodes.loc[node_id]
            if node['Feature'] == 'Leaf':
                condition = ' and '.join(conditions)
                if condition == '':  # a stump that never splits selects everything
                    condition = ' == '.join([feature_names[0]] * 2)
                # for a leaf, Gain holds the value the tree contributes
                rules.append((condition, [float(node['Gain'])])
                             if prediction_values else condition)
                return
            name = resolve(node['Feature'])
            # full precision: rounding here would send rows that sit between the
            # true and rounded threshold down the wrong branch
            threshold = repr(float(node['Split']))
            recurse(node['Yes'], conditions + [f'{name} < {threshold}'])
            recurse(node['No'], conditions + [f'{name} >= {threshold}'])

        recurse(tree['ID'].iloc[0], [])
    return rules
