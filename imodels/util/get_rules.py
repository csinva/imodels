"""Extract the rules of a fitted model as a pandas DataFrame.

imodels models store their rules in whatever form suits the algorithm: a list of
`Rule` objects, a list of dicts, or scikit-learn tree structures. `get_rules`
presents all of them the same way, so that inspecting a model doesn't require
knowing which family it belongs to.
"""

import numpy as np
import pandas as pd

from imodels.util.convert import tree_to_rules
from imodels.util.model_trees import (
    WRAPPED_MODEL_ATTRS,
    is_sklearn_tree,
    sklearn_trees,
)

#: Columns every `get_rules` result has, in the order they appear.
CORE_COLUMNS = ['rule', 'prediction']


def get_rules(model, feature_names=None) -> pd.DataFrame:
    """Return the rules of a fitted model as a DataFrame, one row per rule.

    Parameters
    ----------
    model
        A fitted imodels (or scikit-learn tree) model.
    feature_names : list of str, optional
        Names to use in the rule strings. Defaults to the names the model was
        fitted with, falling back to X0, X1, ... .

    Returns
    -------
    pandas.DataFrame
        One row per rule, always with these columns:

        - ``rule``: the condition as a string, e.g. ``"age <= 30.5 and bmi > 24.1"``.
          The catch-all final rule of a rule list is ``"else"``.
        - ``prediction``: what the rule itself predicts. For a single tree this is
          the value held in the leaf, so it is the model's prediction. For models
          that combine several rules the meaning follows the model: an additive
          model like FIGS contributes its trees' values, so they sum to the
          output, while a boosted ensemble takes a weighted vote and reports each
          tree's own prediction alongside a ``weight`` column. It is ``NaN``
          where a model defines no per-rule prediction.

        Models add their own columns on top of these, for example ``coef``,
        ``support`` and ``importance`` for RuleFit, or ``tree`` and ``depth`` for
        tree-based models. Those extra columns vary by model; only ``rule`` and
        ``prediction`` are guaranteed.

    Raises
    ------
    ValueError
        If the model does not expose rules in any recognized form.

    Examples
    --------
    >>> from imodels import FIGSClassifier, get_rules
    >>> model = FIGSClassifier(max_rules=3).fit(X, y)   # doctest: +SKIP
    >>> get_rules(model)                                # doctest: +SKIP
                                  rule  prediction  tree
    0                  X0 <= 0.011           0.06     0
    1                   X0 > 0.011           0.94     0
    """
    for extractor in (_rules_from_wrapped_model,
                      _rules_from_rulefit,
                      _rules_from_rule_objects,
                      _rules_from_rule_dicts,
                      _rules_from_boosted_single_rules,
                      _rules_from_c45,
                      _rules_from_figs,
                      _rules_from_trees):
        rules = extractor(model, feature_names)
        if rules is not None:
            return _order_columns(rules)

    raise ValueError(
        f"Don't know how to extract rules from {type(model).__name__}. "
        "get_rules supports imodels rule sets, rule lists and tree-based models; "
        "if the model is not fitted yet, fit it first."
    )


def _order_columns(rules: pd.DataFrame) -> pd.DataFrame:
    """Put the guaranteed columns first, keeping any model-specific ones after."""
    for col in CORE_COLUMNS:
        if col not in rules:
            rules[col] = np.nan
    extra = [c for c in rules.columns if c not in CORE_COLUMNS]
    return rules[CORE_COLUMNS + extra].reset_index(drop=True)


def _get_feature_names(model, feature_names, n_features=None):
    """Feature names to use in rule strings, preferring what the caller passed."""
    if feature_names is not None:
        return list(feature_names)
    for attr in ('feature_names_in_', 'feature_names_', 'feature_names'):
        names = getattr(model, attr, None)
        if names is not None and len(names) > 0:
            return [str(name) for name in names]
    if n_features is None:
        n_features = getattr(model, 'n_features_in_', 0)
    return [f'X{i}' for i in range(n_features)]


def _rules_from_wrapped_model(model, feature_names):
    """Models that delegate to another fitted model (shrinkage, CV wrappers)."""
    for attr in WRAPPED_MODEL_ATTRS:
        inner = getattr(model, attr, None)
        # an unfitted sklearn tree has no tree_; don't recurse into it
        if inner is not None and inner is not model and _has_rules(inner):
            names = _get_feature_names(model, feature_names,
                                       getattr(inner, 'n_features_in_', None))
            return get_rules(inner, feature_names=names)
    return None


def _has_rules(model):
    if getattr(model, 'rules_', None) is not None:
        return True
    if is_sklearn_tree(model):
        return True
    return any(hasattr(model, attr) for attr in ('trees_', 'estimators_'))


def _rules_from_rulefit(model, feature_names):
    """RuleFit already builds a rules table, with coefficients and support."""
    if not hasattr(model, '_get_rules') or getattr(model, 'coef', None) is None:
        return None
    rules = model._get_rules().copy()
    if 'rule' not in rules:
        return None
    # 'coef' is RuleFit's per-rule prediction: its contribution to the output
    rules['prediction'] = rules['coef']
    return rules


def _rules_from_rule_objects(model, feature_names):
    """Rule sets and Bayesian rule lists, which store imodels Rule objects."""
    rules = getattr(model, 'rules_', None)
    if not rules or isinstance(rules[0], dict):
        return None

    rows = []
    for rule in rules:
        args = getattr(rule, 'args', None)
        rows.append({
            'rule': str(rule),
            'prediction': args[0] if args else np.nan,
        })
    return pd.DataFrame(rows)


def _rules_from_rule_dicts(model, feature_names):
    """Greedy rule lists (and OneR), which store one dict per list entry."""
    rules = getattr(model, 'rules_', None)
    if not rules or not isinstance(rules[0], dict):
        return None

    names = _get_feature_names(model, feature_names)
    rows = []
    for rule in rules:
        if 'col' not in rule:  # the catch-all entry at the end of the list
            rows.append({'rule': 'else', 'prediction': rule.get('val', np.nan),
                         'depth': len(rows), 'num_pts': rule.get('num_pts', np.nan)})
            continue
        col = rule['col']
        if feature_names is not None and 'index_col' in rule:
            col = names[rule['index_col']]
        comparison = '<=' if rule.get('flip') else '>'
        rows.append({
            'rule': f"{col} {comparison} {np.round(rule['cutoff'], 5)}",
            'prediction': rule.get('val_right', np.nan),
            'depth': rule.get('depth', len(rows)),
            'num_pts': rule.get('num_pts_right', np.nan),
        })
    return pd.DataFrame(rows)


def _rules_from_boosted_single_rules(model, feature_names):
    """SLIPPER boosts base estimators that each hold a single rule."""
    subestimators = getattr(model, 'estimators_', None)
    if not subestimators or not hasattr(subestimators[0], 'rule'):
        return None

    names = _get_feature_names(model, feature_names)
    weights = getattr(model, 'estimator_weights_', None)
    rows = []
    for i, estimator in enumerate(subestimators):
        conditions = []
        for condition in estimator.rule or []:
            name = names[int(condition['feature'])]
            # pivots are sometimes stored as strings
            threshold = np.round(float(condition['pivot']), 5)
            conditions.append(f"{name} {condition['operator']} {threshold}")
        rows.append({
            'rule': ' and '.join(conditions) if conditions else 'else',
            # each boosted rule contributes its weight when it fires
            'prediction': weights[i] if weights is not None else np.nan,
        })
    return pd.DataFrame(rows)


def _rules_from_c45(model, feature_names):
    """C4.5 stores its tree as an XML document rather than an sklearn tree."""
    dom = getattr(model, 'dom_', None)
    if dom is None or not dom.childNodes:
        return None

    # flags mark how a child splits its parent: less-than, right (>=) or equal
    comparisons = {'l': '<', 'r': '>=', 'm': '=='}
    # the tree stores XML-safe names; report the ones the caller used
    original_names = getattr(model, 'xml_name_to_feature_name_', {})
    rows = []

    def walk(node, conditions):
        children = [c for c in node.childNodes if c.nodeType != c.TEXT_NODE]
        if not children:  # a leaf holds its predicted value as text
            text = node.firstChild.nodeValue if node.firstChild else None
            rows.append({
                'rule': ' and '.join(conditions) if conditions else 'else',
                'prediction': float(text) if text is not None else np.nan,
            })
            return
        for child in children:
            flag = child.getAttribute('flag')
            threshold = child.getAttribute('feature')
            comparison = comparisons.get(flag, flag)
            name = original_names.get(child.nodeName, child.nodeName)
            walk(child, conditions + [f"{name} {comparison} {threshold}"])

    walk(dom.childNodes[0], [])
    return pd.DataFrame(rows)


def _rules_from_figs(model, feature_names):
    """FIGS: a sum of trees, walked directly so that leaf values are its own.

    The converted sklearn tree stores class counts rather than predictions, so
    read the FIGS nodes instead. Note a FIGS prediction is the sum over trees, so
    `prediction` here is what this tree contributes when the rule applies.
    """
    trees = getattr(model, 'trees_', None)
    if not trees:
        return None

    names = _get_feature_names(model, feature_names,
                               getattr(model, 'n_features_in_', None))
    rows = []

    def walk(node, tree_num, conditions):
        if node is None:
            return
        if node.left is None and node.right is None:  # leaf
            value = np.ravel(node.value)
            rows.append({
                'rule': ' and '.join(conditions) if conditions else 'else',
                'prediction': value[-1] if value.size > 1 else value[0],
                'tree': tree_num,
            })
            return
        threshold = np.round(node.threshold, 5)
        name = names[node.feature]
        walk(node.left, tree_num, conditions + [f"{name} <= {threshold}"])
        walk(node.right, tree_num, conditions + [f"{name} > {threshold}"])

    for tree_num, tree in enumerate(trees):
        walk(tree, tree_num, [])
    return pd.DataFrame(rows)


def _rules_from_trees(model, feature_names):
    """Single trees, tree ensembles, and FIGS sums of trees."""
    trees = _collect_trees(model)
    if trees is None:
        return None

    n_features = getattr(model, 'n_features_in_', None)
    if n_features is None and trees:
        n_features = trees[0].n_features_in_
    names = _get_feature_names(model, feature_names, n_features)

    # boosted ensembles combine their trees by weighted vote, so report the
    # weight alongside each tree's own prediction
    tree_weights = getattr(model, 'estimator_weights_', None)
    if tree_weights is not None and len(tree_weights) != len(trees):
        tree_weights = None

    rows = []
    for tree_num, tree in enumerate(trees):
        for rule, value in tree_to_rules(tree, names, prediction_values=True):
            row = {
                'rule': rule,
                'prediction': value[-1] if len(value) > 1 else value[0],
                'tree': tree_num,
            }
            if tree_weights is not None:
                row['weight'] = tree_weights[tree_num]
            rows.append(row)
    rules = pd.DataFrame(rows)
    if len(trees) == 1:  # a single tree needs no tree index
        rules = rules.drop(columns='tree')
    return rules


def _collect_trees(model):
    """The sklearn trees making up a model, or None if it isn't tree-based.

    FIGS is excluded: _rules_from_figs reads its nodes directly, because the
    converted sklearn trees hold class counts rather than predictions.
    """
    return sklearn_trees(model, convert_figs=False)
