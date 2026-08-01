"""Find the scikit-learn trees behind a fitted imodels model.

Several features (rule extraction, leaf membership) need the same thing: the
sklearn tree structures a model is built from, whatever shape the model takes.
Keeping that in one place stops those features from disagreeing about which
models are tree-based.
"""

import numpy as np

#: attributes under which a model may keep the fitted model it delegates to
WRAPPED_MODEL_ATTRS = ('figs', 'estimator_', 'model')


def is_sklearn_tree(estimator) -> bool:
    """Whether `estimator` holds a fitted sklearn tree.

    Checks the tree structure itself, since C45TreeClassifier also has a `tree_`
    but stores XML in it.
    """
    return hasattr(getattr(estimator, 'tree_', None), 'feature')


def n_tree_outputs(model) -> int:
    """Number of values each leaf of `model` holds (classes, or 1 for regression)."""
    n_outputs = getattr(model, 'n_outputs', None)
    if n_outputs:
        return int(n_outputs)
    classes = getattr(model, 'classes_', None)
    return len(classes) if classes is not None else 1


def sklearn_trees(model, convert_figs=True):
    """The fitted sklearn trees making up `model`, or None if it has none.

    Parameters
    ----------
    model
        A fitted model.
    convert_figs : bool
        Whether to convert a FIGS model's own tree objects into sklearn trees.
        Set False when the caller reads FIGS nodes directly (the converted trees
        store class counts rather than predictions).
    """
    if hasattr(model, 'trees_'):  # FIGS: a sum of its own tree objects
        if not convert_figs:
            return None
        from imodels.tree.viz_utils import extract_sklearn_tree_from_figs
        n_outputs = n_tree_outputs(model)
        return [extract_sklearn_tree_from_figs(model, i, n_outputs)
                for i in range(len(model.trees_))]

    if is_sklearn_tree(model):
        return [model]

    # models that delegate to another fitted model (shrinkage, CV wrappers, TAO)
    for attr in WRAPPED_MODEL_ATTRS:
        inner = getattr(model, attr, None)
        if inner is not None and inner is not model:
            trees = sklearn_trees(inner, convert_figs=convert_figs)
            if trees is not None:
                return trees

    subestimators = getattr(model, 'estimators_', None)
    if subestimators is not None and len(subestimators) > 0:
        trees = []
        for estimator in subestimators:
            if isinstance(estimator, np.ndarray):  # gradient boosting nests them
                estimator = estimator[0]
            if not is_sklearn_tree(estimator):
                return None
            trees.append(estimator)
        return trees

    return None
