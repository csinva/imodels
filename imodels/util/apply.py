"""Map samples to the leaves they land in, like sklearn's `tree.apply`."""

import numpy as np
from sklearn.utils.validation import check_array


def apply_leaves(model, X) -> np.ndarray:
    """Return the leaf each sample reaches, for a tree-based model.

    Parameters
    ----------
    model
        A fitted tree-based imodels model.
    X : array-like of shape (n_samples, n_features)

    Returns
    -------
    numpy.ndarray
        For a model built from a single tree, shape ``(n_samples,)``: the index
        of the leaf each sample falls in, using the same node numbering as
        scikit-learn's ``DecisionTree.apply``.

        For a model built from several trees (FIGS, boosted rules, TreeGAM),
        shape ``(n_samples, n_trees)``, matching ``RandomForest.apply``: column
        ``t`` holds the leaf reached in tree ``t``.

    Raises
    ------
    ValueError
        If the model is not tree-based, or is not fitted yet.

    Examples
    --------
    >>> model = FIGSClassifier(max_rules=3).fit(X, y)   # doctest: +SKIP
    >>> model.apply(X).shape                            # doctest: +SKIP
    (100, 2)
    """
    trees = _sklearn_trees(model)
    if trees is None:
        raise ValueError(
            f"Don't know how to get leaf membership for {type(model).__name__}. "
            "apply is defined for tree-based models; if the model is not fitted "
            "yet, fit it first."
        )

    X = check_array(X, ensure_all_finite=False) if _accepts_nan() else check_array(X)
    leaves = np.column_stack([tree.apply(X) for tree in trees])
    return leaves[:, 0] if len(trees) == 1 else leaves


def _accepts_nan():
    from inspect import signature
    return 'ensure_all_finite' in signature(check_array).parameters


def _is_sklearn_tree(estimator):
    return hasattr(getattr(estimator, 'tree_', None), 'feature')


def _sklearn_trees(model):
    """The fitted sklearn trees behind a model, or None if there are none."""
    if hasattr(model, 'trees_'):  # FIGS: a sum of trees
        from imodels.tree.viz_utils import extract_sklearn_tree_from_figs
        n_classes = len(getattr(model, 'classes_', [0, 1]))
        return [extract_sklearn_tree_from_figs(model, i, n_classes)
                for i in range(len(model.trees_))]

    if _is_sklearn_tree(model):
        return [model]

    # models that wrap or delegate to another fitted model
    for attr in ('figs', 'estimator_', 'model'):
        inner = getattr(model, attr, None)
        if inner is not None and inner is not model:
            trees = _sklearn_trees(inner)
            if trees is not None:
                return trees

    subestimators = getattr(model, 'estimators_', None)
    if subestimators is not None and len(subestimators) > 0:
        trees = []
        for estimator in subestimators:
            if isinstance(estimator, np.ndarray):  # gradient boosting nests them
                estimator = estimator[0]
            if not _is_sklearn_tree(estimator):
                return None
            trees.append(estimator)
        return trees

    return None
