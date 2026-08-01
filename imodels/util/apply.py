"""Map samples to the leaves they land in, like sklearn's `tree.apply`."""

import numpy as np
from sklearn.utils.validation import check_array

from imodels.util.arguments import _finite_check_kwarg
from imodels.util.model_trees import sklearn_trees


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
    trees = sklearn_trees(model)
    if trees is None:
        raise ValueError(
            f"Don't know how to get leaf membership for {type(model).__name__}. "
            "apply is defined for tree-based models; if the model is not fitted "
            "yet, fit it first."
        )

    # missing values are left to the tree, which handles them for sklearn trees;
    # fit and predict accept them, so apply must too
    X = check_array(X, **_finite_check_kwarg(allow_nan=True))
    leaves = np.column_stack([tree.apply(X) for tree in trees])
    return leaves[:, 0] if len(trees) == 1 else leaves
