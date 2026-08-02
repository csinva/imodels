"""Plot imodels trees with dtreeviz.

dtreeviz draws a `ShadowDecTree`; imodels models are built from scikit-learn
trees underneath, so one can be built with dtreeviz's own `ShadowSKDTree`
(https://github.com/parrt/dtreeviz). dtreeviz is not a dependency of imodels and
is imported only when this is called.
"""

import numpy as np

from imodels.util.model_trees import n_tree_outputs, sklearn_trees


def shadow_tree(model, X, y, feature_names=None, target_name='target',
                class_names=None, tree_num=0):
    """Build a dtreeviz ShadowSKDTree for one of a model's trees.

    Parameters
    ----------
    model
        A fitted tree-based imodels model (FIGS, hierarchical shrinkage, CART,
        TAO, boosted rules, ...).
    X, y : the data to annotate the tree with, as dtreeviz requires.
    feature_names : list of str, optional
        Defaults to the names the model was fitted with, else X0, X1, ... .
    target_name : str
        Name shown for the target.
    class_names : list, optional
        Classification only. Defaults to the model's ``classes_``.
    tree_num : int
        Which tree to draw, for models made of several (FIGS, boosted rules).

    Returns
    -------
    dtreeviz.models.sklearn_decision_trees.ShadowSKDTree

    Raises
    ------
    ImportError
        If dtreeviz is not installed.
    ValueError
        If the model is not tree-based, or tree_num is out of range.

    Examples
    --------
    >>> import dtreeviz                                       # doctest: +SKIP
    >>> from imodels import FIGSClassifier, shadow_tree       # doctest: +SKIP
    >>> model = FIGSClassifier(max_rules=5).fit(X, y)         # doctest: +SKIP
    >>> viz = dtreeviz.trees.DTreeVizAPI(                     # doctest: +SKIP
    ...     shadow_tree(model, X, y, tree_num=0))
    >>> viz.view()                                            # doctest: +SKIP
    """
    try:
        from dtreeviz.models.sklearn_decision_trees import ShadowSKDTree
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise ImportError(
            "dtreeviz is needed to plot imodels trees but is not installed "
            "('pip install dtreeviz'). It is not a dependency of imodels."
        ) from error

    trees = sklearn_trees(model)
    if trees is None:
        raise ValueError(
            f"Don't know how to draw {type(model).__name__}. dtreeviz support "
            "covers tree-based models; if the model is not fitted yet, fit it first."
        )
    if not 0 <= tree_num < len(trees):
        raise ValueError(
            f"tree_num must be in [0, {len(trees)}) for this model, got {tree_num}."
        )

    X = np.asarray(X)
    if feature_names is None:
        feature_names = _feature_names(model, X.shape[1])
    if class_names is None:
        class_names = _class_names(model)

    return ShadowSKDTree(trees[tree_num], X, np.asarray(y),
                         list(feature_names), target_name, class_names)


def _feature_names(model, n_features):
    for attr in ('feature_names_in_', 'feature_names_', 'feature_names'):
        names = getattr(model, attr, None)
        if names is not None and len(names) == n_features:
            return [str(name) for name in names]
    return [f'X{i}' for i in range(n_features)]


def _class_names(model):
    """dtreeviz wants class labels for classifiers and None for regressors."""
    classes = getattr(model, 'classes_', None)
    if classes is None or n_tree_outputs(model) == 1:
        return None
    return [str(label) for label in classes]
