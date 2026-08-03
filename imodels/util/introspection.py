"""Mixins giving models the shared introspection methods.

``imodels.get_rules`` and ``imodels.util.apply.apply_leaves`` work on a model
from the outside. These mixins expose them as methods so that ``model.get_rules()``
also works, without every model repeating the same two-line delegation.

A model mixes in only what it actually supports: ``model_introspection_test``
asserts that a model has ``get_rules`` exactly when ``imodels.get_rules`` supports
it, and likewise for ``apply``.
"""


class RulesMixin:
    """Adds ``get_rules()`` to a model whose rules imodels.get_rules can extract."""

    def get_rules(self, feature_names=None):
        """Return this model's rules as a DataFrame (see imodels.get_rules)."""
        from imodels.util.get_rules import get_rules
        return get_rules(self, feature_names=feature_names)


class LeavesMixin:
    """Adds ``apply()`` to a model built from trees, so leaf membership is defined."""

    def apply(self, X):
        """Return the leaf each sample reaches (see imodels.util.apply.apply_leaves)."""
        from imodels.util.apply import apply_leaves
        return apply_leaves(self, X)


class RuleInspectionMixin(RulesMixin, LeavesMixin):
    """Both of the above, for tree-based models that support each."""
