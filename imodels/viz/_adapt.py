"""Turn any supported model into a view object (TreeInfo for tree-like models)."""

from ._extract import TreeInfo, extract


def to_view(model, X=None, y=None, feature_names=None, class_names=None, target_name=None, output=0):
    from ._views import AdditiveView

    if isinstance(model, (TreeInfo, AdditiveView)):
        return model
    from . import _imodels_models as _imodels

    view = _imodels.adapt(model, X, y, feature_names, class_names, target_name)
    if view is not None:
        return view
    from . import _sklearn

    view = _sklearn.adapt(model, X, y, feature_names, class_names, target_name)
    if view is not None:
        return view
    return extract(model, X, y, feature_names, class_names, target_name, output)
