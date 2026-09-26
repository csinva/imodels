"""Report missing optional dependencies when a model is initialized.

A few imodels estimators rely on packages that are not installed by default
(the ``optional`` extra in pyproject.toml). Importing those packages lazily
keeps ``import imodels`` working without them, but the failure then surfaces
deep inside ``fit`` as an opaque ImportError -- or, worse, as a silent
fallback to a different model. The helpers here move the message up to
``__init__``, where it can name the model, the missing package, and the
command that installs it.
"""

import warnings
from importlib.util import find_spec

__all__ = ['is_installed', 'require_optional_dependency',
           'warn_optional_dependency']


def is_installed(package: str) -> bool:
    """Whether `package` can be imported, without importing it."""
    try:
        return find_spec(package) is not None
    except (ImportError, ValueError):
        # a parent package is missing, or the module is in a broken state
        return False


def _install_hint(package: str) -> str:
    return (f"Install it with `pip install {package}` "
            f"(or `pip install imodels[optional]` for all optional deps).")


def require_optional_dependency(package: str, model_name: str, purpose: str = ''):
    """Raise ImportError if `package` is missing.

    For models that cannot run at all without the dependency.

    Params
    ------
    package: str
        name of the package to import, e.g. 'interpret'
    model_name: str
        name of the model requiring it, used in the message
    purpose: str
        optional clause explaining what the package is used for
    """
    if is_installed(package):
        return
    because = f' ({purpose})' if purpose else ''
    raise ImportError(f"{model_name} requires the optional dependency "
                      f"'{package}'{because}, which is not installed. "
                      f"{_install_hint(package)}")


def warn_optional_dependency(package: str, model_name: str, fallback: str,
                             purpose: str = ''):
    """Warn if `package` is missing.

    For models that still work without the dependency, but behave differently.

    Params
    ------
    package: str
        name of the package to import, e.g. 'cvxpy'
    model_name: str
        name of the model that would use it, used in the message
    fallback: str
        what the model does instead, e.g. 'rounding the coefficients of a Lasso fit'
    purpose: str
        optional clause explaining what the package is used for
    """
    if is_installed(package):
        return
    because = f' ({purpose})' if purpose else ''
    warnings.warn(f"{model_name} uses the optional dependency "
                  f"'{package}'{because}, which is not installed, so it will "
                  f"fall back to {fallback}. {_install_hint(package)}")
