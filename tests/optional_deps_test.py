"""Models that need an optional dependency should say so when they are built,
not deep inside fit.
"""

import warnings

import pytest

import imodels.util.optional_deps as optional_deps
from imodels.algebraic.gam_multitask import MultiTaskGAMClassifier, MultiTaskGAMRegressor
from imodels.algebraic.gam_shap import ShapGAMClassifier, ShapGAMRegressor
from imodels.algebraic.slim import SLIMClassifier, SLIMRegressor


@pytest.fixture
def without_optional_deps(monkeypatch):
    """Pretend every optional dependency is missing."""
    monkeypatch.setattr(optional_deps, 'is_installed', lambda package: False)


class TestOptionalDependencyHelpers:
    def test_is_installed(self):
        assert optional_deps.is_installed('numpy')
        assert not optional_deps.is_installed('a_package_that_does_not_exist')

    def test_error_names_the_model_and_the_install(self):
        with pytest.raises(ImportError) as e:
            optional_deps.require_optional_dependency('a_missing_package', 'FooModel')
        assert 'FooModel' in str(e.value)
        assert 'pip install a_missing_package' in str(e.value)

    def test_no_error_when_installed(self):
        optional_deps.require_optional_dependency('numpy', 'FooModel')
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            optional_deps.warn_optional_dependency('numpy', 'FooModel', fallback='nothing')


@pytest.mark.parametrize('cls', [MultiTaskGAMRegressor, MultiTaskGAMClassifier,
                                 ShapGAMRegressor, ShapGAMClassifier])
def test_gams_raise_without_interpret(cls, without_optional_deps):
    with pytest.raises(ImportError, match='pip install interpret'):
        cls()


@pytest.mark.parametrize('cls', [SLIMRegressor, SLIMClassifier])
def test_slim_warns_without_cvxpy(cls, without_optional_deps):
    """SLIM still fits without cvxpy (it rounds a non-integer fit), so it warns."""
    with pytest.warns(UserWarning, match='pip install cvxpy'):
        m = cls()
    assert m.alpha > 0  # the model is still usable


@pytest.mark.parametrize('cls', [SLIMRegressor, SLIMClassifier])
def test_slim_silent_when_cvxpy_installed(cls, monkeypatch):
    monkeypatch.setattr(optional_deps, 'is_installed', lambda package: True)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        cls()
