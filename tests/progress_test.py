"""Models with a bounded fitting loop should be able to show a progress bar.

The bars are opt-in: nothing is written unless the model is built with a
truthy `verbose`. They go to stderr, so the textual reporting some models
print to stdout (FIGS) is unaffected.
"""

import contextlib
import io

import numpy as np
import pytest

from imodels import (FIGSClassifier, FIGSClassifierCV, HSTreeClassifierCV,
                     MarginalShrinkageLinearModelRegressor, OneRClassifier,
                     TaoTreeClassifier, TreeGAMClassifier)
from imodels.util.progress import progress_bar, progress_iter

RNG = np.random.RandomState(0)
X_CLS = RNG.randn(60, 3)
Y_CLS = (X_CLS[:, 0] > 0).astype(int)


def fit_stderr(model, X=X_CLS, y=Y_CLS):
    """Fit `model`, returning whatever it wrote to stderr."""
    buffer = io.StringIO()
    with contextlib.redirect_stderr(buffer):
        model.fit(X, y)
    return buffer.getvalue()


class TestHelpers:
    def test_iter_is_transparent(self):
        for verbose in (0, 1):
            assert list(progress_iter(range(4), verbose=verbose)) == [0, 1, 2, 3]

    def test_quiet_iter_writes_nothing(self):
        buffer = io.StringIO()
        with contextlib.redirect_stderr(buffer):
            list(progress_iter(range(4), verbose=0, desc='quiet'))
        assert buffer.getvalue() == ''

    def test_loud_iter_writes_the_description(self):
        buffer = io.StringIO()
        with contextlib.redirect_stderr(buffer):
            list(progress_iter(range(4), verbose=1, desc='loud'))
        assert 'loud' in buffer.getvalue()

    def test_manual_bar_is_a_context_manager_either_way(self):
        for verbose in (0, 1):
            buffer = io.StringIO()
            with contextlib.redirect_stderr(buffer):
                with progress_bar(total=2, verbose=verbose, desc='manual') as bar:
                    bar.update(1)
                    bar.set_postfix(step=1)
            assert ('manual' in buffer.getvalue()) == bool(verbose)

    def test_bar_write_goes_to_stdout(self):
        """So a model's own verbose text isn't swallowed by the bar."""
        for verbose in (0, 1):
            out, err = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                with progress_bar(total=1, verbose=verbose) as bar:
                    bar.write('a message')
            assert 'a message' in out.getvalue()


# each model paired with kwargs that keep the fit small
MODELS_WITH_BARS = [
    (FIGSClassifier, dict(max_rules=3)),
    (FIGSClassifierCV, dict(n_rules_list=[2], n_trees_list=[2], cv=2)),
    (TreeGAMClassifier, dict(n_boosting_rounds=3, random_state=0)),
    (TaoTreeClassifier, dict(n_iters=2)),
    (OneRClassifier, dict(max_depth=2)),
    (HSTreeClassifierCV, dict(reg_param_list=[0.1, 1], cv=2)),
]


@pytest.mark.parametrize('cls,kwargs', MODELS_WITH_BARS,
                         ids=[c.__name__ for c, _ in MODELS_WITH_BARS])
def test_silent_by_default(cls, kwargs):
    assert fit_stderr(cls(**kwargs)) == ''


@pytest.mark.parametrize('cls,kwargs', MODELS_WITH_BARS,
                         ids=[c.__name__ for c, _ in MODELS_WITH_BARS])
def test_bar_shown_when_verbose(cls, kwargs):
    assert 'it' in fit_stderr(cls(verbose=1, **kwargs))


@pytest.mark.parametrize('cls,kwargs', MODELS_WITH_BARS,
                         ids=[c.__name__ for c, _ in MODELS_WITH_BARS])
def test_verbose_round_trips_as_a_param(cls, kwargs):
    assert cls(verbose=1, **kwargs).get_params()['verbose'] == 1


def test_regressor_with_a_bar():
    """The marginal fits are per-feature, so the bar is over the columns."""
    y = X_CLS[:, 0] + 0.1 * X_CLS[:, 1]
    assert fit_stderr(MarginalShrinkageLinearModelRegressor(), y=y) == ''
    assert 'it' in fit_stderr(
        MarginalShrinkageLinearModelRegressor(verbose=1), y=y)
