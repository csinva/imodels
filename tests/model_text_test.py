"""print(model) shows the fitted model as readable text (imodels.viz.text), for every model imodels.viz reads."""

import os
import sys

import pytest

import imodels
from imodels import viz

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from viz_imodels_test import CASES, fitted  # noqa: E402  (the fitted models the viz tests use)


@pytest.mark.parametrize("name", sorted(CASES))
def test_printing_shows_the_fitted_model(name):
    m, X, y, _ = fitted(name)
    printed = str(m)
    assert printed == viz.text(m)
    assert printed.splitlines()[0].startswith(type(m).__name__)
    assert "(" not in printed.splitlines()[0] or "shrinkage" in printed.splitlines()[0]  # not the parameter repr
    assert len(printed.splitlines()) >= 4


def test_unfitted_models_print_their_parameters():
    assert str(imodels.FIGSClassifier(max_rules=3)) == repr(imodels.FIGSClassifier(max_rules=3))
    assert "max_rules=3" in str(imodels.FIGSClassifier(max_rules=3))


def test_text_with_data_adds_coverage():
    m, X, y, _ = fitted("RuleFitClassifier")
    with_data = viz.text(m, X, y)
    assert "coverage" in with_data and "coverage" not in viz.text(m)
