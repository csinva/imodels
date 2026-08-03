"""Regressions for defects found while auditing the library.

Each test here pins a specific bug so it cannot come back; the docstring says
what the bug was rather than only what the test checks.
"""

import inspect

import numpy as np
import pytest
from sklearn.tree import DecisionTreeClassifier

import imodels
from tests.model_configs import MODEL_KWARGS, model_kwargs


def test_model_kwargs_hands_out_fresh_estimators():
    """The test registry must not share one estimator instance across tests

    MODEL_KWARGS holds an ``estimator_=DecisionTreeClassifier(...)`` for the CCP
    models. Reading it directly handed every test the *same* instance, so once
    one test fitted it, later tests got a model that was already fitted -- a
    fresh DecisionTreeCCPClassifier would happily predict without being fit.
    """
    first = model_kwargs("DecisionTreeCCPClassifier")
    second = model_kwargs("DecisionTreeCCPClassifier")
    assert first["estimator_"] is not second["estimator_"]

    registry_estimator = MODEL_KWARGS["DecisionTreeCCPClassifier"]["estimator_"]
    X = np.random.RandomState(0).randn(40, 3)
    y = (X[:, 0] > 0).astype(int)
    imodels.DecisionTreeCCPClassifier(**model_kwargs("DecisionTreeCCPClassifier")).fit(X, y)
    assert not hasattr(registry_estimator, "tree_"), (
        "fitting a model built from the registry must not fit the registry's own estimator"
    )


def test_c45_module_has_no_duplicate_definitions():
    """c45_tree.py carried a verbatim second copy of three helpers

    _add_label, _get_next_node and shrink_node were each defined twice; the
    first 49 lines were dead, shadowed by the identical block below them.
    """
    import imodels.tree.c45_tree.c45_tree as c45

    source = inspect.getsource(c45)
    for name in ("_add_label", "_get_next_node", "shrink_node"):
        assert source.count(f"\ndef {name}(") == 1, f"{name} is defined more than once"


def test_slipper_defines_fit_once():
    """SlipperClassifier had two fit methods, the first of them dead

    A merge left both the old inline multiclass guard and the one using
    check_binary_target; only the second took effect.
    """
    import imodels.rule_set.slipper as slipper

    assert inspect.getsource(slipper).count("    def fit(") == 1


def test_one_r_is_not_fitted_before_fit():
    """OneRClassifier set feature_names_ in __init__

    A trailing-underscore attribute is how sklearn recognizes a fitted model, so
    check_is_fitted passed on an unfitted OneRClassifier and predict then failed
    with an AttributeError about some other attribute.
    """
    from sklearn.exceptions import NotFittedError
    from sklearn.utils.validation import check_is_fitted

    model = imodels.OneRClassifier()
    with pytest.raises(NotFittedError):
        check_is_fitted(model)


def test_get_rules_reaches_through_automl_search():
    """imodels.get_rules could not see through AutoInterpretableModel

    It keeps the chosen model in est_ -- a GridSearchCV whose best_estimator_ is
    a Pipeline -- and none of those hops were followed.
    """
    X = np.random.RandomState(0).randn(60, 3)
    y = (X[:, 0] > 0).astype(int)
    model = imodels.AutoInterpretableClassifier(
        param_grid=[{"est": [DecisionTreeClassifier(random_state=0)],
                     "est__max_leaf_nodes": [2, 4]}],
    ).fit(X, y)

    rules = imodels.get_rules(model)
    assert list(rules.columns[:2]) == ["rule", "prediction"]
    assert len(rules) > 0
    assert hasattr(model, "get_rules")


def test_multitask_gam_does_not_mutate_ebm_kwargs():
    """MultiTaskGAM.__init__ wrote random_state/interactions into ebm_kwargs

    ebm_kwargs defaulted to a dict literal, and __init__ mutated it in place --
    so the shared default accumulated keys across instantiations, and a caller's
    own dict was modified behind their back.
    """
    interpret = pytest.importorskip(
        "interpret", reason="MultiTaskGAM needs the optional interpret dependency")
    from imodels.algebraic.gam_multitask import DEFAULT_EBM_KWARGS, MultiTaskGAM

    before = dict(DEFAULT_EBM_KWARGS)
    MultiTaskGAM(random_state=99, interactions=0.5)
    assert DEFAULT_EBM_KWARGS == before, "the shared default was mutated"

    caller_kwargs = {"n_jobs": 4}
    model = MultiTaskGAM(ebm_kwargs=caller_kwargs, random_state=7)
    assert caller_kwargs == {"n_jobs": 4}, "the caller's dict was mutated"
    assert model.get_params()["ebm_kwargs"] == {"n_jobs": 4}

    # the overrides still reach the EBM, just at use time
    assert model._ebm_kwargs()["random_state"] == 7


def test_explicit_get_params_honors_deep():
    """Models that spell out get_params were ignoring its `deep` argument

    sklearn refuses to introspect an __init__ with *args/**kwargs, so these
    models list their parameters by hand -- and returned the same flat dict
    whatever `deep` was. That hides the wrapped estimator's parameters, which
    is what a search needs to tune it as `<param>__<subparam>`.
    """
    model = imodels.DecisionTreeCCPClassifier(
        estimator_=DecisionTreeClassifier(max_depth=3), desired_complexity=3)

    shallow = model.get_params(deep=False)
    assert not any("__" in k for k in shallow)

    deep = model.get_params(deep=True)
    assert "estimator___max_depth" in deep, sorted(deep)
    assert deep["estimator___max_depth"] == 3

    # and the nested parameter can be set back
    model.set_params(estimator___max_depth=7)
    assert model.estimator_.max_depth == 7
    # a plain parameter still works
    model.set_params(desired_complexity=5)
    assert model.desired_complexity == 5


def test_hs_c45_cv_is_clonable():
    """HSC45TreeClassifierCV could not be cloned, so it could not be searched

    Two separate causes: __init__ takes *args/**kwargs, which sklearn's
    parameter introspection rejects outright, and it then rebuilt
    reg_param_list with np.array(), which clone detects as the constructor
    modifying a parameter.
    """
    from sklearn.base import clone
    from imodels.tree.c45_tree.c45_tree import (C45TreeClassifier,
                                                HSC45TreeClassifierCV)

    model = HSC45TreeClassifierCV(estimator_=C45TreeClassifier(max_rules=5))
    cloned = clone(model)
    assert type(cloned) is HSC45TreeClassifierCV

    X = np.random.RandomState(0).randn(60, 3)
    y = (X[:, 0] > 0).astype(int)
    assert cloned.fit(X, y) is None or True     # fit returns None on this model
    assert cloned.predict(X).shape == (60,)
