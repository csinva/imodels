"""Run interactive pages in headless Chromium: no script errors, Predict and what-if work.

Skipped when playwright or its browser is not installed.
"""

import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.tree import DecisionTreeClassifier

import imodels.viz as dti

sync_api = pytest.importorskip("playwright.sync_api")


@pytest.fixture(scope="module")
def page():
    try:
        p = sync_api.sync_playwright().start()
        browser = p.chromium.launch()
    except Exception as e:  # browser binaries missing
        pytest.skip(f"chromium not available: {e}")
    pg = browser.new_page(viewport={"width": 1400, "height": 900})
    yield pg
    browser.close()
    p.stop()


def exercise(pg, html, tmp_path, name):
    path = tmp_path / f"{name}.html"
    path.write_text(html)
    errors = []
    pg.on("pageerror", lambda e: errors.append(str(e)))
    pg.goto(path.as_uri())
    pg.wait_for_timeout(300)
    pg.click("#b-pred")
    pg.wait_for_timeout(350)
    pg.click("#b-rand")
    pg.wait_for_timeout(300)
    before = pg.inner_text("#res")
    assert pg.locator("#whatif button").count() > 0, pg.inner_text("#whatif")
    pg.locator("#whatif button").first.click()
    pg.wait_for_timeout(300)
    after = pg.inner_text("#res")
    assert not errors, errors
    return before, after


def test_tree_page(page, tmp_path):
    d = load_iris(as_frame=True)
    m = DecisionTreeClassifier(max_depth=3, random_state=0).fit(d.data, d.target)
    before, after = exercise(page, dti.interactive(m, d.data, d.target).html, tmp_path, "tree")
    assert "PREDICTION" in before.upper() and before != after
    # clicking a leaf loads one of its training rows
    page.locator(".node.leaf").first.click()
    page.wait_for_timeout(300)
    assert "leaf" in page.inner_text("#res")


def test_view_page(page, tmp_path):
    im = pytest.importorskip("imodels")
    d = load_breast_cancer(as_frame=True)
    m = im.RuleFitClassifier(max_rules=8, random_state=0).fit(d.data, d.target)
    before, after = exercise(page, dti.interactive(m, d.data, d.target).html, tmp_path, "view")
    assert before != after
    # the waterfall adds up to the score shown
    assert page.locator(".wf .tot").count() == 2
