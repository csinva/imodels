"""FastSmallTreeClassifier and FastRiskScoreClassifier without numba, their optional dependency.

numba is in the dev dependencies, so the rest of the suite always has it; this runs
imodels in a fresh interpreter with numba made unimportable, to check that the package
still imports, that other models are unaffected, and that fitting either numba model
fails with an ImportError that says how to install it. Both models share one
interpreter, since starting it and importing imodels is most of the cost.
"""

import subprocess
import sys
import textwrap

import pytest

SCRIPT = textwrap.dedent("""
    import sys
    sys.modules["numba"] = None               # as if numba were not installed
    import numpy as np
    import imodels
    from imodels import FastRiskScoreClassifier, FastSmallTreeClassifier, GreedyTreeClassifier
    X = np.random.RandomState(0).randint(0, 2, (40, 3))
    y = X[:, 0] ^ X[:, 1]
    GreedyTreeClassifier().fit(X, y)
    for model in (FastSmallTreeClassifier, FastRiskScoreClassifier):
        try:
            model().fit(X, y)
        except ImportError as err:
            print(model.__name__, "IMPORTERROR:", err)
        else:
            print(model.__name__, "NO ERROR")
""")


@pytest.fixture(scope="module")
def output():
    out = subprocess.run([sys.executable, "-c", SCRIPT], capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr
    return dict(line.split(" ", 1) for line in out.stdout.splitlines() if " " in line)


@pytest.mark.parametrize("model", ["FastSmallTreeClassifier", "FastRiskScoreClassifier"])
def test_fit_without_numba_raises_an_install_hint(output, model):
    assert output[model].startswith("IMPORTERROR:"), output
    assert "pip install numba" in output[model]
