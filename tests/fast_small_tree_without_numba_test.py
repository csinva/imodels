"""FastSmallTreeClassifier without numba, its optional dependency.

numba is in the dev dependencies, so the rest of the suite always has it; this runs
imodels in a fresh interpreter with numba made unimportable, to check that the package
still imports, that other models are unaffected, and that fitting FastSmallTree fails
with an ImportError that says how to install it.
"""

import subprocess
import sys
import textwrap

SCRIPT = textwrap.dedent("""
    import sys
    sys.modules["numba"] = None               # as if numba were not installed
    import numpy as np
    import imodels
    from imodels import FastSmallTreeClassifier, GreedyTreeClassifier
    X = np.random.RandomState(0).randint(0, 2, (40, 3))
    y = X[:, 0] ^ X[:, 1]
    GreedyTreeClassifier().fit(X, y)
    try:
        FastSmallTreeClassifier().fit(X, y)
    except ImportError as err:
        print("IMPORTERROR:", err)
    else:
        print("NO ERROR")
""")


def test_fit_without_numba_raises_an_install_hint():
    out = subprocess.run([sys.executable, "-c", SCRIPT], capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr
    assert "IMPORTERROR:" in out.stdout, out.stdout
    assert "pip install numba" in out.stdout
