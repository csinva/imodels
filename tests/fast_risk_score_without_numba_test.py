"""FastRiskScoreClassifier without numba, its optional dependency: imodels still imports,
and fitting the model fails with an ImportError that says how to install numba."""

import subprocess
import sys
import textwrap

SCRIPT = textwrap.dedent("""
    import sys
    sys.modules["numba"] = None               # as if numba were not installed
    import numpy as np
    from imodels import FastRiskScoreClassifier, GreedyTreeClassifier
    X = np.random.RandomState(0).randint(0, 2, (40, 3))
    y = X[:, 0] ^ X[:, 1]
    GreedyTreeClassifier().fit(X, y)
    try:
        FastRiskScoreClassifier().fit(X, y)
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
