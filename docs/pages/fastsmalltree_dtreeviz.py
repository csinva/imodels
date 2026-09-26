"""Draw the quickstart's breast-cancer tree with dtreeviz for fastsmalltree.html.

FastSmallTreeClassifier stores its certified tree as an sklearn
DecisionTreeClassifier in ``estimator_``, so dtreeviz reads it like any other
sklearn tree. This writes ``docs/img/fastsmalltree_dtreeviz.svg``; dtreeviz is not
an imodels dependency, so run it with one supplied for the run:

    uv run --with dtreeviz docs/pages/fastsmalltree_dtreeviz.py
"""

import os

import dtreeviz
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from imodels import FastSmallTreeClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "img", "fastsmalltree_dtreeviz.svg")

data = load_breast_cancer(as_frame=True)
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

model = FastSmallTreeClassifier(regularization=0.03).fit(X_train, y_train)
assert model.optimal_, "the example should show a certified tree"

viz = dtreeviz.model(model.estimator_, X_train=X_train, y_train=y_train,
                     feature_names=list(X.columns), target_name="diagnosis",
                     class_names=list(data.target_names))
viz.view(scale=1.3).save(OUT)
# save() also leaves the graphviz source beside the svg, which the page does not use
source = os.path.splitext(OUT)[0]
if os.path.exists(source):
    os.remove(source)
print(f"{model.n_leaves_} leaves, objective {model.objective_:.4f}, "
      f"test accuracy {(model.predict(X_test) == y_test).mean():.3f} -> {os.path.relpath(OUT)}")
