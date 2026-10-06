"""Static and interactive visualizations of fitted models.

`draw` renders a model as a self-contained SVG figure (save it as .svg, .png, .pdf or .html), and
`interactive` builds a single offline HTML page to explore it: fold and unfold trees, highlight a
feature, route a sample through the model, and see the smallest change that would flip its prediction.

Supported models, each drawn so that it reproduces the model's own `predict` / `predict_proba`:

- **imodels**: trees (GreedyTree, DecisionTreeCCP, HSTree, TaoTree, C4.5, FastSmallTree), sums of trees
  (FIGS, IRF), rule lists (GreedyRuleList, OneR, FastFrugalTree, BayesianRuleList), rule sets (RuleFit,
  FPLasso, SkopeRules, FPSkope, BoostedRules, Slipper, BayesianRuleSet), scoring systems
  (FastRiskScore, SLIM) and additive models (TreeGAM, GPGam, MarginalShrinkageLinearRegressor).
- **scikit-learn**: decision trees, random forests, extra trees, gradient boosting, histogram gradient
  boosting, linear and logistic models, GLMs, linear SVMs and isotonic regression. Pipelines whose
  earlier steps only scale features are drawn in raw units.

```python
from imodels import FIGSClassifier
from imodels.viz import draw, interactive

model = FIGSClassifier(max_rules=8).fit(X, y)
draw(model, X, y).save("figs.svg")                # static figure
interactive(model, X, y).save("figs.html")        # interactive page, one offline file
```

Both functions show inline in Jupyter. See the [gallery post](https://csinva.io/imodels/viz.html)
for examples of every kind of model.

Tree-based imodels models (the trees above, FIGS and IRF) are drawn through their exact scikit-learn
export, `imodels.to_sklearn`, so this package only reads scikit-learn trees. The same export makes
every tree-based imodels model drawable by any tool that reads scikit-learn trees, such as
[dtreeviz](https://github.com/parrt/dtreeviz) (`dtreeviz.model(imodels.to_sklearn(model), X, y)`) or
`sklearn.tree.plot_tree`.
"""

from ._interactive import InteractiveTree, interactive
from ._static import TreeFigure, draw

__all__ = ["draw", "interactive", "TreeFigure", "InteractiveTree"]
