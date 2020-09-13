<h1 align="center"> Interpretable machine learning models (imodels) <img src='https://svgshare.com/i/PDf.svg' style="height:1em;"/> </h1>
<p align="center"> Straightforward implementations of interpretable ML models + demos of how to use various interpretability techniques. Code is optimized for readability. Pull requests welcome!
</p>

<p align="center">
  <a href="#implementations-of-interpretable-models"> Implementations of imodels </a> •
  <a href="#demo-notebook">Demo notebook</a> •
  <a href="https://docs.google.com/presentation/d/1RIdbV279r20marRrN0b1bu2z9STkrivsMDa_Dauk8kE/present">Accompanying slides</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg">
  <a href="https://github.com/csinva/interpretability-implementations-demos/actions"><img src="https://github.com/csinva/interpretability-implementations-demos/workflows/tests/badge.svg"></a>
</p>


## Implementations of interpretable models
Provides scikit-learn style wrappers/implementations of different interpretable models - can be easily installed and used:

`pip install git+https://github.com/csinva/interpretability-implementations-demos`

```python
from imodels import RuleListClassifier, RuleFit, GreedyRuleList, SkopeRules, SLIM, IRFClassifier
model = RuleListClassifier()  # initialize Bayesian Rule List
model.fit(X_train, y_train)   # fit model
preds = model.predict(X_test) # discrete predictions: shape is (n_test, 1)
preds_proba = model.predict_proba(X_test) # predicted probabilities: shape is (n_test, n_classes)
```

- [bayesian rule list](https://arxiv.org/abs/1602.08610) (based on [this implementation](https://github.com/tmadl/sklearn-expertsys)) - learn a compact rule list
- [rulefit](http://statweb.stanford.edu/~jhf/ftp/RuleFit.pdf) (based on [this implementation](https://github.com/christophM/rulefit)) - find rules from a decision tree and build a linear model with them
- [sparse integer linear model](https://link.springer.com/article/10.1007/s10994-015-5528-6) (simple implementation with cvxpy)
- greedy rule list (based on [this implementation](https://medium.com/@penggongting/implementing-decision-tree-from-scratch-in-python-c732e7c69aea)) - uses CART to learn a list (only a single path), rather than a decision tree
- [skope-rules](https://github.com/scikit-learn-contrib/skope-rules) (based on [this implementation](https://github.com/scikit-learn-contrib/skope-rules))
- (in progress) [optimal classification tree](https://link.springer.com/article/10.1007/s10994-017-5633-9) (based on [this implementation](https://github.com/pan5431333/pyoptree)) - learns succinct trees using global optimization rather than greedy heuristics
- [iterative random forest](https://www.pnas.org/content/115/8/1943) (based on [this implementation](https://github.com/Yu-Group/iterative-Random-Forest))



## Demo notebook
Demo on all usage is contained in this [notebook](https://github.com/csinva/interpretability-implementations-demos/blob/master/notebooks/1_model_based.ipynb), focusing on the model-based interpretability part of this cheat-sheet: ![cheat_sheet](https://csinva.github.io/interpretability-implementations-demos/docs/cheat_sheet.png)
For post-hoc interpretability, see the [<img src='https://csinva.github.io/assets/github.svg' style="height:1em;" /> github repo](https://github.com/csinva/interpretability-implementations-demos).



## References / further reading

- [high-level review on interpretable machine learning](https://arxiv.org/abs/1901.04592)
- [book on interpretable machine learning](https://christophm.github.io/interpretable-ml-book/)
- [review on black-blox explanation methods](https://hal.inria.fr/hal-02131174v2/document)
- [review on variable importance](https://www.sciencedirect.com/science/article/pii/S0951832015001672)
- for updates, star the repo, [see this related repo](https://github.com/csinva/csinva.github.io), or follow [@chandan_singh96](https://twitter.com/chandan_singh96)

Feel free to cite the following but make sure to give authors of original methods / base implementations credit:
```r
@software{
    singh2020,
    title        = {imodels python package for interpretable modeling},
    publisher    = {Zenodo},
    year         = 2020,
    author       = {Chandan Singh},
    version      = {v0.2.2},
    doi          = {10.5281/zenodo.4026887},
    url          = {https://doi.org/10.5281/zenodo.4026887}
}
```