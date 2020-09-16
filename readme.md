<h1 align="center"> Interpretable machine learning models (imodels) 🔍</h1>
<p align="center"> Straightforward implementations of interpretable ML models + demos of how to use various interpretability techniques. Code is optimized for readability. [Pull requests welcome](contributing.md)!
</p>

<p align="center">
  <a href="https://csinva.github.io/imodels/docs/">Docs</a> •
  <a href="#implementations-of-interpretable-models"> Implementations of imodels </a> •
  <a href="#demo-notebooks">Demo notebooks</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg">
  <a href="https://github.com/csinva/imodels/actions"><img src="https://github.com/csinva/imodels/workflows/tests/badge.svg"></a>
</p>  


## Implementations of interpretable models
Scikit-learn style wrappers/implementations of different interpretable models. The interpretable models within the [imodels](imodels) folder can be easily installed and used:

`pip install git+https://github.com/csinva/imodels` (see [here](csinva.io/docs/troubleshooting.md) for more help)

```python
from imodels import RuleListClassifier, GreedyRuleListClassifier, SkopeRulesClassifier, IRFClassifier
from imodels import SLIMRegressor, RuleFitRegressor

model = RuleListClassifier()  # initialize Bayesian Rule List
model.fit(X_train, y_train)   # fit model
preds = model.predict(X_test) # discrete predictions: shape is (n_test, 1)
preds_proba = model.predict_proba(X_test) # predicted probabilities: shape is (n_test, n_classes)
```

- [bayesian rule list](https://arxiv.org/abs/1602.08610) (based on [this implementation](https://github.com/tmadl/sklearn-expertsys)) - learn a compact rule list
- [rulefit](http://statweb.stanford.edu/~jhf/ftp/RuleFit.pdf) (based on [this implementation](https://github.com/christophM/rulefit)) - find rules from a decision tree and build a linear model with them
- [skope-rules](https://github.com/scikit-learn-contrib/skope-rules) (based on [this implementation](https://github.com/scikit-learn-contrib/skope-rules)) - extracts rules from base estimators (e.g. decision trees) then tries to deduplicate them
- [sparse integer linear model](https://link.springer.com/article/10.1007/s10994-015-5528-6) (simple implementation with cvxpy)
- greedy rule list (based on [this implementation](https://medium.com/@penggongting/implementing-decision-tree-from-scratch-in-python-c732e7c69aea)) - uses CART to learn a list (only a single path), rather than a decision tree
- (in progress) [optimal classification tree](https://link.springer.com/article/10.1007/s10994-017-5633-9) (based on [this implementation](https://github.com/pan5431333/pyoptree)) - learns succinct trees using global optimization rather than greedy heuristics
- [iterative random forest](https://www.pnas.org/content/115/8/1943) (based on [this implementation](https://github.com/Yu-Group/iterative-Random-Forest))
-  see readmes in individual folders within [imodels](imodels) for details


## Demo notebooks
The demos are contained in 3 main [notebooks](notebooks), following this cheat-sheet:![cheat_sheet](docs/cheat_sheet.png)

1. [model_based.ipynb](notebooks/1_model_based.ipynb) - how to use different interpretable models and examples with the **imodels** package
    - see an example of using this package for deriving a clinical decision rule in [this nb](https://github.com/csinva/iai-clinical-decision-rule/blob/master/notebooks/04_fit_interpretable_models.ipynb)
2. [posthoc.ipynb](notebooks/2_posthoc.ipynb) - different simple analyses to interpret a trained model
3. [uncertainty.ipynb](notebooks/3_uncertainty.ipynb) - code to get uncertainty estimates for a model



## References / further reading

- Stop Explaining Black Box Machine Learning Models for High Stakes Decisions and Use Interpretable Models Instead (rudin 2019, [pdf](https://arxiv.org/pdf/1811.10154.pdf)) - expounds why one should use interpretable models
- Interpretable machine learning: definitions, methods, and applications (murdoch et al. 2019, [pdf](https://arxiv.org/pdf/1901.04592.pdf)) - good quick review on interpretable ML 
- Interpretable Machine Learning: A Guide for Making Black Box Models Explainable (molnar 2019, [pdf](https://christophm.github.io/interpretable-ml-book/)) - book on interpretable ML
- Review on evaluating interpretability (doshi-velez & kim 2017, [pdf](https://arxiv.org/pdf/1702.08608.pdf))
- For updates, star the repo, [see this related repo](https://github.com/csinva/csinva.github.io), or follow [@chandan_singh96](https://twitter.com/chandan_singh96)


Feel free to cite the following but definitely make sure to give authors of original methods / base implementations credit:
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