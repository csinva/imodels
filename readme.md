<h1 align="center"> Interpretable machine learning models (imodels) 🔍</h1>
<p align="center"> Python package for concise, transparent, and accurate predictive modeling. All sklearn-compatible and easily understandable. Pull requests <a href="https://github.com/csinva/imodels/blob/master/docs/contributing.md">very welcome</a>!
</p>


<p align="center">
  <a href="https://csinva.github.io/imodels/">Docs</a> •
  <a href="#popular-interpretable-models"> Popular imodels </a> •
  <a href="#custom-interpretable-models"> Custom imodels </a> •
  <a href="#demo-notebooks">Demo notebooks</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg">
  <a href="https://github.com/csinva/imodels/actions"><img src="https://github.com/csinva/imodels/workflows/tests/badge.svg"></a>
</p>  


## Popular interpretable models

Implementations of different interpretable models, all compatible with scikit-learn. The interpretable models can be easily used and installed:

```python
from imodels import RuleListClassifier, GreedyRuleListClassifier, SkopeRulesClassifier, IRFClassifier
from imodels import SLIMRegressor, RuleFitRegressor

model = RuleListClassifier()  # initialize a model
model.fit(X_train, y_train)   # fit model
preds = model.predict(X_test) # discrete predictions: shape is (n_test, 1)
preds_proba = model.predict_proba(X_test) # predicted probabilities: shape is (n_test, n_classes)
```

Install with `pip install git+https://github.com/csinva/imodels` (see [here](https://github.com/csinva/imodels/blob/master/docs/troubleshooting.md) for help). Contains the following models:

| Model                       | Reference                                                    | Description                                                  |
| :--------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Bayesian rule list          | [🗂️](https://csinva.io/imodels/bayesian_rule_list/RuleListClassifier.html), [🔗](https://github.com/tmadl/sklearn-expertsys), [📄](https://arxiv.org/abs/1602.08610) | Learns a compact rule list by sampling rule lists (rather than using a greedy heuristic) |
| Rulefit                     | [🗂️](https://csinva.io/imodels/rule_fit.html), [🔗](https://github.com/christophM/rulefit), [📄](http://statweb.stanford.edu/~jhf/ftp/RuleFit.pdf) | Extracts rules from a decision tree then builds a sparse linear model with them |
| Skope rules                 | [🗂️](https://csinva.io/imodels/skope_rules.html), [🔗](https://github.com/scikit-learn-contrib/skope-rules) | Extracts rules from gradient-boosted trees, deduplicates them, then forms a linear combination of them based on their OOB precision |
| Sparse integer linear model | [🗂️](https://csinva.io/imodels/slim.html), [📄](https://link.springer.com/article/10.1007/s10994-015-5528-6) | Forces coefficients to be integers                           |
| Greedy rule list            | [🗂️](https://csinva.io/imodels/greedy_rule_list.html), [🔗](https://medium.com/@penggongting/implementing-decision-tree-from-scratch-in-python-c732e7c69aea) | Uses CART to learn a list (only a single path), rather than a decision tree |
| Iterative random forest     | [🗂️](https://csinva.io/imodels/iterative_random_forest/iterative_random_forest.html), [🔗](https://github.com/Yu-Group/iterative-Random-Forest), [📄](https://www.pnas.org/content/115/8/1943) | (In progress) Repeatedly fit random forest, giving features with high importance a higher chance of being selected. |
| Optimal classification tree | [🗂️](https://csinva.io/imodels/optimal_classification_tree/index.html), [🔗](https://github.com/pan5431333/pyoptree), [📄](https://link.springer.com/article/10.1007/s10994-017-5633-9) | (In progress) Learns succinct trees using global optimization rather than greedy heuristics |
| Rule sets                   |                                                              | (Coming soon) Many popular rule sets including SLIPPER, Lightweight Rule Induction, MLRules |

<p align="center">
Reference includes docs <a href="https://csinva.io/imodels/">🗂️</a>, reference code implementation 🔗, and research paper 📄
</p>

## Custom interpretable models

The code here contains many useful and readable functions for a variety of rule-based models, contained in the [util folder](https://csinva.io/imodels/util/index.html). This includes functions and simple classes for rule deduplication, rule screening, converting between trees, rulesets, and pytorch neural nets. The final derived rules easily allows for extending any of the following general classes of models:

|           Rule set            |        Rule list        |  (Decision) Rule tree   |        Algebraic models        |
| :---------------------------: | :---------------------: | :---------------------: | :----------------------------: |
| <img src="https://csinva.io/imodels/rule_set.jpg" width="100%"> | <img src="https://csinva.io/imodels/rule_list.jpg"> | <img src="https://csinva.io/imodels/rule_tree.jpg"> | <img src="https://csinva.io/imodels/algebraic_models.jpg"> |

## Demo notebooks
Demos are contained in the [notebooks](notebooks) folder.

- [model_based.ipynb](notebooks/1_model_based.ipynb), demos the imodels package. It shows how to fit, predict, and visualize with different interpretable models
- [this notebook](https://github.com/csinva/iai-clinical-decision-rule/blob/master/notebooks/04_fit_interpretable_models.ipynb) shows an example of using `imodels` for deriving a clinical decision rule
- After fitting models, we can also do posthoc analysis, following the cheat-sheet below 
  - [posthoc.ipynb](notebooks/2_posthoc.ipynb) - shows different simple analyses to interpret a trained model
  - [uncertainty.ipynb](notebooks/3_uncertainty.ipynb) - basic code to get uncertainty estimates for a model
<img src="https://csinva.github.io/imodels/cheat_sheet.svg?sanitize=True">    

## References
- Readings
    - Interpretable machine learning: definitions, methods, and applications (murdoch et al. 2019, [pdf](https://arxiv.org/pdf/1901.04592.pdf)) - good quick review on interpretable ML
    - Interpretable Machine Learning: A Guide for Making Black Box Models Explainable (molnar 2019, [pdf](https://christophm.github.io/interpretable-ml-book/)) - book on interpretable ML
    - Stop Explaining Black Box Machine Learning Models for High Stakes Decisions and Use Interpretable Models Instead (rudin 2019, [pdf](https://arxiv.org/pdf/1811.10154.pdf)) - good explanation of why one should use interpretable models 
    - Review on evaluating interpretability (doshi-velez & kim 2017, [pdf](https://arxiv.org/pdf/1702.08608.pdf))
- Reference implementations (also linked above): the code here heavily derives from (and in some case is just a wrapper for) the wonderful work of previous projects. We seek to to extract out, combine, and maintain select relevant parts of these projects.
    - [sklearn-expertsys](https://github.com/tmadl/sklearn-expertsys) - by [@tmadl](https://github.com/tmadl) and [@kenben](https://github.com/kenben) based on original code by [Ben Letham](http://lethalletham.com/)
    - [rulefit](https://github.com/christophM/rulefit) - by [@christophM](https://github.com/christophM)
    - [skope-rules](https://github.com/scikit-learn-contrib/skope-rules) - by the [skope-rules team](https://github.com/scikit-learn-contrib/skope-rules/blob/master/AUTHORS.rst) (including [@ngoix](https://github.com/ngoix), [@floriangardin](https://github.com/floriangardin), [@datajms](https://github.com/datajms), [Bibi Ndiaye](), [Ronan Gautier]())
- Related packages
    - [gplearn](https://github.com/trevorstephens/gplearn/tree/ad57cb18caafdb02cca861aea712f1bf3ed5016e) for symbolic regression/classification
    - [pygam](https://github.com/dswah/pyGAM) for generative additive models

For updates, star the repo, [see this related repo](https://github.com/csinva/csinva.github.io), or follow [@chandan_singh96](https://twitter.com/chandan_singh96). Please make sure to give authors of original methods / base implementations appropriate credit!

