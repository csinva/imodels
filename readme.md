<h1 align="center"> Interpretable machine-learning models (imodels) 🔍</h1>
<p align="center"> Python package for concise, transparent, and accurate predictive modeling. All sklearn-compatible and easily customizable.
</p>


<p align="center">
  <a href="https://csinva.github.io/imodels/">docs</a> •
  <a href="#imodels-overview">imodels overview</a> •
  <a href="#demo-notebooks">demo notebooks</a>
</p>
<p align="center">
  <img src="https://img.shields.io/badge/license-mit-blue.svg">
  <img src="https://img.shields.io/badge/python-3.6--3.9-blue">
  <a href="https://doi.org/10.21105/joss.03192"><img src="https://joss.theoj.org/papers/10.21105/joss.03192/status.svg"></a>
  <a href="https://github.com/csinva/imodels/actions"><img src="https://github.com/csinva/imodels/workflows/tests/badge.svg"></a>
  <!--img src="https://img.shields.io/github/checks-status/csinva/imodels/master"-->
  <img src="https://img.shields.io/pypi/v/imodels?color=orange">
  <img src="https://static.pepy.tech/personalized-badge/imodels?period=total&units=none&left_color=gray&right_color=orange&left_text=downloads&kill_cache=11">
</p>  





## imodels overview

Modern machine-learning models are increasingly complex, often making them difficult to interpret. This package provides a simple interface for fitting and using state-of-the-art interpretable models, all compatible with scikit-learn. These models can often replace black-box models (e.g. random forests) with simpler models (e.g. rule lists) while improving interpretability and computational efficiency, all without sacrificing predictive accuracy! Simply import a classifier or regressor and use the `fit` and `predict` methods, same as standard scikit-learn models.

```python
from imodels import BayesianRuleListClassifier, GreedyRuleListClassifier, SkopeRulesClassifier # see more models below
from imodels import SLIMRegressor, RuleFitRegressor

model = BayesianRuleListClassifier()  # initialize a model
model.fit(X_train, y_train)   # fit model
preds = model.predict(X_test) # discrete predictions: shape is (n_test, 1)
preds_proba = model.predict_proba(X_test) # predicted probabilities: shape is (n_test, n_classes)
print(model) # print the rule-based model

-----------------------------
# the model consists of the following 3 rules
# if X1 > 5: then 80.5% risk
# else if X2 > 5: then 40% risk
# else: 10% risk
```

### Installation
Install with `pip install imodels` (see [here](https://github.com/csinva/imodels/blob/master/docs/troubleshooting.md) for help). 

### Supported models

| Model                       | Reference                                                    | Description                                                  |
| :-------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Rulefit rule set            | [🗂️](https://csinva.io/imodels/rule_set/rule_fit.html), [🔗](https://github.com/christophM/rulefit), [📄](http://statweb.stanford.edu/~jhf/ftp/RuleFit.pdf) | Extracts rules from a decision tree then builds a sparse linear model with them |
| Skope rule set              | [🗂️](https://csinva.io/imodels/rule_set/skope_rules.html#imodels.rule_set.skope_rules.SkopeRulesClassifier), [🔗](https://github.com/scikit-learn-contrib/skope-rules) | Extracts rules from gradient-boosted trees, deduplicates them, then forms a linear combination of them based on their OOB precision |
| Boosted rule set            | [🗂️](https://csinva.io/imodels/rule_set/boosted_rules.html), [🔗](https://github.com/jaimeps/adaboost-implementation), [📄](https://www.sciencedirect.com/science/article/pii/S002200009791504X) | Uses Adaboost to sequentially learn a set of rules           |
| Slipper rule set            | [🗂️](https://csinva.io/imodels/rule_set/slipper.html), [📄](https://www.aaai.org/Papers/AAAI/1999/AAAI99-049.pdf) | Uses SLIPPER to sequentially learn a set of rules            |
| BOA rule set                | [🗂️](https://csinva.io/imodels/rule_set/boa.html#imodels.rule_set.boa.BOAClassifier), [🔗](https://github.com/wangtongada/BOA), [📄](https://www.jmlr.org/papers/volume18/16-003/16-003.pdf) | Uses a Bayesian or-of-and algorithm to find a concise rule set |
| Corels rule list            | [🗂️]([CorelsRuleListClassifier](https://csinva.io/imodels/rule_list/corels_wrapper.html#imodels.rule_list.corels_wrapper.CorelsRuleListClassifier)), [🔗](https://github.com/corels/pycorels), [📄](https://www.jmlr.org/papers/volume18/17-716/17-716.pdf) | Learns an optimal compact rule list (rather than using a greedy heuristic) |
| Bayesian rule list          | [🗂️](https://csinva.io/imodels/rule_list/bayesian_rule_list/bayesian_rule_list.html#imodels.rule_list.bayesian_rule_list.bayesian_rule_list.BayesianRuleListClassifier), [🔗](https://github.com/tmadl/sklearn-expertsys), [📄](https://arxiv.org/abs/1602.08610) | Learns a compact rule list distribution by sampling rule lists (slow) |
| Greedy rule list            | [🗂️](https://csinva.io/imodels/rule_list/greedy_rule_list.html), [🔗](https://medium.com/@penggongting/implementing-decision-tree-from-scratch-in-python-c732e7c69aea) | Uses CART to learn a list (only a single path), rather than a decision tree |
| OneR rule list              | [🗂️](https://csinva.io/imodels/rule_list/one_r.html), [📄](https://link.springer.com/article/10.1023/A:1022631118932) | Learns rule list restricted to only one feature              |
| Optimal rule tree           | [🗂️](https://csinva.io/imodels/tree/optimal_classification_tree/index.html), [🔗](https://github.com/pan5431333/pyoptree), [📄](https://link.springer.com/article/10.1007/s10994-017-5633-9) | (In progress) Learns succinct trees using global optimization rather than greedy heuristics |
| Iterative random forest     | [🗂️](https://csinva.io/imodels/tree/iterative_random_forest/iterative_random_forest.html), [🔗](https://github.com/Yu-Group/iterative-Random-Forest), [📄](https://www.pnas.org/content/115/8/1943) | (In progress) Repeatedly fit random forest, giving features with high importance a higher chance of being selected |
| Sparse integer linear model | [🗂️](https://csinva.io/imodels/algebraic/slim.html), [📄](https://link.springer.com/article/10.1007/s10994-015-5528-6) | Forces coefficients to be integers                           |
| More models                 | ⌛                                                            | (Coming soon!) Many popular rule sets including Lightweight Rule Induction, MLRules |

<p align="center">
Docs <a href="https://csinva.io/imodels/">🗂️</a>, Reference code implementation 🔗, Research paper 📄
</br>
  See also our <a href="https://csinva.io/imodels/discretization/index.html">fast and effective discretizers</a> for data preprocessing.
</br>
</p>


The final form of the above models takes one of the following forms, which aim to be simultaneously simple to understand and highly predictive:

|                           Rule set                           |                        Rule list                        |                        Rule tree                        |                       Algebraic models                       |
| :----------------------------------------------------------: | :-----------------------------------------------------: | :-----------------------------------------------------: | :----------------------------------------------------------: |
| <img src="https://csinva.io/imodels/img/rule_set.jpg" width="100%"> | <img src="https://csinva.io/imodels/img/rule_list.jpg"> | <img src="https://csinva.io/imodels/img/rule_tree.jpg"> | <img src="https://csinva.io/imodels/img/algebraic_models.jpg"> |

Different models and algorithms vary not only in their final form but also in different choices made during modeling. In particular, many models differ in the 3 steps given by the table below.

<details>
<summary>ex. RuleFit and SkopeRules</summary>
RuleFit and SkopeRules differ only in the way they prune rules: RuleFit uses a linear model whereas SkopeRules heuristically deduplicates rules sharing overlap.
</details>

<details>
<summary>ex. Bayesian rule lists and greedy rule lists</summary>
Bayesian rule lists and greedy rule lists differ in how they select rules; bayesian rule lists perform a global optimization over possible rule lists while Greedy rule lists pick splits sequentially to maximize a given criterion.
</details>

<details>
<summary>ex. FPSkope and SkopeRules</summary>
FPSkope and SkopeRules differ only in the way they generate candidate rules: FPSkope uses FPgrowth whereas SkopeRules extracts rules from decision trees.
</details>

See the docs for individual models for futher descriptions.

|                  Rule candidate generation                   |                       Rule selection                       |                Rule pruning / combination                 |
| :----------------------------------------------------------: | :--------------------------------------------------------: | :-------------------------------------------------------: |
| <img src="https://csinva.io/imodels/img/rule_candidates.jpg" width="100%"> | <img src="https://csinva.io/imodels/img/rule_overfit.jpg"> | <img src="https://csinva.io/imodels/img/rule_pruned.jpg"> |

The code here contains many useful and customizable functions for rule-based learning in the [util folder](https://csinva.io/imodels/util/index.html). This includes functions / classes for rule deduplication, rule screening, and converting between trees, rulesets, and neural networks.

## Demo notebooks

Demos are contained in the [notebooks](notebooks) folder.

<details>
<summary><a href="notebooks/imodels_demo.ipynb">imodels demo</a></summary>
Shows how to fit, predict, and visualize with different interpretable models
</details>

<details>
<summary><a href="https://colab.research.google.com/drive/1WfqvSjegygT7p0gyqiWpRpiwz2ePtiao#scrollTo=bLnLknIuoWtQ">imodels colab demo</a> <a href="https://colab.research.google.com/drive/1WfqvSjegygT7p0gyqiWpRpiwz2ePtiao#scrollTo=bLnLknIuoWtQ"> <img src="https://colab.research.google.com/assets/colab-badge.svg"></a></summary>
Shows how to fit, predict, and visualize with different interpretable models
</details>

<details>
<summary><a href="https://github.com/csinva/iai-clinical-decision-rule/blob/master/notebooks/05_fit_interpretable_models.ipynb">clinical decision rule notebook</a></summary>
Shows an example of using <code>imodels</code> for deriving a clinical decision rule
</details>

<details>
<summary>posthoc analysis</summary>
We also include some demos of posthoc analysis, which occurs after fitting models:
<a href="notebooks/posthoc_analysis.ipynb">posthoc.ipynb</a> shows different simple analyses to interpret a trained model and 
<a href="notebooks/uncertainty_analysis.ipynb">uncertainty.ipynb</a> contains basic code to get uncertainty estimates for a model
</details>



## Support for different tasks

Different models support different machine-learning tasks. Current support for different models is given below (each of these models can be imported directly from imodels (e.g. `from imodels import RuleFitClassifier`):

| Model                       | Binary classification | Regression |
| :-------------------------- | :-------------------: | :--------: |
| Rulefit rule set            |           [RuleFitClassifier](https://csinva.io/imodels/rule_set/rule_fit.html#imodels.rule_set.rule_fit.RuleFitClassifier)           |     [RuleFitRegressor](https://csinva.io/imodels/rule_set/rule_fit.html#imodels.rule_set.rule_fit.RuleFitRegressor)     |
| Skope rule set              |           [SkopeRulesClassifier](https://csinva.io/imodels/rule_set/slipper.html#imodels.rule_set.slipper.SlipperClassifier)           |            |
| Boosted rule set            |           [BoostedRulesClassifier](https://csinva.io/imodels/rule_set/boosted_rules.html#imodels.rule_set.boosted_rules.BoostedRulesClassifier)           |            |
| SLIPPER rule set            |           [SlipperClassifier](https://csinva.io/imodels/rule_set/slipper.html#imodels.rule_set.slipper.SlipperClassifier)           |            |
| BOA rule set                |           [BOAClassifier](https://csinva.io/imodels/rule_set/boa.html#imodels.rule_set.boa.BOAClassifier)           |            |
| Corels rule list            |           [CorelsRuleListClassifier](https://csinva.io/imodels/rule_list/corels_wrapper.html#imodels.rule_list.corels_wrapper.CorelsRuleListClassifier)           |            |
| Bayesian rule list          |           [BayesianRuleListClassifier](https://csinva.io/imodels/rule_list/bayesian_rule_list/bayesian_rule_list.html#imodels.rule_list.bayesian_rule_list.bayesian_rule_list.BayesianRuleListClassifier)           |            |
| Greedy rule list            |           [GreedyRuleListClassifier](https://csinva.io/imodels/rule_list/greedy_rule_list.html#imodels.rule_list.greedy_rule_list.GreedyRuleListClassifier)           |            |
| OneR rule list              |           [OneRClassifier](https://csinva.io/imodels/rule_list/one_r.html#imodels.rule_list.one_r.OneRClassifier)           |            |
| Optimal rule tree           |                       |            |
| Iterative random forest     |                       |            |
| Sparse integer linear model |           [SLIMClassifier](https://csinva.io/imodels/algebraic/slim.html#imodels.algebraic.slim.SLIMClassifier)           |     [SLIMRegressor](https://csinva.io/imodels/algebraic/slim.html#imodels.algebraic.slim.SLIMRegressor)     |

## References
<details>
<summary>Readings</summary>
<ul>
  <li>Interpretable ML good quick overview: murdoch et al. 2019, <a href="https://arxiv.org/pdf/1901.04592.pdf">pdf</a></li>
	<li>Interpretable ML book: molnar 2019, <a href="https://christophm.github.io/interpretable-ml-book/">pdf</a></li>
	<li>Case for interpretable models rather than post-hoc explanation: rudin 2019, <a href="https://arxiv.org/pdf/1811.10154.pdf">pdf</a></li>
	<li>Review on evaluating interpretability: doshi-velez & kim 2017, <a href="https://arxiv.org/pdf/1702.08608.pdf">pdf</a></li>	
</ul>
</details>

<details>
<summary>Reference implementations (also linked above)</summary>
The code here heavily derives from the wonderful work of previous projects. We seek to to extract out, unify, and maintain key parts of these projects.
<ul>
  <li><a href="https://github.com/corels/pycorels">pycorels</a> - by <a href="https://github.com/fingoldin">@fingoldin</a> and the <a href="https://github.com/corels/corels">original CORELS team</a>
  <li><a href="https://github.com/tmadl/sklearn-expertsys">sklearn-expertsys</a> - by <a href="https://github.com/tmadl">@tmadl</a> and <a href="https://github.com/kenben">@kenben</a> based on original code by <a href="http://lethalletham.com/">Ben Letham</a></li>
  <li><a href="https://github.com/christophM/rulefit">rulefit</a> - by <a href="https://github.com/christophM">@christophM</a></li>
  <li><a href="https://github.com/scikit-learn-contrib/skope-rules">skope-rules</a> - by the <a href="https://github.com/scikit-learn-contrib/skope-rules/blob/master/AUTHORS.rst">skope-rules team</a> (including <a href="https://github.com/ngoix">@ngoix</a>, <a href="https://github.com/floriangardin">@floriangardin</a>, <a href="https://github.com/datajms">@datajms</a>, <a href="">Bibi Ndiaye</a>, <a href="">Ronan Gautier</a>)</li>
  <li><a href="https://github.com/wangtongada/BOA">boa</a> - by <a href="https://github.com/wangtongada">@wangtongada</a></li>	
</ul>
</details>

<details>
<summary>Related packages</summary>
<ul>
	<li><a href="https://github.com/trevorstephens/gplearn/tree/ad57cb18caafdb02cca861aea712f1bf3ed5016e">gplearn</a>: symbolic regression/classification</li>
  <li><a href="https://github.com/dswah/pyGAM">pygam</a>: generative additive models</li>
  <li><a href="https://github.com/interpretml/interpret">interpretml</a>: boosting-based gam</li>
</ul>
</details>

<details>
<summary>Updates</summary>
<ul>
  <li>For updates, star the repo, <a href="https://github.com/csinva/csinva.github.io">see this related repo</a>, or follow <a href="https://twitter.com/csinva_">@csinva_</a></li>
  <li>Please make sure to give authors of original methods / base implementations appropriate credit!</li>
  <li>Contributing: pull requests <a href="https://github.com/csinva/imodels/blob/master/docs/contributing.md">very welcome</a>!</li>
</ul>
</details>


If it's useful for you, please star/cite the package, and make sure to give authors of original methods / base implementations credit:

```r
@software{
    imodels2021,
    title        = {{imodels: a python package for fitting interpretable models}},
    journal      = {Journal of Open Source Software}
    publisher    = {The Open Journal},
    year         = {2021},
    author       = {Singh, Chandan and Nasseri, Keyan and Tan, Yan Shuo and Tang, Tiffany and Yu, Bin},
    volume       = {6},
    number       = {61},
    pages        = {3192},
    doi          = {10.21105/joss.03192},
    url          = {https://doi.org/10.21105/joss.03192},
}

```
