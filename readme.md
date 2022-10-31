<p align="center">
	<img align="center" width=60% src="https://csinva.io/imodels/img/imodels_logo.svg?sanitize=True&kill_cache=1"> </img>	 <br/>
	Python package for concise, transparent, and accurate predictive modeling. <br/>
	All sklearn-compatible and easy to use. <br/>
	<i> For interpretability in NLP, check out our new package: <a href="https://github.com/csinva/imodelsX">imodelsX</a> </i>
</p>
<p align="center">
  <a href="https://csinva.github.io/imodels/">📚 docs</a> •
  <a href="#demo-notebooks">📖 demo notebooks</a>
</p>
<p align="center">
  <img src="https://img.shields.io/badge/license-mit-blue.svg">
  <img src="https://img.shields.io/badge/python-3.7--3.10-blue">
  <a href="https://doi.org/10.21105/joss.03192"><img src="https://joss.theoj.org/papers/10.21105/joss.03192/status.svg"></a>
  <a href="https://github.com/csinva/imodels/actions"><img src="https://github.com/csinva/imodels/workflows/tests/badge.svg"></a>
  <!--img src="https://img.shields.io/github/checks-status/csinva/imodels/master"-->
  <img src="https://img.shields.io/pypi/v/imodels?color=orange">
  <img src="https://static.pepy.tech/personalized-badge/imodels?period=total&units=none&left_color=gray&right_color=orange&left_text=downloads&kill_cache=12">
</p>  





<img align="center" width=100% src="https://csinva.io/imodels/img/anim.gif"> </img>

Modern machine-learning models are increasingly complex, often making them difficult to interpret. This package provides a simple interface for fitting and using state-of-the-art interpretable models, all compatible with scikit-learn. These models can often replace black-box models (e.g. random forests) with simpler models (e.g. rule lists) while improving interpretability and computational efficiency, all without sacrificing predictive accuracy! Simply import a classifier or regressor and use the `fit` and `predict` methods, same as standard scikit-learn models.

```python
from sklearn.model_selection import train_test_split
from imodels import get_clean_dataset, HSTreeClassifierCV # import any model here

# prepare data (a sample clinical dataset)
X, y, feature_names = get_clean_dataset('csi_pecarn_pred')
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42)

# fit the model
model = HSTreeRegressorCV(max_leaf_nodes=4)  # initialize a tree model and specify only 4 leaf nodes
model.fit(X_train, y_train, feature_names=feature_names)   # fit model
preds = model.predict(X_test) # discrete predictions: shape is (n_test, 1)
preds_proba = model.predict_proba(X_test) # predicted probabilities: shape is (n_test, n_classes)
print(model) # print the model
```

```
------------------------------
Decision Tree with Hierarchical Shrinkage
Prediction is made by looking at the value in the appropriate leaf of the tree
------------------------------
|--- FocalNeuroFindings2 <= 0.50
|   |--- HighriskDiving <= 0.50
|   |   |--- Torticollis2 <= 0.50
|   |   |   |--- value: [0.10]
|   |   |--- Torticollis2 >  0.50
|   |   |   |--- value: [0.30]
|   |--- HighriskDiving >  0.50
|   |   |--- value: [0.68]
|--- FocalNeuroFindings2 >  0.50
|   |--- value: [0.42]
```

### Installation

Install with `pip install imodels` (see [here](https://github.com/csinva/imodels/blob/master/docs/troubleshooting.md) for help).

### Supported models

| Model                       | Reference                                                    | Description                                                  |
| :-------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Rulefit rule set            | [🗂️](https://csinva.io/imodels/rule_set/rule_fit.html), [🔗](https://github.com/christophM/rulefit), [📄](http://statweb.stanford.edu/~jhf/ftp/RuleFit.pdf) | Fits a sparse linear model on rules extracted from decision trees |
| Skope rule set              | [🗂️](https://csinva.io/imodels/rule_set/skope_rules.html#imodels.rule_set.skope_rules.SkopeRulesClassifier), [🔗](https://github.com/scikit-learn-contrib/skope-rules) | Extracts rules from gradient-boosted trees, deduplicates them,<br/>then linearly combines them based on their OOB precision |
| Boosted rule set            | [🗂️](https://csinva.io/imodels/rule_set/boosted_rules.html), [🔗](https://github.com/jaimeps/adaboost-implementation), [📄](https://www.sciencedirect.com/science/article/pii/S002200009791504X) | Sequentially fits a set of rules with Adaboost           |
| Slipper rule set            | [🗂️](https://csinva.io/imodels/rule_set/slipper.html), ㅤㅤ[📄](https://www.aaai.org/Papers/AAAI/1999/AAAI99-049.pdf) | Sequentially learns a set of rules with SLIPPER            |
| Bayesian rule set           | [🗂️](https://csinva.io/imodels/rule_set/brs.html#imodels.rule_set.brs.BayesianRuleSetClassifier), [🔗](https://github.com/wangtongada/BOA), [📄](https://www.jmlr.org/papers/volume18/16-003/16-003.pdf) | Finds concise rule set with Bayesian sampling (slow)  |
| Optimal rule list           | [🗂️](https://csinva.io/imodels/rule_list/corels_wrapper.html#imodels.rule_list.corels_wrapper.OptimalRuleListClassifier), [🔗](https://github.com/corels/pycorels), [📄](https://www.jmlr.org/papers/volume18/17-716/17-716.pdf) | Fits rule list using global optimization for sparsity (CORELS) |
| Bayesian rule list          | [🗂️](https://csinva.io/imodels/rule_list/bayesian_rule_list/bayesian_rule_list.html#imodels.rule_list.bayesian_rule_list.bayesian_rule_list.BayesianRuleListClassifier), [🔗](https://github.com/tmadl/sklearn-expertsys), [📄](https://arxiv.org/abs/1602.08610) | Fits compact rule list distribution with Bayesian sampling (slow) |
| Greedy rule list            | [🗂️](https://csinva.io/imodels/rule_list/greedy_rule_list.html), [🔗](https://medium.com/@penggongting/implementing-decision-tree-from-scratch-in-python-c732e7c69aea) | Uses CART to fit a list (only a single path), rather than a tree |
| OneR rule list              | [🗂️](https://csinva.io/imodels/rule_list/one_r.html), ㅤㅤ[📄](https://link.springer.com/article/10.1023/A:1022631118932) | Fits rule list restricted to only one feature              |
| Optimal rule tree           | [🗂️](https://csinva.io/imodels/tree/gosdt/pygosdt.html#imodels.tree.gosdt.pygosdt.OptimalTreeClassifier), [🔗](https://github.com/Jimmy-Lin/GeneralizedOptimalSparseDecisionTrees), [📄](https://arxiv.org/abs/2006.08690) | Fits succinct tree using global optimization for sparsity (GOSDT) |
| Greedy rule tree            | [🗂️](https://csinva.io/imodels/tree/cart_wrapper.html), [🔗](https://scikit-learn.org/stable/modules/tree.html), [📄](https://www.taylorfrancis.com/books/mono/10.1201/9781315139470/classification-regression-trees-leo-breiman-jerome-friedman-richard-olshen-charles-stone) | Greedily fits tree using CART                              |
| C4.5 rule tree        | [🗂️](https://csinva.io/imodels/tree/c45_tree/c45_tree.html#imodels.tree.c45_tree.c45_tree.C45TreeClassifier), [🔗](https://github.com/RaczeQ/scikit-learn-C4.5-tree-classifier), [📄](https://link.springer.com/article/10.1007/BF00993309) | Greedily fits tree using C4.5                           |
| TAO rule tree        | [🗂️](https://csinva.io/imodels/tree/tao.html), ㅤㅤ[📄](https://proceedings.neurips.cc/paper/2018/hash/185c29dc24325934ee377cfda20e414c-Abstract.html) | Fits tree using alternating optimization                    |
| Iterative random<br/>forest | [🗂️](https://csinva.io/imodels/tree/iterative_random_forest/iterative_random_forest.html), [🔗](https://github.com/Yu-Group/iterative-Random-Forest), [📄](https://www.pnas.org/content/115/8/1943) | Repeatedly fit random forest, giving features with<br/>high importance a higher chance of being selected |
| Sparse integer<br/>linear model | [🗂️](https://csinva.io/imodels/algebraic/slim.html), ㅤㅤ[📄](https://link.springer.com/article/10.1007/s10994-015-5528-6) | Sparse linear model with integer coefficients                           |
| <b>Greedy tree sums</b> | [🗂️](https://csinva.io/imodels/tree/figs.html#imodels.tree.figs), ㅤㅤ[📄](https://arxiv.org/abs/2201.11931) | Sum of small trees with very few total rules (FIGS)                          |
| <b>Hierarchical<br/> shrinkage wrapper</b> | [🗂️](https://csinva.io/imodels/tree/hierarchical_shrinkage.html), ㅤㅤ[📄](https://arxiv.org/abs/2202.00858) | Improve a decision tree, random forest, or<br/>gradient-boosting ensemble with ultra-fast, post-hoc regularization |
| Distillation<br/>wrapper | [🗂️](https://csinva.io/imodels/util/distillation.html)  | Train a black-box model,<br/>then distill it into an interpretable model |
| More models                 | ⌛                                                            | (Coming soon!) Lightweight Rule Induction, MLRules, ... |

<p align="center">
Docs <a href="https://csinva.io/imodels/">🗂️</a>, Reference code implementation 🔗, Research paper 📄
</br>
</p>

## Demo notebooks

Demos are contained in the [notebooks](notebooks) folder.

<details>
<summary><a href="notebooks/imodels_demo.ipynb">Quickstart demo</a></summary>
Shows how to fit, predict, and visualize with different interpretable models
</details>

<details>
<summary><a href="https://auto.gluon.ai/dev/tutorials/tabular_prediction/tabular-interpretability.html">Autogluon demo</a></summary>
Fit/select an interpretable model automatically using Autogluon AutoML
</details>

<details>
<summary><a href="https://colab.research.google.com/drive/1WfqvSjegygT7p0gyqiWpRpiwz2ePtiao#scrollTo=bLnLknIuoWtQ">Quickstart colab demo</a> <a href="https://colab.research.google.com/drive/1WfqvSjegygT7p0gyqiWpRpiwz2ePtiao#scrollTo=bLnLknIuoWtQ"> <img src="https://colab.research.google.com/assets/colab-badge.svg"></a></summary>
Shows how to fit, predict, and visualize with different interpretable models
</details>

<details>
<summary><a href="https://github.com/csinva/iai-clinical-decision-rule/blob/master/notebooks/05_fit_interpretable_models.ipynb">Clinical decision rule notebook</a></summary>
Shows an example of using <code>imodels</code> for deriving a clinical decision rule
</details>

<details>
<summary>Posthoc analysis</summary>
We also include some demos of posthoc analysis, which occurs after fitting models:
<a href="notebooks/posthoc_analysis.ipynb">posthoc.ipynb</a> shows different simple analyses to interpret a trained model and 
<a href="notebooks/uncertainty_analysis.ipynb">uncertainty.ipynb</a> contains basic code to get uncertainty estimates for a model
</details>

## What's the difference between the models?

The final form of the above models takes one of the following forms, which aim to be simultaneously simple to understand and highly predictive:

|                           Rule set                           |                        Rule list                        |                        Rule tree                        |                       Algebraic models                       |
| :----------------------------------------------------------: | :-----------------------------------------------------: | :-----------------------------------------------------: | :----------------------------------------------------------: |
| <img src="https://csinva.io/imodels/img/rule_set.jpg" width="100%"> | <img src="https://csinva.io/imodels/img/rule_list.jpg"> | <img src="https://csinva.io/imodels/img/rule_tree.jpg"> | <img src="https://csinva.io/imodels/img/algebraic_models.jpg"> |

Different models and algorithms vary not only in their final form but also in different choices made during modeling, such as how they generate, select, and postprocess rules:

|                  Rule candidate generation                   |                       Rule selection                       |                Rule postprocessing|
| :----------------------------------------------------------: | :--------------------------------------------------------: | :-------------------------------------------------------: |
| <img src="https://csinva.io/imodels/img/rule_candidates.jpg"> | <img src="https://csinva.io/imodels/img/rule_overfit.jpg"> | <img src="https://csinva.io/imodels/img/rule_pruned.jpg"> |

<details>
<summary>Ex. RuleFit vs. SkopeRules</summary>
RuleFit and SkopeRules differ only in the way they prune rules: RuleFit uses a linear model whereas SkopeRules heuristically deduplicates rules sharing overlap.
</details>

<details>
<summary>Ex. Bayesian rule lists vs. greedy rule lists</summary>
Bayesian rule lists and greedy rule lists differ in how they select rules; bayesian rule lists perform a global optimization over possible rule lists while Greedy rule lists pick splits sequentially to maximize a given criterion.
</details>

<details>
<summary>Ex. FPSkope vs. SkopeRules</summary>
FPSkope and SkopeRules differ only in the way they generate candidate rules: FPSkope uses FPgrowth whereas SkopeRules extracts rules from decision trees.
</details>

## Support for different tasks

Different models support different machine-learning tasks. Current support for different models is given below (each of these models can be imported directly from imodels (e.g. `from imodels import RuleFitClassifier`):

| Model                       |                    Binary classification                     |                          Regression                          | Notes |
| :-------------------------- | :----------------------------------------------------------: | :----------------------------------------------------------: | --------------------------- |
| Rulefit rule set            | [RuleFitClassifier](https://csinva.io/imodels/rule_set/rule_fit.html#imodels.rule_set.rule_fit.RuleFitClassifier) | [RuleFitRegressor](https://csinva.io/imodels/rule_set/rule_fit.html#imodels.rule_set.rule_fit.RuleFitRegressor) |  |
| Skope rule set              | [SkopeRulesClassifier](https://csinva.io/imodels/rule_set/slipper.html#imodels.rule_set.slipper.SlipperClassifier) |                                                              |  |
| Boosted rule set            | [BoostedRulesClassifier](https://csinva.io/imodels/rule_set/boosted_rules.html#imodels.rule_set.boosted_rules.BoostedRulesClassifier) |                                                              |  |
| SLIPPER rule set            | [SlipperClassifier](https://csinva.io/imodels/rule_set/slipper.html#imodels.rule_set.slipper.SlipperClassifier) |                                                              |  |
| Bayesian rule set           | [BayesianRuleSetClassifier](https://csinva.io/imodels/rule_set/brs.html#imodels.rule_set.brs.BayesianRuleSetClassifier) |                                                              | Fails for large problems |
| Optimal rule list (CORELS)  | [OptimalRuleListClassifier](https://csinva.io/imodels/rule_list/corels_wrapper.html#imodels.rule_list.corels_wrapper.OptimalRuleListClassifier) |                                                              | Requires [corels](https://pypi.org/project/corels/), fails for large problems |
| Bayesian rule list          | [BayesianRuleListClassifier](https://csinva.io/imodels/rule_list/bayesian_rule_list/bayesian_rule_list.html#imodels.rule_list.bayesian_rule_list.bayesian_rule_list.BayesianRuleListClassifier) |                                                              |  |
| Greedy rule list            | [GreedyRuleListClassifier](https://csinva.io/imodels/rule_list/greedy_rule_list.html#imodels.rule_list.greedy_rule_list.GreedyRuleListClassifier) |                                                              |  |
| OneR rule list              | [OneRClassifier](https://csinva.io/imodels/rule_list/one_r.html#imodels.rule_list.one_r.OneRClassifier) |                                                              |  |
| Optimal rule tree (GOSDT)   | [OptimalTreeClassifier](https://csinva.io/imodels/tree/gosdt/pygosdt.html#imodels.tree.gosdt.pygosdt.OptimalTreeClassifier) |                                                              | Requires [gosdt](https://pypi.org/project/gosdt/), fails for large problems |
| Greedy rule tree (CART)     | [GreedyTreeClassifier](https://csinva.io/imodels/tree/cart_wrapper.html#imodels.tree.cart_wrapper.GreedyTreeClassifier) |      [GreedyTreeRegressor](https://csinva.io/imodels/tree/cart_wrapper.html#imodels.tree.cart_wrapper.GreedyTreeRegressor)                                                        |  |
| C4.5 rule tree              | [C45TreeClassifier](https://csinva.io/imodels/tree/c45_tree/c45_tree.html#imodels.tree.c45_tree.c45_tree.C45TreeClassifier) |           |  |
| TAO rule tree              | [TaoTreeClassifier](https://csinva.io/imodels/tree/tao.html#imodels.tree.tao.TaoTreeClassifier) |   [TaoTreeRegressor](https://csinva.io/imodels/tree/tao.html#imodels.tree.tao.TaoTreeRegressor)        |  |
| Iterative random forest     | [IRFClassifier](https://csinva.io/imodels/tree/iterative_random_forest/iterative_random_forest.html#imodels.tree.iterative_random_forest.iterative_random_forest.IRFClassifier)                                                             |                                                              | Requires [irf](https://pypi.org/project/irf/) |
| Sparse integer linear model | [SLIMClassifier](https://csinva.io/imodels/algebraic/slim.html#imodels.algebraic.slim.SLIMClassifier) | [SLIMRegressor](https://csinva.io/imodels/algebraic/slim.html#imodels.algebraic.slim.SLIMRegressor) | Requires extra dependencies for speed |
| Greedy tree sums (FIGS) | [FIGSClassifier](https://csinva.io/imodels/tree/figs.html#imodels.tree.figs.FIGSClassifier) | [FIGSRegressor](https://csinva.io/imodels/tree/figs.html#imodels.tree.figs.FIGSRegressor) |                                                              |
| Hierarchical shrinkage | [HSTreeClassifierCV](https://csinva.io/imodels/tree/hierarchical_shrinkage.html#imodels.tree.hierarchical_shrinkage.HSTreeClassifierCV) | [HSTreeRegressorCV](https://csinva.io/imodels/tree/hierarchical_shrinkage.html#imodels.tree.hierarchical_shrinkage.HSTreeRegressorCV) | Wraps any sklearn tree-based model |
| Distillation |  | [DistilledRegressor](https://csinva.io/imodels/docs/util/distillation.html#imodels.util.distillation.DistilledRegressor) | Wraps any sklearn-compatible models |

### Extras

<details>
<summary><a href="https://csinva.io/imodels/util/data_util.html#imodels.util.data_util.get_clean_dataset">Data-wrangling functions</a> for working with popular tabular datasets (e.g. compas).</summary>
These functions, in conjunction with <a href="https://github.com/csinva/imodels-data">imodels-data</a> and <a href="https://github.com/Yu-Group/imodels-experiments">imodels-experiments</a>, make it simple to download data and run experiments on new models.
</details>

<details>
<summary><a href="https://csinva.io/imodels/util/explain_errors.html">Explain classification errors</a> with a simple posthoc function.</summary>
Fit an interpretable model to explain a previous model's errors (ex. in <a href="https://github.com/csinva/imodels/blob/master/notebooks/error_detection_demo.ipynb">this notebook📓</a>).
</details>

<details>
<summary><a href="https://csinva.io/imodels/discretization/index.html">Fast and effective discretizers</a> for data preprocessing.</summary>
<table>
<thead>
<tr>
<th>Discretizer</th>
<th>Reference</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td>MDLP</td>
<td><a href="https://csinva.io/imodels/discretization/mdlp.html#imodels.discretization.mdlp.MDLPDiscretizer">🗂️</a>, <a href="https://github.com/navicto/Discretization-MDLPC">🔗</a>, <a href="https://trs.jpl.nasa.gov/handle/2014/35171">📄</a></td>
<td>Discretize using entropy minimization heuristic</td>
</tr>
<tr>
<td>Simple</td>
<td><a href="https://csinva.io/imodels/discretization/simple.html#imodels.discretization.simple.SimpleDiscretizer">🗂️</a>, <a href="https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.KBinsDiscretizer.html">🔗</a></td>
<td>Simple KBins discretization</td>
</tr>
<tr>
<td>Random Forest</td>
<td><a href="https://csinva.io/imodels/discretization/discretizer.html#imodels.discretization.discretizer.RFDiscretizer">🗂️</a></td>
<td>Discretize into bins based on random forest split popularity</td>
</tr>
</tbody>
</table>
</details>

<details>
<summary><a href="https://csinva.io/imodels/util/index.html">Rule-based utils</a> for customizing models</summary>
The code here contains many useful and customizable functions for rule-based learning in the <a href="https://csinva.io/imodels/util/index.html">util folder</a>. This includes functions / classes for rule deduplication, rule screening, and converting between trees, rulesets, and neural networks.
</details>

## Our favorite models

After developing and playing with `imodels`, we developed a few new models to overcome limitations of existing interpretable models.

### FIGS: Fast interpretable greedy-tree sums

[📄 Paper](https://arxiv.org/abs/2201.11931), [🔗 Post](https://csinva.io/imodels/figs.html), [📌 Citation](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=fast+interpretable+greedy-tree+sums&oq=fast#d=gs_cit&u=%2Fscholar%3Fq%3Dinfo%3ADnPVL74Rop0J%3Ascholar.google.com%2F%26output%3Dcite%26scirp%3D0%26hl%3Den)

Fast Interpretable Greedy-Tree Sums (FIGS) is an algorithm for fitting concise rule-based models. Specifically, FIGS generalizes CART to simultaneously grow a flexible number of trees in a summation. The total number of splits across all the trees can be restricted by a pre-specified threshold, keeping the model interpretable. Experiments across a wide array of real-world datasets show that FIGS achieves state-of-the-art prediction performance when restricted to just a few splits (e.g. less than 20).

<p align="center">
	<img src="https://demos.csinva.io/figs/diabetes_figs.svg?sanitize=True" width="50%">
</p>  
<p align="center">	
	<i><b>Example FIGS model.</b> FIGS learns a sum of trees with a flexible number of trees; to make its prediction, it sums the result from each tree.</i>
</p>

### Hierarchical shrinkage: post-hoc regularization for tree-based methods

[📄 Paper](https://arxiv.org/abs/2202.00858) (ICML 2022), [🔗 Post](https://csinva.io/imodels/shrinkage.html), [📌 Citation](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=hierarchical+shrinkage+singh&btnG=&oq=hierar#d=gs_cit&u=%2Fscholar%3Fq%3Dinfo%3Azc6gtLx-aL4J%3Ascholar.google.com%2F%26output%3Dcite%26scirp%3D0%26hl%3Den)

Hierarchical shrinkage is an extremely fast post-hoc regularization method which works on any decision tree (or tree-based ensemble, such as Random Forest). It does not modify the tree structure, and instead regularizes the tree by shrinking the prediction over each node towards the sample means of its ancestors (using a single regularization parameter). Experiments over a wide variety of datasets show that hierarchical shrinkage substantially increases the predictive performance of individual decision trees and decision-tree ensembles.

<p align="center">
	<img src="https://demos.csinva.io/shrinkage/shrinkage_intro.svg?sanitize=True" width="75%">
</p>  
<p align="center">	
	<i><b>HS Example.</b> HS applies post-hoc regularization to any decision tree by shrinking each node towards its parent.</i>
</p>

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
  <li><a href="https://github.com/MilesCranmer/PySR">pysr</a>: fast symbolic regression</li>
  <li><a href="https://github.com/dswah/pyGAM">pygam</a>: generative additive models</li>
  <li><a href="https://github.com/interpretml/interpret">interpretml</a>: boosting-based gam</li>
  <li><a href="https://github.com/h2oai/h2o-3">h20 ai</a>: gams + glms (and more)</li>
  <li><a href="https://github.com/guillermo-navas-palencia/optbinning">optbinning</a>: data discretization / scoring models</li>	
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
title        = {imodels: a python package for fitting interpretable models},
journal      = {Journal of Open Source Software},
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
