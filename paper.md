---
title: 'imodels: a python package for fitting interpretable models'
tags:
  - python
  - machine learning
  - interpretability
  - explainability
  - transparency
  - decision rules
authors:
  - name: Chandan Singh^[Equal contribution]
    orcid: 0000-0003-0318-2340
    affiliation: 1 # (Multiple affiliations must be quoted)
  - name: Keyan Nasseri^[Equal contribution]
    affiliation: 1
  - name: Yan Shuo Tan
    affiliation: 2
  - name: Tiffany Tang
    affiliation: 2
  - name: Bin Yu
    affiliation: "1, 2"
affiliations:
 - name: EECS Department, University of California, Berkeley
   index: 1
 - name: Statistics Department, University of California, Berkeley
   index: 2
date: 27 January 2021
bibliography: docs/paper/references.bib
---

# Summary

`imodels` is a Python package for concise, transparent, and accurate predictive modeling.
It provides users a simple interface for fitting and using state-of-the-art interpretable models, all compatible with scikit-learn [@pedregosa2011scikit].
These models can often replace black-box models while improving interpretability and computational efficiency, all without sacrificing predictive accuracy.
In addition, the package provides a framework for developing custom tools and rule-based models for interpretability.

# Statement of need

Recent advancements in machine learning have led to increasingly complex predictive models, often at the cost of interpretability.
There is often a need for models which are inherently interpretable [@rudin2019stop; @murdoch2019definitions], particularly in high-stakes applications such as medicine, biology, and political science.
In these cases, interpretability can ensure that models behave reasonably, identify when models will make errors, and make the models more trusted by domain experts.
Moreover, interpretable models tend to be much more computationally efficient then larger black-box models.

Despite the development of many methods for fitting interpretable models [@molnar2020interpretable], implementations for such models are often difficult to find, use, and compare to one another.
`imodels` aims to fill this gap by providing a simple unified interface and implementation for many state-of-the-art interpretable modeling techniques.

# Features

Interpretable models can take various forms.
\autoref{fig:models} shows four possible forms a model in the `imodels` package can take.
Each form constrains the final model in order to make it interpretable, but there are different methods for fitting the model which differ in their biases and computational costs.
The `imodels` package contains implementations of various such methods and also useful functions for recombining and extending them.

Rule sets consist of a set of rules which each act independently.
    There are different strategies for deriving a rule set, such as Skope-rules [@skope] or Rulefit [@friedman2008predictive].
Rule lists are composed of a set of rules which act in sequence, and include models such as Bayesian rule lists [@letham2015interpretable] or the oneR algorithm [@holte1993very].
Rule trees are similar to rule lists, but allow branching after rules. This includes models such as CART decision trees [@breiman1984classification].
Algebraic models take a final form of simple algebraic expressions, such as supersparse linear integer models [@ustun2016supersparse].

![Examples of different supported model forms. The bottom of each box shows predictions of the corresponding model as a function of $X_1$ and $X_2$.\label{fig:models}](./docs/img/model_table.png){ width=100% }

# Acknowledgements

The code here heavily derives from the wonderful work of previous projects.
In particular, we build upon the following repos and users: [sklearn-expertsys](https://github.com/tmadl/sklearn-expertsys) - by [Tamas Madl](https://github.com/tmadl) and [Benedict](https://github.com/kenben) based on original code by [Ben Letham](http://lethalletham.com/).
We also based many rule-based models on [skope-rules](https://github.com/scikit-learn-contrib/skope-rules) by the [skope-rules team](https://github.com/scikit-learn-contrib/skope-rules/blob/master/AUTHORS.rst) (including [
Nicolas Goix](https://github.com/ngoix), [Florian Gardin](https://github.com/floriangardin), [Jean-Matthieu Schertzer](https://github.com/datajms), Bibi Ndiaye, and Ronan Gautier). 
We also build upon the [rulefit](https://github.com/christophM/rulefit) repository by [Christoph Molnar](https://github.com/christophM).

# References