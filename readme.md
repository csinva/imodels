# Interpretability demos + implementations

Demos of how to use various interpretability techniques (accompanying slides [here](https://docs.google.com/presentation/d/1RIdbV279r20marRrN0b1bu2z9STkrivsMDa_Dauk8kE/present)) and code for implementations of interpretable machine learning models.

### Implementations of interpretable models
Provides scikit-learn style wrappers/implementations of different interpretable models (see readmes in individual folders within [imodels](imodels) for details)

- [bayesian rule list](https://arxiv.org/abs/1602.08610) (based on [this implementation](https://github.com/tmadl/sklearn-expertsys))
- [rulefit](http://statweb.stanford.edu/~jhf/ftp/RuleFit.pdf) (based on [this implementation](https://github.com/christophM/rulefit))
- [sparse integer linear model](https://link.springer.com/article/10.1007/s10994-015-5528-6) (simple implementation based on cvxpy)
- [optimal classification tree](https://link.springer.com/article/10.1007/s10994-017-5633-9) (based on [this implementation](https://github.com/pan5431333/pyoptree))
- greedy rule list (based on [this impelementation](https://medium.com/@penggongting/implementing-decision-tree-from-scratch-in-python-c732e7c69aea)) - grows a CART tree that only goes down one path, resulting in a list rather than a tree

The interpretable models within the [imodels](imodels) folder can be easily installed and used.

`pip install git+https://github.com/csinva/interpretability-implementations-demos`

```python
from imodels import RuleListClassifier, RuleFit
model = RuleListClassifier() # Bayesian Rule List
model.fit(X_train, y_train)
model.score(X_test, y_test)
preds = model.predict(X_test)
```

### Demo notebooks
The demos are contained in 3 main [notebooks](notebooks), following this cheat-sheet:![cheat_sheet](cheat_sheet.png)

1. [model_based.ipynb](notebooks/1_model_based.ipynb) - how to use different interpretable models
2. [posthoc.ipynb](notebooks/2_posthoc.ipynb) - different simple analyses to interpret a trained model
3. [uncertainty.ipynb](notebooks/3_uncertainty.ipynb) - code to get uncertainty estimates for a model



### References / further reading

- [high-level review on interpretable machine learning](https://arxiv.org/abs/1901.04592)
- [book on interpretable machine learning](https://christophm.github.io/interpretable-ml-book/)
- [review on black-blox explanation methods](https://hal.inria.fr/hal-02131174v2/document)
- [review on variable importance](https://www.sciencedirect.com/science/article/pii/S0951832015001672)
- for updates, star the repo, [see this related repo](https://github.com/csinva/csinva.github.io), or follow [@chandan_singh96](https://twitter.com/chandan_singh96)
