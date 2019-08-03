# Interpretability demos + implementations

Demos of how to use various interpretability techniques (with accompanying slides [here](https://docs.google.com/presentation/d/1RIdbV279r20marRrN0b1bu2z9STkrivsMDa_Dauk8kE/present) or pdf [here](slides.pdf)) and code for implementations of interpretable machine learning models.

### Demo notebooks
The demos are contained in 3 main [notebooks](notebooks), summarized in [cheat_sheet.pdf](cheat_sheet.pdf)

1. [model_based.ipynb](notebooks/1_model_based.ipynb) - how to use different interpretable models
2. [posthoc.ipynb](notebooks/2_posthoc.ipynb) - different simple analyses to interpret a trained model
3. [uncertainty.ipynb](notebooks/3_uncertainty.ipynb) - code to get uncertainty estimates for a model

### Code implementations
Provides scikit-learn style wrappers/implementations of different interpretable models (see readmes in individual folders within [imodels](imodels) for details)

- [bayesian rule lists](https://arxiv.org/abs/1602.08610)
- [optimal classification tree](https://link.springer.com/article/10.1007/s10994-017-5633-9)
- sparse integer linear models (simple, unstable implementation)

The interpretable models within the [imodels](imodels) folder can be easily installed and used.

`pip install git+https://github.com/csinva/interpretability-implementations-demos`

```
from imodels.bayesian_rule_lists.RuleListClassifier import RuleListClassifier
model = RuleListClassifier()
model.fit(X_train, y_train)
model.score(X_test, y_test)
preds = model.predict(X_test)
```

### References / further reading

- [book on interpretable machine learning](https://christophm.github.io/interpretable-ml-book/)
- [high-level review on interpretable machine learning](https://arxiv.org/abs/1901.04592)
- [review on black-blox explanation methods](https://hal.inria.fr/hal-02131174v2/document)
- [review on variable importance](https://www.sciencedirect.com/science/article/pii/S0951832015001672)
- for updates, star the repo, [see this related repo](https://github.com/csinva/csinva.github.io), or follow [@chandan_singh96](https://twitter.com/chandan_singh96)