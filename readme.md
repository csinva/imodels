# Interpretability implementations + demos

Code for implementations of interpretable machine learning models and demos of how to use various interpretability techniques (with accompanying slides [here](https://docs.google.com/presentation/d/1RIdbV279r20marRrN0b1bu2z9STkrivsMDa_Dauk8kE/present)).


### Code implementations
Provides scikit-learn style wrappers/implementations of different interpretable models (see readmes in individual folders within [models](models) for details)

- [bayesian rule lists](https://arxiv.org/abs/1602.08610)
- [optimal classification tree](https://link.springer.com/article/10.1007/s10994-017-5633-9)
- sparse integer linear models (simple, unstable implementation)

### Demo notebooks
The demos are contained in 3 main [notebooks](notebooks), summarized in [cheat_sheet.pdf](cheat_sheet.pdf)

1. [model_based.ipynb](notebooks/model_based.ipynb) - how to use different interpretable models
2. [posthoc.ipynb](notebooks/posthoc.ipynb) - different simple analyses to interpret a trained model
3. [uncertainty.ipynb](notebooks/uncertainty.ipynb) - code to get uncertainty estimates for a model

### Installation / quickstart
To install, `pip install git+https://github.com/csinva/interpretability-implementations-demos`

```

```


### References / further reading

- [book on interpretable machine learning](https://christophm.github.io/interpretable-ml-book/)
- [high-level review on interpretable machine learning](https://arxiv.org/abs/1901.04592)
- [review on black-blox explanation methods](https://hal.inria.fr/hal-02131174v2/document)
- [review on variable importance](https://www.sciencedirect.com/science/article/pii/S0951832015001672)
