# Generalized Mean Decrease in Impurity (GMDI)

Generalized Mean Decrease in Impurity (GMDI) is a novel feature importance framework, which generalizes the popular mean decrease in impurity (MDI) importance score for random forests. At its core, GMDI expands upon a recently discovered connection between linear regression and decision trees. In doing so, GMDI enables practitioners to (1) tailor the feature importance computation to the data/problem structure and (2) incorporate additional features or knowledge to mitigate known biases of decision trees. In both real data case studies and extensive real-data-inspired simulations, GMDI outperforms commonly used feature importance measures (e.g., MDI, permutation-based scores, and TreeSHAP) by substantional margins. 

For further details, we refer to [Agarwal et al. (2023)]().

**Regression Example Usage:**

```python
from imodels.importance import RandomForestPlusRegressor

rf_plus_model = RandomForestPlusRegressor()
rf_plus_model.fit(X, y)
gmdi_scores = rf_plus_model.get_gmdi_scores(X, y)
```

**Classification Example Usage:**

```python
from imodels.importance import RandomForestPlusClassifier

rf_plus_model = RandomForestPlusClassifier()
rf_plus_model.fit(X, y)
gmdi_scores = rf_plus_model.get_gmdi_scores(X, y)
```


## Demo notebooks

<details>
<summary><a href="notebooks/gmdi_demo.ipynb">GMDI demo</a></summary>
<ul>
<li>Shows how to compute GMDI importance scores for different tasks (regression and classification) and configurations (with flexible GLMs, scoring metrics, and custom transformations).</li>
<li>Provides starter code on how to choose the GLM and scoring metric within GMDI via a stability metric and/or combine these fits in an ensemble</li>
</ul>
</details>


## Overview of GMDI

*Input:* a collection of fitted trees (e.g., a random forest), data `X` and `y`

For each fitted tree in the forest:

1. Transform `X` using the tree stump features from the fitted tree, and append any additional engineered features (e.g., the raw `X` features) to this transformed dataset.

2. Fit a prediction model on this augmented transformed dataset to predict `y`. Here, we recommend fitting a generalized linear model (GLM) to leverage computational speed-ups.

3. Use this fitted prediction model to make partial model predictions for each feature k. That is, for each feature k, get the model's predictions if the contribution of all other features (except feature k) were zeroed out. Put differently, the kth partial model predictions are the predictions we get when using only the engineered features that are related to feature k.

4. For each feature k, evaluate the similarity between the observed `y` and the kth partial model predictions using any user-defined similarity metric (i.e., a larger value should indicate greater feature importance).

This gives the GMDI scores for a single tree. To get the GMDI scores for the forest, these scores are averaged across all trees in the forest. 

<p align="center">
	<img src="" width="80%">
</p>  
<p align="center">	
	<i>Overview of GMDI.</i>
</p>

## Practical Considerations

We show in [Agarwal et al. (2023)]() that this framework is indeed a proper generalization of the popular MDI feature importance score. However, as a result of the increased flexibility provided by GMDI, there are several choices that must be made by the analyst to run GMDI in practice. In particular,

1. In Step 1: What feature engineering/transformations to include?
	- We recommend including the raw feature (i.e., `X`) in this transformed dataset, which is done by default (via `RandomForestPlus*(include_raw=True)`). To include additional transformations, create custom `BlockTransformerBase` object(s) and use the `add_transformers` argument in `RandomForestPlus*()`.

2. In Step 2: Which GLM?
	- We recommend using `RidgeRegressorPPM()` for regression tasks and `LogisticClassifierPPM()` for classification tasks and thus set these to be the defaults. To use a custom prediction model, use the `prediction_model` argument in `RandomForestPlus*()`.

3. In Step 3: Which sample splitting strategy (if any) to use when making the partial model predictions?
	- We recommend using a leave-one-out (`"loo"`) sample splitting strategy as it overcomes the known correlation and entropy biases suffered by MDI. Out-of-bag (`"oob"`) can also be used to overcome these biases but tends to be more unstable than leave-one-out. MDI uses an in-bag sample splitting scheme and is not recommended. The sample splitting strategy is set to `"loo"` by default but can be changed via the `sample_split` argument in `RandomForestPlus*()`.

4. In Step 4: Which similarity metric to use?
	- We recommend using r-squared for regression tasks and log-loss for classification tasks, which are the defaults. To use a custom metric, use the `scoring_fns` argument in the `get_gmdi_scores()` method.

These recommendations are based on extensive simulations across a wide variety of data-generating processes, data sets, noise levels, and misspecifications. 

Nevertheless, different choices may be better for different problems. For examples on how to implement some of these custom options, see the [GMDI Demo](notebooks/gmdi_demo.ipynb) notebook. This demo also includes examples on how to aggregate feature importances from multiple GMDI configurations in an ensemble as well as how to choose the "best" GLM and metric in a data-driven manner based upon a stability score.


