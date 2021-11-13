# Experiments

This directory contains files pertaining to experimental rule-based / interpretable models.

Follow these steps to benchmark a new model:

1. Write the sklearn-compliant model and put it in the `models` folder
2. Download the appropriate data (see the data folder / notebook)
3. Select which datasets you want by modifying `config.datasets`
4. Select which models you want by editing a list similar to `config.supercart.models`
5. run `00_comparisons.py` then `01_aggregate_comparisons.py` (which just combines pkls into a `combined.pkl` file across datasets)
6. look at the results in notebooks
