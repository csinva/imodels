Highly interpretable, sklearn-compatible classifier based on decision rules
===============

This is a python3 version adapted from [this repo](https://github.com/tmadl/sklearn-expertsys). This is a scikit-learn compatible wrapper for the Bayesian Rule List classifier developed by [Letham et al., 2015](http://projecteuclid.org/euclid.aoas/1446488742) (see [Letham's original code](http://lethalletham.com/)), extended by a minimum description length-based discretizer ([Fayyad & Irani, 1993](http://sci2s.ugr.es/keel/pdf/algorithm/congreso/fayyad1993.pdf)) for continuous data, and by an approach to subsample large datasets for better performance.

It produces rule lists, which makes trained classifiers **easily interpretable to human experts**, and is competitive with state of the art classifiers such as random forests or SVMs.

For example, an easily understood Rule List model of the well-known Titanic dataset:

```
IF male AND adult THEN survival probability: 21% (19% - 23%)
ELSE IF 3rd class THEN survival probability: 44% (38% - 51%)
ELSE IF 1st class THEN survival probability: 96% (92% - 99%)
ELSE survival probability: 88% (82% - 94%)
``` 

Letham et al.'s approach only works on discrete data. However, this approach can still be used on continuous data after discretization. The RuleListClassifier class also includes a discretizer that can deal with continuous data (using [Fayyad & Irani's](http://sci2s.ugr.es/keel/pdf/algorithm/congreso/fayyad1993.pdf) minimum description length principle criterion, based on an implementation by [navicto](https://github.com/navicto/Discretization-MDLPC)).

The inference procedure is slow on large datasets. If you have more than a few thousand data points, and only numeric data, try the included `BigDataRuleListClassifier(training_subset=0.1)`, which first determines a small subset of the training data that is most critical in defining a decision boundary (the data points that are hardest to classify) and learns a rule list only on this subset (you can specify which estimator to use for judging which subset is hardest to classify by passing any sklearn-compatible estimator in the `subset_estimator` parameter - see `examples/diabetes_bigdata_demo.py`). 

Usage
===============

The project requires [pyFIM](http://www.borgelt.net/pyfim.html), [scikit-learn](http://scikit-learn.org/stable/install.html), and [pandas](http://pandas.pydata.org/) to run.

- easiest to just install .so file form pyFIM

The included `RuleListClassifier` works as a scikit-learn estimator, with a `model.fit(X,y)` method which takes training data `X` (numpy array or pandas DataFrame; continuous, categorical or mixed data) and labels `y`. 

The learned rules of a trained model can be displayed simply by casting the object as a string, e.g. `print model`, or by using the `model.tostring(decimals=1)` method and optionally specifying the rounding precision.

Numerical data in `X` is automatically discretized. To prevent discretization (e.g. to protect columns containing categorical data represented as integers), pass the list of protected column names in the `fit` method, e.g. `model.fit(X,y,undiscretized_features=['CAT_COLUMN_NAME'])` (entries in undiscretized columns will be converted to strings and used as categorical values - see `examples/hepatitis_mixeddata_demo.py`). 

Usage example:

```python

import sys
sys.path.append('discretization')
from RuleListClassifier import *
from sklearn.datasets.mldata import fetch_mldata
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


feature_labels = ["#Pregnant","Glucose concentration test","Blood pressure(mmHg)","Triceps skin fold thickness(mm)",
                  "2-Hour serum insulin (mu U/ml)","Body mass index","Diabetes pedigree function","Age (years)"]


np.random.seed(13)
data = fetch_openml("diabetes") # get dataset
y = (data.target == 'tested_positive').astype(np.int) # labels 0-1

Xtrain, Xtest, ytrain, ytest = train_test_split(data.data, y) # split

# train classifier (allow more iterations for better accuracy; use BigDataRuleListClassifier for large datasets)
print('training...')
model = RuleListClassifier(max_iter=10000, class1label="diabetes", verbose=False) # max_iter = 100000
model.fit(Xtrain, ytrain, feature_labels=feature_labels)

print("RuleListClassifier Accuracy:", model.score(Xtest, ytest), "Learned interpretable model:\n", model)
print("RandomForestClassifier Accuracy:", RandomForestClassifier(n_estimators=10).fit(Xtrain, ytrain).score(Xtest, ytest))

"""
Output:

RuleListClassifier Accuracy: 0.7291666666666666 Learned interpretable model:
 Trained RuleListClassifier for detecting diabetes
==================================================
IF Glucose concentration test : 157.5_to_inf THEN probability of diabetes: 81.9% (73.0%-89.4%)
ELSE IF Body mass index : -inf_to_27.45 THEN probability of diabetes: 8.3% (4.4%-13.3%)
ELSE IF Glucose concentration test : 123.5_to_157.5 THEN probability of diabetes: 53.5% (44.9%-62.1%)
ELSE IF Age (years) : -inf_to_28.5 THEN probability of diabetes: 11.6% (6.5%-17.8%)
ELSE probability of diabetes: 36.9% (28.2%-46.1%)
=================================================

RandomForestClassifier Accuracy: 0.7135416666666666
"""
```