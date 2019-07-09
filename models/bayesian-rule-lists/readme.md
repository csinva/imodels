Highly interpretable, sklearn-compatible classifier based on decision rules
===============

Very heavily based on this repo: https://github.com/tmadl/sklearn-expertsys

This is a scikit-learn compatible wrapper for the Bayesian Rule List classifier developed by [Letham et al., 2015](http://projecteuclid.org/euclid.aoas/1446488742) (see [Letham's original code](http://lethalletham.com/)), extended by a minimum description length-based discretizer ([Fayyad & Irani, 1993](http://sci2s.ugr.es/keel/pdf/algorithm/congreso/fayyad1993.pdf)) for continuous data, and by an approach to subsample large datasets for better performance.

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
from RuleListClassifier import *
from sklearn.datasets.mldata import fetch_mldata
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier

feature_labels = ["#Pregnant","Glucose concentration test","Blood pressure(mmHg)","Triceps skin fold thickness(mm)","2-Hour serum insulin (mu U/ml)","Body mass index","Diabetes pedigree function","Age (years)"]
    
data = fetch_mldata("diabetes") # get dataset
y = (data.target+1)/2 # target labels (0 or 1)
Xtrain, Xtest, ytrain, ytest = train_test_split(data.data, y) # split

# train classifier (allow more iterations for better accuracy; use BigDataRuleListClassifier for large datasets)
model = RuleListClassifier(max_iter=10000, class1label="diabetes", verbose=False)
model.fit(Xtrain, ytrain, feature_labels=feature_labels)

print "RuleListClassifier Accuracy:", model.score(Xtest, ytest), "Learned interpretable model:\n", model
print "RandomForestClassifier Accuracy:", RandomForestClassifier().fit(Xtrain, ytrain).score(Xtest, ytest)
"""
**Output:**
RuleListClassifier Accuracy: 0.776041666667 Learned interpretable model:
Trained RuleListClassifier for detecting diabetes
==================================================
IF Glucose concentration test : 157.5_to_inf THEN probability of diabetes: 81.1% (72.5%-72.5%)
ELSE IF Body mass index : -inf_to_26.3499995 THEN probability of diabetes: 5.2% (1.9%-1.9%)
ELSE IF Glucose concentration test : -inf_to_103.5 THEN probability of diabetes: 14.4% (8.8%-8.8%)
ELSE IF Age (years) : 27.5_to_inf THEN probability of diabetes: 59.6% (51.8%-51.8%)
ELSE IF Glucose concentration test : 103.5_to_127.5 THEN probability of diabetes: 15.9% (8.0%-8.0%)
ELSE probability of diabetes: 44.7% (29.5%-29.5%)
=================================================

RandomForestClassifier Accuracy: 0.729166666667
"""
```