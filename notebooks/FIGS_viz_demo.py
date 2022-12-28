# %% [markdown]
# # Setup

# %%
# %load_ext autoreload
# %autoreload 2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree, DecisionTreeClassifier
from sklearn import metrics

# TODo remove when package is updated
import sys,os
sys.path.append(os.path.expanduser('~/imodels'))

# installable with: `pip install imodels`
import imodels
from imodels import FIGSClassifier
import demo_helper
np.random.seed(13)

# %% [markdown]
# Let's start by loading some data in...  
# Note, we need to still load the reg dataset first to get the same splits as in `imodels_demo.ipynb` due to the call to random

# %%
# ames housing dataset: https://www.openml.org/search?type=data&status=active&id=43926
X_train_reg, X_test_reg, y_train_reg, y_test_reg, feat_names_reg = demo_helper.get_ames_data()

# diabetes dataset: https://www.openml.org/search?type=data&sort=runs&id=37&status=active
X_train, X_test, y_train, y_test, feat_names = demo_helper.get_diabetes_data()
    # feat_names meanings:
    # ["#Pregnant", "Glucose concentration test", "Blood pressure(mmHg)",
    # "Triceps skin fold thickness(mm)",
    # "2-Hour serum insulin (mu U/ml)", "Body mass index", "Diabetes pedigree function", "Age (years)"]

# load some data
# print('Regression data training', X_train_reg.shape, 'Classification data training', X_train.shape)

# %% [markdown]
# ***
# # FIGS

# %%
model_figs = FIGSClassifier(max_rules=7)

# %%
# specify a decision tree with a maximum depth
model_figs.fit(X_train, y_train, feature_names=feat_names);

# %%
print(model_figs)

# %%
print(model_figs.print_tree(X_train, y_train))

# %%
model_figs.plot(fig_size=7)

# %% [markdown]
# ***
# # `dtreeviz` Integration
# One tree at a time only, showing tree 0 here

# %%
from dtreeviz import trees
from dtreeviz.models.sklearn_decision_trees import ShadowSKDTree
from imodels.tree.viz_utils import extract_sklearn_tree_from_figs

dt = extract_sklearn_tree_from_figs(model_figs, tree_num=0, n_classes=2)
shadow_dtree = ShadowSKDTree(dt, X_train, y_train, feat_names, 'y', [0, 1])

# %%
trees.dtreeviz(shadow_dtree)

# %%
x_example = X_train[13]

# %%
list(zip(feat_names,x_example))

# %%
print(trees.explain_prediction_path(shadow_dtree, x=x_example, explanation_type='plain_english'))

# %%
trees.dtreeviz(shadow_dtree, X=x_example)

# %%
trees.dtreeviz(shadow_dtree, show_node_labels=True, fancy=False)

# %%
trees.describe_node_sample(shadow_dtree, node_id=8)

# %%
trees.ctreeviz_leaf_samples(shadow_dtree)

# %% [markdown]
# ***
# # `SKompiler` Integration
# One tree at a time only, showing tree 0 here

# %%
from skompiler import skompile
from imodels.tree.viz_utils import extract_sklearn_tree_from_figs

dt = extract_sklearn_tree_from_figs(model_figs, tree_num=0, n_classes=2)
expr = skompile(dt.predict_proba, feat_names)

# %%
print(expr.to('sqlalchemy/sqlite', component=1, assign_to='tree_0'))

# %%
print(expr.to('python/code'))
