# # Setup

# +
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
from imodels import FIGSClassifier
import demo_helper
np.random.seed(13)
# -

# Let's start by loading some data in...  
# Note, we need to still load the reg dataset first to get the same splits as in `imodels_demo.ipynb` due to the call to random

# +
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
# -

# ***
# # FIGS

model_figs = FIGSClassifier(max_rules=7, max_trees=3)

model_figs.fit(X_train, y_train, feature_names=feat_names);

print(model_figs)

print(model_figs.print_tree(X_train, y_train))

model_figs.plot(fig_size=7)

# ## Gini Importance

dfp_importance = pd.DataFrame({'feat_names': feat_names})
dfp_importance['feature'] = dfp_importance.index
dfp_importance_gini = pd.DataFrame({'importance_gini': model_figs.feature_importances_})
dfp_importance_gini['feature'] = dfp_importance_gini.index
dfp_importance_gini['importance_gini_pct'] = dfp_importance_gini['importance_gini'].rank(pct=True)
dfp_importance = pd.merge(dfp_importance, dfp_importance_gini, on='feature', how='left')
dfp_importance = dfp_importance.sort_values(by=['importance_gini', 'feature'], ascending=[False, True]).reset_index(drop=True)
display(dfp_importance)

# ***
# # `dtreeviz` Integration
# One tree at a time only, showing tree 0 here

# +
import dtreeviz
from imodels.tree.viz_utils import extract_sklearn_tree_from_figs

dt = extract_sklearn_tree_from_figs(model_figs, tree_num=0, n_classes=2)
viz_model = dtreeviz.model(dt, X_train=X_train, y_train=y_train, feature_names=feat_names, target_name='y', class_names=[0, 1])
# -

color_params = {'classes': dtreeviz.colors.mpl_colors, 'hist_bar': 'C0', 'legend_edge': None}
for _ in ['axis_label', 'title', 'legend_title', 'text', 'arrow', 'node_label', 'tick_label', 'leaf_label', 'wedge', 'text_wedge']:
    color_params[_] = 'black'
dtv_params_gen = {'colors': color_params, 'fontname': 'Arial', 'figsize': (4, 3)}
dtv_params = {'leaftype': 'barh',
              'label_fontsize': 10,
              'colors': dtv_params_gen['colors'],
              'fontname': dtv_params_gen['fontname']
             }

viz_model.view(**dtv_params)

x_example = X_train[13]
display(pd.DataFrame([{col: value for col,value in zip(feat_names, x_example)}]))

print(viz_model.explain_prediction_path(x=x_example))

viz_model.view(**dtv_params, x=x_example)

viz_model.view(**dtv_params, show_node_labels=True, fancy=False)

viz_model.ctree_leaf_distributions(**dtv_params_gen)

viz_model.leaf_purity(display_type='plot', **dtv_params_gen)

# ***
# # `SKompiler` Integration
# One tree at a time only, showing tree 0 here

# +
from skompiler import skompile
from imodels.tree.viz_utils import extract_sklearn_tree_from_figs

dt = extract_sklearn_tree_from_figs(model_figs, tree_num=0, n_classes=2)
expr = skompile(dt.predict_proba, feat_names)

# +
# Currently broken, see https://github.com/konstantint/SKompiler/issues/16
# print(expr.to('sqlalchemy/sqlite', component=1, assign_to='tree_0'))
# -

print(expr.to('python/code'))
