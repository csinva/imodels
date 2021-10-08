import numpy as np
import pandas as pd

from autogluon.tabular import TabularDataset, TabularPredictor
from autogluon.core.space import Categorical, Int, Real
from autogluon.tabular.configs.hyperparameter_configs import get_hyperparameter_config
from autogluon.tabular.models import RuleFitModel, GreedyTreeModel, CorelsRuleListModel, \
  BayesianRuleSetModel, GlobalSparseTreeModel, BoostedRulesModel


# train data
train_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv')
subsample_size = 100  # subsample subset of data for faster demo, try setting this to much larger values
train_data = train_data.sample(n=subsample_size, random_state=0)
# train_data.head()

# test data
test_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv')
label = 'class'
y_test = test_data[label]  # values to predict
test_data_nolab = test_data.drop(columns=[label])  # delete label column to prove we're not cheating
# test_data_nolab.head()
# print("Summary of class variable: \n", train_data[label].describe())

predictor = TabularPredictor(label=label).fit(train_data, presets='interpretable', time_limit=8, verbosity=2)
print(predictor.leaderboard(test_data))