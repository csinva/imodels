import os
import unittest
from os.path import join as oj
from random import sample

test_dir = os.path.dirname(os.path.abspath(__file__))
import numpy as np
from pandas.io.parsers import read_csv
import random
from imodels import BOAClassifier


class TestBoaClassifier(unittest.TestCase):
    def setup(self):
        np.random.seed(13)
        random.seed(13)

    def test_boa(self):
        '''Test classifiers are properly sklearn-compatible
        '''

        df = read_csv(oj(test_dir, 'test_data', 'tictactoe_X.txt'), header=0, sep=" ")
        Y = np.loadtxt(open(oj(test_dir, 'test_data', 'tictactoe_Y.txt'), "rb"), delimiter=" ")

        lenY = len(Y)
        train_index = sample(range(lenY), int(0.70 * lenY))
        test_index = [i for i in range(lenY) if i not in train_index]

        model = BOAClassifier(n_rules=1000,
                              supp=5,
                              maxlen=3,
                              num_iterations=300,
                              num_chains=2,
                              alpha_pos=500, beta_pos=1,
                              alpha_neg=500, beta_neg=1,
                              alpha_l=None, beta_l=None)
        # model.generate_rules()
        # model.set_parameters(alpha_1, beta_pos, alpha_neg, beta_neg, None, None)
        model.fit(df.iloc[train_index], Y[train_index])

        # test
        print('printing rules...', model)
        Yhat = model.predict(df.iloc[test_index])

        def getConfusion(Yhat, Y):
            if len(Yhat) != len(Y):
                raise NameError('Yhat has different length')
            TP = np.dot(np.array(Y), np.array(Yhat))
            FP = np.sum(Yhat) - TP
            TN = len(Y) - np.sum(Y) - FP
            FN = len(Yhat) - np.sum(Yhat) - TN
            return TP, FP, TN, FN

        TP, FP, TN, FN = getConfusion(Yhat, Y[test_index])
        tpr = float(TP) / (TP + FN)
        fpr = float(FP) / (FP + TN)
        print(
            f'TP = {TP}, FP = {FP}, TN = {TN}, FN = {FN}'
            '\naccuracy = {float(TP + TN) / (TP + TN + FP + FN)}, tpr = {tpr}, fpr = {fpr}'
        )
        assert tpr > 0.8

'''
df = read_csv(oj(test_dir, 'test_data', 'tictactoe_X.txt'), header=0, sep=" ")
Y = np.loadtxt(open(oj(test_dir, 'test_data', 'tictactoe_Y.txt'), "rb"), delimiter=" ")

lenY = len(Y)
train_index = sample(range(lenY), int(0.70 * lenY))
test_index = [i for i in range(lenY) if i not in train_index]

print(df.head())

model = BOAClassifier(df.iloc[train_index], Y[train_index],
                      n_rules=2000,
                      supp=5,
                      maxlen=3,
                      num_iterations=500,
                      num_chains=2,
                      alpha_pos=500, beta_pos=1,
                      alpha_neg=500, beta_neg=1,
                      alpha_l=None, beta_l=None)
# model.generate_rules()
# model.set_parameters(alpha_1, beta_pos, alpha_neg, beta_neg, None, None)
model.fit()

# test
print('printing rules...', model)
Yhat = model.predict(df.iloc[test_index])

def getConfusion(Yhat, Y):
    if len(Yhat) != len(Y):
        raise NameError('Yhat has different length')
    TP = np.dot(np.array(Y), np.array(Yhat))
    FP = np.sum(Yhat) - TP
    TN = len(Y) - np.sum(Y) - FP
    FN = len(Yhat) - np.sum(Yhat) - TN
    return TP, FP, TN, FN

TP, FP, TN, FN = getConfusion(Yhat, Y[test_index])
tpr = float(TP) / (TP + FN)
fpr = float(FP) / (FP + TN)
print(
    f'TP = {TP}, FP = {FP}, TN = {TN}, FN = {FN}'
    '\naccuracy = {float(TP + TN) / (TP + TN + FP + FN)}, tpr = {tpr}, fpr = {fpr}'
)
assert tpr > 0.8
'''