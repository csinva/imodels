import os
from os.path import join as oj
from random import sample
import unittest
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

        """ parameters """
        # The following parameters are recommended to change depending on the size and complexity of the data
        N = 2000  # number of rules to be used in SA_patternbased and also the output of generate_rules
        Niteration = 500  # number of iterations in each chain
        Nchain = 2  # number of chains in the simulated annealing search algorithm

        supp = 5  # 5% is a generally good number. The higher this supp, the 'larger' a pattern is
        maxlen = 3  # maxmum length of a pattern

        # \rho = alpha/(alpha+beta). Make sure \rho is close to one when choosing alpha and beta.
        alpha_1 = 500  # alpha_+
        beta_1 = 1  # beta_+
        alpha_2 = 500  # alpha_-
        beta_2 = 1  # beta_-

        """ input file """
        # notice that in the example, X is already binary coded.
        # Data has to be binary coded and the column name shd have the form: attributename_attributevalue
        filepathX = oj(test_dir, 'test_data', 'tictactoe_X.txt')  # input file X
        filepathY = oj(test_dir, 'test_data', 'tictactoe_Y.txt')  # input file Y
        df = read_csv(filepathX, header=0, sep=" ")
        Y = np.loadtxt(open(filepathY, "rb"), delimiter=" ")

        lenY = len(Y)
        train_index = sample(range(lenY), int(0.70 * lenY))
        test_index = [i for i in range(lenY) if i not in train_index]

        model = BOAClassifier(df.iloc[train_index], Y[train_index])
        model.generate_rules(supp, maxlen, N)
        model.set_parameters(alpha_1, beta_1, alpha_2, beta_2, None, None)
        rules = model.fit(Niteration, Nchain, print_message=True)

        # test
        print('printing rules...', rules)
        Yhat = model.predict(rules, df.iloc[test_index])


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
