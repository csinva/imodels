'''Original implementation at https://github.com/wangtongada/BOA
'''

import itertools
import operator
import time
from bisect import bisect_left
from collections import defaultdict
from itertools import combinations
from random import sample

import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import fpgrowth
from numpy.random import random
from scipy.sparse import csc_matrix
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier

from imodels.rule_set.rule_set import RuleSet


class BOAClassifier(RuleSet, BaseEstimator, ClassifierMixin):
    '''Bayesian or-of-and algorithm.
    Generates patterns that satisfy the minimum support and maximum length and then select the Nrules rules that have the highest entropy.
    In function SA_patternbased, each local maximum is stored in maps and the best BOA is returned.
    Remember here the BOA contains only the index of selected rules from Nrules self.rules
    '''

    def __init__(self, binary_data, Y):
        self.df = binary_data
        self.Y = Y
        self.attributeLevelNum = defaultdict(int)
        self.attributeNames = []
        for i, name in enumerate(binary_data.columns):
            attribute = name.split('_')[0]
            self.attributeLevelNum[attribute] += 1
            self.attributeNames.append(attribute)
        self.attributeNames = list(set(self.attributeNames))

    def getPatternSpace(self):
        print('Computing sizes for pattern space ...')
        start_time = time.time()
        """ compute the rule space from the levels in each attribute """
        for item in self.attributeNames:
            self.attributeLevelNum[item + '_neg'] = self.attributeLevelNum[item]
        self.patternSpace = np.zeros(self.maxlen + 1)
        tmp = [item + '_neg' for item in self.attributeNames]
        self.attributeNames.extend(tmp)
        for k in range(1, self.maxlen + 1, 1):
            for subset in combinations(self.attributeNames, k):
                tmp = 1
                for i in subset:
                    tmp = tmp * self.attributeLevelNum[i]
                self.patternSpace[k] = self.patternSpace[k] + tmp
        print('\tTook %0.3fs to compute patternspace' % (time.time() - start_time))

    # This function generates rules that satisfy supp and maxlen using fpgrowth, then it selects the top N rules that make data have the biggest decrease in entropy
    # there are two ways to generate rules. fpgrowth can handle cases where the maxlen is small. If maxlen<=3, fpgrowth can generates rules much faster than randomforest.
    # If maxlen is big, fpgrowh tends to generate too many rules that overflow the memories.
    def generate_rules(self, supp, maxlen, N, method='randomforest'):
        self.maxlen = maxlen
        self.supp = supp
        df = 1 - self.df  # df has negative associations
        df.columns = [name.strip() + '_neg' for name in self.df.columns]
        df = pd.concat([self.df, df], axis=1)
        if method == 'fpgrowth' and maxlen <= 3:
            itemMatrix = [[item for item in df.columns if row[item] == 1] for i, row in df.iterrows()]
            pindex = np.where(self.Y == 1)[0]
            nindex = np.where(self.Y != 1)[0]
            print('Generating rules using fpgrowth')
            start_time = time.time()
            rules = fpgrowth([itemMatrix[i] for i in pindex], supp=supp, zmin=1, zmax=maxlen)
            rules = [tuple(np.sort(rule[0])) for rule in rules]
            rules = list(set(rules))
            start_time = time.time()
            print('\tTook %0.3fs to generate %d rules' % (time.time() - start_time, len(rules)))
        else:
            rules = []
            start_time = time.time()
            for length in range(1, maxlen + 1, 1):
                n_estimators = min(pow(df.shape[1], length), 4000)
                clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=length)
                clf.fit(self.df, self.Y)
                for n in range(n_estimators):
                    rules.extend(extract_rules(clf.estimators_[n], df.columns))
            rules = [list(x) for x in set(tuple(x) for x in rules)]
            print('\tTook %0.3fs to generate %d rules' % (time.time() - start_time, len(rules)))
        self.screen_rules(rules, df, N)  # select the top N rules using secondary criteria, information gain
        self.getPatternSpace()

    def screen_rules(self, rules, df, N):
        print('Screening rules using information gain')
        start_time = time.time()
        itemInd = {}
        for i, name in enumerate(df.columns):
            itemInd[name] = i
        indices = np.array(list(itertools.chain.from_iterable([[itemInd[x] for x in rule] for rule in rules])))
        len_rules = [len(rule) for rule in rules]
        indptr = list(accumulate(len_rules))
        indptr.insert(0, 0)
        indptr = np.array(indptr)
        data = np.ones(len(indices))
        ruleMatrix = csc_matrix((data, indices, indptr), shape=(len(df.columns), len(rules)))
        mat = np.matrix(df) * ruleMatrix
        lenMatrix = np.matrix([len_rules for i in range(df.shape[0])])
        Z = (mat == lenMatrix).astype(int)
        Zpos = [Z[i] for i in np.where(self.Y > 0)][0]
        TP = np.array(np.sum(Zpos, axis=0).tolist()[0])
        supp_select = np.where(TP >= self.supp * sum(self.Y) / 100)[0]
        FP = np.array(np.sum(Z, axis=0))[0] - TP
        TN = len(self.Y) - np.sum(self.Y) - FP
        FN = np.sum(self.Y) - TP
        p1 = TP.astype(float) / (TP + FP)
        p2 = FN.astype(float) / (FN + TN)
        pp = (TP + FP).astype(float) / (TP + FP + TN + FN)
        tpr = TP.astype(float) / (TP + FN)
        fpr = FP.astype(float) / (FP + TN)
        cond_entropy = -pp * (p1 * np.log(p1) + (1 - p1) * np.log(1 - p1)) - (1 - pp) * (
                p2 * np.log(p2) + (1 - p2) * np.log(1 - p2))
        cond_entropy[p1 * (1 - p1) == 0] = -((1 - pp) * (p2 * np.log(p2) + (1 - p2) * np.log(1 - p2)))[
            p1 * (1 - p1) == 0]
        cond_entropy[p2 * (1 - p2) == 0] = -(pp * (p1 * np.log(p1) + (1 - p1) * np.log(1 - p1)))[p2 * (1 - p2) == 0]
        cond_entropy[p1 * (1 - p1) * p2 * (1 - p2) == 0] = 0
        select = np.argsort(cond_entropy[supp_select])[::-1][-N:]
        self.rules = [rules[i] for i in supp_select[select]]
        self.RMatrix = np.array(Z[:, supp_select[select]])
        print('\tTook %0.3fs to generate %d rules' % (time.time() - start_time, len(self.rules)))

    def set_parameters(self, a1=100, b1=1, a2=1, b2=100, al=None, bl=None):
        # input al and bl are lists
        self.alpha_1 = a1
        self.beta_1 = b1
        self.alpha_2 = a2
        self.beta_2 = b2
        if al == None or bl == None or len(al) != self.maxlen or len(bl) != self.maxlen:
            print('No or wrong input for alpha_l and beta_l. The model will use default parameters!')
            self.C = [1.0 / self.maxlen for i in range(self.maxlen)]
            self.C.insert(0, -1)
            self.alpha_l = [10 for i in range(self.maxlen + 1)]
            self.beta_l = [10 * self.patternSpace[i] / self.C[i] for i in range(self.maxlen + 1)]
        else:
            self.alpha_l = [1] + list(al)
            self.beta_l = [1] + list(bl)

    def fit(self, Niteration=5000, Nchain=3, q=0.1, init=[], print_message=True):
        # print('Searching for an optimal solution...')
        start_time = time.time()
        nRules = len(self.rules)
        self.rules_len = [len(rule) for rule in self.rules]
        maps = defaultdict(list)
        T0 = 1000
        split = 0.7 * Niteration
        for chain in range(Nchain):
            # initialize with a random pattern set
            if init != []:
                rules_curr = init.copy()
            else:
                N = sample(range(1, min(8, nRules), 1), 1)[0]
                rules_curr = sample(range(nRules), N)
            rules_curr_norm = self.normalize(rules_curr)
            pt_curr = -100000000000
            maps[chain].append(
                [-1, [pt_curr / 3, pt_curr / 3, pt_curr / 3], rules_curr, [self.rules[i] for i in rules_curr]])

            for iter in range(Niteration):
                if iter >= split:
                    p = np.array(range(1 + len(maps[chain])))
                    p = np.array(list(accumulate(p)))
                    p = p / p[-1]
                    index = find_lt(p, random())
                    rules_curr = maps[chain][index][2].copy()
                    rules_curr_norm = maps[chain][index][2].copy()
                rules_new, rules_norm = self.propose(rules_curr.copy(), rules_curr_norm.copy(), q)
                cfmatrix, prob = self.compute_prob(rules_new)
                T = T0 ** (1 - iter / Niteration)
                pt_new = sum(prob)
                alpha = np.exp(float(pt_new - pt_curr) / T)

                if pt_new > sum(maps[chain][-1][1]):
                    maps[chain].append([iter, prob, rules_new, [self.rules[i] for i in rules_new]])
                    if print_message:
                        print(
                            '\n** chain = {}, max at iter = {} ** \n accuracy = {}, TP = {},FP = {}, TN = {}, FN = {}\n pt_new is {}, prior_ChsRules={}, likelihood_1 = {}, likelihood_2 = {}\n '.format(
                                chain, iter, (cfmatrix[0] + cfmatrix[2] + 0.0) / len(self.Y), cfmatrix[0], cfmatrix[1],
                                cfmatrix[2], cfmatrix[3], sum(prob), prob[0], prob[1], prob[2]))
                        # print '\n** chain = {}, max at iter = {} ** \n obj = {}, prior = {}, llh = {} '.format(chain, iter,prior+llh,prior,llh)
                        self.print_rules(rules_new)
                        print(rules_new)
                if random() <= alpha:
                    rules_curr_norm, rules_curr, pt_curr = rules_norm.copy(), rules_new.copy(), pt_new
        pt_max = [sum(maps[chain][-1][1]) for chain in range(Nchain)]
        index = pt_max.index(max(pt_max))
        # print('\tTook %0.3fs to generate an optimal rule set' % (time.time() - start_time))
        return maps[index][-1][3]

    def propose(self, rules_curr, rules_norm, q):
        nRules = len(self.rules)
        Yhat = (np.sum(self.RMatrix[:, rules_curr], axis=1) > 0).astype(int)
        incorr = np.where(self.Y != Yhat)[0]
        N = len(rules_curr)
        if len(incorr) == 0:
            clean = True
            move = ['clean']
            # it means the BOA correctly classified all points but there could be redundant patterns, so cleaning is needed
        else:
            clean = False
            ex = sample(incorr.tolist(), 1)[0]
            t = random()
            if self.Y[ex] == 1 or N == 1:
                if t < 1.0 / 2 or N == 1:
                    move = ['add']  # action: add
                else:
                    move = ['cut', 'add']  # action: replace
            else:
                if t < 1.0 / 2:
                    move = ['cut']  # action: cut
                else:
                    move = ['cut', 'add']  # action: replace
        if move[0] == 'cut':
            """ cut """
            if random() < q:
                candidate = list(set(np.where(self.RMatrix[ex, :] == 1)[0]).intersection(rules_curr))
                if len(candidate) == 0:
                    candidate = rules_curr
                cut_rule = sample(candidate, 1)[0]
            else:
                p = []
                all_sum = np.sum(self.RMatrix[:, rules_curr], axis=1)
                for index, rule in enumerate(rules_curr):
                    Yhat = ((all_sum - np.array(self.RMatrix[:, rule])) > 0).astype(int)
                    TP, FP, TN, FN = getConfusion(Yhat, self.Y)
                    p.append(TP.astype(float) / (TP + FP + 1))
                    # p.append(log_betabin(TP,TP+FP,self.alpha_1,self.beta_1) + log_betabin(FN,FN+TN,self.alpha_2,self.beta_2))
                p = [x - min(p) for x in p]
                p = np.exp(p)
                p = np.insert(p, 0, 0)
                p = np.array(list(accumulate(p)))
                if p[-1] == 0:
                    index = sample(range(len(rules_curr)), 1)[0]
                else:
                    p = p / p[-1]
                index = find_lt(p, random())
                cut_rule = rules_curr[index]
            rules_curr.remove(cut_rule)
            rules_norm = self.normalize(rules_curr)
            move.remove('cut')

        if len(move) > 0 and move[0] == 'add':
            """ add """
            if random() < q:
                add_rule = sample(range(nRules), 1)[0]
            else:
                Yhat_neg_index = list(np.where(np.sum(self.RMatrix[:, rules_curr], axis=1) < 1)[0])
                mat = np.multiply(self.RMatrix[Yhat_neg_index, :].transpose(), self.Y[Yhat_neg_index])
                # TP = np.array(np.sum(mat,axis = 0).tolist()[0])
                TP = np.sum(mat, axis=1)
                FP = np.array((np.sum(self.RMatrix[Yhat_neg_index, :], axis=0) - TP))
                TN = np.sum(self.Y[Yhat_neg_index] == 0) - FP
                FN = sum(self.Y[Yhat_neg_index]) - TP
                p = (TP.astype(float) / (TP + FP + 1))
                p[rules_curr] = 0
                add_rule = sample(np.where(p == max(p))[0].tolist(), 1)[0]
            if add_rule not in rules_curr:
                rules_curr.append(add_rule)
                rules_norm = self.normalize(rules_curr)

        if len(move) > 0 and move[0] == 'clean':
            remove = []
            for i, rule in enumerate(rules_norm):
                Yhat = (np.sum(
                    self.RMatrix[:, [rule for j, rule in enumerate(rules_norm) if (j != i and j not in remove)]],
                    axis=1) > 0).astype(int)
                TP, FP, TN, FN = getConfusion(Yhat, self.Y)
                if TP + FP == 0:
                    remove.append(i)
            for x in remove:
                rules_norm.remove(x)
            return rules_curr, rules_norm
        return rules_curr, rules_norm

    def compute_prob(self, rules):
        Yhat = (np.sum(self.RMatrix[:, rules], axis=1) > 0).astype(int)
        TP, FP, TN, FN = getConfusion(Yhat, self.Y)
        Kn_count = list(np.bincount([self.rules_len[x] for x in rules], minlength=self.maxlen + 1))
        prior_ChsRules = sum([log_betabin(Kn_count[i], self.patternSpace[i], self.alpha_l[i], self.beta_l[i]) for i in
                              range(1, len(Kn_count), 1)])
        likelihood_1 = log_betabin(TP, TP + FP, self.alpha_1, self.beta_1)
        likelihood_2 = log_betabin(TN, FN + TN, self.alpha_2, self.beta_2)
        return [TP, FP, TN, FN], [prior_ChsRules, likelihood_1, likelihood_2]

    def normalize_add(self, rules_new, rule_index):
        rules = rules_new.copy()
        for rule in rules_new:
            if set(self.rules[rule]).issubset(self.rules[rule_index]):
                return rules_new.copy()
            if set(self.rules[rule_index]).issubset(self.rules[rule]):
                rules.remove(rule)
        rules.append(rule_index)
        return rules

    def normalize(self, rules_new):
        try:
            rules_len = [len(self.rules[index]) for index in rules_new]
            rules = [rules_new[i] for i in np.argsort(rules_len)[::-1][:len(rules_len)]]
            p1 = 0
            while p1 < len(rules):
                for p2 in range(p1 + 1, len(rules), 1):
                    if set(self.rules[rules[p2]]).issubset(set(self.rules[rules[p1]])):
                        rules.remove(rules[p1])
                        p1 -= 1
                        break
                p1 += 1
            return rules
        except:
            return rules_new.copy()

    def print_rules(self, rules_max):
        for rule_index in rules_max:
            print(self.rules[rule_index])

    def predict(self, rules, df):
        Z = [[] for rule in rules]
        dfn = 1 - df  # df has negative associations
        dfn.columns = [name.strip() + '_neg' for name in df.columns]
        df = pd.concat([df, dfn], axis=1)
        for i, rule in enumerate(rules):
            Z[i] = (np.sum(df[list(rule)], axis=1) == len(rule)).astype(int)
        Yhat = (np.sum(Z, axis=0) > 0).astype(int)
        return Yhat


def accumulate(iterable, func=operator.add):
    'Return running totals'
    # accumulate([1,2,3,4,5]) --> 1 3 6 10 15
    # accumulate([1,2,3,4,5], operator.mul) --> 1 2 6 24 120
    it = iter(iterable)
    total = next(it)
    yield total
    for element in it:
        total = func(total, element)
        yield total


def find_lt(a, x):
    """ Find rightmost value less than x"""
    i = bisect_left(a, x)
    if i:
        return int(i - 1)
    print('in find_lt,{}'.format(a))
    raise ValueError


def log_gampoiss(k, alpha, beta):
    import math
    k = int(k)
    return math.lgamma(k + alpha) + alpha * np.log(beta) - math.lgamma(alpha) - math.lgamma(k + 1) - (
            alpha + k) * np.log(1 + beta)


def log_betabin(k, n, alpha, beta):
    import math
    try:
        Const = math.lgamma(alpha + beta) - math.lgamma(alpha) - math.lgamma(beta)
    except:
        print('alpha = {}, beta = {}'.format(alpha, beta))
    if isinstance(k, list) or isinstance(k, np.ndarray):
        if len(k) != len(n):
            print('length of k is %d and length of n is %d' % (len(k), len(n)))
            raise ValueError
        lbeta = []
        for ki, ni in zip(k, n):
            # lbeta.append(math.lgamma(ni+1)- math.lgamma(ki+1) - math.lgamma(ni-ki+1) + math.lgamma(ki+alpha) + math.lgamma(ni-ki+beta) - math.lgamma(ni+alpha+beta) + Const)
            lbeta.append(math.lgamma(ki + alpha) + math.lgamma(ni - ki + beta) - math.lgamma(ni + alpha + beta) + Const)
        return np.array(lbeta)
    else:
        return math.lgamma(k + alpha) + math.lgamma(n - k + beta) - math.lgamma(n + alpha + beta) + Const
        # return math.lgamma(n+1)- math.lgamma(k+1) - math.lgamma(n-k+1) + math.lgamma(k+alpha) + math.lgamma(n-k+beta) - math.lgamma(n+alpha+beta) + Const


def getConfusion(Yhat, Y):
    if len(Yhat) != len(Y):
        raise NameError('Yhat has different length')
    TP = np.dot(np.array(Y), np.array(Yhat))
    FP = np.sum(Yhat) - TP
    TN = len(Y) - np.sum(Y) - FP
    FN = len(Yhat) - np.sum(Yhat) - TN
    return TP, FP, TN, FN


def extract_rules(tree, feature_names):
    left = tree.tree_.children_left
    right = tree.tree_.children_right
    threshold = tree.tree_.threshold
    features = [feature_names[i] for i in tree.tree_.feature]
    # get ids of child nodes
    idx = np.argwhere(left == -1)[:, 0]

    def recurse(left, right, child, lineage=None):
        if lineage is None:
            lineage = []
        if child in left:
            parent = np.where(left == child)[0].item()
            suffix = '_neg'
        else:
            parent = np.where(right == child)[0].item()
            suffix = ''

        #           lineage.append((parent, split, threshold[parent], features[parent]))
        lineage.append((features[parent].strip() + suffix))

        if parent == 0:
            lineage.reverse()
            return lineage
        else:
            return recurse(left, right, parent, lineage)

    rules = []
    for child in idx:
        rule = []
        for node in recurse(left, right, child):
            rule.append(node)
        rules.append(rule)
    return rules
