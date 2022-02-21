'''Original implementation at https://github.com/wangtongada/BOA
'''

import itertools
import operator
from bisect import bisect_left
from collections import defaultdict
from copy import deepcopy
from itertools import combinations
from random import sample

import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import fpgrowth
from numpy.random import random
from scipy.sparse import csc_matrix
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_X_y, check_is_fitted

from imodels.rule_set.rule_set import RuleSet


class BayesianRuleSetClassifier(RuleSet, BaseEstimator, ClassifierMixin):
    '''Bayesian or-of-and algorithm.
    Generates patterns that satisfy the minimum support and maximum length and then select the Nrules rules that have the highest entropy.
    In function SA_patternbased, each local maximum is stored in maps and the best BOA is returned.
    Remember here the BOA contains only the index of selected rules from Nrules self.rules_
    '''

    def __init__(self, n_rules: int = 2000,
                 supp=5, maxlen: int = 10,
                 num_iterations=5000, num_chains=3, q=0.1,
                 alpha_pos=100, beta_pos=1,
                 alpha_neg=100, beta_neg=1,
                 alpha_l=None, beta_l=None,
                 discretization_method='randomforest', random_state=0):
        '''
        Params
        ------
        n_rules
            number of rules to be used in SA_patternbased and also the output of generate_rules
        supp
            The higher this supp, the 'larger' a pattern is. 5% is a generally good number
        maxlen
            maximum length of a pattern
        num_iterations
            number of iterations in each chain
        num_chains
            number of chains in the simulated annealing search algorithm
        q
        alpha_pos
            $\rho = alpha/(alpha+beta)$. Make sure $\rho$ is close to one when choosing alpha and beta
            The alpha and beta parameters alter the prior distributions for different rules
        beta_pos
        alpha_neg
        beta_neg
        alpha_l
        beta_l
        discretization_method
            discretization method
        '''
        self.n_rules = n_rules
        self.supp = supp
        self.maxlen = maxlen

        self.num_iterations = num_iterations
        self.num_chains = num_chains
        self.q = q

        self.alpha_pos = alpha_pos
        self.beta_pos = beta_pos
        self.alpha_neg = alpha_neg
        self.beta_neg = beta_neg
        self.discretization_method = discretization_method

        self.alpha_l = alpha_l
        self.beta_l = beta_l
        self.random_state = 0

    def fit(self, X, y, feature_names: list = None, init=[], verbose=False):
        '''
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training data
        y : array_like, shape = [n_samples]
            Labels

        feature_names : array_like, shape = [n_features], optional (default: [])
            String labels for each feature.
            If empty and X is a DataFrame, column labels are used.
            If empty and X is not a DataFrame, then features are simply enumerated
        '''
        # check inputs
        self.attr_level_num = defaultdict(int)  # any missing value defaults to 0
        self.attr_names = []

        # get feature names
        if feature_names is None:
            if isinstance(X, pd.DataFrame):
                feature_names = X.columns
            else:
                feature_names = ['X' + str(i) for i in range(X.shape[1])]

        # checks
        X, y = check_X_y(X, y)  # converts df to ndarray
        check_classification_targets(y)
        assert len(feature_names) == X.shape[1], 'feature_names should be same size as X.shape[1]'
        np.random.seed(self.random_state)

        # convert to pandas DataFrame
        X = pd.DataFrame(X, columns=feature_names)

        for i, name in enumerate(X.columns):
            self.attr_level_num[name] += 1
            self.attr_names.append(name)
        self.attr_names_orig = deepcopy(self.attr_names)
        self.attr_names = list(set(self.attr_names))

        # set up patterns
        self._set_pattern_space()

        # parameter checking
        if self.alpha_l is None or self.beta_l is None or len(self.alpha_l) != self.maxlen or len(
                self.beta_l) != self.maxlen:
            print('No or wrong input for alpha_l and beta_l - the model will use default parameters.')
            self.C = [1.0 / self.maxlen] * self.maxlen
            self.C.insert(0, -1)
            self.alpha_l = [10] * (self.maxlen + 1)
            self.beta_l = [10 * self.pattern_space[i] / self.C[i] for i in range(self.maxlen + 1)]
        else:
            self.alpha_l = [1] + list(self.alpha_l)
            self.beta_l = [1] + list(self.beta_l)

        # setup
        self._generate_rules(X, y)
        n_rules_current = len(self.rules_)
        self.rules_len_list = [len(rule) for rule in self.rules_]
        maps = defaultdict(list)
        T0 = 1000  # initial temperature for simulated annealing
        split = 0.7 * self.num_iterations

        # run simulated annealing
        for chain in range(self.num_chains):
            # initialize with a random pattern set
            if init != []:
                rules_curr = init.copy()
            else:
                assert n_rules_current > 1, f'Only {n_rules_current} potential rules found, change hyperparams to allow for more'
                N = sample(range(1, min(8, n_rules_current), 1), 1)[0]
                rules_curr = sample(range(n_rules_current), N)
            rules_curr_norm = self._normalize(rules_curr)
            pt_curr = -100000000000
            maps[chain].append(
                [-1, [pt_curr / 3, pt_curr / 3, pt_curr / 3], rules_curr, [self.rules_[i] for i in rules_curr]])

            for iter in range(self.num_iterations):
                if iter >= split:
                    p = np.array(range(1 + len(maps[chain])))
                    p = np.array(list(_accumulate(p)))
                    p = p / p[-1]
                    index = _find_lt(p, random())
                    rules_curr = maps[chain][index][2].copy()
                    rules_curr_norm = maps[chain][index][2].copy()

                # propose new rules
                rules_new, rules_norm = self._propose(rules_curr.copy(), rules_curr_norm.copy(), self.q, y)

                # compute probability of new rules
                cfmatrix, prob = self._compute_prob(rules_new, y)
                T = T0 ** (1 - iter / self.num_iterations)  # temperature for simulated annealing
                pt_new = sum(prob)
                alpha = np.exp(float(pt_new - pt_curr) / T)

                if pt_new > sum(maps[chain][-1][1]):
                    maps[chain].append([iter, prob, rules_new, [self.rules_[i] for i in rules_new]])
                    if verbose:
                        print((
                            '\n** chain = {}, max at iter = {} ** \n accuracy = {}, TP = {},FP = {}, TN = {}, FN = {}'
                            '\n pt_new is {}, prior_ChsRules={}, likelihood_1 = {}, likelihood_2 = {}\n').format(
                            chain, iter, (cfmatrix[0] + cfmatrix[2] + 0.0) / len(y), cfmatrix[0], cfmatrix[1],
                            cfmatrix[2], cfmatrix[3], sum(prob), prob[0], prob[1], prob[2])
                        )
                        self._print_rules(rules_new)
                        print(rules_new)
                if random() <= alpha:
                    rules_curr_norm, rules_curr, pt_curr = rules_norm.copy(), rules_new.copy(), pt_new
        pt_max = [sum(maps[chain][-1][1]) for chain in range(self.num_chains)]
        index = pt_max.index(max(pt_max))
        self.rules_ = maps[index][-1][3]
        return self

    def __str__(self):
        return ' '.join(str(r) for r in self.rules_)

    def predict(self, X):
        check_is_fitted(self)
        if isinstance(X, np.ndarray):
            df = pd.DataFrame(X, columns=self.attr_names_orig)
        else:
            df = X
        Z = [[]] * len(self.rules_)
        dfn = 1 - df  # df has negative associations
        dfn.columns = [name.strip() + '_neg' for name in df.columns]
        df = pd.concat([df, dfn], axis=1)
        for i, rule in enumerate(self.rules_):
            Z[i] = (np.sum(df[list(rule)], axis=1) == len(rule)).astype(int)
        Yhat = (np.sum(Z, axis=0) > 0).astype(int)
        return Yhat

    def predict_proba(self, X):
        raise Exception('BOA does not support predicted probabilities.')

    def _set_pattern_space(self):
        """Compute the rule space from the levels in each attribute
        """
        # add feat_neg to each existing feature feat
        for item in self.attr_names:
            self.attr_level_num[item + '_neg'] = self.attr_level_num[item]
        tmp = [item + '_neg' for item in self.attr_names]
        self.attr_names.extend(tmp)

        # set up pattern_space
        self.pattern_space = np.zeros(self.maxlen + 1)
        for k in range(1, self.maxlen + 1, 1):
            for subset in combinations(self.attr_names, k):
                tmp = 1
                for i in subset:
                    tmp = tmp * self.attr_level_num[i]
                # print('subset', subset, 'tmp', tmp, 'k', k)
                self.pattern_space[k] = self.pattern_space[k] + tmp

    def _generate_rules(self, X, y):
        '''This function generates rules that satisfy supp and maxlen using fpgrowth, then it selects the top n_rules rules that make data have the biggest decrease in entropy
        there are two ways to generate rules. fpgrowth can handle cases where the maxlen is small. If maxlen<=3, fpgrowth can generates rules much faster than randomforest.
        If maxlen is big, fpgrowh tends to generate too many rules that overflow the memories.
        '''

        df = 1 - X  # df has negative associations
        df.columns = [name.strip() + '_neg' for name in X.columns]
        df = pd.concat([X, df], axis=1)
        if self.discretization_method == 'fpgrowth' and self.maxlen <= 3:
            itemMatrix = [[item for item in df.columns if row[item] == 1] for i, row in df.iterrows()]
            pindex = np.where(y == 1)[0]
            rules = fpgrowth([itemMatrix[i] for i in pindex], supp=self.supp, zmin=1, zmax=self.maxlen)
            rules = [tuple(np.sort(rule[0])) for rule in rules]
            rules = list(set(rules))
        else:
            '''todo: replace this with imodels.RFDiscretizer
            '''
            rules = []
            for length in range(1, self.maxlen + 1, 1):
                n_estimators = min(pow(df.shape[1], length), 4000)
                clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=length)
                clf.fit(X, y)
                for n in range(n_estimators):
                    rules.extend(_extract_rules(clf.estimators_[n], df.columns))
            rules = [list(x) for x in set(tuple(x) for x in rules)]
        self.rules_ = rules

        # select the top n_rules rules using secondary criteria, information gain
        self._screen_rules(df, y)  # updates self.rules_
        self._set_pattern_space()

    def _screen_rules(self, df, y):
        '''Screening rules using information gain
        '''
        item_ind_dict = {}
        for i, name in enumerate(df.columns):
            item_ind_dict[name] = i
        indices = np.array(
            list(itertools.chain.from_iterable([[item_ind_dict[x] for x in rule] for rule in self.rules_])))
        len_rules = [len(rule) for rule in self.rules_]
        indptr = list(_accumulate(len_rules))
        indptr.insert(0, 0)
        indptr = np.array(indptr)
        data = np.ones(len(indices))
        rule_matrix = csc_matrix((data, indices, indptr), shape=(len(df.columns), len(self.rules_)))
        mat = np.matrix(df) @ rule_matrix
        len_matrix = np.array([len_rules] * df.shape[0])
        Z = (mat == len_matrix).astype(int)
        Zpos = [Z[i] for i in np.where(y > 0)][0]
        TP = np.array(np.sum(Zpos, axis=0).tolist()[0])
        supp_select = np.where(TP >= self.supp * sum(y) / 100)[0]
        FP = np.array(np.sum(Z, axis=0))[0] - TP
        TN = len(y) - np.sum(y) - FP
        FN = np.sum(y) - TP
        p1 = TP.astype(float) / (TP + FP)
        p2 = FN.astype(float) / (FN + TN)
        pp = (TP + FP).astype(float) / (TP + FP + TN + FN)
        cond_entropy = -pp * (p1 * np.log(p1) + (1 - p1) * np.log(1 - p1)) - (1 - pp) * (
                p2 * np.log(p2) + (1 - p2) * np.log(1 - p2))
        cond_entropy[p1 * (1 - p1) == 0] = -((1 - pp) * (p2 * np.log(p2) + (1 - p2) * np.log(1 - p2)))[
            p1 * (1 - p1) == 0]
        cond_entropy[p2 * (1 - p2) == 0] = -(pp * (p1 * np.log(p1) + (1 - p1) * np.log(1 - p1)))[p2 * (1 - p2) == 0]
        cond_entropy[p1 * (1 - p1) * p2 * (1 - p2) == 0] = 0
        select = np.argsort(cond_entropy[supp_select])[::-1][-self.n_rules:]
        self.rules_ = [self.rules_[i] for i in supp_select[select]]
        self.RMatrix = np.array(Z[:, supp_select[select]])

    def _propose(self, rules_curr, rules_norm, q, y):
        nRules = len(self.rules_)
        yhat = (np.sum(self.RMatrix[:, rules_curr], axis=1) > 0).astype(int)
        incorr = np.where(y != yhat)[0]
        N = len(rules_curr)

        if len(incorr) == 0:
            # BOA correctly classified all points but there could be redundant patterns, so cleaning is needed
            move = ['clean']
        else:
            ex = sample(incorr.tolist(), 1)[0]
            t = random()
            if y[ex] == 1 or N == 1:
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
                    yhat = ((all_sum - np.array(self.RMatrix[:, rule])) > 0).astype(int)
                    TP, FP, TN, FN = _get_confusion_matrix(yhat, y)
                    p.append(TP.astype(float) / (TP + FP + 1))
                p = [x - min(p) for x in p]
                p = np.exp(p)
                p = np.insert(p, 0, 0)
                p = np.array(list(_accumulate(p)))
                if p[-1] == 0:
                    index = sample(range(len(rules_curr)), 1)[0]
                else:
                    p = p / p[-1]
                index = _find_lt(p, random())
                cut_rule = rules_curr[index]
            rules_curr.remove(cut_rule)
            rules_norm = self._normalize(rules_curr)
            move.remove('cut')

        if len(move) > 0 and move[0] == 'add':
            """ add """
            if random() < q:
                add_rule = sample(range(nRules), 1)[0]
            else:
                Yhat_neg_index = list(np.where(np.sum(self.RMatrix[:, rules_curr], axis=1) < 1)[0])
                mat = np.multiply(self.RMatrix[Yhat_neg_index, :].transpose(), y[Yhat_neg_index])
                TP = np.sum(mat, axis=1)
                FP = np.array((np.sum(self.RMatrix[Yhat_neg_index, :], axis=0) - TP))
                p = (TP.astype(float) / (TP + FP + 1))
                p[rules_curr] = 0
                add_rule = sample(np.where(p == max(p))[0].tolist(), 1)[0]
            if add_rule not in rules_curr:
                rules_curr.append(add_rule)
                rules_norm = self._normalize(rules_curr)

        if len(move) > 0 and move[0] == 'clean':
            remove = []
            for i, rule in enumerate(rules_norm):
                yhat = (np.sum(
                    self.RMatrix[:, [rule for j, rule in enumerate(rules_norm) if (j != i and j not in remove)]],
                    axis=1) > 0).astype(int)
                TP, FP, TN, FN = _get_confusion_matrix(yhat, y)
                if TP + FP == 0:
                    remove.append(i)
            for x in remove:
                rules_norm.remove(x)
            return rules_curr, rules_norm
        return rules_curr, rules_norm

    def _compute_prob(self, rules, y):
        Yhat = (np.sum(self.RMatrix[:, rules], axis=1) > 0).astype(int)
        TP, FP, TN, FN = _get_confusion_matrix(Yhat, y)
        Kn_count = list(np.bincount([self.rules_len_list[x] for x in rules], minlength=self.maxlen + 1))
        prior_ChsRules = sum([_log_betabin(Kn_count[i], self.pattern_space[i], self.alpha_l[i], self.beta_l[i]) for i in
                              range(1, len(Kn_count), 1)])
        likelihood_1 = _log_betabin(TP, TP + FP, self.alpha_pos, self.beta_pos)
        likelihood_2 = _log_betabin(TN, FN + TN, self.alpha_neg, self.beta_neg)
        return [TP, FP, TN, FN], [prior_ChsRules, likelihood_1, likelihood_2]

    def _normalize_add(self, rules_new, rule_index):
        rules = rules_new.copy()
        for rule in rules_new:
            if set(self.rules_[rule]).issubset(self.rules_[rule_index]):
                return rules_new.copy()
            if set(self.rules_[rule_index]).issubset(self.rules_[rule]):
                rules.remove(rule)
        rules.append(rule_index)
        return rules

    def _normalize(self, rules_new):
        try:
            rules_len = [len(self.rules_[index]) for index in rules_new]
            rules = [rules_new[i] for i in np.argsort(rules_len)[::-1][:len(rules_len)]]
            p1 = 0
            while p1 < len(rules):
                for p2 in range(p1 + 1, len(rules), 1):
                    if set(self.rules_[rules[p2]]).issubset(set(self.rules_[rules[p1]])):
                        rules.remove(rules[p1])
                        p1 -= 1
                        break
                p1 += 1
            return rules
        except:
            return rules_new.copy()

    def _print_rules(self, rules_max):
        for rule_index in rules_max:
            print(self.rules_[rule_index])


def _accumulate(iterable, func=operator.add):
    '''Return running totals
    Ex. _accumulate([1,2,3,4,5]) --> 1 3 6 10 15
    Ex. _accumulate([1,2,3,4,5], operator.mul) --> 1 2 6 24 120
    '''
    it = iter(iterable)
    total = next(it)
    yield total
    for element in it:
        total = func(total, element)
        yield total


def _find_lt(a, x):
    """ Find rightmost value less than x"""
    i = bisect_left(a, x)
    if i:
        return int(i - 1)
    print('in _find_lt,{}'.format(a))
    raise ValueError


def _log_gampoiss(k, alpha, beta):
    import math
    k = int(k)
    return math.lgamma(k + alpha) + alpha * np.log(beta) - math.lgamma(alpha) - math.lgamma(k + 1) - (
            alpha + k) * np.log(1 + beta)


def _log_betabin(k, n, alpha, beta):
    import math
    try:
        const = math.lgamma(alpha + beta) - math.lgamma(alpha) - math.lgamma(beta)
    except:
        print('alpha = {}, beta = {}'.format(alpha, beta))
    if isinstance(k, list) or isinstance(k, np.ndarray):
        if len(k) != len(n):
            print('length of k is %d and length of n is %d' % (len(k), len(n)))
            raise ValueError
        lbeta = []
        for ki, ni in zip(k, n):
            lbeta.append(math.lgamma(ki + alpha) + math.lgamma(ni - ki + beta) - math.lgamma(ni + alpha + beta) + const)
        return np.array(lbeta)
    else:
        return math.lgamma(k + alpha) + math.lgamma(n - k + beta) - math.lgamma(n + alpha + beta) + const


def _get_confusion_matrix(Yhat, Y):
    if len(Yhat) != len(Y):
        raise NameError('Yhat has different length')
    TP = np.dot(np.array(Y), np.array(Yhat))
    FP = np.sum(Yhat) - TP
    TN = len(Y) - np.sum(Y) - FP
    FN = len(Yhat) - np.sum(Yhat) - TN
    return TP, FP, TN, FN


def _extract_rules(tree, feature_names):
    left = tree.tree_.children_left
    right = tree.tree_.children_right
    features = [feature_names[i] for i in tree.tree_.feature]

    # get ids of child nodes
    idx = np.argwhere(left == -1)[:, 0]

    def _recurse(left, right, child, lineage=None):
        if lineage is None:
            lineage = []
        if child in left:
            parent = np.where(left == child)[0].item()
            suffix = '_neg'
        else:
            parent = np.where(right == child)[0].item()
            suffix = ''
        lineage.append((features[parent].strip() + suffix))

        if parent == 0:
            lineage.reverse()
            return lineage
        else:
            return _recurse(left, right, parent, lineage)

    rules = []
    for child in idx:
        rule = []
        for node in _recurse(left, right, child):
            rule.append(node)
        rules.append(rule)
    return rules
