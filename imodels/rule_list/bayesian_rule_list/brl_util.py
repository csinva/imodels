#######Supplement for "Interpretable classifiers using rules and Bayesian analysis: Building a better stroke prediction model."

###LICENSE
#
# This software is released under the MIT license.
#
# Copyright (c) 2013-14 Ben Letham
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# The author/copyright holder can be contacted at bletham@mit.edu

####README
#
# This code implements the Bayesian Rule Lists algorithm as described in the
# paper. We include the Titanic dataset in the correct formatting to be used
# by this code.
#
# This code requires the external frequent itemset mining package "PyFIM,"
# available at http://www.borgelt.net/pyfim.html
#
# It is specific to binary classification with binary features (although could
# easily be extended to multiclass).
#
#
# ##OUTPUT
#
# The highest level function, "topscript," returns:
#
# - permsdic - Contains the important information from the MCMC sampling. A 
# dictionary whose keys are a string Pickle-dump of the antecedent list d, and
# whose values are a list [a,b] where a is (proportional to) the log posterior of
# d, and b is the number of times d is present in the MCMC samples.
# - d_star - the BRL-point antecedent list. A list of indicies corresponding to 
# variable "itemsets."
# - itemsets - A list of itemsets. itemsets[d_star[i]] is the antecedent in 
# position i on the BRL-point list
# - theta - A list of the expected value of the posterior consequent 
# distribution for each entry in BRL-point.
# - ci_theta - A list of tuples, each the 95% credible interval for the 
# corresponding theta.
# - preds_d_star - Predictions on the demo data made using d_star and theta.
# - accur_d_star - The accuracy of the BRL-point predictions, with the decision 
# boundary at 0.5.
# - preds_fullpost - Predictions on the demo data using the full posterior 
# (BRL-post)
# - accur_fullpost - The accuracy of the BRL-post predictions, decision boundary 
# at 0.5.
#
import pickle as Pickle
import time
from collections import defaultdict

from numpy import *
from scipy.special import gammaln
from scipy.stats import poisson, beta

try:
    from matplotlib import pyplot as plt
except:
    pass


###############BRL



def default_permsdic():
    '''For producing the defaultdict used for storing MCMC results
    '''
    return [0., 0.]


def reset_permsdic(permsdic):
    '''Resets the number of MCMC samples stored (value[1]) while maintaining the
    log-posterior value (so it doesn't need to be re-computed in future chains).
    '''
    for perm in permsdic:
        permsdic[perm][1] = 0.
    return permsdic


def run_bdl_multichain_serial(numiters, thinning, alpha, lbda, eta, X, Y, nruleslen, lhs_len, maxlhs, permsdic, burnin,
                              nchains, d_inits, verbose=True, seed=42):
    '''Run mcmc for each of the chains in serial
    '''
    # random seed
    random.seed(seed)

    # Run each chain
    t1 = time.process_time()
    if verbose:
        print('Starting mcmc chains')
    res = {}
    for n in range(nchains):
        res[n] = mcmcchain(numiters, thinning, alpha, lbda, eta, X, Y, nruleslen, lhs_len, maxlhs, permsdic, burnin,
                           nchains, d_inits[n])

    if verbose:
        print('Elapsed CPU time', time.process_time() - t1)

    # Check convergence
    Rhat = gelmanrubin(res)

    if verbose:
        print('Rhat for convergence:', Rhat)
    ##plot?
    # plot_chains(res)
    return res, Rhat


def mcmcchain(numiters, thinning, alpha, lbda, eta, X, Y, nruleslen, lhs_len, maxlhs, permsdic, burnin, nchains,
              d_init):
    '''Run and store mcmc chain
    '''
    res = {}
    permsdic, res['perms'] = bayesdl_mcmc(numiters, thinning, alpha, lbda, eta, X, Y, nruleslen, lhs_len, maxlhs,
                                          permsdic, burnin, None, d_init)
    # Store the permsdic results
    res['permsdic'] = {perm: list(vals) for perm, vals in permsdic.items() if vals[1] > 0}
    # Reset the permsdic
    permsdic = reset_permsdic(permsdic)
    return res



def gelmanrubin(res):
    '''Check convergence with GR diagnostic
    '''
    n = 0  # number of samples per chain - to be computed
    m = len(res)  # number of chains
    phi_bar_j = {}
    for chain in res:
        phi_bar_j[chain] = 0.
        for val in res[chain]['permsdic'].values():
            phi_bar_j[chain] += val[1] * val[0]  # numsamples*log posterior
            n += val[1]
    # And normalize
    n = n // m  # Number of samples per chain (assuming all m chains have same number of samples)
    # Normalize, and compute phi_bar
    phi_bar = 0.
    for chain in phi_bar_j:
        phi_bar_j[chain] = phi_bar_j[chain] / float(n)  # normalize
        phi_bar += phi_bar_j[chain]
    phi_bar = phi_bar / float(m)  # phi_bar = average of phi_bar_j
    # Now B
    B = 0.
    for chain in phi_bar_j:
        B += (phi_bar_j[chain] - phi_bar) ** 2
    B = B * (n / float(m - 1))
    # Now W.
    W = 0.
    for chain in res:
        s2_j = 0.
        for val in res[chain]['permsdic'].values():
            s2_j += val[1] * (val[0] - phi_bar_j[chain]) ** 2
        s2_j = (1. / float(n - 1)) * s2_j
        W += s2_j
    W = W * (1. / float(m))
    # Next varhat
    varhat = ((n - 1) / float(n)) * W + (1. / float(n)) * B
    # And finally,
    try:
        Rhat = sqrt(varhat / float(W))
    except:
        print('RuntimeWarning computing Rhat, W=' + str(W) + ', B=' + str(B))
        Rhat = 0.
    return Rhat

def plot_chains(res):
    '''Plot the logposterior values for the samples in the chains.
    '''
    for chain in res:
        plt.plot([res[chain]['permsdic'][a][0] for a in res[chain]['perms']])
    plt.show()
    return

def merge_chains(res):
    '''Merge chains into a single collection of posterior samples
    '''
    permsdic = defaultdict(default_permsdic)
    for n in res:
        for perm, vals in res[n]['permsdic'].items():
            permsdic[perm][0] = vals[0]
            permsdic[perm][1] += vals[1]
    return permsdic

def get_point_estimate(permsdic, lhs_len, X, Y, alpha, nruleslen, maxlhs, lbda, eta, verbose=True):
    '''Get a point estimate with length and width similar to the posterior average, with highest likelihood
    '''
    # Figure out the posterior expected list length and average rule size
    listlens = []
    rulesizes = []
    for perm in permsdic:
        #         with open(perm, 'rb') as file:
        #             d_t = pickle.loads(file)
        #         print('perm', perm, type(perm))
        #         print('perm list', list(perm))
        #         d_t = Pickle.loads(bytes(perm, encoding="latin1")) #, encoding='bytes')
        d_t = Pickle.loads(perm)  # , encoding='bytes')

        listlens.extend([len(d_t)] * int(permsdic[perm][1]))
        rulesizes.extend([lhs_len[j] for j in d_t[:-1]] * int(permsdic[perm][1]))

    # Now compute average
    avglistlen = average(listlens)
    if verbose:
        print('Posterior average length:', avglistlen)
    try:
        avgrulesize = average(rulesizes)
        if verbose:
            print('Posterior average width:', avgrulesize)
        # Prepare the intervals
        minlen = int(floor(avglistlen))
        maxlen = int(ceil(avglistlen))
        minrulesize = int(floor(avgrulesize))
        maxrulesize = int(ceil(avgrulesize))
        # Run through all perms again
        likelihds = []
        d_ts = []
        beta_Z, logalpha_pmf, logbeta_pmf = prior_calculations(lbda, len(X), eta,
                                                               maxlhs)  # get the constants needed to compute the prior
        for perm in permsdic:
            if permsdic[perm][1] > 0:
                d_t = Pickle.loads(perm)  # this is the antecedent list

                # Check the list length
                if len(d_t) >= minlen and len(d_t) <= maxlen:

                    # Check the rule size
                    rulesize = average([lhs_len[j] for j in d_t[:-1]])
                    if rulesize >= minrulesize and rulesize <= maxrulesize:
                        d_ts.append(d_t)

                        # Compute the likelihood
                        R_t = d_t.index(0)
                        N_t = compute_rule_usage(d_t, R_t, X, Y)
                        likelihds.append(
                            fn_logposterior(d_t, R_t, N_t, alpha, logalpha_pmf, logbeta_pmf, maxlhs, beta_Z, nruleslen,
                                            lhs_len))
        likelihds = array(likelihds)
        d_star = d_ts[likelihds.argmax()]
    except RuntimeWarning:
        # This can happen if all perms are identically [0], or if no soln is found within the len and width bounds (probably the chains didn't converge)
        print('No suitable point estimate found')
        d_star = None
    return d_star


#################COMPUTING RESULTS
def get_rule_rhs(Xtrain, Ytrain, d_t, alpha, intervals):
    '''Compute the posterior consequent distributions
    (Basically compute points in each part of rule)
    '''
    N_t = compute_rule_usage(d_t, d_t.index(0), Xtrain, Ytrain)
    theta = []  # P(Y=1)
    ci_theta = [] # confidence interval for Y=1
    for i, j in enumerate(d_t):
        # theta ~ Dirichlet(N[j,:] + alpha)
        # E[theta] = (N[j,:] + alpha)/float(sum(N[j,:] + alpha))
        # NOTE this result is only for binary classification
        # theta = p(y=1)
        theta.append((N_t[i, 1] + alpha[1]) / float(sum(N_t[i, :] + alpha)))
        # And now the 95% interval, for Beta(N[j,1] + alpha[1], N[j,0] + alpha[0])
        if intervals:
            ci_theta.append(beta.interval(0.95, N_t[i, 1] + alpha[1], N_t[i, 0] + alpha[0]))
    return theta, ci_theta


def preds_d_t(X, Y, d_t, theta):
    '''Get predictions from the list d_t
    '''
    # this is binary only. The score is the Prob of 1.
    unused = set(range(Y.shape[0]))
    preds = -1 * ones(Y.shape[0])
    for i, j in enumerate(d_t):
        usedj = unused.intersection(X[j])  # these are the observations in X that make it to rule j
        preds[list(usedj)] = theta[i]
        unused = unused.difference(set(usedj))
    if preds.min() < 0:
        raise Exception  # this means some observation wasn't given a prediction - shouldn't happen
    return preds


##############MCMC core 
def bayesdl_mcmc(numiters, thinning, alpha, lbda, eta, X, Y, nruleslen, lhs_len, maxlhs, permsdic, burnin, rseed,
                 d_init):
    '''Run Metropolis-Hastings algorithm
    '''
    # initialize
    perms = []
    if rseed:
        random.seed(rseed)
        
    # Do some pre-computation for the prior
    beta_Z, logalpha_pmf, logbeta_pmf = prior_calculations(lbda, len(X), eta, maxlhs)
    if d_init:  # If we want to begin our chain at a specific place (e.g. to continue a chain)
        d_t = Pickle.loads(d_init)
        d_t.extend([i for i in range(len(X)) if i not in d_t])
        R_t = d_t.index(0)
        N_t = compute_rule_usage(d_t, R_t, X, Y)
    else:
        d_t, R_t, N_t = initialize_d(X, Y, lbda, eta, lhs_len, maxlhs,
                                     nruleslen)  # Otherwise sample the initial value from the prior
    
    # Add to dictionary which will store the sampling results
    a_t = Pickle.dumps(d_t[:R_t + 1])  # The antecedent list in string form
    if a_t not in permsdic:
        permsdic[a_t][0] = fn_logposterior(d_t, R_t, N_t, alpha, logalpha_pmf, logbeta_pmf, maxlhs, beta_Z, nruleslen,
                                           lhs_len)  # Compute its logposterior
    if burnin == 0:
        permsdic[a_t][1] += 1  # store the initialization sample
    
    # iterate!
    for itr in range(numiters):
        # Sample from proposal distribution
        d_star, Jratio, R_star, step = proposal(d_t, R_t, X, Y, alpha)
        # Compute the new posterior value, if necessary
        a_star = Pickle.dumps(d_star[:R_star + 1])
        if a_star not in permsdic:
            N_star = compute_rule_usage(d_star, R_star, X, Y)
            permsdic[a_star][0] = fn_logposterior(d_star, R_star, N_star, alpha, logalpha_pmf, logbeta_pmf, maxlhs,
                                                  beta_Z, nruleslen, lhs_len)
        # Compute the metropolis acceptance probability
        q = exp(permsdic[a_star][0] - permsdic[a_t][0] + Jratio)
        u = random.random()
        if u < q:
            # then we accept the move
            d_t = list(d_star)
            R_t = int(R_star)
            a_t = a_star
            # else: pass
        if itr > burnin and itr % thinning == 0:
            ##store
            permsdic[a_t][1] += 1
            perms.append(a_t)
    return permsdic, perms


def initialize_d(X, Y, lbda, eta, lhs_len, maxlhs, nruleslen):
    '''Samples a list from the prior
    '''
    m = Inf
    while m >= len(X):
        m = poisson.rvs(lbda)  # sample the length of the list from Poisson(lbda), truncated at len(X)
    # prepare the list
    d_t = []
    empty_rulelens = [r for r in range(1, maxlhs + 1) if r not in nruleslen]
    used_rules = []
    for i in range(m):
        # Sample a rule size.
        r = 0
        while r == 0 or r > maxlhs or r in empty_rulelens:
            r = poisson.rvs(
                eta)  # Sample the rule size from Poisson(eta), truncated at 0 and maxlhs and not using empty rule lens
        # Now sample a rule of that size uniformly at random
        rule_cands = [j for j, lhslen in enumerate(lhs_len) if lhslen == r and j not in used_rules]
        random.shuffle(rule_cands)
        j = rule_cands[0]
        # And add it in
        d_t.append(j)
        used_rules.append(j)
        assert lhs_len[j] == r
        if len(rule_cands) == 1:
            empty_rulelens.append(r)
    # Done adding rules. We have added m rules. Finish up.
    d_t.append(0)  # all done
    d_t.extend([i for i in range(len(X)) if i not in d_t])
    R_t = d_t.index(0)
    assert R_t == m
    # Figure out what rules are used to classify what points
    N_t = compute_rule_usage(d_t, R_t, X, Y)
    return d_t, R_t, N_t


def proposal(d_t, R_t, X, Y, alpha):
    '''Propose a new d_star
    '''
    d_star = list(d_t)
    R_star = int(R_t)
    # We begin with these as the move probabilities, but will renormalize as needed if certain moves are unavailable.
    move_probs_default = array([0.3333333333, 0.3333333333, 0.3333333333])
    # We have 3 moves: move, add, cut. Define the pdf for the probabilities of the moves, in that order:
    if R_t == 0:
        # List is empty. We must add.
        move_probs = array([0., 1., 0.])
        # This is an add transition. The probability of the reverse cut move is the prob of a list of len 1 having
        # a cut (other option for list of len 1 is an add).
        Jratios = array([0., move_probs_default[2] / float(move_probs_default[1] + move_probs_default[2]), 0.])
    elif R_t == 1:
        # List has one rule on it. We cannot move, must add or cut.
        move_probs = array(move_probs_default)  # copy
        move_probs[0] = 0.  # drop move move.
        move_probs = move_probs / sum(move_probs)  # renormalize
        # If add, probability of the reverse cut is the default cut probability
        # If cut, probability of the reverse add is 1.
        inv_move_probs = array([0., move_probs_default[2], 1.])
        Jratios = zeros_like(move_probs)
        Jratios[1:] = inv_move_probs[1:] / move_probs[1:]  # array elementwise division
    elif R_t == len(d_t) - 1:
        # List has all rules on it. We cannot add, must move or cut.
        move_probs = array(move_probs_default)  # copy
        move_probs[1] = 0.  # drop add move.
        move_probs = move_probs / sum(move_probs)  # renormalize
        # If move, probability of reverse move is move_probs[0], so Jratio = 1.
        # if cut, probability of reverse add is move_probs_default
        Jratios = array([1., 0., move_probs_default[1] / move_probs[2]])
    elif R_t == len(d_t) - 2:
        # List has all rules but 1 on it.
        # Move probabilities are the default, but the inverse are a little different.
        move_probs = array(move_probs_default)
        # If move, probability of reverse move is still default, so Jratio = 1.
        # if cut, probability of reverse add is move_probs_default[1],
        # if add, probability of reverse cut is,
        Jratios = array([1., move_probs_default[2] / float(move_probs_default[0] + move_probs_default[2]) / float(
            move_probs_default[1]), move_probs_default[1] / float(move_probs_default[2])])
    else:
        move_probs = array(move_probs_default)
        Jratios = array([1., move_probs[2] / float(move_probs[1]), move_probs[1] / float(move_probs[2])])
    u = random.random()
    # First we will find the indicies for the insertion-deletion. indx1 is the item to be moved, indx2 is the new location
    if u < sum(move_probs[:1]):
        # This is an on-list move.
        step = 'move'
        [indx1, indx2] = random.permutation(range(len(d_t[:R_t])))[:2]  # value error if there are no on list entries
        # print 'move',indx1,indx2
        Jratio = Jratios[0]  # ratio of move/move probabilities is 1.
    elif u < sum(move_probs[:2]):
        # this is an add
        step = 'add'
        indx1 = R_t + 1 + random.randint(0, len(
            d_t[R_t + 1:]))  # this will throw ValueError if there are no off list entries
        indx2 = random.randint(0, len(d_t[:R_t + 1]))  # this one will always work
        # print 'add',indx1,indx2
        # the probability of going from d_star back to d_t is the probability of the corresponding cut.
        # p(d*->d|cut) = 1/|d*| = 1/(|d|+1) = 1./float(R_t+1)
        # p(d->d*|add) = 1/((|a|-|d|)(|d|+1)) = 1./(float(len(d_t)-1-R_t)*float(R_t+1))
        Jratio = Jratios[1] * float(len(d_t) - 1 - R_t)
        R_star += 1
    elif u < sum(move_probs[:3]):
        # this is a cut
        step = 'cut'
        indx1 = random.randint(0, len(d_t[:R_t]))  # this will throw ValueError if there are no on list entries
        indx2 = R_t + random.randint(0, len(d_t[R_t:]))  # this one will always work
        # print 'cut',indx1,indx2
        # the probability of going from d_star back to d_t is the probability of the corresponding add.
        # p(d*->d|add) = 1/((|a|-|d*|)(|d*|+1)) = 1/((|a|-|d|+1)(|d|))
        # p(d->d*|cut) = 1/|d|
        # Jratio =
        Jratio = Jratios[2] * (1. / float(len(d_t) - 1 - R_t + 1))
        R_star -= 1
    else:
        raise Exception
    # Now do the insertion-deletion
    d_star.insert(indx2, d_star.pop(indx1))
    return d_star, log(Jratio), R_star, step


def prior_calculations(lbda, maxlen, eta, maxlhs):
    '''Compute the normalization constants for the prior on rule cardinality
    '''
    # First normalization constants for beta
    beta_Z = poisson.cdf(maxlhs, eta) - poisson.pmf(0, eta)
    # Then the actual un-normalized pmfs
    logalpha_pmf = {}
    for i in range(maxlen + 1):
        try:
            logalpha_pmf[i] = poisson.logpmf(i, lbda)
        except RuntimeWarning:
            logalpha_pmf[i] = -inf
    logbeta_pmf = {}
    for i in range(1, maxlhs + 1):
        logbeta_pmf[i] = poisson.logpmf(i, eta)
    return beta_Z, logalpha_pmf, logbeta_pmf


def fn_logposterior(d_t, R_t, N_t, alpha, logalpha_pmf, logbeta_pmf, maxlhs, beta_Z, nruleslen, lhs_len):
    '''# Compute log posterior
    '''
    logliklihood = fn_logliklihood(d_t, N_t, R_t, alpha)
    logprior = fn_logprior(d_t, R_t, logalpha_pmf, logbeta_pmf, maxlhs, beta_Z, nruleslen, lhs_len)
    return logliklihood + logprior


def fn_logliklihood(d_t, N_t, R_t, alpha):
    '''Compute log likelihood
    '''
    gammaln_Nt_jk = gammaln(N_t + alpha)
    gammaln_Nt_j = gammaln(sum(N_t + alpha, 1))
    logliklihood = sum(gammaln_Nt_jk) - sum(gammaln_Nt_j)
    return logliklihood



def fn_logprior(d_t, R_t, logalpha_pmf, logbeta_pmf, maxlhs, beta_Z, nruleslen, lhs_len):
    '''# Compute log prior
    The prior will be _proportional_ to this -> we drop the normalization for alpha
    beta_Z is the normalization for beta, except the terms that need to be dropped due to running out of rules.
    log p(d_star) = log \alpha(m|lbda) + sum_{i=1...m} log beta(l_i | eta) + log gamma(r_i | l_i)
    The length of the list (m) is R_t
    Get logalpha (length of list) (overloaded notation in this code, unrelated to the prior hyperparameter alpha)
    '''
    logprior = 0.
    logalpha = logalpha_pmf[
        R_t]  # this is proportional to logalpha - we have dropped the normalization for truncating based on total number of rules
    logprior += logalpha
    empty_rulelens = []
    nlens = zeros(maxlhs + 1)
    for i in range(R_t):
        l_i = lhs_len[d_t[i]]
        logbeta = logbeta_pmf[l_i] - log(
            beta_Z - sum([logbeta_pmf[l_j] for l_j in empty_rulelens]))  # The correction for exhausted rule lengths
        # Finally loggamma
        loggamma = -log(nruleslen[l_i] - nlens[l_i])
        # And now check if we have exhausted all rules of a certain size
        nlens[l_i] += 1
        if nlens[l_i] == nruleslen[l_i]:
            empty_rulelens.append(l_i)
        elif nlens[l_i] > nruleslen[l_i]:
            raise Exception
        # Add 'em in
        logprior += logbeta
        logprior += loggamma
    # All done
    return logprior


def compute_rule_usage(d_star, R_star, X, Y):
    '''Compute which rules are being used to classify data points with what labels
    '''
    N_star = zeros((R_star + 1, Y.shape[1]))
    remaining_unused = set(range(Y.shape[0]))
    i = 0
    while remaining_unused:
        j = d_star[i]
        usedj = remaining_unused.intersection(X[j])
        remaining_unused = remaining_unused.difference(set(usedj))
        N_star[i, :] = Y[list(usedj), :].sum(0)
        i += 1
    if int(sum(N_star)) != Y.shape[0]:
        raise Exception  # bug check
    return N_star
