import numpy as np
import pandas as pd
import numbers
from sklearn.ensemble import RandomForestClassifier
from RuleListClassifier import RuleListClassifier

class BigDataRuleListClassifier(RuleListClassifier):
    """
    A scikit-learn compatible wrapper for the Bayesian Rule List
    classifier by Benjamin Letham, adapted to work on large datasets. It 
    trains a linear SVM first, takes the subset of the training data closest
    to the decision boundary (specified by the parameter training_subset), 
    which is most critical to learning a classifier, and then uses this subset
    to learn a rule list. 

    It produces a highly interpretable model (a list of decision rules) of 
    the same form as an expert system. 

    Parameters
    ----------
    training_subset : float, optional (default=0.1)
        Determines the fraction of the data to use for training the Bayesian  
        Rule List classifier (the data points closest to a linear decision
        boundary are selected).
        
    subset_estimator: BaseEstimator, optional (default=RandomForestClassifier)
        An Estimator which is able to produce probabilities, used for finding
        the subset of the data which is closest to the decision boundary
    
    listlengthprior : int, optional (default=3)
        Prior hyperparameter for expected list length (excluding null rule)

    listwidthprior : int, optional (default=1)
        Prior hyperparameter for expected list length (excluding null rule)
        
    maxcardinality : int, optional (default=1)
        Maximum cardinality of an itemset
        
    minsupport : int, optional (default=10)
        Minimum support (%) of an itemset

    alpha : array_like, shape = [n_classes]
        prior hyperparameter for multinomial pseudocounts

    n_chains : int, optional (default=3)
        Number of MCMC chains for inference

    max_iter : int, optional (default=50000)
        Maximum number of iterations
        
    class1label: str, optional (default="class 1")
        Label or description of what the positive class (with y=1) means
        
    verbose: bool, optional (default=True)
        Verbose output
    """
    
    def __init__(self, training_subset=0.1, subset_estimator=RandomForestClassifier(), listlengthprior=3, listwidthprior=1, maxcardinality=2, minsupport=10, alpha = np.array([1.,1.]), n_chains=3, max_iter=50000, class1label="class 1", verbose=True):
        self.training_subset = training_subset
        self.subset_estimator = subset_estimator
        
        self.listlengthprior = listlengthprior
        self.listwidthprior = listwidthprior
        self.maxcardinality = maxcardinality
        self.minsupport = minsupport
        self.alpha = alpha
        self.n_chains = n_chains
        self.max_iter = max_iter
        self.class1label = class1label
        self.verbose = verbose
        self._zmin = 1
        
        self.thinning = 1 #The thinning rate
        self.burnin = self.max_iter//2 #the number of samples to drop as burn-in in-simulation
        
        self.discretizer = None
        self.d_star = None
        
    def _setdata(self, X, y, feature_labels=[], undiscretized_features = []):
        self._setlabels(X, feature_labels)
        
        for fi in range(len(X[0])):
            if not isinstance(X[0][fi], numbers.Number):
                raise Exception("Sorry, only numeric data is supported by BigDataRuleListClassifier at this time")
        
        Xn = np.array(X)
        # train subset estimator if necessary
        try:
            self.subset_estimator.predict_proba(Xn[0])
        except:
            self.subset_estimator.fit(X, y)
        # calculate distances from decision boundary for each point
        dist = np.abs(0.5-self.subset_estimator.predict_proba(Xn)[:, 1])
        ones_idx = np.where(y==1)[0]
        zeros_idx = np.where(y==0)[0]
        dist_ones = dist[ones_idx]
        dist_zeros = dist[zeros_idx]
        
        # take closest training_subset portion of data, preserving class imbalance
        if self.verbose:
            print("Reduced from", len(X))
        n = int(len(y)*self.training_subset)
        bestidx_ones = np.argsort(-dist_ones)
        bestidx_zeros = np.argsort(-dist_zeros)
        one_fraction = len(np.where(y==1)[0])/float(len(y))
        keep_idx = ones_idx[bestidx_ones[:(int(n*one_fraction)+1)]]
        keep_idx = np.hstack((keep_idx, zeros_idx[bestidx_zeros[:(int(n*(1-one_fraction))+1)]]))
        
        if type(X) == pd.DataFrame:
            X = X.iloc[keep_idx, :]
        else:
            X = np.array(X)[keep_idx, :]
        y = np.array(y)[keep_idx].astype(int)
        if self.verbose:
            print("...to", len(X), " data points")
            
        X = self._discretize_mixed_data(X, y, undiscretized_features)
        return X, y