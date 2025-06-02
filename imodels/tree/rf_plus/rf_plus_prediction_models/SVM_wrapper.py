#Generic Imports
import copy, pprint, warnings, imodels
from abc import ABC, abstractmethod
from functools import partial
import numpy as np
import scipy as sp
import pandas as pd


#sklearn imports
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.linear_model import SGDClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import log_loss, mean_squared_error, r2_score, roc_auc_score, log_loss


from scipy.special import expit, logit


class CustomSVMClassifier(LinearSVC):
    '''
    PPM Wrapper for SVM Classifier so that it is compataible with the PPM framework
    Only binary for now. 
    '''
    def __init__(self,n_alphas = 50,penalty="l2",loss="squared_hinge",*,dual="auto",tol=1e-4,multi_class="ovr",
        fit_intercept=True,intercept_scaling=1,class_weight='balanced',verbose=0,random_state=None,max_iter=1000):
        
        # initialize super class
        if loss != "squared_hinge":
            raise ValueError("Different losses currently not supported. Please use loss='squared_hinge'")
        
        self.n_alphas = n_alphas
        self.penalty = penalty
        self.loss = loss
        self.dual = dual
        self.tol = tol
        self.multi_class = multi_class
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling
        self.class_weight = class_weight
        self.verbose = verbose
        self.random_state = random_state
        self.max_iter = max_iter
        self.fitted_ = False
        if self.penalty == "l1":
            self.l1_ratio = 1
        elif self.penalty == "l2":
            self.l1_ratio = 0
        else:
            raise ValueError("penalty must be either l1 or l2")

        self.estimator = LinearSVC(penalty=penalty, loss=loss, dual=dual, tol=tol, multi_class=multi_class,
                                    fit_intercept=fit_intercept, intercept_scaling=intercept_scaling, class_weight=class_weight,
                                    verbose=verbose, random_state=random_state, max_iter=max_iter)
        
        self.lambda_path_ = np.logspace(-4, 3, self.n_alphas)
        self.coef_path_ = None
        self.intercept_path_ = None
        
    def fit(self, X, y,sample_weight=None):
        self.coef_path_ = np.zeros((1,X.shape[1],self.n_alphas))
        self.intercept_path_ = np.zeros((1,self.n_alphas))
        for i,lambda_ in enumerate(self.lambda_path_):
            self.estimator.set_params(C=1/lambda_)
            self.estimator.fit(X, y,sample_weight=sample_weight)
            self.coef_path_[0,:,i] = self.estimator.coef_[0,:]
            self.intercept_path_[0,i] = self.estimator.intercept_[0]
        
        self.fitted_ = True
        self.coef_ = self.estimator.coef_
        self.intercept_ = self.estimator.intercept_
        self.classes_ = self.estimator.classes_



class CustomSVMRegressor(LinearSVR):
    '''
    PPM Wrapper for SVM Classifier so that it is compataible with the PPM framework
    Only binary for now. 
    '''
    def __init__(self,n_alphas = 50,penalty="l2",loss="epsilon_insensitive",*,dual="auto",tol=1e-4,
        fit_intercept=True,intercept_scaling=1,verbose=0,random_state=None,max_iter=1000):
        
        # initialize super class
        if loss != "epsilon_insensitive":
            raise ValueError("Different losses currently not supported. Please use loss='squared_hinge'")
        
        self.n_alphas = n_alphas
        self.penalty = penalty
        self.loss = loss
        self.dual = dual
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling
        self.verbose = verbose
        self.random_state = random_state
        self.max_iter = max_iter
        self.fitted_ = False
        if self.penalty == "l1":
            self.l1_ratio = 1
        elif self.penalty == "l2":
            self.l1_ratio = 0
        else:
            raise ValueError("penalty must be either l1 or l2")

        self.estimator = LinearSVR(penalty=penalty, loss=loss, dual=dual, tol=tol,fit_intercept=fit_intercept, intercept_scaling=intercept_scaling,
                                    verbose=verbose, random_state=random_state, max_iter=max_iter)
        
        self.lambda_path_ = np.logspace(-4, 3, self.n_alphas)
        self.coef_path_ = None
        self.intercept_path_ = None
        
    def fit(self, X, y,sample_weight=None):
        self.coef_path_ = np.zeros((1,X.shape[1],self.n_alphas))
        self.intercept_path_ = np.zeros((1,self.n_alphas))
        for i,lambda_ in enumerate(self.lambda_path_):
            self.estimator.set_params(C=1/lambda_)
            self.estimator.fit(X, y,sample_weight=sample_weight)
            self.coef_path_[0,:,i] = self.estimator.coef_[0,:]
            self.intercept_path_[0,i] = self.estimator.intercept_[0]
        
        self.fitted_ = True
        self.coef_ = self.estimator.coef_
        self.intercept_ = self.estimator.intercept_
        self.classes_ = self.estimator.classes_

    

def derivative_squared_hinge_loss(y, preds):
    """
    Compute the derivative of the weighted squared hinge loss.
    
    Parameters:
    - y: np.array, true labels
    - preds: np.array, predictions
    - w: np.array, sample weights
    
    Returns:
    - np.array: derivative of the weighted squared hinge loss w.r.t. predictions
    """
    # Calculate the condition yf(x) < 1
    condition = y * preds < 1
    
    # Calculate the derivative based on the condition
    derivative = np.where(condition, y * (1 - y * preds), 0)
    
    return derivative
        

def second_derivative_squared_hinge_loss(y,preds):
    """
    Compute the second derivative of the weighted squared hinge loss.
    
    Parameters:
    - y: np.array, true labels
    - f_x: np.array, predictions
    - w: np.array, sample weights
    
    Returns:
    - np.array: second derivative of the weighted squared hinge loss w.r.t. predictions
    """
    # Calculate the condition yf(x) < 1
    condition = y * preds < 1
    
    # For squared hinge loss, the second derivative is the weight where the condition is true, and 0 otherwise
    second_derivative = np.where(condition, 1, 0)
    
    return second_derivative




if __name__ == "__main__":
    import warnings

    # Suppress specific warning
    warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
    X, y = make_classification(n_features=4, random_state=0)
    clf = CustomSVMClassifier()
    clf.fit(X, y)
    #print(clf.predict(X))
    #print(clf.decision_function(X))


    y = np.array([1, -1, 1])  # True labels
    preds = np.array([0.5, -0.2, 1.5])  # Predictions

    derivative = derivative_squared_hinge_loss(y, preds)
    print("Derivative:", derivative)




























































#class SGDClassifierPPM(SGDClassifier):
    #pass
    # def __init__(self, loss="hinge", penalty="elasticnet", n_alphas = 5, l1_ratio=0.15,
    #              fit_intercept=True, max_iter=1000, tol=1e-3, shuffle=True,
    #              verbose=0, epsilon=0.1, n_jobs=None, random_state=None,
    #              learning_rate="optimal", eta0=0.0, power_t=0.5, early_stopping=False,
    #              validation_fraction=0.1, n_iter_no_change=5, class_weight="balanced"):
        
    
    #     self.loss = loss
    #     self.penalty = penalty
    #     self.l1_ratio = l1_ratio
    #     self.fit_intercept = fit_intercept
    #     self.max_iter = max_iter
    #     self.tol = tol
    #     self.shuffle = shuffle
    #     self.verbose = verbose
    #     self.epsilon = epsilon
    #     self.n_jobs = n_jobs
    #     self.random_state = random_state
    #     self.learning_rate = learning_rate
    #     self.eta0 = eta0
    #     self.power_t = power_t
    #     self.early_stopping = early_stopping
    #     self.validation_fraction = validation_fraction
    #     self.n_iter_no_change = n_iter_no_change
    #     self.class_weight = class_weight
    #     self.n_alphas = n_alphas
    #     self.lambda_path_ = np.logspace(-3, 3, self.n_alphas)
    #     self.estimators_ = []
    #     for alpha in self.lambda_path_:
    #         self.estimators_.append(SGDClassifier(loss=loss, penalty=penalty, alpha=alpha, l1_ratio=l1_ratio,
    #                                               fit_intercept=fit_intercept, max_iter=max_iter, tol=tol, shuffle=shuffle,
    #                                               verbose=verbose, epsilon=epsilon, n_jobs=n_jobs, random_state=random_state,
    #                                               learning_rate=learning_rate, eta0=eta0, power_t=power_t, early_stopping=early_stopping,
    #                                               validation_fraction=validation_fraction, n_iter_no_change=n_iter_no_change, class_weight=class_weight))
    #     self.coef_path_ = None
    #     self.intercept_path_ = None
    
    # def fit(self, X, y):
    #     for estimator in self.estimators_:
    #         estimator.fit(X, y)
    #     return self
