# Generic Imports 
import copy, pprint, warnings, imodels
from abc import ABC, abstractmethod
from functools import partial
import time, numbers
import numpy as np
import scipy as sp
import pandas as pd
from collections import OrderedDict


#scipy imports
from scipy.special import softmax, expit
from scipy import linalg

# Sklearn Imports
from sklearn.metrics import log_loss, roc_auc_score, average_precision_score, f1_score, accuracy_score

#Glmnet Imports
from glmnet import LogitNet

#imodels imports
from imodels.tree.rf_plus.rf_plus_prediction_models.aloocv import AloGLM
from imodels.tree.rf_plus.rf_plus_prediction_models.SVM_wrapper import CustomSVMClassifier, derivative_squared_hinge_loss, second_derivative_squared_hinge_loss

def neg_roc_auc_score(y_true, y_pred):
    return -roc_auc_score(y_true, y_pred)

def neg_avg_precision_score(y_true, y_pred):
    return -average_precision_score(y_true, y_pred)

class AloGLMClassifier(AloGLM):
    """
    Only can deal with binary for now. 
    """
    
    def __init__(self, estimator, n_alphas=100, l1_ratio=0.0, standardize = False,
                 inv_link_fn=lambda a: a, l_dot=lambda a, b: b - a, n_splits = 0,
                 l_doubledot=lambda a, b: 1, r_doubledot=lambda a: 1,
                 hyperparameter_scorer=neg_roc_auc_score, class_weight = 'balanced'):
        
        estimator.__init__()
        self.estimator = estimator
        self.n_alphas = n_alphas
        self.inv_link_fn = inv_link_fn
        self.l_dot = l_dot
        self.l_doubledot = l_doubledot
        self.r_doubledot = r_doubledot
        self.hyperparameter_scorer = hyperparameter_scorer
        self._coeffs_for_each_alpha = {}
        self._intercepts_for_each_alpha = {}
        self.alpha_ = {}
        self.loo_coefficients_ = None
        # self.coefficients_ = None
        self.coef_ = None
        self._intercept_pred = None
        self.l1_ratio = l1_ratio
        self.influence_matrix_ = None
        self.support_idxs = None
        self.standardize = standardize
        self._svd = None
        self.class_weight = class_weight
        self.n_splits = n_splits

    def fit(self, X, y,sample_weight = None,max_h = 1 - 1e-5):

        y_train = copy.deepcopy(y)
        self.sample_weight = sample_weight
        
        
        if self.sample_weight is None:
            self.sample_weight = np.ones(len(y_train))

        if self.class_weight is None:
            self.sample_weight = np.ones(len(y_train))
        else:
            if self.class_weight == 'balanced':
                n_pos = np.sum(y_train)
                n_neg = len(y_train) - n_pos
                weight_pos = (1 / n_pos) * (n_pos + n_neg) / 2.0
                weight_neg = (1 / n_neg) * (n_pos + n_neg) / 2.0
                self.sample_weight = np.where(y_train == 1, weight_pos, weight_neg) * self.sample_weight
                self.sample_weight = (self.sample_weight / np.sum(self.sample_weight))*len(y_train)
            else:
                assert len(self.sample_weight) == len(y_train), "Sample weights must be the same length as the target"
        
        self.sample_weight = (self.sample_weight/np.sum(self.sample_weight)) * len(self.sample_weight)
        
        self.estimator.fit(X, y,sample_weight = self.sample_weight)   

        for i,lambda_ in enumerate(self.estimator.lambda_path_):
            self._coeffs_for_each_alpha[lambda_] = self.estimator.coef_path_[0,:, i]
            self._intercepts_for_each_alpha[lambda_] = self.estimator.intercept_path_[0,i]


        #fit the model on the training set and compute the coefficients
        if self.n_splits > 0:
            self._get_aloocv_alpha(X, y_train,max_h,sample_weight = self.sample_weight)
        else:
            self.alpha_ = self.estimator.lambda_max_
            alpha_index = np.where(self.estimator.lambda_path_ == self.alpha_)
            self.cv_scores = self.estimator.cv_mean_score_[alpha_index]
        
        # self.coefficients_ = self._coeffs_for_each_alpha[lambda_]
        self.coef_ = self._coeffs_for_each_alpha[lambda_]
        self.intercept_ = self._intercepts_for_each_alpha[lambda_]
        
        self.loo_coefficients_,self.influence_matrix_ = self._get_loo_coefficients(X, y_train) #contains intercept
        # self.support_idxs_ = np.where(self.coefficients_ != 0)[0]
        self.support_idxs_ = np.where(self.coef_ != 0)[0]
             
    def predict(self, X, threshold=0.5):
        # preds = self.inv_link_fn(X@self.coefficients_ + self.intercept_)
        preds = self.inv_link_fn(X@self.coef_ + self.intercept_)
        return np.where(preds > threshold, 1, 0)
       
    def predict_proba(self, X):
        # preds =  self.inv_link_fn(X@self.coefficients_ + self.intercept_)
        preds =  self.inv_link_fn(X@self.coef_ + self.intercept_)
        return np.stack([1 - preds, preds]).T
    
    def predict_proba_loo(self, X,indices = None):
        if indices is None:
            indices = np.arange(X.shape[0])
        X1 = np.hstack([X, np.ones((X.shape[0], 1))])
        loo_prob_preds = self.inv_link_fn(np.sum(X1[indices,:] * self.loo_coefficients_[indices,:],axis=1))
        return np.stack([1 - loo_prob_preds, loo_prob_preds]).T
    
    def predict_loo(self, X, threshold=0.5):
        proba_preds = self.predict_proba_loo(X)[:,1]
        return np.where(proba_preds > threshold, 1, 0)

    
class AloLogisticElasticNetClassifierCV(AloGLMClassifier):
    """
    PPM class for Logistic Regression with Elastic Net Penalty
    """

    def __init__(self, n_alphas=50, l1_ratio=[0.0,0.99], standardize=False, n_splits = 0,inv_link_fn= expit,
                l_dot=lambda a, b: b - a, l_doubledot=lambda a, b: b * (1-b), r_doubledot=lambda a: 1, 
                 hyperparameter_scorer=neg_roc_auc_score, class_weight='balanced'):
        
        
        self.n_alphas = n_alphas
        self.l1_ratio = l1_ratio
        self.standardize = standardize
        self.inv_link_fn = inv_link_fn
        self.l_dot = l_dot
        self.l_doubledot = l_doubledot
        self.r_doubledot = r_doubledot
        self.hyperparameter_scorer = hyperparameter_scorer
        self.class_weight = class_weight
        self.n_splits = n_splits

    def fit(self,X,y,sample_weight = None):
        """
        Fit the model to the data
        """
        best_cv_scores = np.inf
        self.sample_weight = sample_weight  

        for l1_ratio in self.l1_ratio:
            
            model = AloGLMClassifier(estimator=LogitNet(n_lambda=self.n_alphas, alpha = l1_ratio, standardize = self.standardize,n_splits = self.n_splits), 
                                        standardize= self.standardize,inv_link_fn= self.inv_link_fn, l_dot= self.l_dot, n_splits=self.n_splits,l1_ratio=l1_ratio,
                                        l_doubledot= self.l_doubledot, r_doubledot=lambda a: 1.0 - l1_ratio, hyperparameter_scorer= self.hyperparameter_scorer)
            
                
            model.fit(X = X, y = y,sample_weight = sample_weight)

            if model.cv_scores <  best_cv_scores:
                best_cv_scores = model.cv_scores
                self.l1_ratio = l1_ratio
                self.estimator = model
                self.cv_scores = best_cv_scores
        
        self.r_doubledot = lambda a: 1.0 - self.l1_ratio
        # self.coefficients_ = self.estimator.coefficients_
        self.coef_ = self.estimator.coef_
        self.intercept_ = self.estimator.intercept_
        self.loo_coefficients_ = self.estimator.loo_coefficients_
        self.influence_matrix_ = self.estimator.influence_matrix_
        self.support_idxs_ = self.estimator.support_idxs_
        self.alpha_ = self.estimator.alpha_
        self._coeffs_for_each_alpha = self.estimator._coeffs_for_each_alpha
        self._intercepts_for_each_alpha = self.estimator._intercepts_for_each_alpha


class AloSVCRidgeClassifier(AloGLMClassifier):
    """
    SVM Classifier
    """

    def __init__(self, estimator = CustomSVMClassifier, n_alphas=20, standardize=False,
                  inv_link_fn=expit, l_dot=derivative_squared_hinge_loss, dual="auto",
                 l_doubledot=second_derivative_squared_hinge_loss, r_doubledot=lambda a: 1,
                 hyperparameter_scorer=log_loss, class_weight='balanced'):
        
        super().__init__(CustomSVMClassifier(n_alphas=n_alphas,dual = dual), n_alphas, 0, standardize,
                        inv_link_fn, l_dot, l_doubledot, r_doubledot, hyperparameter_scorer, class_weight)
        
   
if __name__ == "__main__":
    
    warnings.filterwarnings("ignore")
    from sklearn.model_selection import train_test_split

    X,y,f = imodels.get_clean_dataset("enhancer",data_source="imodels")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)


    model = AloLogisticElasticNetClassifierCV()
    model.fit(X_train, y_train)
    probability_preds = model.predict_proba(X_test)[:,1]
    preds = model.predict(X_test)

    pprint.pprint(f"probability_preds: {probability_preds}")   
    pprint.pprint(f"preds: {preds}")    
    pprint.pprint(f"AUC Score {roc_auc_score(y_test,model.predict_proba(X_test)[:,1])}")

