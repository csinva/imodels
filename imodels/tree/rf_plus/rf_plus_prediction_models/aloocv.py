# Generic Imports 
import copy, pprint, warnings, imodels
from abc import ABC, abstractmethod
from functools import partial
import time, numbers
#import numpy as np
import scipy as sp
import pandas as pd
from collections import OrderedDict
import autograd.numpy as np
from autograd import grad, hessian
from sklearn.utils import resample



# Sklearn Imports
from sklearn.linear_model import RidgeCV, Ridge, LogisticRegression, HuberRegressor, Lasso
from sklearn.metrics import  mean_squared_error

#scipy imports
from scipy.special import softmax
from scipy import linalg



def custom_MSE(y,y_pred,sample_weight):
    if sample_weight is None:
        return np.sum((y - y_pred)**2)/len(y)
    else:
        return np.sum(sample_weight * (y - y_pred)**2)/len(y)


class AloGLM():
    """
    Predictive Linear Model with approximate leave-one-out cross-validation

    Assumes estimator is a glmnet model
    """
    def __init__(self, estimator, n_alphas=100, l1_ratio=0.0, standardize = False, n_splits = 0,
                 inv_link_fn=lambda a: a, l_dot=lambda a, b: b - a, bootstrap = False,
                 l_doubledot=lambda a, b: 1, r_doubledot=lambda a: 1, random_state = None,
                 hyperparameter_scorer= custom_MSE):
        
        estimator.__init__()
        self.estimator = estimator 
        self.n_alphas = n_alphas
        self.inv_link_fn = inv_link_fn
        self.l_dot = l_dot
        self.l_doubledot = l_doubledot
        self.r_doubledot = r_doubledot
        self.hyperparameter_scorer = hyperparameter_scorer
        self._coeffs_for_each_alpha = {} #coefficients  for all reg params
        self._intercepts_for_each_alpha = {} #intercepts for all reg params
        self.alpha_ = {}
        self.loo_coefficients_ = None
        self.coef_ = None
        self._intercept_pred = None
        self.l1_ratio = l1_ratio
        self.influence_matrix_ = None
        self.support_idxs = None
        self.standardize = standardize
        self._svd = None
        self.bootstrap = bootstrap
        self.random_state = random_state    
        self.n_splits = n_splits
       

    def fit(self, X, y,sample_weight = None,max_h = 1 - 1e-5):
        y_train = copy.deepcopy(y)
        self.sample_weight = sample_weight  

        if self.sample_weight is None:
            self.sample_weight = np.ones(X.shape[0])
            self.sample_weight = (self.sample_weight/np.sum(self.sample_weight)) * len(self.sample_weight)
            self.estimator.fit(X, y_train)
        else:
            self.sample_weight = (self.sample_weight/np.sum(self.sample_weight)) * len(self.sample_weight)
            self.estimator.fit(X, y_train, sample_weight = self.sample_weight)

        
        for i,lambda_ in enumerate(self.estimator.lambda_path_):
            self._coeffs_for_each_alpha[lambda_] = self.estimator.coef_path_[:, i]
            self._intercepts_for_each_alpha[lambda_] = self.estimator.intercept_path_[i]
            
        #self._get_aloocv_alpha(X, y_train,evaluate_on,max_h)  
        if self.n_splits == 0:
            self._get_aloocv_alpha(X, y_train,max_h)  
        else:
            self.alpha_ = self.estimator.lambda_max_
            alpha_index = np.where(self.estimator.lambda_path_ == self.alpha_)
            self.cv_scores = self.estimator.cv_mean_score_[alpha_index]
            

        self.coef_ = self._coeffs_for_each_alpha[self.alpha_]
        self.intercept_ = self._intercepts_for_each_alpha[self.alpha_]

        self.loo_coefficients_,self.influence_matrix_ = self._get_loo_coefficients(X, y_train) #contains intercept
        self.support_idxs_ = np.where(self.coef_ != 0)[0]
         
    def _get_aloocv_alpha(self, X, y,max_h = 1 - 1e-5):
        #Assume we are solving 1/n l_i + lambda * r
        all_support_idxs = {}
        X1 = np.hstack([X, np.ones((X.shape[0], 1))])
        n = X1.shape[0]
        best_lambda_ = -1 #Assuming regularization parameter is always non-negative 
        best_cv_ = np.inf
        
        sample_weight = self.sample_weight  
 
        #get min of self.estimator.lambda_path
        min_lambda_ = np.min(self.estimator.lambda_path_)

        for i,lambda_ in enumerate(self.estimator.lambda_path_):
            orig_coef_ = self._coeffs_for_each_alpha[lambda_]
            orig_intercept_ = self._intercepts_for_each_alpha[lambda_]
            orig_coef_ = np.hstack([orig_coef_, orig_intercept_])

            support_idxs_lambda_ = orig_coef_ != 0
             
            X1_support = X1[:, support_idxs_lambda_]
            
            if tuple(support_idxs_lambda_) not in all_support_idxs:
                u,s,vh =  linalg.svd(X1_support,full_matrices=False)  
                us = u * s
                ush = us.T
                all_support_idxs[tuple(support_idxs_lambda_)] = s,us,ush
            else:
                s,us,ush = all_support_idxs[tuple(support_idxs_lambda_)]
                        
            orig_preds = self.inv_link_fn(X1 @ orig_coef_) 
            l_doubledot_vals = (self.l_doubledot(y, orig_preds) * sample_weight)/n 
            orig_coef_ = orig_coef_[np.array(support_idxs_lambda_)]
            
            reg_curvature = self._compute_reg_curvature(orig_coef_,lambda_,min_lambda_)
            regularized_diag_elements = self._compute_sigma2_lambda(s,l_doubledot_vals)
            Sigma2_plus_lambda = regularized_diag_elements + reg_curvature
            Sigma2_plus_lambda_inverse = 1.0/(Sigma2_plus_lambda)
            
            h_vals = np.einsum('ij,j,jk->i', us,Sigma2_plus_lambda_inverse,ush,optimize=True) * l_doubledot_vals
            h_vals[h_vals == 1] = max_h
            l_dot_vals = (self.l_dot(y, orig_preds) * sample_weight)/ n         
            loo_preds = orig_preds + h_vals * l_dot_vals / (1 - h_vals)

            if np.isnan(loo_preds).any():
                continue
            sample_scores = self.hyperparameter_scorer(y, loo_preds)

            if sample_scores < best_cv_:
                best_cv_ = sample_scores
                best_lambda_ = lambda_
                self.s,self.us,self.ush = s,us,ush
                self.support_idxs = support_idxs_lambda_
                
                
        self.alpha_ = best_lambda_
        self.cv_scores = best_cv_

    def _compute_sigma2_lambda(self,s,l_doubledot_vals):
        if  isinstance(l_doubledot_vals, float):
            diag_elements = s * l_doubledot_vals * s
        else:
            diag_elements = s * l_doubledot_vals[:len(s)] * s
        return diag_elements
        
    
    def _compute_reg_curvature(self,coef_,lambda_,min_lambda_):
        if self.r_doubledot is not None:
            r_doubledot_vals = self.r_doubledot(coef_) * np.ones_like(coef_)
            r_doubledot_vals[-1] = 0
            reg_curvature =  lambda_ * r_doubledot_vals
        else: 
            r_doubledot_vals = np.ones_like(coef_)
            r_doubledot_vals[-1] = 0
            reg_curvature =  lambda_ * r_doubledot_vals
        return reg_curvature
                                                                              
    
    def _get_loo_coefficients(self,X, y,max_h=1-1e-4):
        """
        Get the coefficient (and intercept) for each LOO model. Since we fit
        one model for each sample, this gives an ndarray of shape (n_samples,
        n_features + 1)
        """

        X1 = np.hstack([X, np.ones((X.shape[0], 1))])
        n = X.shape[0]

        if self.sample_weight is None:
            sample_weight = np.ones(X1.shape[0])
        else:
            sample_weight = self.sample_weight

        orig_coef_ = np.hstack([self.coef_, self.intercept_])
        orig_preds = self.inv_link_fn(X1 @ orig_coef_)#self.estimator.predict(X,self.alpha_)
        support_idxs = orig_coef_ != 0
        if not any(support_idxs):
            return orig_coef_ * np.ones_like(X1)
        X1_support = X1[:, support_idxs]
        orig_coef_ = orig_coef_[support_idxs]
        l_doubledot_vals = (self.l_doubledot(y, orig_preds)*sample_weight)/n
        J = X1_support.T * l_doubledot_vals @ X1_support
        if self.r_doubledot is not None:
            r_doubledot_vals = self.r_doubledot(orig_coef_) * np.ones_like(orig_coef_)
            r_doubledot_vals[-1] = 0 # Do not penalize constant term
            reg_curvature = np.diag(r_doubledot_vals)
            J += self.alpha_ * reg_curvature
        
        normal_eqn_mat = np.linalg.inv(J) @ X1_support.T
        
        h_vals = np.sum(X1_support.T * normal_eqn_mat, axis=0) * l_doubledot_vals/n
        h_vals[h_vals == 1] = max_h
        l_dot_vals = (self.l_dot(y, orig_preds)*sample_weight)/n
        influence_matrix = normal_eqn_mat * l_dot_vals / (1 - h_vals)

        loo_coef_ = orig_coef_[:, np.newaxis] + influence_matrix

        if not all(support_idxs):
            loo_coef_dense_ = np.zeros((X.shape[1] + 1, X.shape[0]))
            loo_coef_dense_[support_idxs, :] = loo_coef_
            loo_coef_ = loo_coef_dense_
        
        self.loo_preds =  np.sum(self.inv_link_fn(X1 *loo_coef_.T),axis=1)
        return loo_coef_.T,influence_matrix

    
    def predict(self, X):
        return self.inv_link_fn(X @ self.coef_ + self.intercept_)
       
    def predict_loo(self, X):
        X1 = np.hstack([X, np.ones((X.shape[0], 1))])
        return self.inv_link_fn(np.sum(X1 * self.loo_coefficients_,axis=1))
                   
    @property
    def intercept_pred(self):
        if self._intercept_pred is None:
            self._intercept_pred = np.array([self.inv_link_fn(self.coef_[-1])])
        return ("constant_model", self._intercept_pred)













# if indices is None:
#             X1 = np.hstack([X, np.ones((X.shape[0], 1))])
#             return np.sum(self.inv_link_fn(X1 * self.loo_coefficients_),axis=1) #returns X.shape[0] matrix 
#         else:
#             X1 = np.hstack([X, np.ones((X.shape[0], 1))])
#             return np.sum(self.inv_link_fn(X1 * self.loo_coefficients_[indices]),axis=1)






























# def elastic_net_loss(X,y,coef_,l1_ratio, alpha,sample_weight):
#     X1 = np.hstack([X, np.ones((X.shape[0], 1))])
#     n = X1.shape[0]
#     residuals = y - X1 @ coef_
#     loss = (0.5/n) * np.sum(sample_weight * (residuals ** 2))
#     l1 = alpha * l1_ratio * np.sum(np.abs(coef_))
#     l2 = 0.5*(1.0 - l1_ratio) * alpha * np.linalg.norm(coef_, ord=2)**2
#     return loss + l1 + l2
#     #np.sqrt(sample_weight)*np.sqrt(sample_weight)
    # def _get_loo_loss(self,X,y,coef_,sample_weight):
        
    #     X1 = np.hstack([X, np.ones((X.shape[0], 1))])
    #     n = X.shape[0]

        
    #     #compute gradient of loss function and hessian with respect to predicted values 
    #     orig_preds = self.inv_link_fn(X1 @ coef_)
    #     l_dot_vals = self.l_dot(y, orig_preds) * sample_weight/n 
    #     l_doubledot_vals = self.l_doubledot(y, orig_preds) * sample_weight/n 

    #     #Compute leverage scores
    #     #coef_ = np.ones(np.sum(self.support_idxs))
    #     coef_ = coef_[self.support_idxs]
    #     reg_curvature = self._compute_reg_curvature(coef_,self.alpha_,np.min(self.estimator.lambda_path_))
    #     regularized_diag_elements = self._compute_sigma2_lambda(self.s,l_doubledot_vals)
    #     Sigma2_plus_lambda = regularized_diag_elements + reg_curvature
    #     Sigma2_plus_lambda_inverse = 1.0/(Sigma2_plus_lambda)
    #     h_vals = np.einsum('ij,j,jk->i', self.us,Sigma2_plus_lambda_inverse,self.ush,optimize=True)
        
    #     h_vals = h_vals * l_doubledot_vals

    #     # #Compute LOO predictions and scores
    #     loo_preds = orig_preds + h_vals * l_dot_vals / (1 - h_vals)
    #     residuals = np.sum((y - loo_preds)**2)/len(y)
        
    #     return residuals

    # def _optimize_sample_weights(self, X, y):
        
    #     #compute hessians and gradients of loss function with respect to coefficients
    #     coefficients_with_intercept = np.hstack([self.coef_, self.intercept_])  
    #     initial_sample_weight = self.sample_weight  
        
    #     def get_loss_grad_coef(X,y,l1_ratio,alpha_,sample_weight_delta):
    #         return grad(lambda coef_: self.loss(X, y, coef_, l1_ratio, alpha_, sample_weight_delta))
        
    #     def get_loss_grad_sw(X,y,coef_,l1_ratio,alpha_):
    #         return grad(lambda sw: self.loss(X, y, coef_, l1_ratio, alpha_,sw))
        
    #     hess = hessian(lambda coef_: self.loss(X, y, coef_, self.l1_ratio, self.alpha_,initial_sample_weight))
    #     inv_hess = np.linalg.inv(hess(coefficients_with_intercept))
        
    #     coef_iter = coefficients_with_intercept
        
    
    #     sample_weight_delta = np.zeros_like(initial_sample_weight)
    #     sample_weight = initial_sample_weight

        # for iter in range(self.max_iter):
        #     random_sample_weight = np.random.multivariate_normal(initial_sample_weight, self.lr*np.eye(len(initial_sample_weight)))
        #     sample_weight_delta = random_sample_weight - initial_sample_weight
        #     coef_w = coefficients_with_intercept - inv_hess @ get_loss_grad_coef(X,y,self.l1_ratio,self.alpha_,sample_weight_delta)(coefficients_with_intercept)
        #     print(self._get_loo_loss(X,y,coef_w,random_sample_weight))

       
        # for iter in range(self.max_iter):

        #     loo_loss_grad_func = grad(lambda sw: self._get_loo_loss(X,y,coef_iter,sw,inv_hess))
        #     loo_grad_sw = loo_loss_grad_func(sample_weight)

        #     sample_weight_delta = sample_weight_delta - self.lr * loo_grad_sw
            # sample_weight = initial_sample_weight + sample_weight_delta
            # sample_weight = np.maximum(sample_weight,0) 
            # sample_weight = sample_weight/np.sum(sample_weight)
 
            # coef_iter = coef_iter - inv_hess @ get_loss_grad_coef(X,y,self.l1_ratio,self.alpha_,sample_weight_delta)(coefficients_with_intercept)
            # print(self._get_loo_loss(X,y,coef_iter,sample_weight))
        