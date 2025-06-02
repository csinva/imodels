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
from scipy.special import softmax
from scipy import linalg

# Sklearn Imports
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score 
from sklearn.linear_model import RidgeCV, HuberRegressor
from sklearn import preprocessing
from sklearn.utils import resample
from sklearn.ensemble._forest import _generate_unsampled_indices, _generate_sample_indices
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

#Glmnet Imports
from glmnet import ElasticNet

#imodels imports
from imodels.tree.rf_plus.rf_plus_prediction_models.aloocv import AloGLM
from imodels.tree.rf_plus.rf_plus_prediction_models.SVM_wrapper import CustomSVMClassifier, derivative_squared_hinge_loss, second_derivative_squared_hinge_loss

#l0 package
import l0learn

class AloGLMRegressor(AloGLM,ABC):
    """
    Alo Regressor
    """
    pass


class AloElasticNetRegressorCV(AloGLMRegressor):
    def __init__(self, n_alphas=100, l1_ratio = [0.0,0.99],standardize = False, random_state = None, n_splits = 0,
                 bootstrap = False,n_bootstraps = 10,hyperparameter_scorer = mean_squared_error, **kwargs):
        
        self.n_alphas = n_alphas
        self.l_dot = lambda a, b: b - a
        self.l_doubledot = lambda a, b: 1
        self.l1_ratio_list = l1_ratio
        self.inv_link_fn= lambda a: a
        self.r_doubledot = None
        self.hyperparameter_scorer=hyperparameter_scorer
        self.l1_ratio = None 
        self.standardize = standardize
        self.estimator_ = None
        self.bootstrap = bootstrap
        self.n_bootstraps = n_bootstraps
        if self.bootstrap:
            self._boostrap_estimators = []
            self._bootstrap_coeffs = []
        else:
            self._bootstrap_coeffs = None
            self._bootstrap_intercepts = None
        self.random_state = None
        self.n_splits = n_splits

    def fit(self, X, y,sample_weight = None):

        best_cv_scores = np.inf

        for l1_ratio in self.l1_ratio_list:
            model = AloGLMRegressor(ElasticNet(n_lambda=self.n_alphas,standardize=self.standardize,n_splits=self.n_splits,
                        alpha=l1_ratio,n_jobs=-1), standardize= self.standardize,inv_link_fn= lambda a: a, n_splits=self.n_splits,
                        l_dot= lambda a, b: b - a, l1_ratio=l1_ratio,l_doubledot= lambda a, b: 1, r_doubledot=lambda a: 1.0 - l1_ratio, 
                        hyperparameter_scorer= self.hyperparameter_scorer)
            model.fit(X,y,sample_weight = sample_weight)
            
            if model.cv_scores < best_cv_scores:
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
        self.loo_preds = self.estimator.loo_preds
        self.support_idxs_ = self.estimator.support_idxs_
        self.alpha_ = self.estimator.alpha_
        self._coeffs_for_each_alpha = self.estimator._coeffs_for_each_alpha
        self._intercepts_for_each_alpha = self.estimator._intercepts_for_each_alpha
   
     
class AloLOL2Regressor(AloGLMRegressor):
    """
    PPM class for LOL2
    """

    def __init__(self, n_alphas=50, max_support_size = 0.025,hyperparameter_scorer = mean_squared_error,penalty = "L0L2", scaler = False,**kwargs):
        
        self.estimator = None
        self.inv_link_fn = lambda a: a
        self.l_dot = lambda a, b: b - a
        self.l_doubledot = lambda a, b: 1
        self.r_doubledot = lambda a: 1
        self.n_alphas = n_alphas
        self.max_support_size = max_support_size
        self._coeffs_for_each_alpha = {} #coefficients  for all reg params
        self._intercepts_for_each_alpha = {} #intercepts for all reg params
        self.penalty = penalty
        self.hyperparameter_scorer = hyperparameter_scorer
        #self.scaler = scaler
    
    def fit(self,X,y,sample_weight = None,max_h = 1-1e-5):
        n = X.shape[0]  
        p = X.shape[1]
        y_train = copy.deepcopy(y)
        self.sample_weight = sample_weight
        
       
        repeated_X = np.repeat(X, self.sample_weight.astype(int), axis=0)
        repeated_y = np.repeat(y_train, self.sample_weight.astype(int), axis=0)
       
        max_support_size = max(int(self.max_support_size*min(n,p)),p)
        self.estimator = l0learn.fit(repeated_X, repeated_y, penalty=self.penalty, num_gamma = self.n_alphas,max_support_size=max_support_size)

        l0_coeff =  np.asarray(self.estimator.coeff().todense())
        
        l0_penalties =  self.estimator.characteristics().to_dict()['l0']
        l2_penalties =  self.estimator.characteristics().to_dict()['l2']

        for i in self.estimator.characteristics().to_dict()['l0'].keys():
            self._coeffs_for_each_alpha[(l0_penalties[i],l2_penalties[i])] = l0_coeff[1:,i]
            self._intercepts_for_each_alpha[(l0_penalties[i],l2_penalties[i])] = l0_coeff[0,i]
    
        self._get_aloocv_alpha(X,y,max_h)  
        # self.coefficients_ = self._coeffs_for_each_alpha[(self.l0_penalty,self.alpha_)]
        self.coef_ = self._coeffs_for_each_alpha[(self.l0_penalty,self.alpha_)]
        self.intercept_ = self._intercepts_for_each_alpha[(self.l0_penalty,self.alpha_)]
        
       
        self.loo_coefficients_,self.influence_matrix_ = self._get_loo_coefficients(X, y) #contains intercept
        # self.support_idxs_ = np.where(self.coefficients_ != 0)[0]
        self.support_idxs_ = np.where(self.coef_ != 0)[0]
        print(f"L0L2 Support Indices: {len(self.support_idxs_),len(self.support_idxs_)/X.shape[1]}")
        print(self.predict_loo(X).shape)
        

    def predict(self, X):
        # X1 = np.hstack([X, np.ones((X.shape[0], 1))])
        # loo_preds = self.inv_link_fn(np.dot(X1, self.loo_coefficients_.T))
        # loo_preds = np.mean(loo_preds, axis=1)
        # return loo_preds
        # return self.inv_link_fn(X@self.coefficients_ + self.intercept_)
        return self.inv_link_fn(X@self.coef_ + self.intercept_)
        #return np.mean(self.predict_loo(X))
        #np.dot(X, )
    
    

    def _get_aloocv_alpha(self, X, y,max_h = 1 - 1e-5):
        #Assume we are solving 1/n l_i + lambda * r
       
        all_support_idxs = {}
        X1 = np.hstack([X, np.ones((X.shape[0], 1))])
        n = X1.shape[0]
        best_lambda_ = -1 #Assuming regularization parameter is always non-negative 
        best_cv_ = np.inf
        
        sample_weight = self.sample_weight  
 
        min_lambda_ = 1e-6

        for i,penalties in enumerate(self._coeffs_for_each_alpha.keys()):
            
            l0_penalty,lambda_ = penalties[0],penalties[1]
            
            orig_coef_ = self._coeffs_for_each_alpha[penalties]
            orig_intercept_ = self._intercepts_for_each_alpha[penalties]
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
                        
            orig_preds = self.inv_link_fn(X1_support @ orig_coef_[support_idxs_lambda_])
            
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
            
            

            sample_scores = self.hyperparameter_scorer(y, loo_preds)

            if sample_scores < best_cv_:
                best_cv_ = sample_scores
                best_lambda_ = lambda_
                best_l0_penalty = l0_penalty
                self.s,self.us,self.ush = s,us,ush
                self.support_idxs = support_idxs_lambda_
                
       
        self.alpha_ = best_lambda_
        self.l0_penalty = best_l0_penalty
        self.loo_scores = best_cv_


def huber_loss(y, preds, epsilon=1.35):
    """
    Evaluates Huber loss function.

    Parameters
    ----------
    y: array-like of shape (n,)
        Vector of observed responses.
    preds: array-like of shape (n,)
        Vector of estimated/predicted responses.
    epsilon: float
        Threshold, determining transition between squared
        and absolute loss in Huber loss function.

    Returns
    -------
    Scalar value, quantifying the Huber loss. Lower loss
    indicates better fit.

    """
    total_loss = 0
    for i in range(len(y)):
        sample_absolute_error = np.abs(y[i] - preds[i])
        if sample_absolute_error < epsilon:
            total_loss += 0.5 * ((y[i] - preds[i]) ** 2)
        else:
            sample_robust_loss = epsilon * sample_absolute_error - 0.5 * \
                                 epsilon ** 2
            total_loss += sample_robust_loss
    return total_loss / len(y)
        

if __name__ == "__main__":
    

    X,y,f = imodels.get_clean_dataset("diabetes_regr",data_source="imodels")
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    pprint.pprint(f"X_train shape: {X_train.shape}")

    l0_model = AloLOL2Regressor()
    l0_model.fit(X_train, y_train)

    l0_preds = l0_model.predict(X_test)

    l2_model = AloElasticNetRegressorCV()
    l2_model.fit(X_train, y_train)

    l2_preds = l2_model.predict(X_test)


    pprint.pprint(f"l0 r2 score: {r2_score(y_test, l0_preds)}")

    pprint.pprint(f"l2 r2 score: {r2_score(y_test, l2_preds)}")




 

















































# def _fit_bootstrap(self,X,y,n_bootstraps = 100):
#     train_indices = self._generate_bootstrap_samples(X,y,n_bootstraps)
#     for i in range(n_bootstraps):
#         train_indices_i = train_indices[i]
#         X_bootstrap =  X
#         y_bootstrap = y
#         unique_train_indices, counts_elements = np.unique(train_indices_i, return_counts=True)    
        
#         sample_weight_i = np.ones(len(y)) * 1e-12
#         sample_weight_i[unique_train_indices] = counts_elements
#         sample_weight_i = (sample_weight_i/np.sum(sample_weight_i))*len(y)

#         model = ElasticNet(lambda_path=np.array([self.alpha_]),standardize=self.standardize,n_splits=0,alpha=self.l1_ratio)
#         model.fit(X_bootstrap,y_bootstrap,sample_weight = sample_weight_i)
#         self._bootstrap_coeffs[i,:-1] = model.coef_path_[:, 0]
#         self._bootstrap_coeffs[i,-1] = model.intercept_path_[0]


# def _generate_bootstrap_samples(self,X,y,n_bootstraps):
#     n_samples = len(y)
#     train_indices = []
#     for i in range(n_bootstraps):
#         train_indices_i = _generate_sample_indices(self.random_state, n_samples, n_samples).astype(int) 
#         train_indices.append(train_indices_i)
        
#     return train_indices

  # if self.bootstrap:
        #     self._bootstrap_coeffs = np.zeros((self.n_bootstraps,len(self.coefficients_)+1))
        #     self._fit_bootstrap(X,y,self.n_bootstraps)


# else:
#     model = AloGLMRegressor(ElasticNet(n_lambda=self.n_alphas,standardize=self.standardize,n_splits=0,alpha=l1_ratio,lambda_path = self.lambda_path), standardize= self.standardize,
#             inv_link_fn= lambda a: a, l_dot= lambda a, b: b - a, l1_ratio=l1_ratio, 
#             l_doubledot= lambda a, b: 1, r_doubledot=lambda a: 1.0 - l1_ratio,
#             hyperparameter_scorer= self.hyperparameter_scorer)
            
        
# class AloGLMRidgeRegressor(AloGLMRegressor):
#     """
#     PPM class for Elastic
#     """

#     def __init__(self,n_alphas=100, l1_ratio = 0.0,standardize = False, **kwargs):
        
#         super().__init__(ElasticNet(n_lambda=n_alphas,standardize=standardize,n_splits=0,alpha=0.0,**kwargs),
#                         inv_link_fn=lambda a: a, l_dot=lambda a, b: b - a, l1_ratio=l1_ratio,
#                         l_doubledot=lambda a, b: 1, r_doubledot=lambda a: 1,
#                         hyperparameter_scorer=mean_squared_error)





# class AloElasticNetRegressor(AloGLMRegressor):
#     """
#     PPM class for Elastic
#     """

#     def __init__(self, n_alphas=100, l1_ratio = 0.5, standardize = False,hyperparameter_scorer = mean_squared_error, **kwargs):

#         super().__init__(ElasticNet(n_lambda=n_alphas,standardize=standardize,n_splits=0,alpha=l1_ratio,**kwargs), standardize= standardize,
#                         inv_link_fn=lambda a: a, l_dot=lambda a, b: b - a, l1_ratio=l1_ratio,
#                         l_doubledot=lambda a, b: 1, r_doubledot=lambda a: 1 - l1_ratio,
#                         hyperparameter_scorer=hyperparameter_scorer)



# class AloLassoRegressor(AloGLMRegressor):
#     """
#     Ppm class for regression that uses lasso as the GLM estimator.
#     """
#     def __init__(self, n_alphas=50, standardize = False,hyperparameter_scorer = mean_squared_error, **kwargs):
#         super().__init__(ElasticNet(n_lambda=n_alphas, n_splits=0, alpha = 1.0,standardize = standardize, **kwargs), 
#                         inv_link_fn=lambda a: a, l_dot=lambda a, b: b - a, l1_ratio = 1.0,
#                         l_doubledot=lambda a, b: 1, r_doubledot=None,
#                         hyperparameter_scorer=hyperparameter_scorer)
    



# class AloRidgeRegressor(AloGLMRegressor):
#     """
#     Ppm class for regression that uses ridge as the GLM estimator.
#     Uses fast scikit-learn LOO implementation
#     """
#     def __init__(self, n_alphas=100, standardize = False,alpha_grid = None,start_alpha = -5, stop_alpha = 5,sample_weight = None,**kwargs):
        
        
#         self.estimator = RidgeCV(**kwargs)
#         self.inv_link_fn = lambda a: a
#         self.l_dot = lambda a, b: b - a
#         self.l_doubledot = lambda a, b: 1
#         self.r_doubledot = lambda a: 1
#         self.start_alpha = start_alpha
#         self.stop_alpha = stop_alpha
#         self.n_alphas = n_alphas
#         if alpha_grid is None:
#             self.alpha_grid = np.logspace(self.start_alpha, self.stop_alpha, self.n_alphas)
#         else:
#             self.alpha_grid = alpha_grid
        

    
#     def fit(self, X, y,sample_weight = None):  

        
#         self.estimator.set_params(alphas = self.alpha_grid)
        
#         y_train = copy.deepcopy(y)
#         if sample_weight is None:
#             self.sample_weight = np.ones_like(y_train)/(2 * len(y_train)) #for consistency with glmnet
#         else:
#             self.sample_weight = sample_weight/(2 * len(y_train))
        
#         self.estimator.fit(X, y_train,sample_weight = self.sample_weight)
#         self.coefficients_ = self.estimator.coef_
#         self.intercept_ = self.estimator.intercept_        
#         self.alpha_ = self.estimator.alpha_
#         self.best_alpha_preds = self.estimator.predict(X)
#         self.loo_coefficients_,self.influence_matrix_ = self._get_loo_coefficients(X, y_train)
        
