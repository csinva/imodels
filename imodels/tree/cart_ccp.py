from sklearn.tree import DecisionTreeClassifier, export_text, DecisionTreeRegressor
from sklearn.base import BaseEstimator
from typing import List
import numpy as np
from imodels.util import checks
from sklearn import datasets
from sklearn.model_selection import cross_val_score,train_test_split
from sklearn.datasets import make_friedman1,make_friedman2,make_friedman3
from imodels.util.tree import compute_tree_complexity
from copy import deepcopy
import sys
sys.path.append(".")
from shrunk_tree import ShrunkTreeRegressor,ShrunkTreeClassifier

class DecisionTreeClassifierCCP(DecisionTreeClassifier):
    def __init__(self, estimator_: BaseEstimator, desired_complexity: int = 1,*args,**kwargs):
        self.desired_complexity = desired_complexity
        #print('est', estimator_)
        self.estimator_ = estimator_
    
    #def fit(self,X,y,sample_weight=None,*args, **kwargs):
    #    path = self.estimator_.cost_complexity_pruning_path(X,y)
    #    ccp_alphas, impurities = path.ccp_alphas, path.impurities
    #    complexities = {}
    #    for alpha in ccp_alphas: 
    #        est_params = self.estimator_.get_params()
    #        est_params['ccp_alpha'] = alpha
    ##        copied_estimator =  deepcopy(self.estimator_).set_params(**est_params)
     #       copied_estimator.fit(X, y)
     #       complexities[alpha] = self._get_complexity(copied_estimator)
     #   closest_alpha, closest_leaves = min(complexities.items(), key=lambda x: abs(self.desired_complexity - x[1]))
     #   params_for_fitting = self.estimator_.get_params()
     #   params_for_fitting['ccp_alpha'] = closest_alpha
     #   self.estimator_.set_params(**params_for_fitting)
     #   self.estimator_.fit(X,y,*args, **kwargs)
    
        
    def _get_alpha(self,X,y,sample_weight = None,*args,**kwargs):
        path = self.estimator_.cost_complexity_pruning_path(X,y)
        ccp_alphas, impurities = path.ccp_alphas, path.impurities
        complexities = {}
        for alpha in ccp_alphas: 
            est_params = self.estimator_.get_params()
            est_params['ccp_alpha'] = alpha
            copied_estimator =  deepcopy(self.estimator_).set_params(**est_params)
            copied_estimator.fit(X, y)
            complexities[alpha] = self._get_complexity(copied_estimator)
        closest_alpha, closest_leaves = min(complexities.items(), key=lambda x: abs(self.desired_complexity - x[1]))
        self.alpha = closest_alpha
    
    def fit(self,X,y,sample_weight=None,*args,**kwargs):
        params_for_fitting = self.estimator_.get_params()
        self._get_alpha(X,y,sample_weight,*args,**kwargs)
        params_for_fitting['ccp_alpha'] = self.alpha
        self.estimator_.set_params(**params_for_fitting)
        self.estimator_.fit(X,y,*args, **kwargs)
    
    
    def _get_complexity(self,BaseEstimator):
        return compute_tree_complexity(BaseEstimator.tree_)
    
    def predict_proba(self, *args, **kwargs):
        if hasattr(self.estimator_, 'predict_proba'):
            return self.estimator_.predict_proba(*args, **kwargs)
        else:
            return NotImplemented

    
    def predict(self,X,*args, **kwargs):
        return self.estimator_.predict(X,*args, **kwargs)
    
    def score(self, *args, **kwargs):
        if hasattr(self.estimator_, 'score'):
            return self.estimator_.score(*args, **kwargs)
        else:
            return NotImplemented
        
        
class DecisionTreeRegressorCCP(BaseEstimator):
    
    def __init__(self, estimator_: BaseEstimator, desired_complexity: int = 1,*args,**kwargs):
        self.desired_complexity = desired_complexity
        #print('est', estimator_)
        self.estimator_ = estimator_
        self.alpha = 0.0
        
    def _get_alpha(self,X,y,sample_weight = None,*args,**kwargs):
        path = self.estimator_.cost_complexity_pruning_path(X,y)
        ccp_alphas, impurities = path.ccp_alphas, path.impurities
        complexities = {}
        for alpha in ccp_alphas: 
            est_params = self.estimator_.get_params()
            est_params['ccp_alpha'] = alpha
            copied_estimator =  deepcopy(self.estimator_).set_params(**est_params)
            copied_estimator.fit(X, y)
            complexities[alpha] = self._get_complexity(copied_estimator)
        closest_alpha, closest_leaves = min(complexities.items(), key=lambda x: abs(self.desired_complexity - x[1]))
        self.alpha = closest_alpha
    
    def fit(self,X,y,sample_weight=None,*args,**kwargs):
        params_for_fitting = self.estimator_.get_params()
        self._get_alpha(X,y,sample_weight,*args,**kwargs)
        params_for_fitting['ccp_alpha'] = self.alpha
        self.estimator_.set_params(**params_for_fitting)
        self.estimator_.fit(X,y,*args, **kwargs)
        
    
    #def fit(self,X,y,sample_weight=None,*args, **kwargs):
    #    path = self.estimator_.cost_complexity_pruning_path(X,y)
    #    ccp_alphas, impurities = path.ccp_alphas, path.impurities
    #    complexities = {}
    #    for alpha in ccp_alphas: 
    #        est_params = self.estimator_.get_params()
    #        est_params['ccp_alpha'] = alpha
    #        copied_estimator =  deepcopy(self.estimator_).set_params(**est_params)
    #        copied_estimator.fit(X, y)
    #        complexities[alpha] = self._get_complexity(copied_estimator)
    #    closest_alpha, closest_leaves = min(complexities.items(), key=lambda x: abs(self.desired_complexity - x[1]))
    #    params_for_fitting = self.estimator_.get_params()
    #    params_for_fitting['ccp_alpha'] = closest_alpha
    #    self.estimator_.set_params(**params_for_fitting)
    #    self.estimator_.fit(X,y,*args, **kwargs)
    #    print(self.estimator_.score(X,y))
    #    m = ShrunkTreeRegressorCV(estimator_=self.estimator_,reg_param_list=[100.0])
    #    print(m.score(X,y))
    #    self.estimator_ = m.estimator_
    #def shrink_tree(self,)
        
    
    def _get_complexity(self,BaseEstimator):
        return compute_tree_complexity(BaseEstimator.tree_)
    
    def predict(self,X,*args, **kwargs):
        return self.estimator_.predict(X,*args, **kwargs)
    
    def score(self, *args, **kwargs):
        if hasattr(self.estimator_, 'score'):
            return self.estimator_.score(*args, **kwargs)
        else:
            return NotImplemented
        
        
class ShrunkDecisionTreeRegressorCCP_CV(ShrunkTreeRegressor):
    def __init__(self,estimator_:BaseEstimator,reg_param_list: List[float] = [0.1, 1, 10, 50, 100, 500],
                 desired_complexity:int = 1, cv: int = 3, scoring=None, *args, **kwargs):
        super().__init__(estimator_ = estimator_,reg_param = None)
        self.reg_param_list = np.array(reg_param_list)
        self.cv = cv
        self.scoring = scoring
        self.desired_complexity = desired_complexity
    
    def fit(self,X,y,sample_weight = None,*args,**kwargs):
        m = DecisionTreeRegressorCCP(self.estimator_,desired_complexity = self.desired_complexity)
        m.fit(X,y,sample_weight,*args,**kwargs)
        self.scores_ = []
        for reg_param in self.reg_param_list:
            est = ShrunkTreeRegressor(deepcopy(m.estimator_),reg_param)
            cv_scores = cross_val_score(est, X, y, cv=self.cv, scoring=self.scoring)
            self.scores_.append(np.mean(cv_scores))
        self.reg_param = self.reg_param_list[np.argmax(self.scores_)]
        super().fit(X=X,y=y)

class ShrunkDecisionTreeClassifierCCP_CV(ShrunkTreeClassifier):
    def __init__(self,estimator_:BaseEstimator,reg_param_list: List[float] = [0.1, 1, 10, 50, 100, 500],
                 desired_complexity:int = 1, cv: int = 3, scoring=None, *args, **kwargs):
        super().__init__(estimator_ = estimator_,reg_param = None)
        self.reg_param_list = np.array(reg_param_list)
        self.cv = cv
        self.scoring = scoring
        self.desired_complexity = desired_complexity
    
    def fit(self,X,y,sample_weight = None,*args,**kwargs):
        m = DecisionTreeClassifierCCP(self.estimator_,desired_complexity = self.desired_complexity)
        m.fit(X,y,sample_weight,*args,**kwargs)
        self.scores_ = []
        for reg_param in self.reg_param_list:
            est = ShrunkTreeClassifier(deepcopy(m.estimator_),reg_param)
            cv_scores = cross_val_score(est, X, y, cv=self.cv, scoring=self.scoring)
            self.scores_.append(np.mean(cv_scores))
        self.reg_param = self.reg_param_list[np.argmax(self.scores_)]
        super().fit(X=X,y=y)

        

if __name__ == '__main__':
    m = DecisionTreeClassifierCCP(estimator_=DecisionTreeClassifier(random_state = 1),desired_complexity = 10)
    #X,y = make_friedman1() #For regression 
    X, y = datasets.load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42)
    m.fit(X_train,y_train)
    m.predict(X_test)
    print(m.score(X_test,y_test))
    
    m = ShrunkDecisionTreeClassifierCCP_CV(estimator_=DecisionTreeClassifier(random_state=1),desired_complexity = 10,reg_param_list = [0.0,0.1,1.0,5.0,10.0,25.0,50.0,100.0])
    m.fit(X_train,y_train)
    print(m.score(X_test,y_test))


